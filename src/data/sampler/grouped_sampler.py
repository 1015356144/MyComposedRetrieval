import random
from collections import defaultdict
from typing import List, Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


class DistributedGroupedBatchSampler(Sampler[List[int]]):
    """
    DDP-friendly 分组打包采样器：
      - 先按 reference_image 分组
      - 全局洗牌 + 打包成 batch（每个 batch 的样本数 <= batch_size）
      - 按 rank 对 batch 列表进行均匀切片（i % world_size == rank）
      - 可选：将不足 batch_size 的 batch 进行“就地重复”补齐到定长
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        pad_to_full: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        ref_key: str = "reference_image",
        no_ref_bucket: str = "__no_ref__",
        # 调试参数（可选）
        debug: bool = False,
        debug_max_batches: int = 5,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.pad_to_full = pad_to_full
        self.drop_last = drop_last
        self.seed = int(seed)
        self.ref_key = ref_key
        self.no_ref_bucket = no_ref_bucket
        # 新增：调试参数
        self.debug = bool(debug)
        self.debug_max_batches = int(debug_max_batches)

        # DDP info
        if world_size is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.epoch = 0  # will be set via set_epoch()

        # 全局索引池：用于“去重式补齐”从全局挑选未在当前批中的样本
        self.all_indices: List[int] = list(range(len(self.dataset)))

        # 新增：按“样本键签名”建立兼容性桶，避免补进不兼容样本导致 KeyError（如缺少 caption）
        self.schema_buckets: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        self.index_signature: Dict[int, Tuple[str, ...]] = {}
        # 新增：记录索引是否增广
        self.index_is_augmented: Dict[int, bool] = {}

        # 1) 分组（一次性）
        groups: Dict[str, List[int]] = defaultdict(list)
        for i in range(len(self.dataset)):
            try:
                ex = self.dataset[i]
                # 记录样本的键签名
                sig = self._signature_of_example(ex)
                if sig is not None:
                    self.index_signature[i] = sig
                    self.schema_buckets[sig].append(i)
                # 记录是否增广
                self.index_is_augmented[i] = bool(ex.get("is_augmented", False))

                ref = ex.get(self.ref_key)
                if ref:
                    groups[ref].append(i)
                else:
                    groups[self.no_ref_bucket].append(i)
            except Exception as e:
                # 静默容错：跳过坏样本
                # 也可以 log warning
                continue

        self.groups: List[List[int]] = list(groups.values())
        self.num_groups = len(self.groups)

        # 2) 预缓存一个“本 epoch 的 my_batches”（在 __iter__ 首次构建）
        self._cached_batches = None
        self._cached_len = None

        # 调试：分组与样本概览（仅 rank0 打印）
        if self.debug and (not dist.is_initialized() or self.rank == 0):
            total_aug = sum(1 for v in self.index_is_augmented.values() if v)
            total = len(self.all_indices)
            both_cnt = 0
            only_aug = 0
            only_orig = 0
            for g in self.groups:
                has_aug = any(self.index_is_augmented.get(x, False) for x in g)
                has_orig = any(not self.index_is_augmented.get(x, False) for x in g)
                if has_aug and has_orig:
                    both_cnt += 1
                elif has_aug:
                    only_aug += 1
                elif has_orig:
                    only_orig += 1
            print(
                f"[Sampler][rank{self.rank}] groups_built: num_groups={self.num_groups}, total={total}, "
                f"augmented={total_aug}; group_types: both={both_cnt}, only_aug={only_aug}, only_orig={only_orig}"
            )

        # 调试：补齐计数器
        self._debug_pad_same_sig = 0
        self._debug_pad_global = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        # 触发重新构建
        self._cached_batches = None
        self._cached_len = None
        # 重置调试计数器
        self._debug_pad_same_sig = 0
        self._debug_pad_global = 0

    # 新增：根据样本字典的键集合创建“签名”，用于匹配同结构的数据
    def _signature_of_example(self, ex) -> Optional[Tuple[str, ...]]:
        try:
            if isinstance(ex, dict):
                keys = [k for k in ex.keys() if isinstance(k, str)]
                return tuple(sorted(keys))
        except Exception:
            pass
        return None

    # 新增：批内“去重式补齐”，优先从同“键签名”的索引中补足，其次再全局，最后回退允许重复
    def _pad_unique(self, base: List[int], target_size: int, rnd: random.Random) -> List[int]:
        if len(base) >= target_size:
            return base[:target_size]
        if not base:
            # 极端情况：没有基样本，退化为全局填充
            batch = []
            exclude = set()
            candidates_pool = self.all_indices
        else:
            batch = list(base)
            exclude = set(batch)
            # 以首个样本的签名作为本批“兼容结构”
            sig = self.index_signature.get(batch[0])
            if sig is not None and sig in self.schema_buckets:
                candidates_pool = self.schema_buckets[sig]
            else:
                candidates_pool = self.all_indices

        need = target_size - len(batch)
        # 先在同签名池中找不重复候选
        candidates = [i for i in candidates_pool if i not in exclude]
        take = min(need, len(candidates))
        if take > 0:
            batch.extend(rnd.sample(candidates, take))
            need -= take
            exclude.update(batch)
            if candidates_pool is self.all_indices:
                self._debug_pad_global += 1
            else:
                self._debug_pad_same_sig += 1

        # 若仍不足，尝试在全局池中找不重复候选
        if need > 0:
            global_cands = [i for i in self.all_indices if i not in exclude]
            take2 = min(need, len(global_cands))
            if take2 > 0:
                batch.extend(rnd.sample(global_cands, take2))
                need -= take2
                self._debug_pad_global += 1

        # 若仍不足（小数据集或 batch_size 超大），回退允许重复以保证定长
        while len(batch) < target_size:
            pool = candidates_pool if candidates_pool else (batch if batch else self.all_indices)
            batch.append(rnd.choice(pool))
        return batch

    def _build_all_batches(self):
        """
        基于当前 epoch 构建全局 batch 列表，然后切片到本 rank。
        """
        # 洗牌 groups 的顺序（全局一致）
        group_indices = list(range(self.num_groups))
        if self.shuffle:
            rnd = random.Random(self.seed + self.epoch)
            rnd.shuffle(group_indices)

        # 打包：<= batch_size
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_sz = 0
        for gi in group_indices:
            g = self.groups[gi]
            gsz = len(g)

            # 如果加入会超标且当前已有样本，则先收一个 batch
            if cur_sz > 0 and (cur_sz + gsz) > self.batch_size:
                batches.append(cur)
                cur = []
                cur_sz = 0

            # 加入当前组
            cur.extend(g)
            cur_sz += gsz

            # 刚好等于也收一个 batch
            if cur_sz == self.batch_size:
                batches.append(cur)
                cur = []
                cur_sz = 0

        # 末尾残留
        if cur:
            # 不足的情况：根据配置 pad 或 drop
            if cur_sz < self.batch_size:
                if self.drop_last:
                    pass  # 丢弃
                elif self.pad_to_full and cur_sz > 0:
                    # 去重式补齐（优先同结构再全局）
                    rnd = random.Random(self.seed + self.epoch + 2024)
                    cur = self._pad_unique(cur, self.batch_size, rnd)
                    batches.append(cur)
                else:
                    # 保持不满（不推荐用于 DDP 对比学习）
                    batches.append(cur)
            else:
                batches.append(cur)

        # === 关键：按 rank 均匀切片 ===
        my_batches = [b for i, b in enumerate(batches) if (i % self.world_size) == self.rank]

        # 如果不 pad_to_full 又不 drop_last，可能导致不同 rank 的 batch 长度仍不一致
        # 为 DDP 稳定，强烈建议 pad_to_full=True
        if self.pad_to_full:
            # 双保险：确保每个 batch 都等长
            fixed = []
            rnd = random.Random(self.seed + self.epoch + 4096 + self.rank)
            for b in my_batches:
                if 0 < len(b) < self.batch_size:
                    c = self._pad_unique(b, self.batch_size, rnd)
                    fixed.append(c)
                elif len(b) > self.batch_size:
                    # 理论上不会发生（因为我们控制了<=batch_size），但以防万一截断
                    fixed.append(b[: self.batch_size])
                else:
                    fixed.append(b)
            my_batches = fixed

        self._cached_batches = my_batches
        self._cached_len = len(my_batches)

        # 调试：本 rank 批次中增广分布与补齐来源（仅 rank0 打印）
        if self.debug and (not dist.is_initialized() or self.rank == 0):
            try:
                num_batches = len(my_batches)
                batches_with_aug = 0
                total_aug_items = 0
                preview_lines = []
                for bi, b in enumerate(my_batches[: self.debug_max_batches]):
                    aug_flags = [self.index_is_augmented.get(x, False) for x in b]
                    aug_count = sum(1 for f in aug_flags if f)
                    total_aug_items += aug_count
                    if aug_count > 0:
                        batches_with_aug += 1
                    preview_lines.append(
                        f"  batch[{bi}] size={len(b)}, aug={aug_count}, ids_sample={b[:min(3, len(b))]}"
                    )
                if num_batches > self.debug_max_batches:
                    for b in my_batches[self.debug_max_batches:]:
                        if any(self.index_is_augmented.get(x, False) for x in b):
                            batches_with_aug += 1
                print(
                    f"[Sampler][rank{self.rank}][epoch{self.epoch}] my_batches={num_batches}, "
                    f"batches_with_aug={batches_with_aug}, total_aug_items_in_preview={total_aug_items}, "
                    f"pad_same_sig_calls={self._debug_pad_same_sig}, pad_global_calls={self._debug_pad_global}\n" +
                    "\n".join(preview_lines)
                )
            except Exception:
                pass

    def __iter__(self) -> Iterator[List[int]]:
        if self._cached_batches is None:
            self._build_all_batches()
        for b in self._cached_batches:
            yield b

    def __len__(self) -> int:
        if self._cached_len is None:
            # 需要构建一次以获得精确长度
            self._build_all_batches()
        return self._cached_len
