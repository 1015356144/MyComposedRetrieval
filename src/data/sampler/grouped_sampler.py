import random
from collections import defaultdict
from typing import List, Dict, Iterator, Optional

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

        # 1) 分组（一次性）
        groups: Dict[str, List[int]] = defaultdict(list)
        for i in range(len(self.dataset)):
            try:
                ex = self.dataset[i]
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

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        # 触发重新构建
        self._cached_batches = None
        self._cached_len = None

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
                    # 在当前批内部重复补齐
                    rnd = random.Random(self.seed + self.epoch + 2024)
                    while len(cur) < self.batch_size:
                        cur.append(rnd.choice(cur))
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
                if len(b) < self.batch_size and len(b) > 0:
                    c = b[:]
                    while len(c) < self.batch_size:
                        c.append(rnd.choice(c))
                    fixed.append(c)
                elif len(b) > self.batch_size:
                    # 理论上不会发生（因为我们控制了<=batch_size），但以防万一截断
                    fixed.append(b[: self.batch_size])
                else:
                    fixed.append(b)
            my_batches = fixed

        self._cached_batches = my_batches
        self._cached_len = len(my_batches)

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
