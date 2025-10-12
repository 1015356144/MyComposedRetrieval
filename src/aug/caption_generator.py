# aug/caption_generator.py
import os
import time
import json
import shutil
import traceback
import torch
import torch.distributed as dist
from src.utils import print_rank
from .validators import CaptionValidator
from .batchers import CaptionBatcher
from src.utils.path_utils import get_full_image_path  # ✅ NEW: 统一路径解析


class CaptionGenerator:
    """
    分布式 / 单卡 caption 增强生成器
    - 逻辑保持与原始大文件一致，仅拆分为模块化实现
    - ✅ 新增：按 CIRR 的 image_base_dir + image_splits 规范化样本路径
    """

    def __init__(
        self,
        foundation_model,
        model_args,
        experiment_dir,
        iteration_round,
        prepare_fns,
        generate_fns,
        # ===== NEW: 为路径规范化提供上下文 =====
        image_base_dir: str = "",
        image_splits: dict | None = None,
    ):
        """
        Args:
            foundation_model: 底层生成模型（带 .processor 与 .generate）
            model_args: 含 foundation_model_backbone 等字段
            experiment_dir: 实验目录（用于缓存 / 同步落盘）
            iteration_round: 当前迭代轮次（生成写入下一轮编号的文件）
            prepare_fns: dict, {"qwen": fn, "llava": fn, "generic": fn}
            generate_fns: dict, {"qwen": fn, "llava": fn, "generic": fn}
            image_base_dir: 数据集图片根目录（如 /path/to/CIRR）
            image_splits: 映射 {样本ID或相对键 -> 相对路径}，用于把键转换为路径
        """
        self.foundation_model = foundation_model
        self.model_args = model_args
        self.experiment_dir = experiment_dir
        self.iteration_round = iteration_round
        self.augmented_samples = []

        # 依赖 batchers 和 validators
        self.batcher = CaptionBatcher(foundation_model, model_args, prepare_fns, generate_fns)
        self.validator = CaptionValidator()

        # ===== NEW: 路径上下文 =====
        self.image_base_dir = image_base_dir or ""
        self.image_splits = image_splits or {}

    # ===== NEW: 统一把 {ref/target/hn} 三类路径规范到绝对路径 =====
    def _resolve_image_path(self, p: str) -> str:
        """
        - 若 p 是绝对路径：原样返回
        - 若 p 是 splits 的键：先映射为相对路径，再拼 base_dir
        - 否则：按相对路径处理并拼 base_dir
        """
        if not isinstance(p, str) or p == "":
            return p
        if os.path.isabs(p):
            return p
        mapped = self.image_splits.get(p, p)
        return get_full_image_path(mapped, self.image_base_dir)

    def _normalize_item_paths(self, item: dict) -> dict:
        out = dict(item)
        if "reference_image" in out:
            out["reference_image"] = self._resolve_image_path(out["reference_image"])
        if "target_image" in out and out["target_image"]:
            out["target_image"] = self._resolve_image_path(out["target_image"])
        if "hard_negative_image" in out and out["hard_negative_image"]:
            out["hard_negative_image"] = self._resolve_image_path(out["hard_negative_image"])
        return out

    # =========================
    #         单卡逻辑
    # =========================
    def generate_augmented_captions(self, hard_negatives):
        """单卡增强 caption 生成"""
        if not self.foundation_model:
            print_rank("No foundation model, skipping caption generation")
            return []

        next_iter = self.iteration_round + 1
        aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iter}.json")

        # 读缓存
        if os.path.exists(aug_file):
            print_rank(f"Loading existing augmented samples from {aug_file}")
            try:
                with open(aug_file, "r") as f:
                    saved = json.load(f)
                samples = saved.get("samples", [])
                samples = self.validator.filter_valid_samples(samples)
                self.augmented_samples = samples
                return samples
            except Exception as e:
                print_rank(f"Error loading augmented samples: {e}, regenerating...")

        print_rank(f"Generating augmented captions for {len(hard_negatives)} hard negatives")
        augmented_samples = []
        batch_size = 4
        total_batches = (len(hard_negatives) + batch_size - 1) // batch_size
        start_time = time.time()

        for i in range(0, len(hard_negatives), batch_size):
            batch_idx = i // batch_size + 1
            batch = hard_negatives[i:i+batch_size]
            # ETA
            if batch_idx > 1:
                elapsed = time.time() - start_time
                avg_tpb = elapsed / (batch_idx - 1)
                remain = total_batches - batch_idx + 1
                eta = f"ETA {int((avg_tpb*remain)//60):02d}:{int((avg_tpb*remain)%60):02d}"
            else:
                eta = "ETA calculating..."

            print_rank(f"Processing batch {batch_idx}/{total_batches} ({len(batch)}) - {eta}")
            try:
                batch_start = time.time()
                batch_aug = self._generate_caption_batch(batch)
                augmented_samples.extend(batch_aug)
                print_rank(f"Batch {batch_idx}/{total_batches} done in {time.time()-batch_start:.1f}s, +{len(batch_aug)}")
            except Exception as e:
                print_rank(f"Error in batch {batch_idx}: {e}")
                continue

        augmented_samples = self.validator.filter_valid_samples(augmented_samples)
        self._save_augmented_samples(augmented_samples)
        total_time = time.time() - start_time
        print_rank(f"✅ Generated {len(augmented_samples)} samples in {total_time:.1f}s")
        return augmented_samples

    def _generate_caption_batch(self, hard_negatives_batch):
        """批量生成 caption（与单卡/多卡复用）"""
        from PIL import Image

        augmented = []
        foundation_processor = getattr(self.foundation_model, "processor", None)
        if foundation_processor is None:
            print_rank("Foundation model has no processor")
            return []

        device = next(self.foundation_model.parameters()).device
        ref_images, tgt_images, texts, meta = [], [], [], []

        # 打包
        for hard_neg in hard_negatives_batch:
            try:
                norm_item = self._normalize_item_paths(hard_neg)

                ref_path = norm_item["reference_image"]
                tgt_path = norm_item.get("hard_negative_image", norm_item.get("target_image"))

                ref_img = Image.open(ref_path).convert("RGB")
                tgt_img = Image.open(tgt_path).convert("RGB")

                ref_images.append(ref_img)
                tgt_images.append(tgt_img)
                texts.append(norm_item["modification_text"])
                meta.append(norm_item)  # ✅ 后续写回时使用规范化后的绝对路径
            except Exception as e:
                print_rank(f"Error preparing sample: {e}")

        # 生成
        if ref_images:
            generated_texts = self.batcher.generate_batch(
                ref_images, tgt_images, texts, foundation_processor, device
            )
            for hard_neg, gen_text in zip(meta, generated_texts):
                if gen_text and self.validator.is_valid(gen_text):
                    augmented.append({
                        "reference_image": hard_neg["reference_image"],
                        "modification_text": gen_text,
                        "target_image": hard_neg.get("hard_negative_image", hard_neg.get("target_image")),
                        "original_mod_text": hard_neg["modification_text"],
                        "is_augmented": True,
                        "hard_negative_rank": hard_neg.get("rank_position"),
                        "similarity_score": hard_neg.get("similarity_score")
                    })
        return augmented

    # =========================
    #        分布式逻辑
    # =========================
    def generate_augmented_captions_distributed(self, hard_negatives):
        """
        多卡分布式 caption 生成（文件式同步与聚合）
        完整复刻你原始实现的行为，但模块化并复用验证/保存函数。
        """
        if not self.foundation_model:
            print_rank("No foundation model provided, skipping caption generation")
            return []

        next_iter = self.iteration_round + 1
        final_aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iter}.json")

        # 读缓存：所有 rank 都尝试直接读
        if os.path.exists(final_aug_file):
            print_rank(f"Loading existing augmented samples from {final_aug_file}")
            try:
                with open(final_aug_file, "r") as f:
                    saved = json.load(f)
                samples = self.validator.filter_valid_samples(saved.get("samples", []))
                self.augmented_samples = samples
                return samples
            except Exception as e:
                print_rank(f"Error loading augmented samples: {e}, regenerating...")

        # 无分布式则退化为单卡
        if not dist.is_initialized():
            return self.generate_augmented_captions(hard_negatives)

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print_rank(f"Starting distributed caption generation for {len(hard_negatives)} samples using {world_size} GPUs")

        # 任务切分
        total = len(hard_negatives)
        per_gpu = (total + world_size - 1) // world_size
        start_idx = rank * per_gpu
        end_idx = min(start_idx + per_gpu, total)
        local_list = hard_negatives[start_idx:end_idx]
        print_rank(f"GPU {rank}: Assigned {len(local_list)} samples [{start_idx}, {end_idx})")

        # 设备放置
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        if hasattr(self.foundation_model, "to"):
            self.foundation_model = self.foundation_model.to(device)

        # 本地生成
        local_aug = []
        if local_list:
            batch_size = 4
            total_batches = (len(local_list) + batch_size - 1) // batch_size
            start_time = time.time()
            for i in range(0, len(local_list), batch_size):
                bidx = i // batch_size + 1
                batch = local_list[i:i+batch_size]

                # 降低输出噪声：rank0 全打，其他每 5 个批次打一次
                if bidx % 5 == 1 or rank == 0:
                    if bidx > 1:
                        elapsed = time.time() - start_time
                        avg_tpb = elapsed / (bidx - 1)
                        remain = total_batches - bidx + 1
                        eta = f"ETA {int((avg_tpb*remain)//60):02d}:{int((avg_tpb*remain)%60):02d}"
                    else:
                        eta = "ETA calculating..."
                    print_rank(f"GPU {rank}: 🔄 Batch {bidx}/{total_batches} ({len(batch)}) - {eta}")

                try:
                    t0 = time.time()
                    batch_aug = self._generate_caption_batch(batch)
                    local_aug.extend(batch_aug)
                    if bidx % 5 == 0 or rank == 0 or bidx == total_batches:
                        print_rank(f"GPU {rank}: ✅ Batch {bidx}/{total_batches} in {time.time()-t0:.1f}s, +{len(batch_aug)}")
                except Exception as e:
                    print_rank(f"❌ GPU {rank}: Error in batch {bidx}: {e}")
                    print_rank(traceback.format_exc())
        else:
            print_rank(f"GPU {rank}: No local samples, skip generation")

        print_rank(f"GPU {rank}: 🎯 Local generation done: {len(local_aug)} samples")

        # ============ 文件式同步：阶段 1（完成标记） ============
        sync_dir = os.path.join(self.experiment_dir, "sync_caption_gen")
        if rank == 0:
            os.makedirs(sync_dir, exist_ok=True)
            print_rank(f"GPU 0: Created sync dir {sync_dir}")

        _wait_dir(sync_dir, rank, max_wait_s=36000)
        completion_flag = os.path.join(sync_dir, f"gpu_{rank}_completed.txt")
        try:
            with open(completion_flag, "w") as f:
                f.write(f"GPU {rank} completed {len(local_aug)} samples at {time.time()}")
            print_rank(f"GPU {rank}: Wrote completion flag: {completion_flag}")
        except Exception as e:
            print_rank(f"GPU {rank}: Error writing completion flag: {e}")
            os.makedirs(sync_dir, exist_ok=True)
            with open(completion_flag, "w") as f:
                f.write(f"GPU {rank} completed {len(local_aug)} samples at {time.time()}")
            print_rank(f"GPU {rank}: Retried flag ok")

        # 等待所有 rank 完成
        print_rank(f"GPU {rank}: Waiting all completion flags")
        _wait_all_flags(sync_dir, world_size, rank, max_wait_s=36000)

        # ============ 文件式同步：阶段 2（各自写结果） ============
        tmp_dir = os.path.join(self.experiment_dir, "temp_caption_results")
        if rank == 0:
            os.makedirs(tmp_dir, exist_ok=True)
            print_rank(f"GPU 0: Created tmp dir {tmp_dir}")
        _wait_dir(tmp_dir, rank, max_wait_s=36000)

        # 本地写文件
        local_file = os.path.join(tmp_dir, f"gpu_{rank}_samples.json")
        try:
            with open(local_file, "w") as f:
                json.dump({
                    "rank": rank,
                    "samples": local_aug,
                    "count": len(local_aug),
                    "timestamp": time.time()
                }, f, indent=2)
            print_rank(f"GPU {rank}: Saved {len(local_aug)} to {local_file}")
        except Exception as e:
            print_rank(f"GPU {rank}: Error saving local file: {e}")

        # 等待所有本地文件 ready
        print_rank(f"GPU {rank}: Waiting all gpu files")
        _wait_all_files(tmp_dir, world_size, rank, max_wait_s=36000)

        # ============ 主进程聚合 ============
        if rank == 0:
            all_aug = [local_aug]  # include rank0
            for r in range(1, world_size):
                fp = os.path.join(tmp_dir, f"gpu_{r}_samples.json")
                try:
                    if os.path.exists(fp):
                        with open(fp, "r") as f:
                            data = json.load(f)
                        samples = data.get("samples", [])
                        all_aug.append(samples)
                        print_rank(f"GPU 0: Loaded {len(samples)} from GPU {r}")
                    else:
                        print_rank(f"GPU 0: Missing file for GPU {r}, append []")
                        all_aug.append([])
                except Exception as e:
                    print_rank(f"GPU 0: Error reading GPU {r} file: {e}")
                    all_aug.append([])

            merged = []
            for chunk in all_aug:
                if chunk and isinstance(chunk, list):
                    merged.extend(chunk)
                    print_rank(f"GPU 0: merge chunk +{len(chunk)}")

            # 过滤无效
            print_rank(f"GPU 0: Filtering {len(merged)} samples")
            merged = self.validator.filter_valid_samples(merged)

            # 保存最终文件
            self._save_augmented_samples(merged)
            print_rank(f"✅ GPU 0: Saved {len(merged)} merged samples")

            # 清理
            try:
                shutil.rmtree(tmp_dir)
                print_rank(f"GPU 0: Cleaned tmp dir {tmp_dir}")
                if os.path.exists(sync_dir):
                    shutil.rmtree(sync_dir)
                    print_rank(f"GPU 0: Cleaned sync dir {sync_dir}")
            except Exception as e:
                print_rank(f"GPU 0: Cleanup warning: {e}")

        # ============ 全部 rank 等待最终文件 ============
        print_rank(f"GPU {rank}: Waiting final file")
        _wait_file(final_aug_file, rank, max_wait_s=36000)

        # 全部 rank 读最终文件（保持一致）
        final_aug = []
        if os.path.exists(final_aug_file):
            try:
                with open(final_aug_file, "r") as f:
                    saved = json.load(f)
                final_aug = self.validator.filter_valid_samples(saved.get("samples", []))
                print_rank(f"GPU {rank}: Final loaded {len(final_aug)} samples")
            except Exception as e:
                print_rank(f"GPU {rank}: Error loading final file: {e}")
        else:
            print_rank(f"GPU {rank}: Final file not found")

        self.augmented_samples = final_aug
        print_rank(f"GPU {rank}: 🎯 Distributed caption generation completed: {len(final_aug)}")
        return final_aug

    # =========================
    #        公共方法
    # =========================
    def _save_augmented_samples(self, samples):
        """保存增强样本（写下一轮编号）"""
        next_iter = self.iteration_round + 1
        out_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iter}.json")
        # 附带基础统计
        summary = {
            "total_samples": len(samples),
            "generation_timestamp": time.time(),
            "iteration_round": next_iter,
            "sample_statistics": {
                "avg_original_length": (sum(len(s.get("original_mod_text", "")) for s in samples) / len(samples)) if samples else 0,
                "avg_generated_length": (sum(len(s.get("modification_text", "")) for s in samples) / len(samples)) if samples else 0,
                "unique_reference_images": len(set(s.get("reference_image", "") for s in samples)),
                "unique_target_images": len(set(s.get("target_image", "") for s in samples)),
            },
            "samples": samples
        }
        # 落盘并强制 flush
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print_rank(f"✅ Saved {len(samples)} samples to {out_file}")


# --------------------------
# 辅助函数（分布式轮询）
# --------------------------
def _wait_dir(path, rank, max_wait_s=36000):
    """等待目录出现（文件轮询），避免 barrier 触发 NCCL 看门狗"""
    waited = 0
    while not os.path.exists(path) and waited < max_wait_s:
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            print_rank(f"GPU {rank}: Waiting dir {path}... {waited}s")
    if not os.path.exists(path):
        print_rank(f"GPU {rank}: ❌ Dir wait timeout, try make locally: {path}")
        os.makedirs(path, exist_ok=True)


def _wait_all_flags(sync_dir, world_size, rank, max_wait_s=36000):
    """等待所有 GPU 的完成标记"""
    start = time.time()
    while time.time() - start < max_wait_s:
        all_ok = True
        for r in range(world_size):
            if not os.path.exists(os.path.join(sync_dir, f"gpu_{r}_completed.txt")):
                all_ok = False
                break
        if all_ok:
            print_rank(f"GPU {rank}: ✅ All completion flags ready")
            return
        time.sleep(5)
        elapsed = int(time.time() - start)
        if elapsed % 120 == 0:
            done = [r for r in range(world_size)
                    if os.path.exists(os.path.join(sync_dir, f"gpu_{r}_completed.txt"))]
            print_rank(f"GPU {rank}: Waiting flags... done={done}, elapsed={elapsed}s")
    print_rank(f"GPU {rank}: ❌ Timeout waiting completion flags")


def _wait_all_files(tmp_dir, world_size, rank, max_wait_s=36000):
    """等待所有 GPU 写出临时结果文件"""
    start = time.time()
    while time.time() - start < max_wait_s:
        all_ok = True
        missing = []
        for r in range(world_size):
            fp = os.path.join(tmp_dir, f"gpu_{r}_samples.json")
            if not os.path.exists(fp):
                all_ok = False
                missing.append(r)
        if all_ok:
            print_rank(f"GPU {rank}: ✅ All GPU files ready")
            return
        time.sleep(2)
        elapsed = int(time.time() - start)
        if elapsed % 60 == 0 and elapsed > 0:
            print_rank(f"GPU {rank}: Waiting files... missing={missing} elapsed={elapsed}s")
    print_rank(f"GPU {rank}: ❌ Timeout waiting gpu files")


def _wait_file(path, rank, max_wait_s=36000):
    """等待最终合并文件"""
    start = time.time()
    while time.time() - start < max_wait_s:
        if os.path.exists(path):
            print_rank(f"GPU {rank}: ✅ Final file ready")
            return
        time.sleep(2)
        elapsed = int(time.time() - start)
        if elapsed % 10 == 0 and elapsed > 0:
            print_rank(f"GPU {rank}: Waiting final file... {elapsed}s")
    print_rank(f"GPU {rank}: ❌ Timeout waiting final file {path}")
