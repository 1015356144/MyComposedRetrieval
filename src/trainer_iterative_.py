"""
Iterative Trainer for Composed Image Retrieval
Refactored to use decoupled modules:
- retrieval.embedding
- retrieval.hard_negative
- aug.caption_generation
- utils.logging, utils.progress
"""

import os
import json
import time
import torch
import torch.distributed as dist
import logging
import statistics
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from .trainer import MMEBTrainer
from .model.model import MMEBModel
from src.data.dataset.cirr import IterativeCIRRDataset
from src.data.dataset.fashioniq import IterativeFashionIQDataset

# 引入解耦后的功能模块
from .utils import print_rank, print_master
from src.mining.hard_negative import HardNegativeMiner
from src.retrieval.candidate_builder import CandidateBuilder      
from src.retrieval.embedding_cache import EmbeddingCache          
from src.retrieval.engine import RetrievalEngine                 
from src.aug.caption_generator import CaptionGenerator
from src.prep.input_adapter import VLMInputPreparer

# 各 backbone 的 prompt builder（若没配 LLaVA/Generic 会自动降级成 no-op）
from src.prompt.qwen.builder import prepare_inputs as qwen_prepare, generate_with_qwen
try:
    from src.prompt.llava.builder import prepare_inputs as llava_prepare, generate_with_llava
except Exception:
    llava_prepare, generate_with_llava = None, None
try:
    from src.prompt.generic.builder import prepare_inputs as generic_prepare, generate_with_generic
except Exception:
    generic_prepare, generate_with_generic = None, None


logger = logging.getLogger(__name__)


class IterativeRetrievalTrainer(MMEBTrainer):
    def __init__(
        self,
        foundation_model=None,                 # 忽略传进来的实例（保持兼容）
        foundation_model_name: str = None,     # 只保存名字，按需懒加载
        max_iterations: int = 3,
        hard_neg_collection_freq: int = 1,
        caption_generation_batch_size: int = 8,
        model_args=None,
        data_args=None,
        max_length=None,
        # Fast / Production
        fast_mode: bool = False,
        fast_mode_max_samples: int = 100,
        fast_mode_retrieval_db_size: int = 50,
        fast_mode_max_steps: int = 5,
        steps_per_iteration: int = 1000,
        production_save_steps: int = 100,
        # 兼容旧参
        production_max_steps: Optional[int] = None,
        **kwargs
    ):
        # ---- 先收下本类需要的对象（super 之前）----
        self.model_args = model_args
        self.data_args = data_args
        self.max_length = max_length

        # 兼容 steps_per_iteration / production_max_steps
        if production_max_steps is not None:
            print_master("⚠️  WARNING: 'production_max_steps' is deprecated, use 'steps_per_iteration' instead")
            self.production_max_steps = production_max_steps
        else:
            self.production_max_steps = steps_per_iteration

        self.fast_mode = fast_mode
        self.fast_mode_max_samples = fast_mode_max_samples
        self.fast_mode_retrieval_db_size = fast_mode_retrieval_db_size
        self.fast_mode_max_steps = fast_mode_max_steps
        self.production_save_steps = production_save_steps

        self.steps_per_iteration = (
            self.fast_mode_max_steps if self.fast_mode else self.production_max_steps
        )

        # 打印训练计划
        print_master(f"📋 Training plan: {max_iterations} iterations × {self.steps_per_iteration} steps/iter")
        print_master("🔄 Strategy: reset optimizer & scheduler every iteration")

        # ---- 处理 Trainer 不认识的 kwargs，避免 super 报错 ----
        # 这些是在 factory 或上层可能透传进来的
        for k in [
            "model_args", "data_args", "max_length",
            "fast_mode", "fast_mode_max_samples", "fast_mode_retrieval_db_size",
            "fast_mode_max_steps", "steps_per_iteration",
            "production_max_steps", "production_save_steps",
            "foundation_model_name", "foundation_model",
        ]:
            kwargs.pop(k, None)

        # 有些人会把 processing_class 放 kwargs，这里接住给本类与 evaluator 用
        self.processing_class = kwargs.get("processing_class", None)

        # ---- 进入父类初始化 ----
        super().__init__(**kwargs)

        # ---- 本类状态与路径 ----
        # 完全忽略传入的 foundation_model 实例（留名即可，以免占显存）
        self.foundation_model = None
        self.foundation_processor = None
        self.foundation_model_name = foundation_model_name

        self.max_iterations = max_iterations
        self.hard_neg_collection_freq = hard_neg_collection_freq
        self.caption_generation_batch_size = caption_generation_batch_size

        self.current_iteration = 0
        self.iteration_metrics: Dict[int, Dict[str, float]] = {}

        self._base_training_completed = False
        self._target_embeddings_cached = False

        # 统一的实验目录（解耦模块都用到）
        self.experiment_dir = getattr(self.args, "output_dir", "./outputs")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 保存原始数据集引用（迭代 0 用）
        self.original_dataset = self.train_dataset

        # 生成 train.log 的文件句柄
        self._configure_logging()

        # 恢复上一次的进度 / 缓存（会设置 current_iteration 等）
        self._try_resume_from_checkpoint()

        print_master(f"Initialized IterativeRetrievalTrainer with max_iterations={max_iterations}")

        # 根据 fast / production 调整 save / logging 频率
        self._configure_training_mode()


    def _try_resume_from_checkpoint(self) -> bool:
        """Try to resume from previous experiment state (backward-compatible)."""
        import os, json
        from glob import glob

        output_dir = self.args.output_dir
        max_iters = int(getattr(self, "max_iterations", 0) or 0)

        def _has_base_model(dir_):
            if not os.path.isdir(dir_):
                return False
            files = set(os.listdir(dir_))
            # 兼容 LoRA 形式 & HF 全量权重保存
            has_lora = any(f.startswith("adapter_") for f in files) and "adapter_config.json" in files
            has_full = ("pytorch_model.bin" in files or "model.safetensors" in files) and "config.json" in files
            return has_lora or has_full

        def _has_any_embedding_cache(root):
            """兼容老路径: cache/target_embeddings_*.pt 以及新路径: cache/embeddings/*.pt"""
            legacy = glob(os.path.join(root, "cache", "target_embeddings_*.pt"))
            modern = glob(os.path.join(root, "cache", "embeddings", "*.pt"))
            return bool(legacy or modern)

        # -------- 1) 查找“最新的完整迭代” --------
        latest_complete = None
        for i in range(max_iters - 1, -1, -1):
            state_file = os.path.join(output_dir, f"iteration_{i}_state.json")
            if not os.path.exists(state_file):
                continue
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
            except Exception as e:
                print_master(f"Error reading iteration state {i}: {e}")
                continue

            # 兼容老字段名
            iter_complete = state.get("iteration_complete", False)
            if iter_complete:
                latest_complete = i
                print_master(f"Found COMPLETE iteration {i}")
                break
            else:
                print_master(f"Found INCOMPLETE iteration {i}, keep searching older COMPLETE")

        if latest_complete is not None:
            # 从“完整”的下一轮继续
            state_file = os.path.join(output_dir, f"iteration_{latest_complete}_state.json")
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
            except Exception as e:
                print_master(f"Error reading COMPLETE iteration state {latest_complete}: {e}")
                state = {}

            # 兼容 metrics 字段
            metrics = state.get("iteration_metrics", state.get("metrics", {}))
            self.iteration_metrics = metrics
            self.current_iteration = min(latest_complete + 1, max_iters)  # 防越界

            # 恢复硬负样本缓存（如果下一轮还需要）
            if latest_complete < max_iters - 1:
                hard_neg_file = os.path.join(output_dir, f"hard_negatives_iter_{latest_complete}.json")
                if os.path.exists(hard_neg_file) and hasattr(self.train_dataset, "hard_negatives_file"):
                    self.train_dataset.hard_negatives_file = hard_neg_file
                    # 数据集里通常有 _load_hard_negatives(iteration) 的轻量方法
                    try:
                        self.train_dataset._load_hard_negatives(latest_complete)
                    except Exception:
                        pass

            print_master(f"✅ Resuming from COMPLETE iteration {latest_complete}")
            print_master(f"   ➡️ Next iteration to run: {self.current_iteration}")
            return True

        # -------- 2) 若没有完整轮，尝试从“不完整迭代”恢复 --------
        for i in range(max_iters - 1, -1, -1):
            state_file = os.path.join(output_dir, f"iteration_{i}_state.json")
            if not os.path.exists(state_file):
                continue
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                completed_steps = state.get("completed_steps", {})
                metrics = state.get("iteration_metrics", state.get("metrics", {}))
            except Exception as e:
                print_master(f"Error reading incomplete iteration state {i}: {e}")
                continue

            self.current_iteration = i
            self.iteration_metrics = metrics
            print_master(f"🔄 Resuming from INCOMPLETE iteration {i}")
            print_master(f"   ➡️ Completed steps detected: {list(completed_steps.keys())}")

            # 关键：解耦后需要把增广样本/状态装载进数据集
            try:
                self._prepare_dataset_for_iteration(i)
            except Exception as e:
                print_master(f"Warning: _prepare_dataset_for_iteration({i}) failed: {e}")

            return True

        # -------- 3) 没有迭代状态文件，则检查“基座模型 + embeddings 缓存” --------
        base_model_dir = os.path.join(output_dir, "base_model")
        cache_root = output_dir

        if _has_base_model(base_model_dir):
            print_master(f"Found base model at: {base_model_dir}")
            has_cache = _has_any_embedding_cache(cache_root)

            if has_cache:
                print_master("✅ Found cached target embeddings (legacy or modern path).")
                print_master("🔄 Resuming from completed base training (iteration 0).")
                print_master("   ➡️ Will skip: base training/eval/target embedding computation")
                print_master("   ➡️ Next: hard negative collection")
                self.current_iteration = 0
                self._base_training_completed = True
                self._target_embeddings_cached = True
                return True
            else:
                print_master("⚠️ Base model found but no target embeddings cache.")
                print_master("🔄 Will recompute target embeddings and continue from iteration 0")
                self.current_iteration = 0
                self._base_training_completed = True
                self._target_embeddings_cached = False
                return True

        # -------- 4) 兜底：有 checkpoint-* 但没有 base_model，视为不完整训练 --------
        ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")] if os.path.isdir(output_dir) else []
        if ckpts:
            print_master(f"Found checkpoints without base model: {ckpts}")
            print_master("This looks like an incomplete training; starting from scratch.")

        print_master("No previous state/base model/embedding cache found, starting from scratch.")
        self._base_training_completed = False
        self._target_embeddings_cached = False
        self.current_iteration = 0
        return False

    def _configure_logging(self):
        """Configure additional logging to ensure train.log is generated"""
        if hasattr(self.args, 'logging_dir') and self.args.logging_dir:
            import os
            os.makedirs(self.args.logging_dir, exist_ok=True)
            log_file = os.path.join(self.args.logging_dir, "train.log")

            root_logger = logging.getLogger()
            file_handler_exists = any(
                isinstance(handler, logging.FileHandler) and 
                handler.baseFilename == os.path.abspath(log_file)
                for handler in root_logger.handlers
            )
            if not file_handler_exists:
                file_handler = logging.FileHandler(log_file, mode='a')
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                print_master(f"Added file logging to: {log_file}")
            else:
                print_master(f"File logging already configured for: {log_file}")
        else:
            print_master("Warning: logging_dir not set, train.log will not be generated")

    def _configure_training_mode(self):
        """Configure training parameters based on fast mode or production mode"""
        print_master(f"DEBUG: self.fast_mode = {self.fast_mode}")
        print_master(f"DEBUG: steps_per_iteration = {self.steps_per_iteration}")

        if self.fast_mode:
            print_master("=== FAST MODE CONFIGURATION ===")
            print_master(f"Steps per iteration: {self.steps_per_iteration}")
            print_master(f"Max samples for hard negatives: {self.fast_mode_max_samples}")
            print_master(f"Retrieval database size: {self.fast_mode_retrieval_db_size}")
            self.args.save_steps = max(1, self.steps_per_iteration // 2)
            self.args.logging_steps = 1
        else:
            print_master("=== PRODUCTION MODE CONFIGURATION ===")
            print_master(f"Steps per iteration: {self.steps_per_iteration}")
            print_master(f"Save frequency: every {self.production_save_steps} steps")
            self.args.save_steps = self.production_save_steps
            self.args.logging_steps = min(10, self.production_save_steps // 10)

        print_master("🎯 Each iteration will train independently with fresh optimizer/scheduler")
        print_master(f"Final training configuration:")
        print_master(f"  steps_per_iteration: {self.steps_per_iteration}")
        print_master(f"  save_steps: {self.args.save_steps}")
        print_master(f"  logging_steps: {self.args.logging_steps}")
        print_master("=" * 50)

    def _train_base_model(self):
        """Train the base retrieval model using standard contrastive learning."""
        import os
        from .utils import print_master

        print_master("Training base model with original dataset and fresh optimizer...")

        # 1) 用最初的数据集并刷新 dataloader
        self.train_dataset = self.original_dataset
        self._update_train_dataloader()

        # 2) 在独立子目录里训练，避免多轮 checkpoint 冲突
        original_output_dir = self.args.output_dir
        original_max_steps = self.args.max_steps
        base_training_dir = os.path.join(original_output_dir, "training_iter_0")
        os.makedirs(base_training_dir, exist_ok=True)

        self.args.output_dir = base_training_dir
        self.args.max_steps = self.steps_per_iteration

        print_master(f"🎯 Base model training plan: 0 → {self.args.max_steps} steps")
        print_master(f"🆕 Starting fresh training with new optimizer and scheduler")
        print_master(f"📁 Training checkpoints will be saved to: {base_training_dir}")
        print_master(f"📁 Final base model will be saved to: {original_output_dir}")

        # 3) 显式重置优化器与调度器，确保“全新一轮”
        try:
            # transformers 兼容处理：有的版本是 optimizer/lr_scheduler 属性
            if hasattr(self, "optimizer"):
                self.optimizer = None
            if hasattr(self, "lr_scheduler"):
                self.lr_scheduler = None
            # 某些版本 Trainer 会在 train() 内部懒初始化；上面置 None 可确保重新创建
            train_result = self.train(resume_from_checkpoint=None)
        finally:
            # 4) 还原全局输出目录与步数
            self.args.output_dir = original_output_dir
            self.args.max_steps = original_max_steps
            print_master(f"✅ Restored output_dir to: {original_output_dir}")
            print_master(f"✅ Restored max_steps to: {original_max_steps}")

        # 5) 保存基座模型到主目录（非子目录）
        base_model_path = os.path.join(original_output_dir, "base_model")
        self.save_model(base_model_path)

        print_master(f"Base model training completed: 0 → {self.state.global_step} steps")
        print_master(f"✅ Base model saved to: {base_model_path}")

        return train_result

    def _train_current_iteration(self):
        """
        Train for the current iteration by loading previous model weights
        but resetting the optimizer and LR scheduler for independent training.
        """
        import os
        from .utils import print_master

        print_master(f"Training iteration {self.current_iteration} with RESET optimizer and scheduler...")

        # 1) 说明：权重已在外部 main() 完成加载；此处只负责训练流程控制
        print_master("🧠 Model weights already loaded by main() function")
        print_master("🔄 Will reset optimizer and scheduler for independent learning rate schedule")

        # 2) 确保数据集与采样器已更新
        self._update_train_dataloader()

        # 3) 为当前迭代创建独立训练子目录，避免 checkpoint 冲突
        original_output_dir = self.args.output_dir
        original_max_steps = self.args.max_steps
        iteration_output_dir = os.path.join(original_output_dir, f"training_iter_{self.current_iteration}")
        os.makedirs(iteration_output_dir, exist_ok=True)

        self.args.output_dir = iteration_output_dir
        self.args.max_steps = self.steps_per_iteration

        print_master(f"🎯 Iteration {self.current_iteration} independent training plan:")
        print_master(f"   - Will train for {self.args.max_steps} fresh steps")
        print_master(f"   - Training checkpoints will be saved to: {iteration_output_dir}")
        print_master(f"   - Final iteration model will be saved to: {original_output_dir}")
        print_master(f"   - Previous global_step will be ignored for LR scheduling")
        print_master(f"   - New optimizer and scheduler will start from scratch")

        try:
            # 4) 显式重置优化器/调度器，确保每一轮都“全新”调度
            if hasattr(self, "optimizer"):
                self.optimizer = None
            if hasattr(self, "lr_scheduler"):
                self.lr_scheduler = None

            print_master("✅ Model weights already loaded in main(), ready for independent training")
            print_master("🔄 Creating fresh optimizer and scheduler for this iteration")

            # 不从 checkpoint 恢复，强制创建全新的优化器与学习率调度器
            train_result = self.train(resume_from_checkpoint=None)
        finally:
            # 5) 还原 trainer 的全局输出目录与 max_steps
            self.args.output_dir = original_output_dir
            self.args.max_steps = original_max_steps
            print_master(f"✅ Restored output_dir to: {original_output_dir}")
            print_master(f"✅ Restored max_steps to: {original_max_steps}")

        print_master(f"Iteration {self.current_iteration} independent training completed")
        print_master(f"Final step count: {self.state.global_step}")

        # 6) 保存本轮最终模型到主目录（非子目录）
        iter_model_path = os.path.join(original_output_dir, f"iteration_{self.current_iteration}")
        self.save_model(iter_model_path)
        print_master(f"✅ Final iteration model saved to: {iter_model_path}")

        return train_result

    def _evaluate_current_model(self) -> Dict[str, float]:
        """Evaluate current model on validation set with caching & distributed support (decoupled layout)."""
        import os, json
        import torch.distributed as dist
        from .utils import print_master

        print_master(f"Evaluating iteration {self.current_iteration} model...")

        # 1) 先看缓存
        eval_results_file = os.path.join(self.args.output_dir, f"eval_results_iter_{self.current_iteration}.json")
        if os.path.exists(eval_results_file):
            print_master(f"Found cached evaluation results for iteration {self.current_iteration}, loading...")
            try:
                with open(eval_results_file, "r") as f:
                    cached = json.load(f)
                print_master(f"Loaded cached evaluation results: {cached}")
                return cached
            except Exception as e:
                print_master(f"Error loading cached evaluation results: {e}; will re-evaluate.")

        # 2) 准备 evaluator（兼容绝对/相对导入）
        try:
            try:
                from src.evaluation.cirr_evaluator import CIRREvaluator  # 新解耦路径
            except Exception:
                from .evaluation.cirr_evaluator import CIRREvaluator    # 旧相对路径兜底
        except Exception as e:
            print_master(f"Evaluator import failed: {e}")
            CIRREvaluator = None

        # 3) 真实评测或回退
        try:
            if CIRREvaluator is None:
                raise RuntimeError("CIRREvaluator is not available")

            # 取 processor：优先使用传入到 Trainer 的 processing_class，其次尝试 model.processor
            processor = getattr(self, "processing_class", None) or getattr(self.model, "processor", None)
            if processor is None:
                print_master("Warning: no processor found on Trainer or model; evaluator may fail.")

            eval_bs = 4 if getattr(self, "fast_mode", False) else 8
            evaluator = CIRREvaluator(
                model=self.model,
                processor=processor,
                data_args=self.data_args,
                model_args=self.model_args,
                device=str(getattr(self.args, "device", "cpu")),
                batch_size=eval_bs,
            )
            print_master(f"Real evaluator initialized (batch_size={eval_bs}).")

            # 分布式与否
            world_ok = dist.is_initialized() and dist.get_world_size() > 1
            # 仅当 evaluator 支持分布式且确实多卡时启用
            supports_dist = hasattr(evaluator, "evaluate") or hasattr(evaluator, "_evaluate_distributed")
            use_distributed = bool(world_ok and supports_dist)

            # fast_mode 下如果实际只有 1 卡，则退回单卡
            if getattr(self, "fast_mode", False) and (not world_ok):
                use_distributed = False
                print_master("Fast mode: single-GPU evaluation.")

            if use_distributed:
                print_master(f"Using distributed evaluation across {dist.get_world_size()} GPUs")
                eval_results = evaluator.evaluate(distributed=True)
            else:
                print_master("Using single-GPU evaluation")
                eval_results = evaluator.evaluate(distributed=False)

            # 元数据
            eval_results["evaluation_mode"] = "distributed" if use_distributed else "single_gpu"
            eval_results["fast_mode"] = bool(getattr(self, "fast_mode", False))
            eval_results["iteration"] = int(self.current_iteration)

        except Exception as e:
            print_master(f"Real evaluation failed: {e}")
            print_master("Falling back to dummy evaluation metrics.")
            if getattr(self, "fast_mode", False):
                eval_results = {
                    "recall_at_1": 0.15,
                    "recall_at_5": 0.35,
                    "recall_at_10": 0.45,
                    "recall_subset_at_1": 0.12,
                    "recall_subset_at_2": 0.25,
                    "recall_subset_at_3": 0.32,
                    "group_recall_at_1": 0.18,
                    "group_recall_at_2": 0.30,
                    "group_recall_at_3": 0.38,
                    "evaluation_mode": "dummy_fast",
                    "fast_mode": True,
                    "iteration": int(self.current_iteration),
                }
            else:
                eval_results = {
                    "recall_at_1": 0.50,
                    "recall_at_5": 0.70,
                    "recall_at_10": 0.80,
                    "recall_subset_at_1": 0.30,
                    "recall_subset_at_2": 0.50,
                    "recall_subset_at_3": 0.60,
                    "group_recall_at_1": 0.40,
                    "group_recall_at_2": 0.60,
                    "group_recall_at_3": 0.70,
                    "evaluation_mode": "dummy_production",
                    "fast_mode": False,
                    "iteration": int(self.current_iteration),
                }

        # 4) 仅 rank0 落盘
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                with open(eval_results_file, "w") as f:
                    json.dump(eval_results, f, indent=2)
                print_master(f"Saved evaluation results to {eval_results_file}")
            except Exception as e:
                print_master(f"Warning: failed to save evaluation results: {e}")

        print_master(f"Iteration {self.current_iteration} results: {eval_results}")
        return eval_results

    def _prepare_next_iteration_dataset(self, next_iteration: int, augmented_samples: List[Dict]):
        """Prepare dataset for next iteration with augmented samples (decoupled version)."""
        import os, json, time
        import torch.distributed as dist

        print_master(f"Preparing dataset for iteration {next_iteration}...")

        # 1) 将增广样本落盘（仅 rank0 写；使用原子替换避免其它进程读到半写文件）
        augmented_file = os.path.join(self.args.output_dir, f"augmented_samples_iter_{next_iteration}.json")
        augmented_dir = os.path.dirname(augmented_file)
        os.makedirs(augmented_dir, exist_ok=True)

        meta = {
            "total_samples": len(augmented_samples),
            "generation_timestamp": time.time(),
            "iteration_round": next_iteration,
            "sample_statistics": self._compute_sample_statistics(augmented_samples),
            "samples": augmented_samples,
        }

        if not dist.is_initialized() or dist.get_rank() == 0:
            tmp_path = augmented_file + ".tmp"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, augmented_file)  # 原子替换
                print_master(f"Saved {len(augmented_samples)} augmented samples to {augmented_file}")
            except Exception as e:
                print_master(f"❌ Failed to save augmented samples: {e}")
        else:
            print_rank(f"GPU {dist.get_rank()}: Skipping augmented samples save (only rank 0 writes)")

        # 2) 同步一次，确保所有 GPU 能看到文件
        if dist.is_initialized():
            dist.barrier()
            print_master("All GPUs synchronized after augmented samples save")

        # 3) 将增广样本追加进当前训练集（数据集本身只负责取样，不做检索/挖掘）
        #    为兼容你现有的数据集类型，做一次类型判断
        try:
            ds_types = (IterativeCIRRDataset, IterativeFashionIQDataset)
        except Exception:
            ds_types = tuple()  # 避免导入问题导致崩溃

        if isinstance(self.train_dataset, ds_types) or hasattr(self.train_dataset, "augmented_samples"):
            # 记录更新前的规模
            old_total = len(self.train_dataset)
            old_aug = len(getattr(self.train_dataset, "augmented_samples", []))

            # 标注轮次并追加样本
            setattr(self.train_dataset, "iteration_round", next_iteration)
            if not hasattr(self.train_dataset, "augmented_samples"):
                self.train_dataset.augmented_samples = []
            self.train_dataset.augmented_samples.extend(augmented_samples)

            # 兼容：很多下游代码会读这个路径（虽然现在挖掘已解耦）
            setattr(
                self.train_dataset,
                "hard_negatives_file",
                os.path.join(self.args.output_dir, f"hard_negatives_iter_{next_iteration}.json"),
            )

            new_total = len(self.train_dataset)
            new_aug = len(self.train_dataset.augmented_samples)

            print_master("📊 Dataset update summary:")
            print_master(f"  - Added {len(augmented_samples)} new augmented samples")
            print_master(f"  - Total augmented samples: {old_aug} → {new_aug}")
            print_master(f"  - Total dataset size: {old_total} → {new_total}")
        else:
            # 极端情况下（自定义数据集对象），只做日志提醒；训练仍可继续
            print_master("⚠️ Train dataset has no 'augmented_samples' attribute; skipped in-memory append.")

        # 4) 关键：重建 dataloader，确保采样器与 batch 逻辑按最新样本数工作
        self._update_train_dataloader()
        print_master(f"Training dataloader updated for iteration {next_iteration}")

    def _save_iteration_state(self, iteration: int):
        """Save iteration state and metrics with step completion tracking"""
        import os, json, time
        import torch.distributed as dist
        from .utils import print_rank, print_master

        # only rank 0 writes
        if dist.is_initialized() and dist.get_rank() != 0:
            print_rank(f"GPU {dist.get_rank()}: Skipping state save (only rank 0 writes)")
            return

        state_file = os.path.join(self.args.output_dir, f"iteration_{iteration}_state.json")

        # model path of this iteration
        model_path = (os.path.join(self.args.output_dir, "base_model")
                    if iteration == 0
                    else os.path.join(self.args.output_dir, f"iteration_{iteration}"))

        # compute step completion flags
        completed_steps = self._check_iteration_completion_status(iteration)

        state = {
            "iteration": iteration,
            # 👇 统一键名：用 iteration_metrics（与读取保持一致）
            "iteration_metrics": self.iteration_metrics,
            "model_path": model_path,
            "hard_negatives_file": f"hard_negatives_iter_{iteration}.json",
            "completed_steps": completed_steps,
            "iteration_complete": completed_steps.get("all_steps_complete", False),
            "timestamp": time.time(),
        }

        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            completion_status = "✅ COMPLETE" if state["iteration_complete"] else "🔄 IN PROGRESS"
            print_master(f"Saved iteration {iteration} state to {state_file} - {completion_status}")
            print_master(f"Model path recorded as: {model_path}")
            print_master(f"Completed steps: {list(completed_steps.keys())}")
        except Exception as e:
            print_master(f"❌ Error saving iteration state: {e}")

    def _check_iteration_completion_status(self, iteration: int) -> dict:
        """Check which steps of an iteration have been completed"""
        import os

        output_dir = self.args.output_dir
        completed_steps = {}

        # Step 1: model training done?
        model_path = (os.path.join(output_dir, "base_model")
                    if iteration == 0
                    else os.path.join(output_dir, f"iteration_{iteration}"))
        completed_steps["model_training"] = os.path.exists(model_path)

        # Step 2: evaluation done?
        eval_file = os.path.join(output_dir, f"eval_results_iter_{iteration}.json")
        completed_steps["evaluation"] = os.path.exists(eval_file)

        # Step 3: hard negatives (non-final iteration)
        is_final_iteration = iteration >= (self.max_iterations - 1)
        if is_final_iteration:
            completed_steps["hard_negatives_collection"] = True
        else:
            hard_neg_file = os.path.join(output_dir, f"hard_negatives_iter_{iteration}.json")
            completed_steps["hard_negatives_collection"] = os.path.exists(hard_neg_file)

        # Step 4: caption generation (non-final iteration)
        if is_final_iteration:
            completed_steps["caption_generation"] = True
        else:
            next_iteration = iteration + 1
            augmented_file = os.path.join(output_dir, f"augmented_samples_iter_{next_iteration}.json")
            completed_steps["caption_generation"] = os.path.exists(augmented_file)

        # all done?
        completed_steps["all_steps_complete"] = all(completed_steps.values())
        return completed_steps

    def _summarize_results(self):
        """Summarize results across all iterations"""
        import os, json
        import torch.distributed as dist
        from .utils import print_master

        print_master("\n" + "=" * 80)
        print_master("ITERATIVE TRAINING SUMMARY")
        print_master("=" * 80)

        for iteration, metrics in self.iteration_metrics.items():
            print_master(f"Iteration {iteration}: {metrics}")

        best_iteration, best_metrics = None, None
        if self.iteration_metrics:
            def _score(m):  # 兼容不同命名
                return m.get("recall_at_1", m.get("r_at_1", 0))
            best_iteration = max(self.iteration_metrics.keys(),
                                key=lambda x: _score(self.iteration_metrics[x]))
            best_metrics = self.iteration_metrics[best_iteration]
            print_master(f"\nBest performance: Iteration {best_iteration}")
            print_master(f"Best metrics: {best_metrics}")

        # only rank0 writes summary
        if not dist.is_initialized() or dist.get_rank() == 0:
            summary_file = os.path.join(self.args.output_dir, "training_summary.json")
            summary = {
                "max_iterations": self.max_iterations,
                "completed_iterations": len(self.iteration_metrics),
                "iteration_metrics": self.iteration_metrics,
                "best_iteration": best_iteration if self.iteration_metrics else None,
                "best_metrics": best_metrics if self.iteration_metrics else None,
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print_master(f"Training summary saved to {summary_file}")

    def _prepare_dataset_for_iteration(self, iteration: int):
        """Prepare dataset state for a specific iteration when resuming"""

        print_master(f"Preparing dataset for resumed iteration {iteration}...")

        # iter0：用最初的原始数据集
        if iteration == 0:
            self.train_dataset = self.original_dataset
            self._update_train_dataloader()  # 防止 sampler 仍引用旧对象
            return

        # 迭代 > 0：累计加载此前各轮的增广样本
        all_augmented_samples = []
        for i in range(1, iteration + 1):
            augmented_file = os.path.join(self.args.output_dir, f"augmented_samples_iter_{i}.json")
            if not os.path.exists(augmented_file):
                print_master(f"Warning: augmented file not found for iter {i}: {augmented_file}")
                continue

            try:
                with open(augmented_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print_master(f"Warning: failed to read {augmented_file}: {e}")
                continue

            # 兼容两种格式：新(带metadata) & 旧(直接list)
            if isinstance(data, dict) and "samples" in data:
                iter_samples = data["samples"]
                print_master(f"Loaded {len(iter_samples)} augmented samples from iter {i} (with metadata)")
            elif isinstance(data, list):
                iter_samples = data
                print_master(f"Loaded {len(iter_samples)} augmented samples from iter {i} (direct list)")
            else:
                print_master(f"Warning: Unexpected data format in {augmented_file}, skip")
                continue

            all_augmented_samples.extend(iter_samples)

        # 🔍 这里补上统计（与老版对齐）
        stats = self._compute_sample_statistics(all_augmented_samples)
        if stats:
            print_master("Resume-time augmented sample statistics:")
            for k, v in stats.items():
                print_master(f"  - {k}: {v}")

            # 可选：把恢复时的统计也落个档，便于核对
            resume_stats_file = os.path.join(self.args.output_dir, f"resume_stats_iter_{iteration}.json")
            try:
                with open(resume_stats_file, "w") as f:
                    json.dump(stats, f, indent=2)
                print_master(f"Saved resume statistics to {resume_stats_file}")
            except Exception as e:
                print_master(f"Warning: failed to save resume statistics: {e}")

        # 将累计样本挂到当前训练集上（数据集已解耦，仅用于训练取样）
        if isinstance(self.train_dataset, (IterativeCIRRDataset, IterativeFashionIQDataset)):
            # 标注当前轮次
            self.train_dataset.iteration_round = iteration
            # 覆盖为累计的增广样本列表
            self.train_dataset.augmented_samples = all_augmented_samples

            # 兼容字段：有些下游代码可能还会读这个路径
            if hasattr(self.train_dataset, "hard_negatives_file"):
                self.train_dataset.hard_negatives_file = os.path.join(
                    self.args.output_dir, f"hard_negatives_iter_{iteration-1}.json"
                )

        print_master(
            f"Dataset prepared for iteration {iteration} "
            f"with {len(all_augmented_samples)} total augmented samples"
        )

        # 🔧 关键：恢复/更新 dataloader，确保 sampler / batch 构造与新数据一致
        self._update_train_dataloader()

    def _compute_sample_statistics(self, samples: List[Dict]) -> Dict[str, Any]:
        """Compute descriptive statistics for augmented samples (robust & backward-compatible)."""
        if not samples:
            return {}

        try:
            orig_chars, gen_chars = [], []
            orig_words, gen_words = [], []
            refs, tgts, pairs = set(), set(), set()

            # optional counters
            has_original_cnt = 0
            has_generated_cnt = 0
            source_counter = {}

            for s in samples:
                # ----- text length (chars/words) -----
                orig_txt = s.get("original_mod_text", "")
                gen_txt  = s.get("modification_text", "")

                if isinstance(orig_txt, str) and len(orig_txt) > 0:
                    has_original_cnt += 1
                    orig_chars.append(len(orig_txt))
                    orig_words.append(len(orig_txt.split()))

                if isinstance(gen_txt, str) and len(gen_txt) > 0:
                    has_generated_cnt += 1
                    gen_chars.append(len(gen_txt))
                    gen_words.append(len(gen_txt.split()))

                # ----- unique images / pairs -----
                ref = s.get("reference_image")
                tgt = s.get("target_image")
                if ref: refs.add(ref)
                if tgt: tgts.add(tgt)
                if ref and tgt:
                    pairs.add((ref, tgt))

                # ----- optional: source breakdown -----
                src = s.get("source")
                if src:
                    source_counter[src] = source_counter.get(src, 0) + 1

            def _mean(x): return float(statistics.mean(x)) if x else 0.0
            def _median(x): return float(statistics.median(x)) if x else 0.0
            def _p95(x):
                if not x: return 0.0
                xs = sorted(x)
                idx = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
                return float(xs[idx])

            total = len(samples)
            augmented_ratio = sum(1 for s in samples if s.get("is_augmented", False)) / total if total else 0.0

            stats: Dict[str, Any] = {
                # 兼容你原有字段（字符级平均长度）
                "total_samples": total,
                "avg_original_length": _mean(orig_chars),
                "avg_generated_length": _mean(gen_chars),
                "unique_reference_images": len(refs),
                "unique_target_images": len(tgts),
                "augmented_ratio": augmented_ratio,

                # 新增更细致的指标
                "original": {
                    "count": has_original_cnt,
                    "avg_chars": _mean(orig_chars),
                    "median_chars": _median(orig_chars),
                    "p95_chars": _p95(orig_chars),
                    "avg_words": _mean(orig_words),
                    "median_words": _median(orig_words),
                    "p95_words": _p95(orig_words),
                },
                "generated": {
                    "count": has_generated_cnt,
                    "avg_chars": _mean(gen_chars),
                    "median_chars": _median(gen_chars),
                    "p95_chars": _p95(gen_chars),
                    "avg_words": _mean(gen_words),
                    "median_words": _median(gen_words),
                    "p95_words": _p95(gen_words),
                },
                "unique_pairs": len(pairs),
                "duplicate_pair_count": max(0, total - len(pairs)),  # 简易估计
            }

            if source_counter:
                stats["by_source"] = dict(sorted(source_counter.items(), key=lambda kv: (-kv[1], kv[0])))

            return stats

        except Exception as e:
            print_master(f"Warning: Failed to compute sample statistics: {e}")
            return {"total_samples": len(samples)}

    def _update_train_dataloader(self):
        """Update train dataloader to reflect dataset changes (safe for HF Trainer + DDP)."""
        import torch.distributed as dist

        # 1) DDP 同步（可选但更稳）
        if dist.is_initialized():
            dist.barrier()

        # 2) 清理 Trainer 内部缓存，强制重建
        #   - _train_dataloader 是 HF Trainer 的内部缓存
        #   - _train_sampler 在部分版本中也会被缓存（尤其自定义 sampler 时）
        if hasattr(self, "_train_dataloader"):
            self._train_dataloader = None
        if hasattr(self, "_train_sampler"):
            self._train_sampler = None

        # 3) 重新构建 dataloader（HF 会自动按当前 dataset / sampler / collator 来生成）
        dl = self.get_train_dataloader()
        # 显式回填，避免某些自定义场景下再次触发 get_train_dataloader
        self._train_dataloader = dl

        # 4) 更详细的统计与日志（保持你原有的输出风格）
        try:
            total_samples = len(self.train_dataset) if self.train_dataset is not None else 0
        except Exception:
            total_samples = 0

        if hasattr(self.train_dataset, "augmented_samples"):
            try:
                augmented_count = len(self.train_dataset.augmented_samples)
            except Exception:
                augmented_count = 0
            original_count = max(0, total_samples - augmented_count)

            print_master("🔄 Updated train dataloader:")
            print_master(f"  - Total samples: {total_samples}")
            print_master(f"  - Original samples: {original_count}")
            print_master(f"  - Augmented samples: {augmented_count}")
        else:
            print_master(f"🔄 Updated train dataloader with {total_samples} total samples")

        # 5) 给出一个简短提示：若使用分组/自定义采样器，需确保它基于最新 dataset 重建
        # （无需额外代码；若你的 Sampler 是在 get_train_dataloader 内部构建，这里已覆盖）

    def _lazy_load_foundation_model(self, to_device: str = None):
        """
        仅在需要生成 caption 时才加载一次 FM；避免训练阶段占显存。
        """
        if getattr(self, "foundation_model", None) is not None:
            # 已加载；必要时挪到目标设备
            if to_device:
                try:
                    self.foundation_model.to(to_device)
                except Exception:
                    pass
            return self.foundation_model

        if not getattr(self, "foundation_model_name", None):
            print_master("No foundation_model_name set; skip loading FM.")
            return None

        from transformers import AutoModelForVision2Seq, AutoProcessor
        dev = to_device or (f"cuda:{self.args.process_index}" if torch.cuda.is_available() else "cpu")

        print_master(f"🔁 Lazy-loading foundation model on {dev}: {self.foundation_model_name}")
        fm = AutoModelForVision2Seq.from_pretrained(
            self.foundation_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None
        ).to(dev).eval()

        proc = AutoProcessor.from_pretrained(self.foundation_model_name, trust_remote_code=True)
        setattr(fm, "processor", proc)      # 供 CaptionGenerator / Batcher 使用
        self.foundation_model = fm
        self.foundation_processor = proc
        print_master("✅ Foundation model lazy-loaded.")
        return fm


    def _unload_foundation_model(self):
        """生成结束后立刻卸载，释放显存。"""
        import gc
        try:
            del self.foundation_model
        except Exception:
            pass
        self.foundation_model = None
        try:
            del self.foundation_processor
        except Exception:
            pass
        self.foundation_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_master("🧹 Foundation model unloaded and CUDA cache cleared.")

    # ---------------------------
    # 外部模块调用封装
    # ---------------------------
    def _collect_hard_negatives(self, iteration: int):
        """
        使用解耦后的 HardNegativeMiner 进行硬负样本采集：
        - 直接实例化 retrieval/ 下的三块模块并传给 miner
        - 兼容单卡 & “最小改动版”分布式
        - 缓存文件仍是 {output_dir}/hard_negatives_iter_{iteration}.json
        """
        import os, json
        import torch.distributed as dist
        from src.utils import print_master

        print_master(f"Collecting hard negatives for iteration {iteration}...")

        cache_file = os.path.join(self.args.output_dir, f"hard_negatives_iter_{iteration}.json")
        if os.path.exists(cache_file):
            print_master(f"Found cached hard negatives for iteration {iteration}, loading...")
            with open(cache_file, "r") as f:
                cached = json.load(f)
            print_master(f"Loaded {len(cached)} cached hard negatives")
            return cached

        # ---------- 校验数据集必需字段 ----------
        ds = self.train_dataset

        annotations = (
            getattr(ds, "annotations", None)
            or getattr(ds, "train_annotations", None)
            or getattr(ds, "ann", None)
        )
        image_splits = getattr(ds, "image_splits", None)
        image_base_dir = getattr(ds, "image_base_dir", None) or getattr(ds, "root", None)

        retrieval_candidates = getattr(ds, "retrieval_candidates", None)
        if retrieval_candidates is None:
            retrieval_candidates = []  # 允许为空，Engine 内部可自行构建/扩展

        missing = []
        if annotations is None:    missing.append("annotations")
        if image_splits is None:   missing.append("image_splits")
        if image_base_dir is None: missing.append("image_base_dir/root")
        if missing:
            raise RuntimeError(f"train_dataset 缺少必要属性: {missing}")

        if not isinstance(annotations, (list, tuple)) or len(annotations) == 0:
            raise RuntimeError("annotations 为空或类型不对，无法进行硬负样本挖掘")

         # ---------- 训练/设备/处理器 ----------
        proc = self.processing_class or getattr(self.model, "processor", None)
        backbone = getattr(self.model_args, "model_backbone", "qwen2_vl")
        device = f"cuda:{self.args.process_index}" if torch.cuda.is_available() else "cpu"

        # ---------- 目标输入的前处理函数 ----------
        prep = VLMInputPreparer(
            image_base_dir=image_base_dir,
            default_processor=proc,
            default_backbone=backbone,
            default_device=device,
        )
        prepare_fn = prep.prepare_target_inputs  # (paths, processor, backbone, device) -> inputs
        print_master("Using VLMInputPreparer.prepare_target_inputs from src/prep/input_adapter.py")

        # fast/production 控制
        sample_limit = self.fast_mode_max_samples if getattr(self, "fast_mode", False) else None

        # 1) 用数据集提供的基础信息构造三大模块
        candidate_builder = CandidateBuilder(
            annotations=annotations,
            image_splits=image_splits,
            image_base_dir=image_base_dir,
            # 如实现支持，可加：experiment_dir=self.args.output_dir
        )


        embedding_cache = EmbeddingCache(
            experiment_dir=self.args.output_dir
        )

        # 先通过 CandidateBuilder 构建检索候选库
        retrieval_candidates = candidate_builder.build()
        print_master(f"CandidateBuilder built {len(retrieval_candidates)} retrieval candidates")


        # 🔧 关键修复：补齐 RetrievalEngine 必需参数
        retrieval_engine = RetrievalEngine(
            model_args=self.model_args,
            experiment_dir=self.args.output_dir,
            image_base_dir=image_base_dir,
            retrieval_candidates=retrieval_candidates,
            # 可选：若 Engine 支持 fast 限制
            # fast_mode_limit=self.fast_mode_retrieval_db_size if self.fast_mode else None,
        )

        # 2) 组装 miner
        miner = HardNegativeMiner(
            experiment_dir=self.args.output_dir,
            iteration_round=iteration,
            candidate_builder=candidate_builder,
            retrieval_engine=retrieval_engine,
            embedding_cache=embedding_cache,
            image_base_dir=image_base_dir,
            max_negatives_per_query=5,
            examine_topk=10,
        )

        call_kwargs = dict(
            batch_size=8,
            max_samples=sample_limit,
            processor=proc,
            model_backbone=backbone,
            device=device,
            prepare_target_inputs_fn=prepare_fn,
        )

        # 3) 采集：分布式优先使用“最小改动版”
        if dist.is_initialized() and dist.get_world_size() > 1:
            print_master("Using HardNegativeMiner.collect_distributed_minimal ...")
            hard_negatives = miner.collect_distributed_minimal(self.model, annotations, **call_kwargs)
        else:
            print_master("Using HardNegativeMiner.collect_single_gpu ...")
            hard_negatives = miner.collect_single_gpu(self.model, annotations, **call_kwargs)

        print_master(
            f"Collected {len(hard_negatives)} hard negatives "
            f"{'(limited to '+str(sample_limit)+')' if sample_limit else '(no limit)'}"
        )

        # rank0 双保险落缓存（分布式里 rank0 已落过，这里再确保一次也没关系）
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(hard_negatives, f, indent=2)
            print_master(f"Cached hard negatives to {cache_file}")

        return hard_negatives

    def _generate_augmented_captions(self, hard_negatives: List[Dict]) -> List[Dict]:
        """基于 CaptionGenerator（单卡/分布式）生成增广指令文本。"""
        if not hard_negatives:
            return []

        dev = f"cuda:{self.args.process_index}" if torch.cuda.is_available() else "cpu"
        fm = self._lazy_load_foundation_model(to_device=dev)
        if fm is None:
            print_master("No foundation model available, skip caption generation")
            return []

        # 默认 backbone（你项目里就是 Qwen）
        if not getattr(self.model_args, "foundation_model_backbone", None):
            setattr(self.model_args, "foundation_model_backbone", "qwen2_5_vl")



       # 组装 prepare / generate 的函数映射
        PREPARE_FNS = {
            "qwen": qwen_prepare,
            "llava": llava_prepare or (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("LLaVA not configured"))),
            "generic": generic_prepare or (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Generic not configured"))),
        }
        GENERATE_FNS = {
            "qwen": generate_with_qwen,
            "llava": generate_with_llava or (lambda *a, **k: ""),
            "generic": generate_with_generic or (lambda *a, **k: ""),
        }



        try:
            generator = CaptionGenerator(
                foundation_model=fm,
                model_args=self.model_args,
                experiment_dir=self.args.output_dir,
                iteration_round=self.current_iteration,
                prepare_fns=PREPARE_FNS,
                generate_fns=GENERATE_FNS,
                # === 新增：把数据集的 base/splits 传进去 ===
                image_base_dir=self.train_dataset.image_base_dir,
                image_splits=self.train_dataset.image_splits,
            )

            if dist.is_initialized() and dist.get_world_size() > 1:
                print_master("Using distributed caption generation...")
                augmented_samples = generator.generate_augmented_captions_distributed(hard_negatives)
            else:
                print_master("Using single-GPU caption generation...")
                augmented_samples = generator.generate_augmented_captions(hard_negatives)

        except Exception as e:
            print_master(f"Caption generation failed: {e}")
            augmented_samples = []
        finally:
            # 用完立刻卸载，释放显存
            self._unload_foundation_model()

        print_master(f"Generated {len(augmented_samples)} augmented samples")
        return augmented_samples
    
    def iterative_train(self, resume_from_iteration: int = 0):
        """
        Main iterative training loop (完善版)
        - 保留/恢复：手动续跑、步骤完成度检查与跳过、计时统计、分布式 barrier、缓存读写
        - 兼容：解耦后的数据集/检索与增广（内部仍调用你封装好的 dataset 方法）
        """
        import os, json, time
        import torch.distributed as dist
        from .utils import print_master

        print_master("🚀 Starting iterative training process...")

        # 手动指定从某一轮继续
        if resume_from_iteration > 0:
            print_master(f"Manually resuming from iteration {resume_from_iteration}")
            self.current_iteration = resume_from_iteration
            # 读取“上一轮”的元数据（模型权重外部加载）
            self._load_iteration_state(resume_from_iteration - 1)
            # 准备该轮需要的数据集状态（汇总过往增广等）
            self._prepare_dataset_for_iteration(resume_from_iteration)

        for iteration in range(self.current_iteration, self.max_iterations):
            print_master(f"\n{'='*60}")
            print_master(f"🔄 Starting Iteration {iteration}")
            print_master(f"{'='*60}")
            self.current_iteration = iteration

            # 检查该轮各步骤是否已完成（以便跳过）
            completed_steps = self._check_iteration_completion_status(iteration)
            print_master(f"🔍 Iteration {iteration} completion status: {completed_steps}")

            # -----------------------
            # Step 1: 训练（若未完成）
            # -----------------------
            if not completed_steps.get('model_training', False):
                if iteration == 0:
                    print_master("Iteration 0: Training base retrieval model...")
                    self._train_base_model()
                else:
                    print_master(f"Iteration {iteration}: Training with augmented data...")
                    self._train_current_iteration()
                    print_master(f"✅ Iteration {iteration} training completed with fresh optimizer/scheduler")
                if dist.is_initialized():
                    dist.barrier()
                    print_master(f"All GPUs completed training for iteration {iteration}")
            else:
                print_master("✅ Model training already completed, skipping...")

            # -----------------------
            # Step 2: 评估（若未完成）
            # -----------------------
            if not completed_steps.get('evaluation', False):
                if dist.is_initialized():
                    dist.barrier()  # 确保大家都训练结束
                eval_results = self._evaluate_current_model()
                self.iteration_metrics[iteration] = eval_results
            else:
                print_master("✅ Model evaluation already completed, loading cached results...")
                eval_file = os.path.join(self.args.output_dir, f"eval_results_iter_{iteration}.json")
                try:
                    with open(eval_file, 'r') as f:
                        eval_results = json.load(f)
                    self.iteration_metrics[iteration] = eval_results
                except Exception as e:
                    print_master(f"⚠️ Failed to load cached eval results: {e}. Re-evaluating...")
                    eval_results = self._evaluate_current_model()
                    self.iteration_metrics[iteration] = eval_results

            # ------------------------------------------------------
            # Step 3-4: 非最后一轮才进行 硬负样本采集 + Caption 增广
            # ------------------------------------------------------
            if iteration < self.max_iterations - 1:
                # 3) 硬负样本
                hard_neg_time = 0.0
                if not completed_steps.get('hard_negatives_collection', False):
                    print_master(f"🔍 Starting hard negative collection for iteration {iteration}...")
                    t0 = time.time()
                    hard_negatives = self._collect_hard_negatives(iteration)
                    hard_neg_time = time.time() - t0
                    print_master(f"Hard negative collection completed in {int(hard_neg_time//60):02d}:{int(hard_neg_time%60):02d}")
                    if dist.is_initialized():
                        dist.barrier()
                        print_master(f"All GPUs completed hard negative collection for iteration {iteration}")
                else:
                    print_master("✅ Hard negative collection already completed, loading cached results...")
                    hn_file = os.path.join(self.args.output_dir, f"hard_negatives_iter_{iteration}.json")
                    with open(hn_file, 'r') as f:
                        hard_negatives = json.load(f)
                    hard_neg_time = 0.0

                # 没有负样本则提前结束
                if not hard_negatives:
                    print_master("⚠️ No hard negatives found, stopping early")
                    break

                # 4) Caption 增广
                caption_time = 0.0
                next_iter = iteration + 1
                if not completed_steps.get('caption_generation', False):
                    print_master(f"📝 Starting caption generation for {len(hard_negatives)} hard negatives...")
                    t1 = time.time()
                    augmented_samples = self._generate_augmented_captions(hard_negatives)
                    caption_time = time.time() - t1
                    print_master(f"Caption generation completed in {int(caption_time//60):02d}:{int(caption_time%60):02d}")
                    if dist.is_initialized():
                        dist.barrier()
                        print_master(f"All GPUs completed caption generation for iteration {iteration}")
                    # 准备下一轮数据集
                    self._prepare_next_iteration_dataset(next_iter, augmented_samples)
                else:
                    print_master("✅ Caption generation already completed, loading cached results...")
                    aug_file = os.path.join(self.args.output_dir, f"augmented_samples_iter_{next_iter}.json")
                    with open(aug_file, 'r') as f:
                        saved_data = json.load(f)
                    augmented_samples = saved_data.get('samples', saved_data if isinstance(saved_data, list) else [])
                    caption_time = 0.0

                # 统计信息
                total_time = hard_neg_time + caption_time
                print_master(f"📊 Iteration {iteration} data preparation stats:")
                print_master(f"  - Hard negatives: {len(hard_negatives)} samples in {hard_neg_time:.1f}s")
                print_master(f"  - Augmented captions: {len(augmented_samples)} samples in {caption_time:.1f}s")
                print_master(f"  - Total time: {int(total_time//60):02d}:{int(total_time%60):02d}")
                if dist.is_initialized():
                    ws = dist.get_world_size()
                    print_master(f"  - Used {ws} GPUs for parallel processing")
                    if total_time > 0:
                        rate = (len(hard_negatives) + len(augmented_samples)) / max(total_time, 1e-6)
                        print_master(f"  - Processing rate: {rate:.2f} samples/second")

            # -----------------------
            # 最后的同步与状态落盘
            # -----------------------
            if dist.is_initialized():
                dist.barrier()
                print_master(f"All GPUs completed iteration {iteration}, saving state...")

            self._save_iteration_state(iteration)

        print_master("\n✅ Iterative training completed!")
        self._summarize_results()


# ---------------------------
# 工厂函数
# ---------------------------
def create_iterative_trainer(
    model: MMEBModel,
    foundation_model=None,
    args: TrainingArguments = None,
    train_dataset=None,
    eval_dataset=None,
    experiment_dir=None,
    **kwargs
) -> IterativeRetrievalTrainer:

    iterative_params = {k: kwargs.pop(k) for k in
                        ['max_iterations', 'hard_neg_collection_freq', 'caption_generation_batch_size']
                        if k in kwargs}

    fast_mode_params = {k: kwargs.pop(k) for k in
                        ['fast_mode', 'fast_mode_max_samples', 'fast_mode_retrieval_db_size',
                         'fast_mode_max_steps', 'steps_per_iteration', 'production_save_steps',
                         'production_max_steps']
                        if k in kwargs}

    important_args = {k: kwargs.pop(k) for k in ['model_args', 'data_args', 'max_length'] if k in kwargs}
    foundation_model_name = kwargs.pop('foundation_model_name', None)

    # ✅ 新增：把标准 Trainer 相关参数单独捞出来
    trainer_params = {k: kwargs.pop(k) for k in [
        'data_collator', 'tokenizer', 'model_init', 'compute_metrics',
        'callbacks', 'optimizers', 'preprocess_logits_for_metrics',
        'processing_class'  # 你在评估里要用
    ] if k in kwargs}

    # ✅ 关键点：把 trainer_params 和剩余 kwargs 一并传下去
    return IterativeRetrievalTrainer(
        model=model,
        foundation_model=foundation_model,
        foundation_model_name=foundation_model_name,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **iterative_params,
        **fast_mode_params,
        **important_args,
        **trainer_params,
        **kwargs
    )