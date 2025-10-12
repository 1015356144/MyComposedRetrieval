"""
Composed Image Retrieval Dataset for iterative training
Supports CIRR and FashionIQ datasets with iterative hard negative mining
"""

import os
import json
import time
import random
import re
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import traceback
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
from datasets import load_dataset

from .base_pair_dataset import AutoPairDataset, add_metainfo_hook
from ...model.processor import VLM_IMAGE_TOKENS, process_input_text
from ...utils import print_rank, print_master


class IterativeCIRRDataset(Dataset):
    """
    Iterative CIRR Dataset for composed image retrieval with hard negative mining
    """
    
    def __init__(self, 
                 model_args,
                 data_args, 
                 training_args,
                 iteration_round: int = 0,
                 foundation_model=None,
                 experiment_dir=None,
                 **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # Store dataset configuration from kwargs
        self.dataset_config = kwargs
        
        self.iteration_round = iteration_round
        self.foundation_model = foundation_model
        
        # Set experiment directory for storing hard negatives
        self.experiment_dir = experiment_dir or getattr(training_args, 'output_dir', "./experiments/default")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Hard negatives file based on experiment directory
        self.hard_negatives_file = os.path.join(self.experiment_dir, f"hard_negatives_iter_{iteration_round}.json")
        
        self.hard_negatives_cache = {}
        self.augmented_samples = []
        
        # Load original CIRR data
        self._load_cirr_data()
        
        # Load hard negatives from previous iterations if exists
        if iteration_round > 0:
            self._load_hard_negatives(iteration_round)
    
    def _load_cirr_data(self):
        """Load CIRR dataset"""
        print_rank(f"Loading CIRR dataset...")
        
        # Load from local files
        self._load_cirr_local()
    
    def _load_cirr_local(self):
        """Load CIRR from local files"""
        data_dir = self.dataset_config.get('data_dir', './data/CIRR')
        image_base_dir = self.dataset_config.get('image_base_dir', './data/CIRR')
        captions_file = self.dataset_config.get('captions_file')
        image_splits_file = self.dataset_config.get('image_splits_file')
        
        # Load captions
        with open(captions_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Load image paths mapping
        with open(image_splits_file, 'r') as f:
            self.image_splits = json.load(f)
        
        self.image_base_dir = image_base_dir
        print_rank(f"Loaded {len(self.annotations)} CIRR training samples")
        print_rank(f"Loaded {len(self.image_splits)} image path mappings")
        
        # Extract all possible target candidates for retrieval
        # This is crucial for proper hard negative mining
        self._build_retrieval_candidates()
        
        # Add num_rows property for VLM2Vec compatibility
        self.num_rows = len(self.annotations)
    
    def _build_retrieval_candidates(self):
        """
        Build the complete set of retrieval candidates from CIRR image_splits_file.
        
        According to CIRR evaluation protocol, image_splits_file contains ALL possible 
        candidate images that can be retrieved. This is essential for proper hard negative mining.
        """
        print_rank("Building CIRR retrieval candidate set...")
        
        # CIRR image_splits_file structure: {"image_name": "./path/to/image.png", ...}
        # This contains ALL ~16,939 candidate images for retrieval
        if not isinstance(self.image_splits, dict):
            raise ValueError(f"Expected image_splits to be a dict, got {type(self.image_splits)}")
        
        print_rank(f"CIRR image_splits contains {len(self.image_splits)} total images")
        
        # Extract all image paths as retrieval candidates
        # Each value in image_splits is a relative path like "./train/34/train-11041-2-img0.png"
        retrieval_candidates = []
        valid_candidates = 0
        
        for img_name, img_path in self.image_splits.items():
            # Validate that the image file exists
            if self._image_exists(img_path):
                retrieval_candidates.append(img_path)
                valid_candidates += 1
            else:
                # Log missing images for debugging (but don't fail)
                if len(retrieval_candidates) < 5:  # Only log first few to avoid spam
                    print_rank(f"Warning: Image not found: {img_name} -> {img_path}")
        
        # Store as list for easier indexing during retrieval
        self.retrieval_candidates = retrieval_candidates
        
        print_rank(f"Built CIRR retrieval candidate set:")
        print_rank(f"  • Total candidates from image_splits: {len(self.image_splits)}")
        print_rank(f"  • Valid candidates (files exist): {valid_candidates}")
        print_rank(f"  • Missing files: {len(self.image_splits) - valid_candidates}")
        
        # Verify we have sufficient candidates for hard negative mining
        if len(self.retrieval_candidates) < 1000:
            print_rank(f"⚠️  Warning: Only {len(self.retrieval_candidates)} retrieval candidates found.")
            print_rank(f"    This might be insufficient for high-quality hard negative mining.")
            print_rank(f"    Expected ~16,000+ candidates for CIRR dataset.")
        else:
            print_rank(f"✅ Excellent! {len(self.retrieval_candidates)} candidates available for hard negative mining.")
            print_rank(f"    This is {len(self.retrieval_candidates)/200:.1f}x more than the previous limited approach.")
        
        # Verify that training samples are covered by retrieval candidates
        self._validate_candidate_coverage()
        
        return self.retrieval_candidates
    
    def _get_full_image_path(self, image_path: str) -> str:
        """
        统一处理图片路径，将相对路径转换为绝对路径
        
        Args:
            image_path: 图片路径，可能是相对路径（如 './train/34/image.png'）或绝对路径
            
        Returns:
            完整的绝对路径
        """
        if not isinstance(image_path, str):
            return str(image_path)
        
        if image_path.startswith('./'):
            # 移除开头的 './' 并与base_dir合并
            return os.path.join(self.image_base_dir, image_path[2:])
        elif os.path.isabs(image_path):
            # 已经是绝对路径
            return image_path
        else:
            # 相对路径，与base_dir合并
            return os.path.join(self.image_base_dir, image_path)
    
    def _validate_candidate_coverage(self):
        """Validate that all reference and target images from training data are in candidate set"""
        print_rank("Validating candidate set coverage...")
        
        # Create a set of candidate paths for fast lookup
        candidate_paths_set = set(self.retrieval_candidates)
        
        # Check coverage of training samples
        missing_refs = 0
        missing_targets = 0
        total_refs = set()
        total_targets = set()
        
        for ann in self.annotations:
            # Get reference and target paths
            ref_path = self.image_splits.get(ann['reference'], ann['reference'])
            target_path = self.image_splits.get(ann['target_hard'], ann['target_hard'])
            
            total_refs.add(ref_path)
            total_targets.add(target_path)
            
            if ref_path not in candidate_paths_set:
                missing_refs += 1
            if target_path not in candidate_paths_set:
                missing_targets += 1
        
        print_rank(f"Coverage validation results:")
        print_rank(f"  • Unique reference images: {len(total_refs)}")
        print_rank(f"  • Unique target images: {len(total_targets)}")
        print_rank(f"  • Missing reference images: {missing_refs}")
        print_rank(f"  • Missing target images: {missing_targets}")
        
        if missing_refs == 0 and missing_targets == 0:
            print_rank(f"✅ Perfect coverage! All training images are in the candidate set.")
        else:
            print_rank(f"⚠️  Coverage issues detected. This may affect hard negative mining quality.")
        
        coverage_rate = (len(total_refs | total_targets) - missing_refs - missing_targets) / len(total_refs | total_targets) * 100
        print_rank(f"  • Overall coverage rate: {coverage_rate:.1f}%")
    
    def _load_hard_negatives(self, iteration_round: int):
        """Load hard negatives from previous iteration"""
        # For iteration 0, there are no previous hard negatives
        if iteration_round <= 0:
            print_rank(f"Skipping hard negatives loading for iteration {iteration_round} (no previous iterations)")
            return
            
        cache_file = os.path.join(self.experiment_dir, f"hard_negatives_iter_{iteration_round-1}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.hard_negatives_cache = json.load(f)
            print_rank(f"Loaded hard negatives from iteration {iteration_round-1}: {cache_file}")
        else:
            print_rank(f"No hard negatives cache found for iteration {iteration_round-1}")
    
    def collect_hard_negatives_batch(self, retrieval_model, batch_size: int = 8, max_samples: int = None):
        """
        单卡硬负样本收集方法
        
        Args:
            retrieval_model: 检索模型
            batch_size: 批次大小
            max_samples: 最大样本数（用于fast mode）
        """
        print_rank(f"Starting single-GPU hard negative collection for iteration {self.iteration_round}")
        
        # 检查是否已有缓存
        if os.path.exists(self.hard_negatives_file):
            print_rank(f"Loading existing hard negatives from {self.hard_negatives_file}")
            with open(self.hard_negatives_file, 'r') as f:
                hard_negatives = json.load(f)
            
            # 应用max_samples限制
            if max_samples is not None and len(hard_negatives) > max_samples:
                hard_negatives = hard_negatives[:max_samples]
            
            self.hard_negatives_cache = hard_negatives
            print_rank(f"Loaded {len(hard_negatives)} existing hard negatives")
            return hard_negatives
        
        retrieval_model.eval()
        
        # 1. 确定样本范围
        if max_samples is not None:
            sample_limit = min(max_samples, len(self.annotations))
            print_rank(f"Using max_samples limit: {sample_limit}")
        else:
            sample_limit = len(self.annotations)
            print_rank(f"Processing all {sample_limit} samples")
        
        sample_annotations = self.annotations[:sample_limit]
        
        # 2. 预计算target embeddings（单卡模式）
        target_embeddings = self._get_or_compute_target_embeddings(
            self.retrieval_candidates, retrieval_model,
            getattr(retrieval_model, 'processor', None),
            getattr(self.model_args, 'model_backbone', 'qwen2_vl'),
            next(retrieval_model.parameters()).device
        )
        
        # 3. 处理所有查询
        all_hard_negatives = []
        with torch.no_grad():
            for i in range(0, len(sample_annotations), batch_size):
                batch_annotations = sample_annotations[i:i+batch_size]
                
                print_rank(f"Processing batch {i//batch_size + 1}/{(len(sample_annotations) + batch_size - 1)//batch_size}")
                
                # 转换为检索格式
                batch = []
                for ann in batch_annotations:
                    batch.append({
                        'reference_image': self.image_splits.get(ann['reference'], ann['reference']),
                        'modification_text': ann['caption'],
                        'target_image': self.image_splits.get(ann['target_hard'], ann['target_hard'])
                    })
                
                # 运行检索
                retrieval_results = self._run_real_retrieval_with_cached_targets(
                    retrieval_model, batch, target_embeddings, max_samples
                )
                
                # 识别硬负样本
                batch_hard_negs = self._identify_hard_negatives(batch, retrieval_results)
                all_hard_negatives.extend(batch_hard_negs)
        
        print_rank(f"Collected {len(all_hard_negatives)} total hard negatives")
        
        # 4. 保存到文件
        with open(self.hard_negatives_file, 'w') as f:
            json.dump(all_hard_negatives, f, indent=2)
        
        print_rank(f"✅ Saved hard negatives to {self.hard_negatives_file}")
        self.hard_negatives_cache = all_hard_negatives
        
        return all_hard_negatives

    def collect_hard_negatives_batch_distributed(self, retrieval_model, batch_size: int = 8, max_samples: int = None):
        """
        (最小改动版) 分布式硬负样本收集：
        仅 rank0 计算整库 target embeddings 并落盘，其它 rank 不进入 barrier，轮询等待文件生成后再加载；
        避免长时间 barrier 导致 NCCL watchdog 超时。
        """
        import torch.distributed as dist
        import time
        from datetime import datetime
        if not dist.is_initialized():
            return self.collect_hard_negatives_batch(retrieval_model, batch_size, max_samples)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        print_rank(f"GPU {rank}: Starting minimal-change distributed hard negative collection (Iter {self.iteration_round})")
        # 若已有最终硬负样本文件直接加载
        if os.path.exists(self.hard_negatives_file):
            try:
                with open(self.hard_negatives_file, 'r') as f:
                    hard_negatives = json.load(f)
                if max_samples is not None and len(hard_negatives) > max_samples:
                    hard_negatives = hard_negatives[:max_samples]
                self.hard_negatives_cache = hard_negatives
                print_rank(f"GPU {rank}: Loaded existing hard negatives ({len(hard_negatives)})")
                return hard_negatives
            except Exception as e:
                print_rank(f"GPU {rank}: Failed loading existing hard negatives, will recompute: {e}")
        target_embeddings_cache_file = self._get_cache_file_path(self.retrieval_candidates)
        done_flag = target_embeddings_cache_file + '.done'
        # rank0 计算
        if rank == 0:
            print_rank("GPU 0: Computing / loading target embeddings (single-process)...")
            start_t = time.time()
            self._get_or_compute_target_embeddings(
                self.retrieval_candidates,
                retrieval_model,
                getattr(retrieval_model, 'processor', None),
                getattr(self.model_args, 'model_backbone', 'qwen2_vl'),
                'cuda:0' if torch.cuda.is_available() else 'cpu'
            )
            # 写完成标志（先确保主文件存在）
            if not os.path.exists(target_embeddings_cache_file):
                raise RuntimeError(f"GPU 0: Expected embeddings file not found after computation: {target_embeddings_cache_file}")
            with open(done_flag, 'w') as f:
                f.write(datetime.utcnow().isoformat())
            print_rank(f"GPU 0: Target embeddings ready. File + flag written. Time {(time.time()-start_t)/60:.1f} min")
        else:
            print_rank(f"GPU {rank}: Waiting for embeddings flag file (no barrier)")
        # 其它 rank 轮询等待
        if rank != 0:
            poll_start = time.time()
            max_wait = 3 * 3600  # 最长等待 3 小时
            last_log = 0
            while True:
                if os.path.exists(done_flag) and os.path.exists(target_embeddings_cache_file):
                    try:
                        # 试读一次防止写尚未完成
                        _ = os.path.getsize(target_embeddings_cache_file)
                        time.sleep(1)  # small grace
                        break
                    except Exception:
                        time.sleep(2)
                        continue
                time.sleep(2)
                waited = time.time() - poll_start
                if waited - last_log > 60:
                    print_rank(f"GPU {rank}: Still waiting embeddings... waited {int(waited//60)}m")
                    last_log = waited
                if waited > max_wait:
                    raise TimeoutError(f"GPU {rank}: Timeout waiting embeddings (> {max_wait/3600:.1f}h)")
        # 全部加载
        try:
            cached_data = torch.load(target_embeddings_cache_file, map_location=device)
            target_embeddings = cached_data['embeddings'].to(next(retrieval_model.parameters()).dtype)
            print_rank(f"GPU {rank}: Loaded shared target embeddings: {target_embeddings.shape}")
        except Exception as e:
            raise RuntimeError(f"GPU {rank}: Failed to load embeddings after flag present: {e}")
        # 查询划分
        sample_annotations = self.annotations[:max_samples] if max_samples is not None else self.annotations
        total_samples = len(sample_annotations)
        samples_per_gpu = (total_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_gpu
        end_idx = min(start_idx + samples_per_gpu, total_samples)
        local_annotations = sample_annotations[start_idx:end_idx]
        print_rank(f"GPU {rank}: Assigned {len(local_annotations)} / {total_samples} queries")
        retrieval_model.eval()
        local_hard_negatives = []
        if local_annotations:
            with torch.no_grad():
                for i in range(0, len(local_annotations), batch_size):
                    batch_annotations = local_annotations[i:i+batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(local_annotations) + batch_size - 1)//batch_size
                    print_rank(f"GPU {rank}: Query batch {batch_num}/{total_batches}")
                    batch = []
                    for ann in batch_annotations:
                        batch.append({
                            'reference_image': self.image_splits.get(ann['reference'], ann['reference']),
                            'modification_text': ann['caption'],
                            'target_image': self.image_splits.get(ann['target_hard'], ann['target_hard'])
                        })
                    try:
                        start_time = time.time()
                        retrieval_results = self._run_real_retrieval_with_cached_targets(
                            retrieval_model, batch, target_embeddings, max_samples
                        )
                        batch_hard_negs = self._identify_hard_negatives(batch, retrieval_results)
                        local_hard_negatives.extend(batch_hard_negs)
                        print_rank(f"GPU {rank}: Batch {batch_num} done ({time.time()-start_time:.1f}s), hard_negs +{len(batch_hard_negs)}")
                    except Exception as e:
                        print_rank(f"GPU {rank}: Error batch {batch_num}: {e}")
                        continue
        else:
            print_rank(f"GPU {rank}: No queries assigned (idle)")
        # 收集结果 (仍需一次同步，可接受；若 rank0 很慢这里只发生在查询阶段之后)
        gathered = [None] * world_size
        try:
            dist.all_gather_object(gathered, local_hard_negatives)
        except Exception as e:
            print_rank(f"GPU {rank}: all_gather_object failed: {e}")
            gathered = [[] for _ in range(world_size)]
            gathered[rank] = local_hard_negatives
        final_hard_negatives = []
        if rank == 0:
            for gidx, gres in enumerate(gathered):
                if gres is not None:
                    final_hard_negatives.extend(gres)
            with open(self.hard_negatives_file, 'w') as f:
                json.dump(final_hard_negatives, f, indent=2)
            print_rank(f"GPU 0: Saved {len(final_hard_negatives)} hard negatives -> {self.hard_negatives_file}")
            self.hard_negatives_cache = final_hard_negatives
        # 广播最终结果
        broadcast_container = [final_hard_negatives if rank == 0 else []]
        dist.broadcast_object_list(broadcast_container, src=0)
        if rank != 0:
            self.hard_negatives_cache = broadcast_container[0]
        print_rank(f"GPU {rank}: Done. Total hard negatives = {len(self.hard_negatives_cache)}")
        return self.hard_negatives_cache
    
    def collect_hard_negatives_batch_old(self, retrieval_model, batch_size: int = 8, max_samples: int = None):
        """
        (旧版) 分布式硬负样本收集。
        
        该版本与 collect_hard_negatives_batch_distributed 功能相同，但实现方式不同。
        """
        import torch.distributed as dist
        import time

        # 检查是否为分布式环境
        if not dist.is_initialized():
            return self.collect_hard_negatives_batch(retrieval_model, batch_size, max_samples)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        
        print_rank(f"GPU {rank}: Starting distributed hard negative collection (Iteration {self.iteration_round})")

        # 检查是否已有最终缓存文件
        if os.path.exists(self.hard_negatives_file):
            print_rank(f"GPU {rank}: Loading existing hard negatives from {self.hard_negatives_file}")
            try:
                with open(self.hard_negatives_file, 'r') as f:
                    hard_negatives = json.load(f)
                
                # 应用max_samples限制
                if max_samples is not None and len(hard_negatives) > max_samples:
                    hard_negatives = hard_negatives[:max_samples]
                
                self.hard_negatives_cache = hard_negatives
                print_rank(f"GPU {rank}: Loaded {len(hard_negatives)} existing hard negatives")
                return hard_negatives
            except Exception as e:
                print_rank(f"GPU {rank}: Error loading existing hard negatives: {e}, will recompute")

        # ====================================================================================
        # 步骤 1-3: 准备共享数据 (Target Embeddings)，采用 "Rank 0 计算，全体加载" 模式
        # ====================================================================================
        
        target_embeddings_cache_file = self._get_cache_file_path(self.retrieval_candidates)

        if rank == 0:
            print_rank("GPU 0 is preparing shared target embeddings...")
            # 调用单卡的 _get_or_compute_target_embeddings 函数，它内部会处理缓存和计算
            self._get_or_compute_target_embeddings(
                self.retrieval_candidates, 
                retrieval_model,
                getattr(retrieval_model, 'processor', None),
                getattr(self.model_args, 'model_backbone', 'qwen2_vl'),
                'cuda:0'  # 明确在 rank 0 的设备上
            )

        # 全体在此等待，确保 rank 0 已将最新的 embeddings 文件写入磁盘
        print_rank(f"GPU {rank}: Waiting at barrier for target embeddings to be ready...")
        dist.barrier()
        print_rank(f"GPU {rank}: Barrier passed. Loading shared embeddings...")

        # 所有进程从共享文件加载到各自的 GPU
        try:
            cached_data = torch.load(target_embeddings_cache_file, map_location=device, weights_only=True)
            target_embeddings = cached_data['embeddings'].to(next(retrieval_model.parameters()).dtype)
            print_rank(f"GPU {rank}: ✅ Successfully loaded {target_embeddings.shape[0]} shared target embeddings.")
        except Exception as e:
            raise RuntimeError(f"GPU {rank}: CRITICAL ERROR - Failed to load shared embeddings file '{target_embeddings_cache_file}': {e}")
            
        # ====================================================================================
        # 步骤 4: 并行处理查询任务
        # ====================================================================================
        
        sample_annotations = self.annotations[:max_samples] if max_samples is not None else self.annotations
        total_samples = len(sample_annotations)
        samples_per_gpu = (total_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_gpu
        end_idx = min(start_idx + samples_per_gpu, total_samples)
        local_annotations = sample_annotations[start_idx:end_idx]

        print_rank(f"GPU {rank}: Processing {len(local_annotations)} queries ({start_idx}-{end_idx})...")
        
        retrieval_model.eval()
        local_hard_negatives = []
        
        if local_annotations:
            with torch.no_grad():
                for i in range(0, len(local_annotations), batch_size):
                    batch_annotations = local_annotations[i:i+batch_size]
                    batch_num = i//batch_size + 1
                    total_batches = (len(local_annotations) + batch_size - 1)//batch_size
                    
                    print_rank(f"GPU {rank}: 🔍 Processing hard negative batch {batch_num}/{total_batches}")
                    
                    # 转换为检索格式
                    batch = []
                    for ann in batch_annotations:
                        batch.append({
                            'reference_image': self.image_splits.get(ann['reference'], ann['reference']),
                            'modification_text': ann['caption'],
                            'target_image': self.image_splits.get(ann['target_hard'], ann['target_hard'])
                        })
                    
                    try:
                        # 运行检索，带超时保护和错误处理
                        start_time = time.time()
                        retrieval_results = self._run_real_retrieval_with_cached_targets(
                            retrieval_model, batch, target_embeddings, max_samples
                        )
                        batch_time = time.time() - start_time
                        
                        # 识别硬负样本
                        batch_hard_negs = self._identify_hard_negatives(batch, retrieval_results)
                        local_hard_negatives.extend(batch_hard_negs)
                        
                        print_rank(f"GPU {rank}: ✅ Batch {batch_num} completed in {batch_time:.1f}s, found {len(batch_hard_negs)} hard negatives")
                        
                    except Exception as e:
                        print_rank(f"GPU {rank}: ❌ Error in batch {batch_num}: {e}")
                        print_rank(f"GPU {rank}: Skipping batch {batch_num} and continuing...")
                        continue
        else:
            print_rank(f"GPU {rank}: No local annotations to process")

        # ====================================================================================
        # 步骤 5-6: 高效收集结果并返回
        # ====================================================================================

        print_rank(f"GPU {rank}: Finished local processing with {len(local_hard_negatives)} hard negatives. Gathering results...")
        
        gathered_results = [None] * world_size
        try:
            # 使用原生通信收集结果（PyTorch 2.6.0 不支持 timeout 参数，使用默认超时）
            dist.all_gather_object(gathered_results, local_hard_negatives)
            print_rank(f"GPU {rank}: ✅ Successfully gathered results from all GPUs")
        except Exception as e:
            print_rank(f"GPU {rank}: ⚠️ dist.all_gather_object failed: {e}. One or more GPUs may have failed.")
            # 即使失败，也创建一个空列表，让主进程可以继续，避免程序崩溃
            gathered_results = [[] for _ in range(world_size)]
            gathered_results[rank] = local_hard_negatives  # 至少保证当前GPU的结果

        final_hard_negatives = []
        if rank == 0:
            for gpu_idx, gpu_result in enumerate(gathered_results):
                if gpu_result is not None:
                    final_hard_negatives.extend(gpu_result)
                    print_rank(f"Merged {len(gpu_result)} hard negatives from GPU {gpu_idx}")
                else:
                    print_rank(f"⚠️ WARNING: Received no result from GPU {gpu_idx}")
            
            print_rank(f"Collected {len(final_hard_negatives)} total hard negatives.")
            with open(self.hard_negatives_file, 'w') as f:
                json.dump(final_hard_negatives, f, indent=2)
            print_rank(f"✅ Saved hard negatives to {self.hard_negatives_file}")
            self.hard_negatives_cache = final_hard_negatives
        
        # 使用 broadcast_object_list 将最终列表广播给所有进程，确保大家返回的是同样的内容
        # 这对于后续的数据集状态一致性非常重要
        final_list_container = [final_hard_negatives if rank == 0 else []]
        dist.broadcast_object_list(final_list_container, src=0)
        
        # 非rank 0 进程也需要更新缓存
        if rank != 0:
            self.hard_negatives_cache = final_list_container[0]
        
        print_rank(f"🎯 GPU {rank}: Distributed hard negative collection completed with {len(final_list_container[0])} total samples")
        return final_list_container[0]
    
    def _run_retrieval_batch(self, model, batch, max_samples=None):
        """Run real retrieval for a batch of queries using the actual VLM2Vec model"""
        import torch.nn.functional as F
        
        batch_size = len(batch)
        print_rank(f"Running real retrieval for {batch_size} queries")
        
        try:
            return self._run_real_retrieval(model, batch, max_samples)
        except Exception as e:
            print_rank(f"Real retrieval failed: {e}, falling back to simplified retrieval")
            return self._run_simplified_retrieval(batch)
    
    def _run_real_retrieval(self, model, batch, max_samples=None):
        """Run real retrieval using VLM2Vec model"""
        import torch.nn.functional as F
        
        batch_size = len(batch)
        device = next(model.parameters()).device
        
        # Get model backbone and processor
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        processor = getattr(model, 'processor', None)
        
        if processor is None:
            print_rank("Warning: No processor found in model")
            raise Exception("No processor available")
        
        # Collect target images for retrieval database
        # Use the proper retrieval candidates built during initialization
        if max_samples is not None and max_samples <= 100:
            # Fast mode: use a reasonable subset for quick testing
            # But still ensure we have enough candidates for meaningful hard negative mining
            min_candidates = min(1000, len(self.retrieval_candidates))  # At least 1000 or all available
            candidate_targets = self.retrieval_candidates[:min_candidates]
            # 只有单卡模式或rank 0打印，避免分布式环境下的重复输出
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank(f"Fast mode: using {len(candidate_targets)} target candidates (subset of {len(self.retrieval_candidates)})")
        else:
            # Production mode: use ALL available retrieval candidates
            # This is the correct approach for finding true hard negatives
            candidate_targets = self.retrieval_candidates
            # 只有单卡模式或rank 0打印，避免分布式环境下的重复输出
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank(f"Production mode: using full retrieval candidate set ({len(candidate_targets)} images)")
        
        target_database = candidate_targets
        target_paths = candidate_targets
        
        # 只有单卡模式或rank 0打印，避免分布式环境下的重复输出
        if not dist.is_initialized() or dist.get_rank() == 0:
            print_rank(f"Retrieval database: {len(target_database)} target images")
        
        if len(target_database) == 0:
            raise Exception("No valid target images found")
        
        # 检查是否有缓存的target embeddings
        target_embeddings = self._get_or_compute_target_embeddings(
            target_database, model, processor, model_backbone, device
        )
        
        # 编码查询批次（参考图片 + 修改文本）
        with torch.no_grad():
            query_inputs = self._prepare_query_inputs(batch, processor, model_backbone, device)
            
            try:
                query_embeddings = model.encode_input(query_inputs)
                query_embeddings = self._process_embeddings(query_embeddings, len(batch), "query_embeddings")
                query_embeddings = query_embeddings.cpu()
                
            except Exception as e:
                print_rank(f"Error encoding queries: {e}")
                # 使用dummy embeddings作为fallback
                query_embeddings = torch.randn(len(batch), target_embeddings.size(1))
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.mm(query_embeddings, target_embeddings.t())
        
        # Get top-k retrieval results
        k = min(10, len(target_database))
        top_k_similarities, top_k_indices = torch.topk(similarities, k, dim=1, largest=True)
        
        # Find ground truth indices for each query
        gt_indices = []
        for query in batch:
            gt_target_path = query['target_image']
            try:
                gt_idx = target_paths.index(gt_target_path)
            except ValueError:
                # If GT not found in target database, use -1
                gt_idx = -1
            gt_indices.append(gt_idx)
        
        # Format results
        results = {
            "top_k_indices": top_k_indices.tolist(),
            "gt_indices": gt_indices,
            "similarities": top_k_similarities.tolist(),
            "target_paths": target_paths  # Include actual target paths for reference
        }
        
        # 只有单卡模式或rank 0打印，避免分布式环境下的重复输出
        if not dist.is_initialized() or dist.get_rank() == 0:
            print_rank(f"Real retrieval completed. Average top-1 similarity: {top_k_similarities[:, 0].mean():.4f}")
        return results
    
    def _get_cache_file_path(self, target_database):
        """生成target embeddings缓存文件路径"""
        import hashlib
        
        # 基于target database内容生成hash
        content_hash = hashlib.md5(str(sorted(target_database)).encode()).hexdigest()[:8]
        cache_filename = f"target_embeddings_{len(target_database)}_{content_hash}.pt"
        return os.path.join(self.experiment_dir, "cache", cache_filename)
    
    def _get_or_compute_target_embeddings(self, target_database, model, processor, model_backbone, device):
        """
        获取或计算target embeddings，使用缓存机制提高性能
        
        Args:
            target_database: 目标图片路径列表
            model: 编码模型
            processor: 模型processor
            model_backbone: 模型backbone名称
            device: 设备
            
        Returns:
            target_embeddings tensor
        """
        # 检查缓存
        cache_dir = os.path.join(self.experiment_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = self._get_cache_file_path(target_database)
        
        # 尝试加载缓存
        if os.path.exists(cache_file):
            try:
                print_rank(f"Loading cached target embeddings from {cache_file}")
                cached_data = torch.load(cache_file, map_location='cpu', weights_only=True)
                
                # 验证缓存有效性
                if (cached_data['target_paths'] == target_database and 
                    cached_data['embeddings'].size(0) == len(target_database)):
                    embeddings = cached_data['embeddings']
                    
                    # 确保数据类型与当前模型匹配
                    model_dtype = next(model.parameters()).dtype
                    embeddings = embeddings.to(dtype=model_dtype, device=device)
                    
                    print_rank(f"✅ Cache hit! Loaded {len(target_database)} target embeddings (dtype: {embeddings.dtype})")
                    return embeddings
                else:
                    print_rank("Cache validation failed, will recompute embeddings")
            except Exception as e:
                print_rank(f"Error loading cache: {e}, will recompute embeddings")
        
        # 计算新的embeddings
        print_rank(f"Computing target embeddings for {len(target_database)} images...")
        target_embeddings = self._compute_target_embeddings_batch(
            target_database, model, processor, model_backbone, device
        )
            
        # 保存到缓存
        try:
            cache_data = {
                'target_paths': target_database,
                'embeddings': target_embeddings.cpu(),
                'timestamp': time.time(),
                'model_backbone': model_backbone
            }
            torch.save(cache_data, cache_file)
            print_rank(f"💾 Cached target embeddings to {cache_file}")
        except Exception as e:
            print_rank(f"Warning: Failed to cache embeddings: {e}")
        
        return target_embeddings
    
    def _compute_target_embeddings_batch(self, target_database, model, processor, model_backbone, device):
        """
        批量计算target embeddings - 带进度显示
        """
        import time
        
        target_embeddings = []
        target_batch_size = 8  # 小批次以避免内存问题
        total_batches = (len(target_database) + target_batch_size - 1) // target_batch_size
        
        print_rank(f"Computing embeddings for {len(target_database)} target images in {total_batches} batches...")
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, len(target_database), target_batch_size):
                batch_idx = i // target_batch_size + 1
                batch_targets = target_database[i:i+target_batch_size]
                
                # 计算进度和ETA
                if batch_idx > 1:
                    elapsed = time.time() - start_time
                    avg_time_per_batch = elapsed / (batch_idx - 1)
                    remaining_batches = total_batches - batch_idx + 1
                    eta_seconds = avg_time_per_batch * remaining_batches
                    eta_str = f"ETA: {int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
                    progress_pct = (batch_idx - 1) / total_batches * 100
                else:
                    eta_str = "ETA: calculating..."
                    progress_pct = 0
                
                print_rank(f"  Batch {batch_idx:4d}/{total_batches} ({progress_pct:5.1f}%) - Processing {len(batch_targets)} images - {eta_str}")
                
                # 创建target输入（仅图片）
                target_inputs = self._prepare_target_inputs(batch_targets, processor, model_backbone, device)
                
                try:
                    target_embs = model.encode_input(target_inputs)
                    target_embs = self._process_embeddings(target_embs, len(batch_targets), f"target_batch_{batch_idx}")
                    target_embeddings.append(target_embs.cpu())
                    
                except Exception as e:
                    print_rank(f"Error encoding target batch {batch_idx}: {e}")
                    # 使用dummy embeddings作为fallback
                    dummy_embs = torch.randn(len(batch_targets), 768)
                    target_embeddings.append(dummy_embs)
        
        total_time = time.time() - start_time
        print_rank(f"✅ Target embeddings computation completed in {int(total_time//60):02d}:{int(total_time%60):02d}")
        print_rank(f"   Average speed: {len(target_database)/total_time:.1f} images/second")
        
        # 连接所有target embeddings
        final_embeddings = torch.cat(target_embeddings, dim=0)
        print_rank(f"   Final embeddings shape: {final_embeddings.shape}")
        return final_embeddings
    
    def _compute_target_embeddings_batch_local(self, target_candidates, model, processor, model_backbone, device, rank=0):
        """
        在单个GPU上计算target embeddings的子集（用于分布式计算）
        
        Args:
            target_candidates: 分配给当前GPU的候选图像列表
            model: 编码模型
            processor: 模型processor
            model_backbone: 模型backbone名称
            device: 设备
            rank: GPU编号（用于日志）
            
        Returns:
            当前GPU计算的embeddings tensor
        """
        import time
        
        target_embeddings = []
        target_batch_size = 8  # 小批次以避免内存问题
        total_batches = (len(target_candidates) + target_batch_size - 1) // target_batch_size
        
        print_rank(f"GPU {rank}: Computing embeddings for {len(target_candidates)} images in {total_batches} batches...")
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, len(target_candidates), target_batch_size):
                batch_idx = i // target_batch_size + 1
                batch_targets = target_candidates[i:i+target_batch_size]
                
                # 计算进度和ETA
                # 只有rank 0打印详细进度，避免输出混乱
                if rank == 0:
                    if batch_idx > 1:
                        elapsed = time.time() - start_time
                        avg_time_per_batch = elapsed / (batch_idx - 1)
                        remaining_batches = total_batches - batch_idx + 1
                        eta_seconds = avg_time_per_batch * remaining_batches
                        eta_str = f"ETA: {int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
                        progress_pct = (batch_idx - 1) / total_batches * 100
                    else:
                        eta_str = "ETA: calculating..."
                        progress_pct = 0
                    
                    print_rank(f"  📊 Batch {batch_idx:4d}/{total_batches} ({progress_pct:5.1f}%) - Processing {len(batch_targets)} images - {eta_str}")
                
                # 创建target输入（仅图片）
                target_inputs = self._prepare_target_inputs(batch_targets, processor, model_backbone, device)
                
                try:
                    target_embs = model.encode_input(target_inputs)
                    target_embs = self._process_embeddings(target_embs, len(batch_targets), f"target_batch_{batch_idx}")
                    target_embeddings.append(target_embs.cpu())
                    
                except Exception as e:
                    print_rank(f"GPU {rank}: Error encoding target batch {batch_idx}: {e}")
                    # 使用dummy embeddings作为fallback
                    dummy_embs = torch.randn(len(batch_targets), 768)
                    target_embeddings.append(dummy_embs)
        
        total_time = time.time() - start_time
        print_rank(f"GPU {rank}: ✅ Local embeddings computation completed in {int(total_time//60):02d}:{int(total_time%60):02d}")
        print_rank(f"GPU {rank}: Average speed: {len(target_candidates)/total_time:.1f} images/second")
        
        # 连接所有target embeddings
        if target_embeddings:
            final_embeddings = torch.cat(target_embeddings, dim=0)
            print_rank(f"GPU {rank}: Final local embeddings shape: {final_embeddings.shape}")
            return final_embeddings.to(device)
        else:
            # 返回空tensor
            return torch.empty(0, 768, device=device)
    
    def _run_real_retrieval_with_cached_targets(self, model, batch, target_embeddings, max_samples=None):
        """
        使用预缓存的target embeddings进行检索（避免重复计算）
        
        Args:
            model: 检索模型
            batch: 查询批次
            target_embeddings: 预计算的target embeddings
            max_samples: 最大样本数（用于确定候选集大小）
            
        Returns:
            检索结果
        """
        batch_size = len(batch)
        device = next(model.parameters()).device
        
        # 获取模型配置
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        processor = getattr(model, 'processor', None)
        
        if processor is None:
            raise Exception("No processor available")
        
        # 确定使用的候选集大小
        if max_samples is not None and max_samples <= 100:
            # Fast mode: 使用子集
            min_candidates = min(1000, len(self.retrieval_candidates))
            candidate_targets = self.retrieval_candidates[:min_candidates]
            used_target_embeddings = target_embeddings[:min_candidates].to(device)
            # 只有rank 0打印，避免分布式环境下的重复输出
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank(f"Fast mode: using {len(candidate_targets)} target candidates")
        else:
            # Production mode: 使用完整集合
            candidate_targets = self.retrieval_candidates
            used_target_embeddings = target_embeddings.to(device)
            # 只有rank 0打印，避免分布式环境下的重复输出
            if not dist.is_initialized() or dist.get_rank() == 0:
                print_rank(f"Production mode: using full candidate set ({len(candidate_targets)} images)")
        
        # 编码查询
        with torch.no_grad():
            query_inputs = self._prepare_query_inputs(batch, processor, model_backbone, device)
            
            try:
                query_embeddings = model.encode_input(query_inputs)
                query_embeddings = self._process_embeddings(query_embeddings, len(batch), "query_embeddings")
                query_embeddings = query_embeddings.to(device)
            except Exception as e:
                print_rank(f"Error encoding queries: {e}")
                query_embeddings = torch.randn(len(batch), used_target_embeddings.size(1), device=device)
        
        # 标准化embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        used_target_embeddings = F.normalize(used_target_embeddings, p=2, dim=1)
        
        # 确保数据类型一致（统一使用query_embeddings的数据类型，通常是bfloat16）
        used_target_embeddings = used_target_embeddings.to(query_embeddings.dtype)
        
        # 计算相似度
        similarities = torch.mm(query_embeddings, used_target_embeddings.t())
        
        # 获取top-k结果
        k = min(10, len(candidate_targets))
        top_k_similarities, top_k_indices = torch.topk(similarities, k, dim=1, largest=True)
        
        # 找到ground truth索引
        gt_indices = []
        for query in batch:
            gt_target_path = query['target_image']
            try:
                gt_idx = candidate_targets.index(gt_target_path)
            except ValueError:
                gt_idx = -1
            gt_indices.append(gt_idx)
        
        # 格式化结果
        results = {
            "top_k_indices": top_k_indices.tolist(),
            "gt_indices": gt_indices,
            "similarities": top_k_similarities.tolist(),
            "target_paths": candidate_targets
        }
        
        return results
    
    def _image_exists(self, image_path):
        """检查图片文件是否存在"""
        if not isinstance(image_path, str):
            return False
        
        full_path = self._get_full_image_path(image_path)
        return os.path.exists(full_path)
    
    def _process_embeddings(self, embeddings, expected_batch_size: int, embedding_type: str = "embeddings") -> torch.Tensor:
        """
        统一处理embeddings的维度检查、None值处理和尺寸验证
        
        Args:
            embeddings: 模型输出的embeddings
            expected_batch_size: 期望的batch size
            embedding_type: embeddings类型（用于日志）
            
        Returns:
            处理后的embeddings tensor
        """
        if embeddings is None:
            print_rank(f"Warning: {embedding_type} returned None, using dummy embeddings")
            return torch.randn(expected_batch_size, 768)
        
        # 处理0维tensor
        if embeddings.dim() == 0:
            print_rank(f"Warning: Got 0-d tensor for {embedding_type}, reshaping")
            if embeddings.numel() > 0:
                embeddings = embeddings.view(1, -1)
            else:
                return torch.randn(expected_batch_size, 768)
        
        # 处理1维tensor
        elif embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # 处理高维tensor（展平）
        elif embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
        
        # 检查batch size
        if embeddings.size(0) != expected_batch_size:
            print_rank(f"Warning: {embedding_type} batch size {embeddings.size(0)} != expected {expected_batch_size}")
            
            if embeddings.size(0) == 1 and expected_batch_size > 1:
                # 重复单个embedding到期望的batch size
                embeddings = embeddings.repeat(expected_batch_size, 1)
            else:
                # 使用dummy embeddings
                print_rank(f"Using dummy {embedding_type} due to size mismatch")
                return torch.randn(expected_batch_size, embeddings.size(-1) if embeddings.numel() > 0 else 768)
        
        return embeddings
    
    def _prepare_vlm_inputs(self, image_paths, texts, processor, model_backbone, device, input_type="general"):
        """
        统一准备VLM模型输入的函数
        
        Args:
            image_paths: 图片路径列表
            texts: 文本列表
            processor: 模型processor
            model_backbone: 模型backbone名称
            device: 设备
            input_type: 输入类型（"target", "query", "general"）
            
        Returns:
            处理好的模型inputs
        """
        from PIL import Image
        
        images = []
        processed_texts = []
        
        # 加载图片和准备文本
        for img_path, text in zip(image_paths, texts):
            try:
                full_path = self._get_full_image_path(img_path)
                image = Image.open(full_path).convert('RGB')
                images.append(image)
                processed_texts.append(text)
            except Exception as e:
                print_rank(f"Error loading image {img_path}: {e}")
                # 使用dummy图片和文本
                images.append(Image.new('RGB', (224, 224), color='white'))
                processed_texts.append(text if text else "")
        
        # 使用VLM2Vec的官方processor
        try:
            from src.model.processor import process_vlm_inputs_fns
            
            model_inputs = {
                'text': processed_texts,
                'images': images
            }
            
            if model_backbone in process_vlm_inputs_fns:
                inputs = process_vlm_inputs_fns[model_backbone](model_inputs, processor)
            else:
                raise ValueError(f"Model backbone {model_backbone} not supported in VLM2Vec")
            
            # 移动到设备
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to(device)
            
            return inputs
            
        except Exception as e:
            print_rank(f"Error in VLM2Vec processor for {input_type}: {e}")
            raise e
    
    def _prepare_target_inputs(self, target_paths, processor, model_backbone, device):
        """为目标图片准备输入（仅图片）"""
        
        # 为目标图片创建简单的描述性文本
        texts = []
        for _ in target_paths:
            target_text = process_input_text(
                instruction="Represent the given image", 
                model_backbone=model_backbone, 
                text="", 
                add_image_token=True
            )
            texts.append(target_text)
        
        return self._prepare_vlm_inputs(target_paths, texts, processor, model_backbone, device, "target")
    
    def _prepare_query_inputs(self, batch, processor, model_backbone, device):
        """为查询准备输入（参考图片 + 修改文本）"""
        
        # 提取图片路径和文本
        image_paths = [query['reference_image'] for query in batch]
        texts = []
        
        for query in batch:
            # 组合修改文本和图片token
            query_text = process_input_text(
                instruction="Represent the given image with the following modification", 
                model_backbone=model_backbone, 
                text=query['modification_text'], 
                add_image_token=True
            )
            texts.append(query_text)
        
        return self._prepare_vlm_inputs(image_paths, texts, processor, model_backbone, device, "query")
    
    def _run_simplified_retrieval(self, batch):
        """Fallback simplified retrieval with dummy results"""
        print_rank("Running simplified retrieval (fallback)")
        batch_size = len(batch)
        
        # Create more realistic dummy retrieval results
        dummy_results = {
            "top_k_indices": [],
            "gt_indices": [],  
            "similarities": []
        }
        
        for i in range(batch_size):
            # Simulate retrieval results where GT might not be top-1
            if i % 3 == 0:  # 1/3 of cases: GT not in top-1, but in top-3
                gt_idx = 2  # GT at position 2
                top_k = [1, 5, gt_idx, 7, 9, 3, 8, 4, 6, 0]  # GT at index 2
            elif i % 3 == 1:  # 1/3 of cases: GT at top-1 (perfect retrieval)
                gt_idx = 0
                top_k = [gt_idx, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:  # 1/3 of cases: GT not in top-1, but in top-5
                gt_idx = 4  # GT at position 4  
                top_k = [2, 7, 1, 9, gt_idx, 5, 3, 8, 6, 0]  # GT at index 4
            
            dummy_results["top_k_indices"].append(top_k)
            dummy_results["gt_indices"].append(gt_idx)
            dummy_results["similarities"].append([0.9 - j*0.1 for j in range(10)])
        
        return dummy_results
    
    def _identify_hard_negatives(self, batch, retrieval_results):
        """Identify hard negatives from retrieval results with reference image filtering"""
        hard_negatives = []
        
        top_k_indices = retrieval_results['top_k_indices']
        gt_indices = retrieval_results['gt_indices']
        similarities = retrieval_results['similarities']
        
        # Check if we have real target paths (from real retrieval) or just indices (from simulation)
        target_paths = retrieval_results.get('target_paths', None)
        is_real_retrieval = target_paths is not None
        
        for idx, (query, gt_target, top_k, sims) in enumerate(zip(
            batch, gt_indices, top_k_indices, similarities
        )):
            # 🔥 新增：过滤掉参考图像本身
            query_ref_path = query['reference_image']
            filtered_hard_negatives = []
            
            # Helper function to check if two images are the same
            def is_same_image(path1, path2):
                """Check if two image paths refer to the same image"""
                if not path1 or not path2:
                    return False
                
                # Convert to full paths for comparison
                full_path1 = self._get_full_image_path(path1)
                full_path2 = self._get_full_image_path(path2)
                
                # Normalize paths to handle symlinks, '../' etc.
                try:
                    norm_path1 = os.path.normpath(os.path.realpath(full_path1))
                    norm_path2 = os.path.normpath(os.path.realpath(full_path2))
                    return norm_path1 == norm_path2
                except:
                    # Fallback to simple basename comparison
                    return os.path.basename(path1) == os.path.basename(path2)
            
            # 🔥 Process each retrieved result and filter
            def process_negative_candidate(neg_pos, neg_idx, gt_position=-1):
                """Process a negative candidate and add if valid"""
                if is_real_retrieval:
                    hard_negative_image = target_paths[neg_idx] if neg_idx < len(target_paths) else f"target_{neg_idx}"
                else:
                    hard_negative_image = neg_idx
                
                # 🔥 关键过滤：检查是否为参考图像本身
                if is_real_retrieval and is_same_image(query_ref_path, hard_negative_image):
                    print_rank(f"Filtered out reference image as hard negative: {query_ref_path}")
                    return False  # 跳过这个候选
                
                # 🔥 可选：额外的过滤条件
                # 检查是否与ground truth相同（避免将正样本当作负样本）
                if is_real_retrieval and 'target_image' in query and is_same_image(query['target_image'], hard_negative_image):
                    print_rank(f"Filtered out ground truth as hard negative: {hard_negative_image}")
                    return False
                
                # 添加有效的硬负样本
                filtered_hard_negatives.append({
                    'reference_image': query['reference_image'],
                    'modification_text': query['modification_text'],
                    'target_image': query['target_image'],  # GT
                    'hard_negative_image': hard_negative_image,
                    'rank_position': neg_pos + 1,
                    'gt_rank': gt_position + 1 if gt_position >= 0 else -1,
                    'similarity_score': sims[neg_pos] if neg_pos < len(sims) else 0.0,
                    'is_real_retrieval': is_real_retrieval
                })
                return True
            
            # Handle different GT scenarios with filtering
            collected_negatives = 0  # 跟踪收集到的有效负样本数量
            max_negatives = 5       # 最大负样本数量
            
            if gt_target == -1:
                # GT not found in retrieval database, all top results are potential hard negatives
                for neg_pos in range(min(len(top_k), 10)):  # 检查更多候选以补偿过滤
                    if collected_negatives >= max_negatives:
                        break
                    if process_negative_candidate(neg_pos, top_k[neg_pos]):
                        collected_negatives += 1
                        
            elif gt_target in top_k:
                gt_position = top_k.index(gt_target)
                
                # If GT is not in top-1 but within top-k, collect hard negatives ranked higher
                if gt_position > 0 :
                    for neg_pos in range(gt_position):
                        if collected_negatives >= max_negatives:
                            break
                        if process_negative_candidate(neg_pos, top_k[neg_pos], gt_position):
                            collected_negatives += 1
            else:
                # GT not found in top-k, all top results are potential hard negatives
                for neg_pos in range(min(len(top_k), 10)):  # 检查更多候选
                    if collected_negatives >= max_negatives:
                        break
                    if process_negative_candidate(neg_pos, top_k[neg_pos]):
                        collected_negatives += 1
            
            # # 🔥 如果过滤后没有足够的负样本，尝试从排名更低的候选中选择
            # if collected_negatives < max_negatives and len(top_k) > max_negatives:
            #     print_rank(f"Only collected {collected_negatives} valid negatives, trying lower-ranked candidates...")
            #     for neg_pos in range(max_negatives, min(len(top_k), 20)):
            #         if collected_negatives >= max_negatives:
            #             break
            #         if process_negative_candidate(neg_pos, top_k[neg_pos]):
            #             collected_negatives += 1
            
            # Add filtered results to main list
            hard_negatives.extend(filtered_hard_negatives)
            
            if len(filtered_hard_negatives) > 0:
                print_rank(f"Query {idx}: Collected {len(filtered_hard_negatives)} valid hard negatives after filtering")
            else:
                print_rank(f"Query {idx}: No valid hard negatives found after filtering")
        
        return hard_negatives
    
    def generate_augmented_captions_distributed(self, hard_negatives: List[Dict]):
        """
        多卡并行的增强caption生成
        
        Args:
            hard_negatives: 硬负样本列表
            
        Returns:
            增强样本列表
        """
        import torch.distributed as dist
        import time
        import json
        import traceback
        
        if not self.foundation_model:
            print_rank("No foundation model provided, skipping caption generation")
            return []
        
        # 检查缓存
        next_iteration = self.iteration_round + 1
        aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iteration}.json")
        
        if os.path.exists(aug_file):
            # 所有GPU都直接从文件读取，避免broadcast
            print_rank(f"Loading existing augmented samples from {aug_file}")
            try:
                with open(aug_file, 'r') as f:
                    saved_data = json.load(f)
                augmented_samples = saved_data.get('samples', [])
            except Exception as e:
                print_rank(f"Error loading augmented samples: {e}, regenerating...")
                augmented_samples = []
            
            if augmented_samples:
                # 过滤掉无效的caption
                print_rank(f"Filtering loaded augmented samples from {len(augmented_samples)} samples...")
                augmented_samples = self._filter_valid_augmented_samples(augmented_samples)
                
                self.augmented_samples = augmented_samples
                print_rank(f"Loaded {len(augmented_samples)} valid existing augmented samples")
                return augmented_samples
        
        if not dist.is_initialized():
            # 单卡模式，使用原有逻辑
            return self.generate_augmented_captions(hard_negatives)
        
        # 多卡模式
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        print_rank(f"Starting distributed caption generation for {len(hard_negatives)} hard negatives using {world_size} GPUs")
        
        # Debug: check type of hard_negatives
        print_rank(f"DEBUG: hard_negatives type = {type(hard_negatives)}")
        
        # Ensure hard_negatives is a list
        if not isinstance(hard_negatives, list):
            print_rank(f"WARNING: hard_negatives is not a list, converting from {type(hard_negatives)}")
            if hasattr(hard_negatives, '__iter__'):
                hard_negatives = list(hard_negatives)
            else:
                print_rank(f"ERROR: hard_negatives is not iterable: {hard_negatives}")
                return []
        
        # 1. 分配任务到各个GPU
        total_negatives = len(hard_negatives)
        per_gpu_negatives = (total_negatives + world_size - 1) // world_size
        start_idx = rank * per_gpu_negatives
        end_idx = min(start_idx + per_gpu_negatives, total_negatives)
        local_hard_negatives = hard_negatives[start_idx:end_idx]
        
        print_rank(f"GPU {rank}: Processing hard negatives {start_idx}-{end_idx} ({len(local_hard_negatives)} samples)")
        
        # 2. 每个GPU独立生成captions
        local_augmented_samples = []
        print_rank(f"GPU {rank}: Starting caption generation for {len(local_hard_negatives)} samples")
        
        if local_hard_negatives:
            # 确保foundation model在当前GPU上
            device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            if hasattr(self.foundation_model, 'to'):
                self.foundation_model = self.foundation_model.to(device)
            
            # 批量处理
            batch_size = 4  # 每个GPU的batch size
            total_batches = (len(local_hard_negatives) + batch_size - 1) // batch_size
            start_time = time.time()
            
            for i in range(0, len(local_hard_negatives), batch_size):
                batch_idx = i // batch_size + 1
                batch_hard_negs = local_hard_negatives[i:i+batch_size]
                
                # 所有GPU都打印进度信息（但添加GPU标识避免混乱）
                if batch_idx % 5 == 1 or rank == 0:  # rank 0 打印所有，其他GPU每5个batch打印一次
                    # 计算ETA
                    if batch_idx > 1:
                        elapsed = time.time() - start_time
                        avg_time_per_batch = elapsed / (batch_idx - 1)
                        remaining_batches = total_batches - batch_idx + 1
                        eta_seconds = avg_time_per_batch * remaining_batches
                        eta_str = f"ETA: {int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
                    else:
                        eta_str = "ETA: calculating..."
                    
                    print_rank(f"GPU {rank}: 🔄 Processing caption batch {batch_idx}/{total_batches} ({len(batch_hard_negs)} samples) - {eta_str}")
                
                try:
                    batch_start_time = time.time()
                    batch_augmented = self._generate_caption_batch_single_gpu(batch_hard_negs)
                    batch_time = time.time() - batch_start_time
                    local_augmented_samples.extend(batch_augmented)
                    
                    # 所有GPU都打印批次完成信息（但降低频率）
                    if batch_idx % 5 == 0 or rank == 0 or batch_idx == total_batches:
                        print_rank(f"GPU {rank}: ✅ Batch {batch_idx}/{total_batches} completed in {batch_time:.1f}s, generated {len(batch_augmented)} samples")
                    
                except Exception as e:
                    # 错误信息所有GPU都打印，因为需要调试
                    print_rank(f"❌ GPU {rank}: Error in batch {batch_idx}: {e}")
                    print_rank(f"GPU {rank}: Traceback: {traceback.format_exc()}")
                    continue
        else:
            print_rank(f"GPU {rank}: No samples to process, skipping caption generation")
        
        # 所有GPU都打印完成信息，用于调试同步问题
        print_rank(f"GPU {rank}: 🎯 Local caption generation completed: {len(local_augmented_samples)} samples")
        
        # 3. 文件式同步：避免NCCL超时问题
        sync_dir = os.path.join(self.experiment_dir, "sync_caption_gen")
        
        # 确保所有GPU都能看到同步目录
        if rank == 0:
            os.makedirs(sync_dir, exist_ok=True)
            print_rank(f"GPU {rank}: Created sync directory: {sync_dir}")
        
        # 等待同步目录创建完成（文件轮询）
        wait_count = 0
        max_wait_dir = 36000  # 10小时等待目录创建
        while not os.path.exists(sync_dir) and wait_count < max_wait_dir:
            time.sleep(1)
            wait_count += 1
            if wait_count % 10 == 0:
                print_rank(f"GPU {rank}: Still waiting for sync directory creation... ({wait_count}s)")
        
        if not os.path.exists(sync_dir):
            print_rank(f"GPU {rank}: ❌ Sync directory creation timeout, creating locally...")
            os.makedirs(sync_dir, exist_ok=True)
        
        print_rank(f"GPU {rank}: Sync directory ready: {sync_dir}")
        
        # 每个GPU创建完成标记文件
        completion_file = os.path.join(sync_dir, f"gpu_{rank}_completed.txt")
        try:
            with open(completion_file, 'w') as f:
                f.write(f"GPU {rank} completed caption generation with {len(local_augmented_samples)} samples at {time.time()}")
            print_rank(f"GPU {rank}: Created completion marker: {completion_file}")
        except Exception as e:
            print_rank(f"GPU {rank}: Error creating completion marker: {e}")
            # 尝试再次创建目录和文件
            try:
                os.makedirs(sync_dir, exist_ok=True)
                with open(completion_file, 'w') as f:
                    f.write(f"GPU {rank} completed caption generation with {len(local_augmented_samples)} samples at {time.time()}")
                print_rank(f"GPU {rank}: Successfully created completion marker after retry: {completion_file}")
            except Exception as e2:
                print_rank(f"GPU {rank}: Failed to create completion marker after retry: {e2}")
        
        # 等待所有GPU完成（文件轮询）
        print_rank(f"GPU {rank}: Waiting for all GPUs to complete caption generation...")
        max_wait_time = 36000  # 10个小时最大等待时间（caption生成非常耗时）
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait_time:
            all_completed = True
            for check_rank in range(world_size):
                check_file = os.path.join(sync_dir, f"gpu_{check_rank}_completed.txt")
                if not os.path.exists(check_file):
                    all_completed = False
                    break
            
            if all_completed:
                print_rank(f"GPU {rank}: ✅ All GPUs completed caption generation")
                break
            
            time.sleep(5)  # 每5秒检查一次
            
            # 每2分钟打印一次等待状态（由于等待时间较长，增加报告频率）
            if int(time.time() - start_wait) % 120 == 0:
                completed_gpus = []
                for check_rank in range(world_size):
                    check_file = os.path.join(sync_dir, f"gpu_{check_rank}_completed.txt")
                    if os.path.exists(check_file):
                        completed_gpus.append(check_rank)
                print_rank(f"GPU {rank}: Still waiting... Completed GPUs: {completed_gpus}")
        else:
            print_rank(f"GPU {rank}: ❌ Timeout waiting for all GPUs to complete")
            # 继续执行，不等待其他GPU
        
        # 4. 文件式数据收集：避免大数据量NCCL传输
        # 每个GPU直接保存到独立文件，避免网络传输超时
        temp_dir = os.path.join(self.experiment_dir, "temp_caption_results")
        
        # 确保所有GPU都能看到临时目录
        if rank == 0:
            os.makedirs(temp_dir, exist_ok=True)
            print_rank(f"GPU {rank}: Created temp directory: {temp_dir}")
        
        # 等待目录创建完成（文件轮询）
        wait_count = 0
        max_wait_temp = 36000  # 10小时等待临时目录创建
        while not os.path.exists(temp_dir) and wait_count < max_wait_temp:
            time.sleep(1)
            wait_count += 1
            if wait_count % 10 == 0:
                print_rank(f"GPU {rank}: Still waiting for temp directory creation... ({wait_count}s)")
        
        if not os.path.exists(temp_dir):
            print_rank(f"GPU {rank}: ❌ Temp directory creation timeout, creating locally...")
            os.makedirs(temp_dir, exist_ok=True)
        
        print_rank(f"GPU {rank}: Temp directory ready: {temp_dir}")
        
        # 每个GPU保存自己的结果到独立文件
        local_file = os.path.join(temp_dir, f"gpu_{rank}_samples.json")
        try:
            with open(local_file, 'w') as f:
                json.dump({
                    'rank': rank,
                    'samples': local_augmented_samples,
                    'count': len(local_augmented_samples),
                    'timestamp': time.time()
                }, f, indent=2)
            print_rank(f"GPU {rank}: Saved {len(local_augmented_samples)} samples to {local_file}")
        except Exception as e:
            print_rank(f"GPU {rank}: Error saving samples to file: {e}")
        
        # 文件式同步：等待所有GPU完成文件写入
        print_rank(f"GPU {rank}: Waiting for all GPUs to save their files...")
        max_file_wait = 36000  # 10小时最大等待时间（文件保存也可能较慢）
        file_wait_start = time.time()
        
        while time.time() - file_wait_start < max_file_wait:
            all_files_exist = True
            missing_files = []
            for check_rank in range(world_size):
                check_file = os.path.join(temp_dir, f"gpu_{check_rank}_samples.json")
                if not os.path.exists(check_file):
                    all_files_exist = False
                    missing_files.append(check_rank)
            
            if all_files_exist:
                print_rank(f"GPU {rank}: ✅ All GPU files are ready")
                break
            
            # 每60秒打印一次等待状态，帮助调试
            elapsed = time.time() - file_wait_start
            if int(elapsed) % 60 == 0 and int(elapsed) > 0:
                print_rank(f"GPU {rank}: Still waiting for GPU files... Missing GPUs: {missing_files} (elapsed: {int(elapsed)}s)")
                # 每5分钟提供更详细的文件状态信息
                if int(elapsed) % 300 == 0:
                    print_rank(f"GPU {rank}: Detailed file status check at {int(elapsed)}s:")
                    for check_rank in range(world_size):
                        check_file = os.path.join(temp_dir, f"gpu_{check_rank}_samples.json")
                        exists = os.path.exists(check_file)
                        try:
                            size = os.path.getsize(check_file) if exists else 0
                            print_rank(f"GPU {rank}: - GPU {check_rank}: exists={exists}, size={size} bytes")
                        except:
                            print_rank(f"GPU {rank}: - GPU {check_rank}: exists={exists}, size=unknown")
            
            time.sleep(2)  # 每2秒检查一次
        else:
            # 超时后检查实际可用的文件
            available_files = []
            for check_rank in range(world_size):
                check_file = os.path.join(temp_dir, f"gpu_{check_rank}_samples.json")
                if os.path.exists(check_file):
                    available_files.append(check_rank)
            print_rank(f"GPU {rank}: ❌ Timeout waiting for all GPU files. Available GPUs: {available_files}")
            # 继续执行，处理可用的文件
        
        # 仅rank 0收集所有文件
        if rank == 0:
            all_augmented_samples = [local_augmented_samples]  # 先添加自己的
            
            # 从文件读取其他GPU的结果
            for source_rank in range(1, world_size):
                source_file = os.path.join(temp_dir, f"gpu_{source_rank}_samples.json")
                try:
                    if os.path.exists(source_file):
                        with open(source_file, 'r') as f:
                            gpu_data = json.load(f)
                        gpu_samples = gpu_data.get('samples', [])
                        all_augmented_samples.append(gpu_samples)
                        print_rank(f"Loaded {len(gpu_samples)} samples from GPU {source_rank} file")
                    else:
                        print_rank(f"Warning: No file found for GPU {source_rank}")
                        all_augmented_samples.append([])
                except Exception as e:
                    print_rank(f"Error reading from GPU {source_rank} file: {e}")
                    all_augmented_samples.append([])
        else:
            all_augmented_samples = [[] for _ in range(world_size)]  # 占位符
        
        # 5. 主进程合并和保存
        if rank == 0:
            merged_augmented_samples = []
            for gpu_samples in all_augmented_samples:
                if gpu_samples and isinstance(gpu_samples, list):
                    merged_augmented_samples.extend(gpu_samples)
                    print_rank(f"Merged {len(gpu_samples)} samples from a GPU")
            
            # 过滤掉无效的caption
            print_rank(f"Filtering invalid captions from {len(merged_augmented_samples)} samples...")
            merged_augmented_samples = self._filter_valid_augmented_samples(merged_augmented_samples)
            
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            print_rank(f"Caption generation completed in {int(total_time//60):02d}:{int(total_time%60):02d}")
            print_rank(f"Generated {len(merged_augmented_samples)} total valid augmented samples from {len(hard_negatives)} hard negatives")
            
            if total_time > 0:
                print_rank(f"Average generation rate: {len(merged_augmented_samples)/total_time:.2f} samples/second")
            
            # 保存到文件
            if merged_augmented_samples:
                self._save_augmented_samples(merged_augmented_samples)
            
            print_rank(f"✅ Saved {len(merged_augmented_samples)} total augmented samples from {world_size} GPUs")
            self.augmented_samples = merged_augmented_samples
            
            # 确保文件写入完成
            time.sleep(100)  # 给文件系统一点时间完成写入
        else:
            merged_augmented_samples = []
        
        # 6. 清理临时文件和同步目录
        if rank == 0:
            # 清理临时文件
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print_rank(f"Cleaned up temporary directory: {temp_dir}")
                
                # 清理同步目录
                if os.path.exists(sync_dir):
                    shutil.rmtree(sync_dir)
                    print_rank(f"Cleaned up sync directory: {sync_dir}")
            except Exception as e:
                print_rank(f"Warning: Could not clean up directories: {e}")
        
        # 文件轮询等待rank 0完成文件写入
        final_aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{self.iteration_round + 1}.json")
        print_rank(f"GPU {rank}: Waiting for final augmented samples file...")
        
        final_wait_start = time.time()
        max_final_wait = 36000  # 10小时最大等待时间（文件合并和写入）

        while time.time() - final_wait_start < max_final_wait:
            if os.path.exists(final_aug_file):
                print_rank(f"GPU {rank}: ✅ Final augmented samples file is ready")
                break
            
            # 每10秒打印一次等待状态
            elapsed = time.time() - final_wait_start
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                print_rank(f"GPU {rank}: Still waiting for final file... (elapsed: {int(elapsed)}s)")
            
            time.sleep(2)
            time.sleep(2)
        else:
            print_rank(f"GPU {rank}: ❌ Timeout waiting for final augmented samples file after {max_final_wait}s")
        
        # 7. 完全避免NCCL通信 - 所有GPU直接从文件读取
        # 减少文件系统等待时间
        time.sleep(0.5)  # 减少文件系统等待时间
        
        # 所有GPU（包括rank 0）都从最终文件读取，确保一致性
        final_augmented_samples = []
        
        if os.path.exists(final_aug_file):
            try:
                with open(final_aug_file, 'r') as f:
                    saved_data = json.load(f)
                final_augmented_samples = saved_data.get('samples', [])
                
                # 过滤掉无效的caption
                print_rank(f"GPU {rank}: Filtering loaded augmented samples from {len(final_augmented_samples)} samples...")
                final_augmented_samples = self._filter_valid_augmented_samples(final_augmented_samples)
                
                self.augmented_samples = final_augmented_samples
                print_rank(f"GPU {rank}: Successfully loaded {len(final_augmented_samples)} valid augmented samples from file")
            except Exception as e:
                print_rank(f"GPU {rank}: Error loading augmented samples from file: {e}")
                final_augmented_samples = []
                self.augmented_samples = []
        else:
            print_rank(f"GPU {rank}: Augmented samples file not found")
            final_augmented_samples = []
            self.augmented_samples = []
        
        print_rank(f"🎯 Distributed caption generation completed: {len(final_augmented_samples)} total samples")
        return final_augmented_samples
    
    def _generate_caption_batch_single_gpu(self, hard_negatives_batch: List[Dict]) -> List[Dict]:
        """
        单GPU的caption生成（用于分布式环境中的每个GPU）
        这是原有_generate_caption_batch逻辑的简化版本，包含内存优化
        """
        from PIL import Image
        
        augmented_samples = []
        
        # 获取foundation model详情
        foundation_processor = getattr(self.foundation_model, 'processor', None)
        foundation_backbone = getattr(self.model_args, 'foundation_model_backbone', 'qwen2_vl')
        
        if foundation_processor is None:
            print_rank("Foundation model has no processor")
            return []
        
        device = next(self.foundation_model.parameters()).device
        
        # 内存优化：清空缓存
        torch.cuda.empty_cache()
        
        # 处理每个样本
        for idx, hard_neg in enumerate(hard_negatives_batch):
            try:
                # 加载参考和目标图片
                ref_image = self._load_pil_image(hard_neg['reference_image'])
                
                # 确定目标图片路径
                if hard_neg.get('is_real_retrieval', False):
                    target_image_path = hard_neg['hard_negative_image']
                    target_image = self._load_pil_image(target_image_path)
                else:
                    target_image = self._load_pil_image(hard_neg['target_image'])
                
                # 生成新的修改文本
                new_mod_text = self._generate_modification_text(
                    ref_image, target_image, hard_neg['modification_text'],
                    foundation_processor, foundation_backbone, device, is_hard_negative=True
                )
                
                if new_mod_text and self._is_valid_caption(new_mod_text):
                    augmented_samples.append({
                        'reference_image': hard_neg['reference_image'],
                        'modification_text': new_mod_text,
                        'target_image': hard_neg['hard_negative_image'],
                        'original_mod_text': hard_neg['modification_text'],
                        'is_augmented': True,
                        'hard_negative_rank': hard_neg['rank_position'],
                        'similarity_score': hard_neg['similarity_score']
                    })
                
                # 内存优化：每个样本处理后清理
                del ref_image, target_image
                if idx % 5 == 0:  # 每5个样本清理一次缓存
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError as e:
                print_rank(f"CUDA OOM while processing hard negative {idx}: {e}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print_rank(f"Error processing hard negative {idx}: {e}")
                continue
        
        # 最终内存清理
        torch.cuda.empty_cache()
        
        return augmented_samples
    
    def generate_augmented_captions(self, hard_negatives: List[Dict]):
        """
        Use foundation model to generate new modification texts for hard negatives
        """
        if not self.foundation_model:
            print_rank("No foundation model provided, skipping caption generation")
            return []
        
        # Check if augmented samples already exist for this iteration
        # Note: We generate samples for the NEXT iteration, so use iteration_round + 1
        next_iteration = self.iteration_round + 1
        aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iteration}.json")
        if os.path.exists(aug_file):
            print_rank(f"Augmented samples already exist for iteration {next_iteration}, loading from {aug_file}")
            try:
                with open(aug_file, 'r') as f:
                    saved_data = json.load(f)
                augmented_samples = saved_data.get('samples', [])
                
                # 过滤掉无效的caption
                print_rank(f"Filtering loaded augmented samples from {len(augmented_samples)} samples...")
                augmented_samples = self._filter_valid_augmented_samples(augmented_samples)
                
                print_rank(f"Loaded {len(augmented_samples)} valid existing augmented samples")
                self.augmented_samples = augmented_samples
                return augmented_samples
            except Exception as e:
                print_rank(f"Error loading augmented samples: {e}, regenerating...")
        
        print_rank(f"No existing augmented samples found, generating new ones...")
        print_rank(f"Generating augmented captions for {len(hard_negatives)} hard negatives")
        print_rank(f"Processing in {(len(hard_negatives) + 3) // 4} batches of 4 samples each")
        augmented_samples = []
        
        # Batch process for efficiency
        batch_size = 4  # Small batch size to avoid memory issues
        total_batches = (len(hard_negatives) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for i in range(0, len(hard_negatives), batch_size):
            batch_idx = i // batch_size + 1
            batch_hard_negs = hard_negatives[i:i+batch_size]
            
            # Calculate ETA
            if batch_idx > 1:
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / (batch_idx - 1)
                remaining_batches = total_batches - batch_idx + 1
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_str = f"ETA: {int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
            else:
                eta_str = "ETA: calculating..."
            
            print_rank(f"Processing caption generation batch {batch_idx}/{total_batches} ({len(batch_hard_negs)} samples) - {eta_str}")
            
            try:
                batch_start_time = time.time()
                batch_augmented = self._generate_caption_batch(batch_hard_negs)
                batch_time = time.time() - batch_start_time
                augmented_samples.extend(batch_augmented)
                print_rank(f"Batch {batch_idx}/{total_batches} completed in {batch_time:.1f}s, generated {len(batch_augmented)} augmented samples")
                
                # 增量保存：每100个批次保存一次
                if batch_idx % 100 == 0 or batch_idx == total_batches:
                    print_rank(f"Performing incremental save at batch {batch_idx}/{total_batches}")
                    try:
                        self._save_augmented_samples_incremental(augmented_samples, batch_idx)
                    except Exception as save_e:
                        print_rank(f"Warning: Incremental save failed: {save_e}")
                
            except Exception as e:
                print_rank(f"Error generating captions for batch {batch_idx}/{total_batches}: {e}")
                # Skip this batch and continue
                continue
        
        # 过滤掉无效的caption
        print_rank(f"Filtering invalid captions from {len(augmented_samples)} samples...")
        augmented_samples = self._filter_valid_augmented_samples(augmented_samples)
        
        self.augmented_samples = augmented_samples
        total_time = time.time() - start_time
        print_rank(f"Caption generation completed in {int(total_time//60):02d}:{int(total_time%60):02d}")
        print_rank(f"Generated {len(augmented_samples)} valid augmented samples from {len(hard_negatives)} hard negatives")
        print_rank(f"Average generation rate: {len(augmented_samples)/total_time:.2f} samples/second")
        
        # Save augmented samples to experiment directory
        if len(augmented_samples) > 0:
            self._save_augmented_samples(augmented_samples)  # 这里才保存
        
        return augmented_samples
    
    def _generate_caption_batch(self, hard_negatives_batch: List[Dict]) -> List[Dict]:
        """批量生成标题以提高效率"""
        from PIL import Image
        
        augmented_samples = []
        
        # 获取foundation model详情
        foundation_processor = getattr(self.foundation_model, 'processor', None)
        foundation_backbone = getattr(self.model_args, 'foundation_model_backbone', 'qwen2_vl')
        
        if foundation_processor is None:
            print_rank("Foundation model has no processor")
            return []
        
        device = next(self.foundation_model.parameters()).device
        
        # 批量处理：准备所有图片和文本
        ref_images = []
        target_images = []
        original_texts = []
        hard_neg_data = []
        
        for hard_neg in hard_negatives_batch:
            try:
                # 加载参考和目标图片
                ref_image = self._load_pil_image(hard_neg['reference_image'])
                
                # 确定目标图片路径
                if hard_neg.get('is_real_retrieval', False):
                    target_image_path = hard_neg['hard_negative_image']
                    target_image = self._load_pil_image(target_image_path)
                else:
                    target_image = self._load_pil_image(hard_neg['target_image'])
                
                ref_images.append(ref_image)
                target_images.append(target_image)
                original_texts.append(hard_neg['modification_text'])
                hard_neg_data.append(hard_neg)
                
            except Exception as e:
                print_rank(f"Error preparing hard negative: {e}")
                continue
        
        # 批量生成修改文本
        if ref_images:
            generated_texts = self._generate_modification_texts_batch(
                ref_images, target_images, original_texts, 
                foundation_processor, foundation_backbone, device
            )
            
            # 构建增强样本
            for i, (hard_neg, generated_text) in enumerate(zip(hard_neg_data, generated_texts)):
                if generated_text and self._is_valid_caption(generated_text):
                    print_rank(f"  - Generated: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
                    augmented_samples.append({
                        'reference_image': hard_neg['reference_image'],
                        'modification_text': generated_text,
                        'target_image': hard_neg['hard_negative_image'],
                        'original_mod_text': hard_neg['modification_text'],
                        'is_augmented': True,
                        'hard_negative_rank': hard_neg['rank_position'],
                        'similarity_score': hard_neg['similarity_score']
                    })
        
        return augmented_samples
    
    def _generate_modification_texts_batch(self, ref_images, target_images, original_texts, 
                                         processor, model_backbone, device):
        """
        批量生成修改文本，充分利用GPU并行性
        
        Args:
            ref_images: 参考图片列表
            target_images: 目标图片列表  
            original_texts: 原始文本列表
            processor: foundation model processor
            model_backbone: foundation model backbone
            device: 设备
            
        Returns:
            生成的文本列表
        """
        generated_texts = []
        batch_size = min(4, len(ref_images))  # 适中的批次大小
        
        for i in range(0, len(ref_images), batch_size):
            batch_ref = ref_images[i:i+batch_size]
            batch_target = target_images[i:i+batch_size]
            batch_original = original_texts[i:i+batch_size]
            
            try:
                # 为当前批次生成文本
                batch_generated = self._generate_batch_with_foundation_model(
                    batch_ref, batch_target, batch_original,
                    processor, model_backbone, device
                )
                generated_texts.extend(batch_generated)
                
            except Exception as e:
                print_rank(f"Error in batch generation: {e}")
                # 失败时逐个处理
                for j in range(len(batch_ref)):
                    try:
                        single_text = self._generate_modification_text(
                            batch_ref[j], batch_target[j], batch_original[j],
                            processor, model_backbone, device, is_hard_negative=True
                        )
                        generated_texts.append(single_text)
                    except:
                        generated_texts.append(None)
        
        return generated_texts
    
    def _generate_batch_with_foundation_model(self, ref_images, target_images, original_texts,
                                            processor, model_backbone, device):
        """
        使用foundation model批量生成文本
        """
        try:
            foundation_model = self.foundation_model
            
            # 为批次中的每个样本创建prompt
            prompts = []
            for original_text in original_texts:
                if model_backbone in ['qwen2_vl', 'qwen']:
                    #将此处的prompt改成原修改文本
                    # prompt = self._create_qwen_prompt(original_text, is_hard_negative_context=True)
                    prompt = original_text
                elif model_backbone in ['llava', 'llava_next']:
                    prompt = self._create_llava_prompt_enhanced(original_text, is_hard_negative_context=True)
                else:
                    prompt = self._create_generic_prompt_enhanced(original_text, is_hard_negative_context=True)
                prompts.append(prompt)
            
            # 根据模型类型准备批量输入
            if model_backbone in ['qwen2_vl', 'qwen']:
                return self._generate_qwen_batch(ref_images, target_images, prompts, processor, device, foundation_model)
            elif model_backbone in ['llava', 'llava_next']:
                return self._generate_llava_batch(ref_images, target_images, prompts, processor, device, foundation_model)
            else:
                return self._generate_generic_batch(ref_images, target_images, prompts, processor, device, foundation_model)
                
        except Exception as e:
            print_rank(f"Error in foundation model batch generation: {e}")
            return [None] * len(ref_images)
    
    def _generate_qwen_batch(self, ref_images, target_images, prompts, processor, device, foundation_model):
        """批量使用Qwen2-VL生成文本"""
        generated_texts = []
        
        for ref_img, target_img, prompt in zip(ref_images, target_images, prompts):
            try:
                inputs = self._prepare_qwen_inputs(ref_img, target_img, prompt, processor, device)
                generated_text = self._generate_with_qwen(inputs, device, foundation_model)
                generated_texts.append(generated_text)
            except Exception as e:
                print_rank(f"Error generating with Qwen: {e}")
                generated_texts.append(None)
        
        return generated_texts
    
    def _generate_llava_batch(self, ref_images, target_images, prompts, processor, device, foundation_model):
        """批量使用LLaVA生成文本"""
        generated_texts = []
        
        for ref_img, target_img, prompt in zip(ref_images, target_images, prompts):
            try:
                inputs = self._prepare_llava_inputs(ref_img, target_img, prompt, processor, device)
                generated_text = self._generate_with_llava(inputs, device, foundation_model)
                generated_texts.append(generated_text)
            except Exception as e:
                print_rank(f"Error generating with LLaVA: {e}")
                generated_texts.append(None)
        
        return generated_texts
    
    def _generate_generic_batch(self, ref_images, target_images, prompts, processor, device, foundation_model):
        """批量使用通用模型生成文本"""
        generated_texts = []
        
        for ref_img, target_img, prompt in zip(ref_images, target_images, prompts):
            try:
                inputs = self._prepare_generic_inputs(ref_img, target_img, prompt, processor, device)
                generated_text = self._generate_with_generic_model(inputs, device, foundation_model)
                generated_texts.append(generated_text)
            except Exception as e:
                print_rank(f"Error generating with generic model: {e}")
                generated_texts.append(None)
        
        return generated_texts
    
    def _load_pil_image(self, image_path: str) -> Image.Image:
        """从路径加载PIL图片"""
        from PIL import Image
        
        try:
            full_path = self._get_full_image_path(image_path)
            return Image.open(full_path).convert('RGB')
        except Exception as e:
            print_rank(f"Error loading image {image_path}: {e}")
            # 返回dummy图片
            return Image.new('RGB', (224, 224), color='white')
    
    def _generate_modification_text(self, ref_image, target_image, original_text, 
                                  processor, model_backbone, device, is_hard_negative=False):
        """Generate new modification text using foundation model with hard negative awareness"""
        try:
            # Use the original foundation model (not MMEBModel) for generation
            foundation_model = self.foundation_model
            
            # Create prompt based on model type with hard negative context
            if model_backbone in ['qwen2_vl', 'qwen']:
                # 使用原修改文本作为prompt，与批量生成保持一致
                prompt = original_text
                inputs = self._prepare_qwen_inputs(ref_image, target_image, prompt, processor, device)
                return self._generate_with_qwen(inputs, device, foundation_model)
            elif model_backbone in ['llava', 'llava_next']:
                prompt = self._create_llava_prompt_enhanced(original_text, is_hard_negative_context=is_hard_negative) 
                inputs = self._prepare_llava_inputs(ref_image, target_image, prompt, processor, device)
                return self._generate_with_llava(inputs, device, foundation_model)
            else:
                # Generic approach with hard negative context
                prompt = self._create_generic_prompt_enhanced(original_text, is_hard_negative_context=is_hard_negative)
                inputs = self._prepare_generic_inputs(ref_image, target_image, prompt, processor, device)
                return self._generate_with_generic_model(inputs, device, foundation_model)
                
        except Exception as e:
            print_rank(f"Error in caption generation: {e}")
            return None
    
    def _create_qwen_prompt(self, original_text: str, is_hard_negative_context: bool = False) -> str:
        """
        Create improved prompt for Qwen2-VL model based on hard negative mining theory.
        
        Theory: When generating captions for hard negatives, the foundation model should understand
        that the original query (I_ref + T_mod) incorrectly retrieved I_tgt_i (hard negative).
        We need to generate new T_mod_i such that (I_ref, T_mod_i, I_tgt_i) becomes a valid positive triplet.
        """
        if is_hard_negative_context:
            # Hard negative mining context: generate T_mod_i to make hard negative become positive
            return f"""You are analyzing a retrieval error case. Here's the situation:

RETRIEVAL ERROR ANALYSIS:
- Reference image (first image) + Original query: "{original_text}"
- This combination INCORRECTLY retrieved the target image (second image)
- The target image should NOT have been retrieved with the original query

TASK: Generate a NEW modification text that would make this retrieval CORRECT.
- Create a description that, when combined with the reference image, should CORRECTLY retrieve the target image
- Focus on the visual differences that make the target image the RIGHT match
- Use completely different vocabulary and approach than the original query

New correct modification text:"""
        else:
            # Standard diversity prompts for regular samples
            diversity_prompts = [
                f"""You are an expert at describing visual changes. Looking at these two images, I need a modification instruction.

Original instruction: "{original_text}"

Create a NEW instruction that describes how to transform the reference image to match the target image. Make it different from the original but achieve the same visual transformation. Use varied vocabulary and phrasing.

New instruction:""",
                
                f"""Based on comparing the reference and target images, write a fresh description of the visual change needed.

Given instruction: "{original_text}"

Generate an alternative instruction that would lead to the same visual outcome but uses different words and structure. Focus on different aspects or details.

Alternative instruction:""",
                
                f"""I have two images and need a modification description. The original was: "{original_text}"

Looking at both images, write a NEW way to describe this same transformation. Use different terminology, focus on different visual elements, or approach from a different angle.

New description:"""
            ]
            
            import random
            return random.choice(diversity_prompts)
    
    def _create_llava_prompt_enhanced(self, original_text: str, is_hard_negative_context: bool = False) -> str:
        """Create improved prompt for LLaVA model based on hard negative mining theory"""
        return create_llava_prompt_enhanced(original_text, is_hard_negative_context)
    
    def _create_generic_prompt_enhanced(self, original_text: str, is_hard_negative_context: bool = False) -> str:
        """Create improved generic prompt based on hard negative mining theory"""
        return create_generic_prompt_enhanced(original_text, is_hard_negative_context)
    
    def _create_llava_prompt(self, original_text: str) -> str:
        """Create improved prompt for LLaVA model with more diversity"""
        prompt_templates = [
            f"""USER: I'm looking at two images. The original description was: "{original_text}". Generate a different way to describe the same visual transformation, using varied language and focusing on different details.

ASSISTANT:""",
            
            f"""USER: Compare these images. Original instruction: "{original_text}". Create an alternative description for the same change but with different words and approach.

ASSISTANT:""",
            
            f"""USER: Given this modification text: "{original_text}", write a fresh version that describes the same visual change using different vocabulary and perspective.

ASSISTANT:"""
        ]
        
        import random
        return random.choice(prompt_templates)
    
    def _create_generic_prompt(self, original_text: str) -> str:
        """Create improved generic prompt with more diversity"""
        prompt_templates = [
            f"""Original: "{original_text}". Generate a different description for the same visual change:""",
            f"""Rephrase this modification: "{original_text}". Use different words for the same transformation:""",
            f"""Alternative to "{original_text}": Create a varied description for the same change:"""
        ]
        
        import random
        return random.choice(prompt_templates)
    
    def _prepare_qwen_inputs(self, ref_image, target_image, prompt, processor, device):
        """Prepare inputs for Qwen2-VL"""
        modification_text = prompt
        # Qwen2-VL can handle multiple images in conversation
        system_prompt = """
        You are a multimodal edit auditor. You receive TWO images (Picture 1 = REFERENCE, Picture 2 = TARGET) and ONE input edit text.
        Your only goal: minimally rewrite the edit text so it is true in the TARGET image.

        ##Global rules (read carefully)
        -Image roles: The Picture 1 is REFERENCE (original), the Picture 2 is TARGET (to match).
        - Grounding:
            - Absolute requirements use TARGET-only evidence.
            - Relative requirements compare REFERENCE vs TARGET (e.g., "different breed", "face the other direction", "add/remove/replace/move").
        - No speculation: If something is blurry/occluded/unreadable, say "uncertain" and do not guess.
        - One QA per intent: The number of QAs must equal the number of atomic intents, with the same IDs and order (|qa| == |intents|; I1→I1, I2→I2, …).
        - Polar questions only: QA must be yes/no questions (general questions). Banned words: how, how many, how much, how long, how far, how big, how tall, how wide, what, which, where, when, why, who, whom, whose.
        - Prefixes for QA:
            - Absolute: start with In the TARGET image, ...
            - Relative: start with Comparing REFERENCE and TARGET, ...
            - Allowed auxiliaries immediately after the prefix: is/are/does/do/has/have/can/was/were/did.
        - Keep it minimal: Keep the final rewrite concise, grammatical, and directly supported by TARGET evidence.

        ##Output format contract (STRICT — no extra sections, no code fences, no JSON)
        - Output exactly these 6 sections(stages), in this order, with the exact section headers shown below.
        - Use ASCII quotes (") in key/value fields. Keep IDs unique (I1, I2, ...).
        - If any field is not applicable, write uncertain.

        ##Guidelines on determining the response <Response>
        - Responses include the Generate Captions, Decompose Intents, Generate Polar Questions, Answer Questions via Image Comparison, Local Edits, and Generate Text_new:
        - Every stage should be included in the response and can not be empty.
        Generate Captions:
        Reference: [REFERENCE] ...
        Target: [TARGET] ...

        Decompose Intents:
        I1 | span="..." | intent="..." | type=absolute|relative | objects=[...] | note="..."
        I2 | span="..." | intent="..." | type=absolute|relative | objects=[...] | note="..."

        Generate Polar Questions:
        I1 | scope=target_only|compare_ref_target | Q="In the TARGET image, ...?"
        I2 | scope=target_only|compare_ref_target | Q="Comparing REFERENCE and TARGET, ...?"

        Answer Questions via Image Comparison:
        I1 | A=Yes|No|Uncertain | evidence_target="..." | evidence_ref=""
        I2 | A=Yes|No|Uncertain | evidence_target="..." | evidence_ref="..."

        Local Edits:
        I1 | action=keep|rewrite|generalize|delete | before="..." | after="..." | reason="..."
        I2 | action=keep|rewrite|generalize|delete | before="..." | after="..." | reason="..."

        Generate Text_new:
        text_new="..."
        """
        #---Example 1---
        fs_input1 = """
        <Input>
            "REFERENCE": <image_path>,
            "TARGET": <image_path>,
            "Input edit text": "Shows a dog of a different breed of the same fur color standing on a white table."
            """
        fs_output1 = """
        Generate Captions:
            Reference: [REFERENCE] A short-haired brown dog with a collar sits outdoors on a paved area; tongue out; background is blurred urban/stone texture.
            Target: [TARGET] A long-haired cream/beige dog lies stretched out on a light-colored tabletop/bench indoors; floppy ears; long fur drapes over the edge.

        Decompose Intents:
            I1 | span="a dog of a different breed" | intent="target dog breed differs from REFERENCE" | type=relative | objects=["dog","breed_difference"] | note="requires comparing REFERENCE vs TARGET appearance"
            I2 | span="of the same fur color" | intent="target dog has same fur color as REFERENCE" | type=relative | objects=["dog","fur_color"] | note="color equality is a comparison claim"
            I3 | span="standing" | intent="the dog should be standing" | type=absolute | objects=["dog","pose"] | note="verify from target only"
            I4 | span="on a white table" | intent="the dog is on a white table" | type=absolute | objects=["dog","table","table_color=white"] | note="presence+color in target only"

        Generate Polar Questions:
            I1 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, is the dog in the TARGET a different breed than the dog in the REFERENCE?"
            I2 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, does the TARGET dog have the same fur color as the REFERENCE dog?"
            I3 | scope=target_only | Q="In the TARGET image, is the dog standing?"
            I4 | scope=target_only | Q="In the TARGET image, is the dog on a white table?"

        Answer Questions via Image Comparison:
            I1 | A=Yes | evidence_target="long hair, floppy ears, different head shape" | evidence_ref="short coat, upright ears"
            I2 | A=No | evidence_target="fur appears light cream/beige" | evidence_ref="fur appears medium brown"
            I3 | A=No | evidence_target="body and legs lying on the surface; not upright" | evidence_ref=""
            I4 | A=No | evidence_target="table surface is light-colored but not clearly white" | evidence_ref=""

        Local Edits:
            I1 | action=keep | before="a dog of a different breed" | after="a dog of a different breed" | reason="confirmed by comparison"
            I2 | action=rewrite | before="of the same fur color" | after="with light cream fur" | reason="target color differs from REFERENCE"
            I3 | action=rewrite | before="standing" | after="lying" | reason="target pose is lying"
            I4 | action=rewrite | before="on a white table" | after="on a light-colored table" | reason="color not clearly white"

        Generate Text_new:
            text_new="Shows a dog of a different breed with light cream fur lying on a light-colored table."
        """
        #---Example 2---
        fs_input2 = """
        <Input>
            "REFERENCE": <image_path>,
            "TARGET": <image_path>,
            "Input edit text": "Smaller dog, no background" 
        """

        fs_output2 = """
        Generate Captions:
            Reference: [REFERENCE] A slender grey dog stands on green grass in a yard. A wooden fence and trees are visible behind it. Full body is shown.
            Target: [TARGET] A close-up of a white-and-tan spaniel face with long ears. The background is pink like a poster, with white headline text at the top.

        Decompose Intents:
            I1 | span="Smaller dog" | intent="dog should appear smaller than in the REFERENCE" | type=relative | objects=["dog","size in frame"] | note="relative change of apparent size"
            I2 | span="no background" | intent="there should be no visible background (plain/transparent)" | type=absolute | objects=["background","emptiness/plain"] | note="absolute requirement about background presence"

        Generate Polar Questions:
            I1 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, does the dog appear smaller in the TARGET image than in the REFERENCE?"
            I2 | scope=target_only | Q="In the TARGET image, is there no background visible behind the dog?"

        Answer Questions via Image Comparison:
            I1 | A=No | evidence_target="the dog is a tight head close-up occupying most of the frame" | evidence_ref="full-body dog occupies a smaller portion of the frame"
            I2 | A=No | evidence_target="pink poster background with headline text is clearly visible" | evidence_ref=""

        Local Edits:
            I1 | action=rewrite | before="Smaller dog" | after="dog" | reason="target shows enlarged."
            I2 | action=rewrite | before="no background" | after="on a pink background with text" | reason="background is pink with headline text"

        Generate Text_new:
            text_new="dog on a pink background with text"
        """
        #---Example 3---
        fs_input3 = """
        <Input>
            "REFERENCE": <image_path>,
            "TARGET": <image_path>,
            "Input edit text": "Add two dogs."
        """

        fs_output3 = """
        Generate Captions:
            Reference: [REFERENCE] two white puppies are playing in the snow.
            Target: [TARGET] Three white dogs are playing on a snowy field.

        Decompose Intents:
            I1 | span="Add two dogs" | intent="add two dogs" | type=relative | objects=["dog"] | note="relative requirement"

        Generate Polar Questions:
            I1 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, does the TARGET image have two more dogs than the REFERENCE?"

        Answer Questions via Image Comparison:
            I1 | A=No | evidence_target="three white dogs are playing on a snowy field" | evidence_ref="two white puppies are playing in the snow"

        Local Edits:
            I1 | action=rewrite | before="Add two dogs" | after="Add a dog" | reason="target has one more dogs than REFERENCE"

        Generate Text_new:
            text_new="Add a dog"
        """
        input_text=f"""
        <Input>
            "REFERENCE": Picture 1,
            "TARGET": Picture 2,
            "Input edit text": {modification_text}
        
        =====================
        STAGES (run in order)
        =====================
        ##Generate Captions:
        - You must not refer to the input edit text in this stage.
        - You must mention their unique features in their own captitons.
        - Do not reuse sentences across Reference and Target. If a sentence would be identical, rewrite it to include a detail unique to that image, or write "uncertain".
        - Tag mapping: "Reference" ALWAYS refers to Picture 1; "Target" ALWAYS refers to Picture 2.
        - Reference: [REFERENCE] only describe the reference image(REFERENCE, Picture 1) in 3-5 literal sentences. Be thorough and accurate; list visible objects, attributes (color/material), counts, spatial relations, and readable text. Do not mention TARGET here. If unreadable, say "uncertain".
        - Target: [TARGET] only describe the target image(TARGET, Picture 2) in 3-5 literal sentences. Same constraints; do not mention REFERENCE here.

        ##Decompose Intents(only use edit text:{modification_text}):
        - One line per atomic intent, covering all content-bearing phrases in the input edit text (split by commas/"and/then"/parentheses).
        - you must only decompose the input edit text, do not refer to any information from the REFERENCE or TARGET.
        - Format (one per line):
            I# | span="..." | intent="..." | type=absolute|relative | objects=[... ] | note="one-line reason"
        - Hints:
            - Treat parenthetical modifiers as their own intents if they encode states/gestures/poses (e.g., "(fist closed)").
            - Count words exactly ("a/one/two/…") and keep them in span.
            - Max 6 intents; if more, merge logically but keep atomicity for verifiable units.

        ##Generate Polar Questions:
        - Create exactly one yes/no question per intent, same order and same ID.
        - Format (one per line):
            I# | scope=target_only|compare_ref_target | Q="...?"
        - Rules:
            - If type=absolute → scope=target_only and start with In the TARGET image, ...
            - If type=relative → scope=compare_ref_target and start with Comparing REFERENCE and TARGET, ...
            - Use only allowed auxiliaries after the prefix; no wh-words.

        ##Answer Questions via Image Comparison:
        - Answer each question (same IDs and order) with Yes/No/Uncertain and brief evidence based on the TARGET image(Picture 2).
        - Carefully consider the visibility of the object in the TARGET image(Picture 2),and only when the evidence is clear, you can answer "Yes".
        - Format (one per line):
            I# | A=Yes|No|Uncertain | evidence_target="..." | evidence_ref="..."
        - every question should be answered, and the answer should be based on the TARGET image only.
        - Notes:
            - If scope=target_only, fill evidence_ref="".
            - If scope=compare_ref_target, you can refer to the REFERENCE image for relative information, but the answer should be based on the TARGET image only.
            - If visibility is insufficient, choose Uncertain.

        ##Local Edits:
        - Map each intent to a local edit: Yes → keep, No → rewrite to match TARGET.
        - Format (one per line):
            I# | action=keep|rewrite|generalize|delete | before="(original span or clause)" | after="(new fragment)" | reason="(short why)"

        ##Generate Text_new:
        - Output a single fluent sentence (or two short clauses) by merging all local edits with action=keep or rewrite (skip delete; generalize if you marked generalize).
        - Format (one per line):
            text_new="..."
        - Rules:
            - Please keep all intents that you answered "Yes" in the local edits.
            - Use the TARGET image as the evidence source.
            - do not contain 'REFERENCE' or 'TARGET' in the text_new.
            - if rewrite, please rewrite it with target image's that feature.(e.g. "four dogs" but in target image only two dogs -> "two dogs")
            - try to keep the original format of the input edit text.(e.g. "a bird of a yellow color" -> "a bird of blue color")
        """
        conversation =[
            {"role": "system", "content": system_prompt},
            #---Example 1---
            {"role": "user", "content": fs_input1},
            {"role": "assistant", "content": fs_output1},
            #---Example 2---
            {"role": "user", "content": fs_input2},
            {"role": "assistant", "content": fs_output2},
            #---Example 3---
            {"role": "user", "content": fs_input3},
            {"role": "assistant", "content": fs_output3},
            #---User Prompt--- 
            {"role": "user",
                "content": [
                    {"type": "image", "image": ref_image},
                    {"type": "image", "image": target_image},
                    {"type": "text", "text": input_text}
                ]}
        ]
        
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True,add_vision_id=True)
        
        inputs = processor(
            text=[text_prompt],
            images=[ref_image, target_image],
            return_tensors="pt",
            padding=True,
        )
        
        return {k: v.to(device) for k, v in inputs.items()}
    
    def _prepare_llava_inputs(self, ref_image, target_image, prompt, processor, device):
        """Prepare inputs for LLaVA"""
        # Concatenate images horizontally for LLaVA
        from PIL import Image
        import numpy as np
        
        ref_array = np.array(ref_image)
        target_array = np.array(target_image)
        
        # Resize to same height
        h = min(ref_array.shape[0], target_array.shape[0])
        ref_resized = Image.fromarray(ref_array).resize((int(ref_array.shape[1] * h / ref_array.shape[0]), h))
        target_resized = Image.fromarray(target_array).resize((int(target_array.shape[1] * h / target_array.shape[0]), h))
        
        # Concatenate horizontally
        combined_width = ref_resized.width + target_resized.width
        combined_image = Image.new('RGB', (combined_width, h))
        combined_image.paste(ref_resized, (0, 0))
        combined_image.paste(target_resized, (ref_resized.width, 0))
        
        inputs = processor(
            text=prompt,
            images=combined_image,
            return_tensors="pt",
            padding=True
        )
        
        return {k: v.to(device) for k, v in inputs.items()}
    
    def _prepare_generic_inputs(self, ref_image, target_image, prompt, processor, device):
        """Prepare inputs for generic model"""
        # Use first image as primary, mention second in text
        enhanced_prompt = f"{prompt} (Comparing reference and target images)"
        
        inputs = processor(
            text=enhanced_prompt,
            images=ref_image,
            return_tensors="pt",
            padding=True
        )
        
        return {k: v.to(device) for k, v in inputs.items()}
    
    def _output_translator(self, generated_text):
        """Extract text_new from model output"""
        import json
        import re
        
        if not generated_text:
            return None
        
        try:
            # 方法1: 尝试直接解析JSON
            parsed_json = json.loads(generated_text.strip())
            if isinstance(parsed_json, dict) and "text_new" in parsed_json:
                return parsed_json["text_new"]
        except json.JSONDecodeError:
            pass
        
        try:
            # 方法2: 尝试匹配```json代码块中的JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed_json = json.loads(json_str)
                if isinstance(parsed_json, dict) and "text_new" in parsed_json:
                    return parsed_json["text_new"]
        except json.JSONDecodeError:
            pass
        
        try:
            # 方法3: 尝试匹配大括号内的JSON内容
            brace_match = re.search(r'\{[^{}]*"text_new"\s*:\s*"([^"]*)"[^{}]*\}', generated_text, re.DOTALL)
            if brace_match:
                return brace_match.group(1)
        except:
            pass
        
        try:
            # 方法4: 直接匹配 "text_new": "内容" 格式
            text_new_match = re.search(r'"text_new"\s*:\s*"([^"]*)"', generated_text)
            if text_new_match:
                return text_new_match.group(1)
        except:
            pass
        
        try:
            # 方法5: 匹配 text_new="内容" 格式（不带引号的key）
            text_new_match = re.search(r'text_new\s*[=:]\s*"([^"]*)"', generated_text)
            if text_new_match:
                return text_new_match.group(1)
        except:
            pass
        
        try:
            # 方法6: 匹配 "Text_new": "内容" 格式（不带引号的key）
            text_new_match = re.search(r'"Text_new"\s*[:=]\s*"([^"]*)"', generated_text)
            if text_new_match:
                return text_new_match.group(1)
        except:
            pass
        
        # 如果所有方法都失败，返回空字符串
        return ' '
    
    def _generate_with_qwen(self, inputs, device, foundation_model):
        """Generate text with Qwen2-VL with memory optimization"""
        try:
            with torch.no_grad():
                # Clear cache before generation to free up memory
                torch.cuda.empty_cache()
                
                output_ids = foundation_model.generate(
                    **inputs,
                    max_new_tokens=2048,# 增加了最大输出token数
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=foundation_model.config.eos_token_id
                )
                 
                #  Decode only the new tokens
                input_length = inputs['input_ids'].shape[1]
                generated_ids = output_ids[:, input_length:]
                
                generated_text = foundation_model.processor.decode(
                    generated_ids[0], skip_special_tokens=True
                ).strip()
                # generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
                # generated_text = foundation_model.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

                # Clear intermediate results from memory
                del output_ids, generated_ids
                torch.cuda.empty_cache()
                
                # Post-process the generated text
                # cleaned_text = self._post_process_generated_text(generated_text)
                #重写新的适配的模型输出解析器。
                cleaned_text=self._output_translator(generated_text)
                return cleaned_text
                
        except torch.cuda.OutOfMemoryError as e:
            # Handle OOM gracefully
            print_rank(f"CUDA OOM in caption generation: {e}")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            print_rank(f"Error in Qwen generation: {e}")
            torch.cuda.empty_cache()
            return None
    
    def _post_process_generated_text(self, text: str) -> str:
        """Post-process generated text to improve quality"""
        if not text:
            return None
            
        # Remove quotes at the beginning and end
        text = text.strip('"\'')
        
        # Filter out problematic patterns
        problematic_patterns = [
            "create a new query",
            "retrieve the",
            "can you please",
            "look at both images",
            "instruction",
            "task:",
            "solution:",
            "problem:",
            "new correct modification",
            "explore the majestic",
            "handcrafted",
            "collection of"
        ]
        
        text_lower = text.lower()
        for pattern in problematic_patterns:
            if pattern in text_lower:
                return None
        
        # Remove coordinate patterns like (123,456)
        import re
        text = re.sub(r'\(\d+,\d+\)', '', text)
        text = re.sub(r'\(\d+,\d+\),\(\d+,\d+\)', '', text)
        
        # Remove overly long texts (likely descriptions rather than modification instructions)
        if len(text) > 150:
            return None
            
        # Remove texts that are too short to be meaningful
        if len(text) < 5:
            return None
            
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        # Ensure it starts with a capital letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            
        return text
    
    def _generate_with_llava(self, inputs, device, foundation_model):
        """Generate text with LLaVA with memory optimization"""
        try:
            with torch.no_grad():
                # Clear cache before generation
                torch.cuda.empty_cache()
                
                output_ids = foundation_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
                generated_text = foundation_model.processor.decode(
                    output_ids[0], skip_special_tokens=True
                ).strip()
                
                # Clear intermediate results
                del output_ids
                torch.cuda.empty_cache()
                
                # Extract only the assistant response
                if "ASSISTANT:" in generated_text:
                    generated_text = generated_text.split("ASSISTANT:")[-1].strip()
                
                # Post-process the generated text
                cleaned_text = self._post_process_generated_text(generated_text)
                
                return cleaned_text
                
        except torch.cuda.OutOfMemoryError as e:
            print_rank(f"CUDA OOM in LLaVA generation: {e}")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            print_rank(f"Error in LLaVA generation: {e}")
            torch.cuda.empty_cache()
            return None
    
    def _generate_with_generic_model(self, inputs, device, foundation_model):
        """Generate text with generic model with memory optimization"""
        try:
            with torch.no_grad():
                # Clear cache before generation
                torch.cuda.empty_cache()
                
                output_ids = foundation_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7
                )
                
                generated_text = foundation_model.processor.decode(
                    output_ids[0], skip_special_tokens=True
                ).strip()
                
                # Clear intermediate results
                del output_ids
                torch.cuda.empty_cache()
                
                return generated_text
                
        except torch.cuda.OutOfMemoryError as e:
            print_rank(f"CUDA OOM in generic generation: {e}")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            print_rank(f"Error in generic generation: {e}")
            torch.cuda.empty_cache()
            return None
      
    
    def __len__(self):
        base_len = len(self.annotations) if hasattr(self, 'annotations') else 0
        aug_len = len(self.augmented_samples)
        total_len = base_len + aug_len
        
        # Update num_rows for VLM2Vec compatibility
        self.num_rows = total_len
        return total_len
    
    def __getitem__(self, idx):
        # Return original samples or augmented samples
        if idx < len(self.annotations):
            return self._get_original_sample(idx)
        else:
            aug_idx = idx - len(self.annotations)
            return self._get_augmented_sample(aug_idx)
    
    # HuggingFace Trainer兼容性属性
    @property
    def _distributed(self):
        """标识此数据集支持分布式"""
        return True
    
    @property
    def _ex_iterable(self):
        """标识此数据集不是可迭代的"""
        return False
    
    def shard(self, num_shards: int, index: int):
        """
        HuggingFace Trainer兼容的数据集分片方法 - 高效实现
        
        Args:
            num_shards: 总的分片数量（通常等于GPU数量）
            index: 当前分片的索引（0到num_shards-1）
            
        Returns:
            分片后的数据集实例
        """
        import copy
        
        # 计算每个shard应该包含的样本范围
        total_samples = len(self.annotations)
        per_shard_samples = (total_samples + num_shards - 1) // num_shards  # 向上取整
        start_idx = index * per_shard_samples
        end_idx = min(start_idx + per_shard_samples, total_samples)
        
        # 创建当前对象的浅拷贝，避免重新运行__init__中的耗时操作
        sharded_dataset = copy.copy(self)
        
        # 只覆盖需要分片的数据
        sharded_dataset.annotations = self.annotations[start_idx:end_idx]
        
        # 动态更新 __len__ 依赖的 num_rows
        sharded_dataset.num_rows = len(sharded_dataset.annotations)
        
        # 由于是浅拷贝，其他所有属性（如image_splits, retrieval_candidates等）都已存在，无需再次赋值
        
        print_rank(f"Sharded dataset: GPU {index}/{num_shards} gets samples {start_idx}-{end_idx} ({len(sharded_dataset.annotations)} samples)")
        return sharded_dataset
    
    def _get_original_sample(self, idx):
        """Get original CIRR sample"""
        sample = self.annotations[idx]
        
        # Get image paths from the mapping
        ref_image_path = self.image_splits.get(sample['reference'], sample['reference'])
        target_image_path = self.image_splits.get(sample['target_hard'], sample['target_hard'])
        
        # Get model backbone from model_args
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        
        # Use VLM2Vec's unified text processing for query (reference image + modification text)
        query_text = process_input_text(
            instruction="",  # No specific instruction for CIRR
            model_backbone=model_backbone,
            text=sample['caption'],
            add_image_token=True  # Add image token for reference image
        )
        
        # For positive (target), just add image token
        pos_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",  # Empty text for target image
            add_image_token=True
        )
        
        # For negative, also just add image token  
        neg_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True
        )
        
        return {
            'query_text': query_text,
            'query_image': self._load_image(ref_image_path),
            'pos_text': pos_text,
            'pos_image': self._load_image(target_image_path),
            'neg_text': neg_text,
            'neg_image': self._load_image(ref_image_path),  # Use reference as negative for now
            'global_dataset_name': 'CIRR',
            # 🔥 添加 reference_image 路径，供 GroupedBatchSampler 使用
            'reference_image': ref_image_path
        }
    
    def _get_augmented_sample(self, idx):
        """Get augmented sample"""
        sample = self.augmented_samples[idx]
        
        # Get model backbone from model_args
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        
        # Use VLM2Vec's unified text processing
        query_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text=sample['modification_text'],
            add_image_token=True
        )
        
        pos_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True
        )
        
        neg_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True
        )
        
        return {
            'query_text': query_text,
            'query_image': self._load_image(sample['reference_image']),
            'pos_text': pos_text,
            'pos_image': self._load_image(sample['target_image']),
            'neg_text': neg_text,
            'neg_image': self._load_image(sample['reference_image']),  # Use reference as negative
            'global_dataset_name': 'CIRR',
            'is_augmented': True,
            'original_mod_text': sample['original_mod_text'],
            # 🔥 添加 reference_image 路径，供 GroupedBatchSampler 使用
            'reference_image': sample['reference_image']
        }
    
    def _load_image(self, image_path):
        """加载并处理图片，返回VLM2Vec格式"""
        if isinstance(image_path, str):
            full_path = self._get_full_image_path(image_path)
            
            if not os.path.exists(full_path):
                print_rank(f"Warning: Image not found at {full_path}")
                # 使用placeholder路径处理缺失图片
                full_path = "dummy_image"
        else:
            full_path = str(image_path)
        
        # 返回VLM2Vec期望的格式 - collator会处理实际的图片加载
        return {
            "paths": [full_path],
            "bytes": [None],  # 让collator处理图片加载
            "resolutions": [None]  # 让processor处理resize
        }

    def _save_augmented_samples(self, augmented_samples: List[Dict]):
        """Save augmented samples to experiment directory"""
        import time
        import json
        
        if not hasattr(self, 'experiment_dir') or not self.experiment_dir:
            print_rank("No experiment directory set, skipping augmented samples save")
            return
        
        # Save with iteration round (for next iteration)
        next_iteration = self.iteration_round + 1
        aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iteration}.json")
        
        # Create summary statistics
        summary = {
            'total_samples': len(augmented_samples),
            'generation_timestamp': time.time(),
            'iteration_round': next_iteration,  # Use next iteration number
            'sample_statistics': {
                'avg_original_length': sum(len(s.get('original_mod_text', '')) for s in augmented_samples) / len(augmented_samples) if augmented_samples else 0,
                'avg_generated_length': sum(len(s.get('modification_text', '')) for s in augmented_samples) / len(augmented_samples) if augmented_samples else 0,
                'unique_reference_images': len(set(s.get('reference_image', '') for s in augmented_samples)),
                'unique_target_images': len(set(s.get('target_image', '') for s in augmented_samples))
            },
            'samples': augmented_samples
        }
        
        try:
            with open(aug_file, 'w') as f:
                json.dump(summary, f, indent=2)

                # --- 关键修改：强制同步到物理存储 ---
                # 2. 确保所有Python内部缓冲区的数据都已写入操作系统
                f.flush()
                # 3. 强制操作系统将文件的所有缓存数据和元数据立即同步到底层存储设备（例如网络文件服务器）
                #    这是一个阻塞操作，直到同步完成才会返回。
                os.fsync(f.fileno())
                # --- 修改结束 ---
                
            print_rank(f"Saved {len(augmented_samples)} augmented samples to {aug_file}")
            print_rank(f"Statistics: avg_original_len={summary['sample_statistics']['avg_original_length']:.1f}, "
                      f"avg_generated_len={summary['sample_statistics']['avg_generated_length']:.1f}")
        except Exception as e:
            print_rank(f"Error saving augmented samples: {e}")
    
    def _save_augmented_samples_incremental(self, all_augmented_samples: List[Dict], batch_idx: int):
        """Incrementally save augmented samples to experiment directory"""
        import json
        import os
        import time
        
        if not hasattr(self, 'experiment_dir') or not self.experiment_dir:
            print_rank("No experiment directory set, skipping augmented samples save")
            return
        
        # Save with iteration round (for next iteration)
        next_iteration = self.iteration_round + 1
        aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{next_iteration}.json")
        
        # Create summary statistics
        summary = {
            'total_samples': len(all_augmented_samples),
            'generation_timestamp': time.time(),
            'iteration_round': next_iteration,  # Use next iteration number
            'last_saved_batch': batch_idx,
            'sample_statistics': {
                'avg_original_length': sum(len(s.get('original_mod_text', '')) for s in all_augmented_samples) / len(all_augmented_samples) if all_augmented_samples else 0,
                'avg_generated_length': sum(len(s.get('modification_text', '')) for s in all_augmented_samples) / len(all_augmented_samples) if all_augmented_samples else 0,
                'unique_reference_images': len(set(s.get('reference_image', '') for s in all_augmented_samples)),
                'unique_target_images': len(set(s.get('target_image', '') for s in all_augmented_samples))
            },
            'samples': all_augmented_samples  # 保存所有当前的样本
        }
        
        try:
            # 创建临时文件，然后原子性替换
            temp_file = aug_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # 原子性替换
            import os
            if os.path.exists(aug_file):
                os.remove(aug_file)
            os.rename(temp_file, aug_file)
            
            print_rank(f"Incrementally saved {len(all_augmented_samples)} augmented samples to {aug_file} (batch {batch_idx})")
            
        except Exception as e:
            print_rank(f"Error incrementally saving augmented samples: {e}")
            # 清理临时文件
            temp_file = aug_file + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def _is_valid_caption(self, caption_text):
        """
        检查生成的caption是否有效（非空且有意义）
        
        Args:
            caption_text: 生成的caption文本
            
        Returns:
            bool: True表示有效，False表示无效（应该被过滤掉）
        """
        if not caption_text:
            return False
        
        # 去除前后空格
        caption_text = caption_text.strip()
        
        # 检查是否为空或只包含空白字符
        if not caption_text or caption_text.isspace():
            return False
        
        # 检查是否只是单个空格字符（_output_translator的默认返回值）
        if caption_text == ' ':
            return False
        
        # 检查长度是否太长
        if len(caption_text) > 300:
            return False
        
        # 检查是否包含一些无意义的标记或占位符
        invalid_patterns = [
            'uncertain',
            'Generate Captions',
            'Generate Text_new',
            'Generate Polar Questions',
            'Answer Questions via Image Comparison',
            'Local Edits',
            'Decompose Intents',
        ]
        
        caption_lower = caption_text.lower()
        for pattern in invalid_patterns:
            if caption_lower == pattern or caption_lower.startswith(pattern):
                return False
        
        return True
    
    def _filter_valid_augmented_samples(self, augmented_samples):
        """
        过滤掉无效caption的增强样本
        """
        if not augmented_samples:
            return []
        
        original_count = len(augmented_samples)
        valid_samples = []
        
        for sample in augmented_samples:
            modification_text = sample.get('modification_text', '')
            
            if self._is_valid_caption(modification_text):
                valid_samples.append(sample)
            else:
                # 记录被过滤的样本（用于调试）
                print_rank(f"Filtered invalid caption: '{modification_text}' (original: '{sample.get('original_mod_text', '')}')") if len(valid_samples) < 5 else None
        
        filtered_count = original_count - len(valid_samples)
        if filtered_count > 0:
            print_rank(f"Filtered out {filtered_count} samples with invalid captions ({filtered_count/original_count*100:.1f}%)")
            print_rank(f"Remaining valid samples: {len(valid_samples)}/{original_count}")
        
        return valid_samples


class IterativeFashionIQDataset(IterativeCIRRDataset):
    """
    Iterative FashionIQ Dataset - similar structure to CIRR but for fashion domain
    """
    
    def _load_cirr_data(self):
        """Override to load FashionIQ data"""
        print_rank(f"Loading FashionIQ dataset...")
        
        # FashionIQ specific loading logic
        data_dir = self.dataset_config.get('data_dir', './data/FashionIQ')
        
        # Load different categories (dress, shirt, toptee)
        categories = ['dress', 'shirt', 'toptee']
        all_annotations = []
        
        for category in categories:
            cat_file = os.path.join(data_dir, 'captions', f'cap.{category}.train.json')
            if os.path.exists(cat_file):
                with open(cat_file, 'r') as f:
                    cat_data = json.load(f)
                    # Add category info
                    for item in cat_data:
                        item['category'] = category
                    all_annotations.extend(cat_data)
        
        self.annotations = all_annotations
        self.image_dir = os.path.join(data_dir, 'images')
        print_rank(f"Loaded {len(self.annotations)} FashionIQ training samples")


@add_metainfo_hook
def cirr_data_prepare(batch_dict, *args, **kwargs):
    """Data preparation function for CIRR"""
    image_dir = kwargs['image_dir']
    model_backbone = kwargs['model_backbone']
    
    # Process batch similar to mmeb_dataset but for CIRR format
    query_texts, query_images, pos_texts, pos_images = [], [], [], []
    
    for sample in batch_dict:
        # Add image token based on model backbone
        mod_text = sample['modification_text']
        if model_backbone in VLM_IMAGE_TOKENS:
            mod_text = VLM_IMAGE_TOKENS[model_backbone] + '\n' + mod_text
        
        query_texts.append(mod_text)
        query_images.append(sample['reference_image'])
        pos_texts.append('')  # Empty for composed retrieval
        pos_images.append(sample['target_image'])
    
    return {
        "query_text": query_texts,
        "query_image": query_images,
        "pos_text": pos_texts,
        "pos_image": pos_images
    }


# Register the dataset class manually
# AutoPairDataset.registry["IterativeCIRRDataset"] = IterativeCIRRDataset


def create_llava_prompt_enhanced(original_text: str, is_hard_negative_context: bool = False) -> str:
    """Create improved prompt for LLaVA model based on hard negative mining theory"""
    if is_hard_negative_context:
        hard_negative_prompts = [
            f"""USER: RETRIEVAL ERROR CORRECTION TASK

Situation: The modification query "{original_text}" applied to the left image incorrectly retrieved the right image as a match.

Task: Generate a NEW modification description that would make the right image the CORRECT match for the left image. Focus on visual differences that justify this pairing.

A:""",
            
            f"""USER: I have a retrieval failure case. The query "{original_text}" with the left image wrongly retrieved the right image. 

Create a corrected modification text that would make this retrieval correct. Use different vocabulary and approach.

A:""",
            
            f"""USER: Transform this retrieval error into success. Original failed query: "{original_text}". Generate a new query that makes the right image the correct target for the left image.

A:"""
        ]
        
        import random
        return random.choice(hard_negative_prompts)
    else:
        diversity_prompts = [
            f"""USER: I'm looking at two images. The original description was: "{original_text}". Generate a different way to describe the same visual transformation, using varied language and focusing on different details.

A:""",
            
            f"""USER: Compare these images. Original instruction: "{original_text}". Create an alternative description for the same change but with different words and approach.

A:""",
            
            f"""USER: Given this modification text: "{original_text}", write a fresh version that describes the same visual change using different vocabulary and perspective.

A:"""
        ]
        
        import random
        return random.choice(diversity_prompts)


def create_generic_prompt_enhanced(original_text: str, is_hard_negative_context: bool = False) -> str:
    """Create improved generic prompt based on hard negative mining theory"""
    if is_hard_negative_context:
        hard_negative_prompts = [
            f"""RETRIEVAL ERROR FIX: "{original_text}" wrongly retrieved this target. Generate new modification for correct retrieval:""",
            f"""CORRECTION NEEDED: Transform failed query "{original_text}" into successful one for this image pair:""",
            f"""HARD NEGATIVE TO POSITIVE: Original "{original_text}" was wrong. Create correct modification:"""
        ]
        
        import random
        return random.choice(hard_negative_prompts)
    else:
        diversity_prompts = [
            f"""Original: "{original_text}". Generate a different description for the same visual change:""",
            f"""Rephrase this modification: "{original_text}". Use different words for the same transformation:""",
            f"""Alternative to "{original_text}": Create a varied description for the same change:"""
        ]
        
        import random
        return random.choice(diversity_prompts)

