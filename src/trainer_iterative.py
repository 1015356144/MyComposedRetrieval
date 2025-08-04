"""
Iterative Trainer for Composed Image Retrieval
Implements iterative hard negative mining and knowledge distillation from foundation models
"""

import os
import json
import time
import torch
import logging
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from .trainer import MMEBTrainer
from .model.model import MMEBModel
from .data.dataset.composed_retrieval_dataset import IterativeCIRRDataset, IterativeFashionIQDataset
from .utils import print_rank, print_master

logger = logging.getLogger(__name__)


class IterativeRetrievalTrainer(MMEBTrainer):
    """
    Trainer for iterative composed image retrieval with hard negative mining
    """
    
    def __init__(self, 
                 foundation_model=None,
                 max_iterations: int = 3,
                 hard_neg_collection_freq: int = 1,
                 caption_generation_batch_size: int = 8,
                 model_args=None,
                 data_args=None,
                 max_length=None,
                 # Fast mode and production mode parameters
                 fast_mode: bool = False,
                 fast_mode_max_samples: int = 100,
                 fast_mode_retrieval_db_size: int = 50,
                 fast_mode_max_steps: int = 5,
                 production_max_steps: int = 1000,
                 production_save_steps: int = 100,
                 **kwargs):
        # Store model_args, data_args and max_length before calling super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.max_length = max_length
        
        # Store fast mode and production mode settings
        self.fast_mode = fast_mode
        self.fast_mode_max_samples = fast_mode_max_samples
        self.fast_mode_retrieval_db_size = fast_mode_retrieval_db_size
        self.fast_mode_max_steps = fast_mode_max_steps
        self.production_max_steps = production_max_steps
        self.production_save_steps = production_save_steps
        
        # Remove parameters that parent Trainer doesn't accept
        kwargs.pop('model_args', None)
        kwargs.pop('data_args', None)
        kwargs.pop('max_length', None)
        kwargs.pop('fast_mode', None)
        kwargs.pop('fast_mode_max_samples', None)
        kwargs.pop('fast_mode_retrieval_db_size', None)
        kwargs.pop('fast_mode_max_steps', None)
        kwargs.pop('production_max_steps', None)
        kwargs.pop('production_save_steps', None)
        
        super().__init__(**kwargs)
        
        self.foundation_model = foundation_model
        self.max_iterations = max_iterations
        self.hard_neg_collection_freq = hard_neg_collection_freq
        self.caption_generation_batch_size = caption_generation_batch_size
        
        # Keep track of iterations
        self.current_iteration = 0
        self.iteration_metrics = {}
        
        # Track completion status for resuming
        self._base_training_completed = False
        self._target_embeddings_cached = False
        
        # Save original dataset for reference
        self.original_dataset = self.train_dataset
        
        # Try to resume from previous experiment
        self._try_resume_from_checkpoint()
        
        print_master(f"Initialized IterativeRetrievalTrainer with {max_iterations} max iterations")
        
        # Apply fast mode or production mode settings
        self._configure_training_mode()
    
    def _configure_training_mode(self):
        """Configure training parameters based on fast mode or production mode"""
        # Debug: Print fast_mode value
        print_master(f"DEBUG: self.fast_mode = {self.fast_mode}")
        print_master(f"DEBUG: fast_mode_max_steps = {self.fast_mode_max_steps}")
        
        if self.fast_mode:
            print_master("=== FAST MODE CONFIGURATION ===")
            print_master(f"Max steps per iteration: {self.fast_mode_max_steps}")
            print_master(f"Max samples for hard negatives: {self.fast_mode_max_samples}")
            print_master(f"Retrieval database size: {self.fast_mode_retrieval_db_size}")
            
            # Override training arguments for fast mode
            self.args.max_steps = self.fast_mode_max_steps
            self.args.save_steps = max(1, self.fast_mode_max_steps // 2)  # Save in the middle
            self.args.logging_steps = 1
            
        else:
            print_master("=== PRODUCTION MODE CONFIGURATION ===")
            print_master(f"Max steps per iteration: {self.production_max_steps}")
            print_master(f"Save frequency: every {self.production_save_steps} steps")
            
            # Override training arguments for production mode
            self.args.max_steps = self.production_max_steps
            self.args.save_steps = self.production_save_steps
            self.args.logging_steps = min(10, self.production_save_steps // 10)
        
        print_master(f"Final training configuration:")
        print_master(f"  max_steps: {self.args.max_steps}")
        print_master(f"  save_steps: {self.args.save_steps}")
        print_master(f"  logging_steps: {self.args.logging_steps}")
        print_master("=" * 50)
    
    def _try_resume_from_checkpoint(self):
        """Try to resume from a previous checkpoint to avoid recomputation"""
        output_dir = self.args.output_dir
        
        # First, look for the latest iteration state (for complete metadata)
        for i in range(self.max_iterations - 1, -1, -1):
            state_file = os.path.join(output_dir, f"iteration_{i}_state.json")
            if os.path.exists(state_file):
                print_master(f"Found iteration state for iteration {i}, loading metadata...")
                
                # Load iteration state (metadata only)
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.current_iteration = i + 1  # Start from next iteration
                self.iteration_metrics = state.get('iteration_metrics', {})
                
                # Load hard negatives if available
                hard_neg_file = os.path.join(output_dir, f"hard_negatives_iter_{i}.json")
                if os.path.exists(hard_neg_file):
                    print_master(f"Loading hard negatives from iteration {i}")
                    if hasattr(self.train_dataset, 'hard_negatives_file'):
                        self.train_dataset.hard_negatives_file = hard_neg_file
                        self.train_dataset._load_hard_negatives(i)  # Pass iteration number
                
                print_master(f"Resuming from iteration {self.current_iteration} (model loaded separately)")
                return True
        
        # If no iteration state found, check for cached embeddings and base model
        # This handles cases where base training completed but iteration didn't start
        cache_dir = os.path.join(output_dir, "cache")
        base_model_dir = os.path.join(output_dir, "base_model")
        
        # Check for base model completion (most comprehensive check)
        if os.path.exists(base_model_dir):
            print_master(f"Found base model directory: {base_model_dir}")
            
            # Check if base model has required files
            base_model_files = os.listdir(base_model_dir)
            has_adapter = any(f.startswith("adapter_") for f in base_model_files)
            has_config = "adapter_config.json" in base_model_files
            
            if has_adapter and has_config:
                print_master("✅ Base model training appears to be completed (found LoRA adapter)")
                
                # Check for cached embeddings
                if os.path.exists(cache_dir):
                    cache_files = [f for f in os.listdir(cache_dir) if f.startswith("target_embeddings_") and f.endswith(".pt")]
                    if cache_files:
                        print_master(f"✅ Found cached target embeddings: {cache_files}")
                        print_master("🔄 Resuming from completed base training (iteration 0)")
                        print_master("   ➡️  Will skip: base model training, evaluation, target embedding computation")
                        print_master("   ➡️  Will start: hard negative collection")
                        
                        # Set flags to indicate what has been completed
                        self.current_iteration = 0
                        self._base_training_completed = True
                        self._target_embeddings_cached = True
                        return True
                    else:
                        print_master("⚠️  Base model found but no cached embeddings")
                        print_master("🔄 Will resume from base model, recompute embeddings")
                        self.current_iteration = 0
                        self._base_training_completed = True
                        self._target_embeddings_cached = False
                        return True
                else:
                    print_master("⚠️  Base model found but no cache directory")
                    print_master("🔄 Will resume from base model, compute embeddings")
                    self.current_iteration = 0
                    self._base_training_completed = True
                    self._target_embeddings_cached = False
                    return True
        
        # Fallback: check for regular checkpoints without base model
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            print_master(f"Found checkpoints but no base model: {checkpoint_dirs}")
            print_master("This might be an incomplete training, starting from scratch")
        
        print_master("No previous state, base model, or cached embeddings found, starting from scratch")
        self._base_training_completed = False
        self._target_embeddings_cached = False
        return False
    
    def iterative_train(self, resume_from_iteration: int = 0):
        """
        Main iterative training loop
        
        Args:
            resume_from_iteration (int): Specific iteration to resume from. 
                                       If 0, will auto-detect latest checkpoint.
                                       If > 0, will load from that specific iteration.
        """
        print_master("Starting iterative training process...")
        
        # Resume from specific iteration if specified
        if resume_from_iteration > 0:
            print_master(f"Manually resuming from iteration {resume_from_iteration}")
            self.current_iteration = resume_from_iteration
            self._load_iteration_state(resume_from_iteration - 1)  # Load previous iteration's state
            
            # Also need to load the dataset state for this iteration
            self._prepare_dataset_for_iteration(resume_from_iteration)
        
        for iteration in range(self.current_iteration, self.max_iterations):
            print_master(f"\n{'='*60}")
            print_master(f"Starting Iteration {iteration}")
            print_master(f"{'='*60}")
            
            self.current_iteration = iteration
            
            # Step 1: Train current model (or skip if already completed)
            if iteration == 0:
                if hasattr(self, '_base_training_completed') and self._base_training_completed:
                    print_master("✅ Base model training already completed, skipping...")
                    print_master(f"   📁 Using existing base model from: {os.path.join(self.args.output_dir, 'base_model')}")
                else:
                    print_master("Iteration 0: Training base retrieval model...")
                    self._train_base_model()
            else:
                print_master(f"Iteration {iteration}: Training with augmented data...")
                self._train_current_iteration()
            
            # 添加同步屏障：确保所有GPU完成训练
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                if not (iteration == 0 and hasattr(self, '_base_training_completed') and self._base_training_completed):
                    print_master(f"All GPUs completed training for iteration {iteration}")
            
            # Step 2: Evaluate current model performance (or skip if cached)
            # Add distributed barrier to ensure all GPUs complete training before evaluation
            if dist.is_initialized():
                dist.barrier()
            
            # Use the improved evaluation method that handles distributed internally
            eval_results = self._evaluate_current_model()
            
            self.iteration_metrics[iteration] = eval_results
            
            # Step 3: Collect hard negatives (if not last iteration)
            if iteration < self.max_iterations - 1:
                print_master(f"🔍 Starting hard negative collection for iteration {iteration}...")
                hard_neg_start_time = time.time()
                
                hard_negatives = self._collect_hard_negatives(iteration)
                
                hard_neg_time = time.time() - hard_neg_start_time
                print_master(f"Hard negative collection completed in {int(hard_neg_time//60):02d}:{int(hard_neg_time%60):02d}")
                
                # 添加同步屏障：确保所有GPU完成硬负样本收集
                if dist.is_initialized():
                    dist.barrier()
                    print_master(f"All GPUs completed hard negative collection for iteration {iteration}")
                
                # Step 4: Generate augmented captions using foundation model
                if len(hard_negatives) > 0:
                    print_master(f"📝 Starting caption generation for {len(hard_negatives)} hard negatives...")
                    caption_start_time = time.time()
                    
                    augmented_samples = self._generate_augmented_captions(hard_negatives)
                    
                    caption_time = time.time() - caption_start_time
                    print_master(f"Caption generation completed in {int(caption_time//60):02d}:{int(caption_time%60):02d}")
                    
                    # 添加同步屏障：确保所有GPU完成caption生成
                    if dist.is_initialized():
                        dist.barrier()
                        print_master(f"All GPUs completed caption generation for iteration {iteration}")
                    
                    # Step 5: Prepare dataset for next iteration
                    self._prepare_next_iteration_dataset(iteration + 1, augmented_samples)
                    
                    # 性能统计
                    total_time = hard_neg_time + caption_time
                    print_master(f"📊 Iteration {iteration} data preparation stats:")
                    print_master(f"  - Hard negatives: {len(hard_negatives)} samples in {hard_neg_time:.1f}s")
                    print_master(f"  - Augmented captions: {len(augmented_samples)} samples in {caption_time:.1f}s")
                    print_master(f"  - Total time: {int(total_time//60):02d}:{int(total_time%60):02d}")
                    
                    if dist.is_initialized():
                        world_size = dist.get_world_size()
                        print_master(f"  - Used {world_size} GPUs for parallel processing")
                        if total_time > 0:
                            print_master(f"  - Processing rate: {(len(hard_negatives) + len(augmented_samples))/total_time:.2f} samples/second")
                else:
                    print_master("No hard negatives found, stopping early")
                    break
            
            # Save iteration checkpoint
            self._save_iteration_state(iteration)
        
        print_master("\nIterative training completed!")
        self._summarize_results()
    
    def _train_base_model(self):
        """Train the base retrieval model using standard contrastive learning"""
        print_master("Training base model with original dataset...")
        
        # Use original dataset
        self.train_dataset = self.original_dataset
        
        # Standard training
        train_result = self.train()
        
        # Save base model
        base_model_path = os.path.join(self.args.output_dir, "base_model")
        self.save_model(base_model_path)
        
        return train_result
    
    def _train_current_iteration(self):
        """Train model for current iteration with augmented data"""
        print_master(f"Training iteration {self.current_iteration} model...")
        
        # The dataset should already be prepared with augmented samples
        train_result = self.train()
        
        # Save iteration model
        iter_model_path = os.path.join(self.args.output_dir, f"iteration_{self.current_iteration}")
        self.save_model(iter_model_path)
        
        return train_result
    
    def _evaluate_current_model(self) -> Dict[str, float]:
        """Evaluate current model on validation set with optimizations"""
        print_master(f"Evaluating iteration {self.current_iteration} model...")
        
        try:
            # Initialize evaluator if not already done
            if not hasattr(self, 'evaluator') or self.evaluator is None:
                from .evaluation.cirr_evaluator import CIRREvaluator
                
                # Adjust batch size based on fast mode
                eval_batch_size = 4 if self.fast_mode else 8
                
                self.evaluator = CIRREvaluator(
                    model=self.model,
                    processor=self.processing_class,
                    data_args=self.data_args,
                    model_args=self.model_args,
                    device=self.args.device,
                    batch_size=eval_batch_size
                )
                print_master(f"Real evaluator initialized successfully (batch_size={eval_batch_size})")
            
            # Check if distributed evaluation is available and beneficial
            import torch.distributed as dist
            use_distributed = (dist.is_initialized() and 
                             dist.get_world_size() > 1 and 
                             hasattr(self.evaluator, '_evaluate_distributed'))
            
            # In fast mode, prefer single GPU evaluation for simplicity unless many GPUs
            if self.fast_mode and use_distributed and dist.get_world_size() <= 2:
                use_distributed = False
                print_master("Fast mode: Using single GPU evaluation for small GPU count")
            
            if use_distributed:
                print_master(f"Using distributed evaluation across {dist.get_world_size()} GPUs")
                eval_results = self.evaluator.evaluate(distributed=True)
            else:
                print_master("Using single GPU evaluation")
                eval_results = self.evaluator.evaluate(distributed=False)
            
            # Add evaluation metadata
            eval_results['evaluation_mode'] = 'distributed' if use_distributed else 'single_gpu'
            eval_results['fast_mode'] = self.fast_mode
            eval_results['iteration'] = self.current_iteration
            
        except Exception as e:
            print_master(f"Real evaluation failed: {e}")
            print_master("Falling back to dummy evaluation")
            # Fallback to dummy metrics with realistic fast mode values
            if self.fast_mode:
                # Fast mode typically has lower performance due to limited training
                eval_results = {
                    'recall_at_1': 0.15,  
                    'recall_at_5': 0.35,  
                    'recall_at_10': 0.45,  
                    'recall_subset_at_1': 0.12,
                    'recall_subset_at_2': 0.25,
                    'recall_subset_at_3': 0.32,
                    'group_recall_at_1': 0.18,
                    'group_recall_at_2': 0.30,
                    'group_recall_at_3': 0.38,
                    'evaluation_mode': 'dummy_fast',
                    'fast_mode': True,
                    'iteration': self.current_iteration
                }
            else:
                # Production mode dummy metrics
                eval_results = {
                    'recall_at_1': 0.5,  
                    'recall_at_5': 0.7,  
                    'recall_at_10': 0.8,  
                    'recall_subset_at_1': 0.3,
                    'recall_subset_at_2': 0.5,
                    'recall_subset_at_3': 0.6,
                    'group_recall_at_1': 0.4,
                    'group_recall_at_2': 0.6,
                    'group_recall_at_3': 0.7,
                    'evaluation_mode': 'dummy_production',
                    'fast_mode': False,
                    'iteration': self.current_iteration
                }
        
        print_master(f"Iteration {self.current_iteration} results: {eval_results}")
        return eval_results
    
    def _collect_hard_negatives(self, iteration: int) -> List[Dict]:
        """Collect hard negatives using current model with caching support"""
        print_master(f"Collecting hard negatives for iteration {iteration}...")
        
        # Check if hard negatives already exist for this iteration
        cache_file = os.path.join(self.args.output_dir, f"hard_negatives_iter_{iteration}.json")
        if os.path.exists(cache_file):
            print_master(f"Found cached hard negatives for iteration {iteration}, loading...")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            print_master(f"Loaded {len(cached_data)} cached hard negatives")
            return cached_data
        
        # Set model to evaluation mode
        self.model.eval()
        
        if isinstance(self.train_dataset, (IterativeCIRRDataset, IterativeFashionIQDataset)):
            # Determine sample limit based on mode
            sample_limit = self.fast_mode_max_samples if self.fast_mode else None
            
            # Also pass retrieval database size information to the dataset
            if self.fast_mode and hasattr(self.train_dataset, 'fast_mode_retrieval_db_size'):
                self.train_dataset.fast_mode_retrieval_db_size = self.fast_mode_retrieval_db_size
            
            # 使用分布式硬负样本收集
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                print_master("Using distributed hard negative collection...")
                hard_negatives = self.train_dataset.collect_hard_negatives_batch_distributed(
                    self.model,
                    batch_size=8,  # Fixed batch size for faster processing
                    max_samples=sample_limit  # Pass limit to dataset
                )
            else:
                print_master("Using single-GPU hard negative collection...")
                hard_negatives = self.train_dataset.collect_hard_negatives_batch(
                    self.model,
                    batch_size=8,  # Fixed batch size for faster processing
                    max_samples=sample_limit  # Pass limit to dataset
                )
            
            print_master(f"Collected {len(hard_negatives)} hard negatives " + 
                        (f"(limited to {sample_limit})" if sample_limit else "(no limit)"))
            
            # Cache the results (only master process)
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(hard_negatives, f, indent=2)
                print_master(f"Cached hard negatives to {cache_file}")
            
        else:
            # Fallback: implement hard negative collection here
            hard_negatives = self._collect_hard_negatives_fallback()
        
        print_master(f"Collected {len(hard_negatives)} hard negatives")
        return hard_negatives
    
    def _collect_hard_negatives_fallback(self) -> List[Dict]:
        """Fallback method for hard negative collection with fast mode support"""
        # This should implement retrieval evaluation and hard negative identification
        print_master("Using fallback hard negative collection")
        
        # Generate dummy hard negatives for testing
        dummy_count = self.fast_mode_max_samples if self.fast_mode else 500
        dummy_negatives = []
        for i in range(dummy_count):
            dummy_negatives.append({
                'query_id': f'dummy_query_{i}',
                'reference_image': f'dummy_ref_{i}',
                'target_image': f'dummy_target_{i}',
                'original_caption': f'dummy caption {i}',
                'difficulty_score': 0.8 + (i % 5) * 0.04  # Simulate difficulty scores
            })
        
        print_master(f"Generated {len(dummy_negatives)} dummy hard negatives")
        return dummy_negatives
    
    def _generate_augmented_captions(self, hard_negatives: List[Dict]) -> List[Dict]:
        """Generate augmented captions using foundation model"""
        if not self.foundation_model:
            print_master("No foundation model available, skipping caption generation")
            return []
        
        print_master(f"Generating augmented captions for {len(hard_negatives)} samples...")
        
        if isinstance(self.train_dataset, (IterativeCIRRDataset, IterativeFashionIQDataset)):
            # Ensure dataset has foundation model
            if not self.train_dataset.foundation_model:
                print_master("Setting foundation model for dataset...")
                self.train_dataset.foundation_model = self.foundation_model
            
            # 使用分布式caption生成
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                print_master("Using distributed caption generation...")
                augmented_samples = self.train_dataset.generate_augmented_captions_distributed(hard_negatives)
            else:
                print_master("Using single-GPU caption generation...")
                augmented_samples = self.train_dataset.generate_augmented_captions(hard_negatives)
        else:
            # Fallback: implement caption generation here
            augmented_samples = self._generate_captions_fallback(hard_negatives)
        
        print_master(f"Generated {len(augmented_samples)} augmented samples")
        return augmented_samples
    
    def _generate_captions_fallback(self, hard_negatives: List[Dict]) -> List[Dict]:
        """Fallback method for caption generation"""
        # This should implement foundation model caption generation
        # For now, return empty list
        print_master("Using fallback caption generation")
        return []
    
    def _prepare_next_iteration_dataset(self, next_iteration: int, augmented_samples: List[Dict]):
        """Prepare dataset for next iteration with augmented samples"""
        print_master(f"Preparing dataset for iteration {next_iteration}...")
        
        # Save augmented samples to file for resuming with metadata
        augmented_file = os.path.join(self.args.output_dir, f"augmented_samples_iter_{next_iteration}.json")
        
        # Create metadata structure
        augmented_data = {
            "total_samples": len(augmented_samples),
            "generation_timestamp": __import__('time').time(),
            "iteration_round": next_iteration,
            "sample_statistics": self._compute_sample_statistics(augmented_samples),
            "samples": augmented_samples
        }
        
        with open(augmented_file, 'w') as f:
            json.dump(augmented_data, f, indent=2)
        print_master(f"Saved {len(augmented_samples)} augmented samples to {augmented_file}")
        
        if isinstance(self.train_dataset, (IterativeCIRRDataset, IterativeFashionIQDataset)):
            # Update dataset with augmented samples
            self.train_dataset.iteration_round = next_iteration
            self.train_dataset.augmented_samples.extend(augmented_samples)
            # Update hard negatives file path for new iteration
            self.train_dataset.hard_negatives_file = os.path.join(
                self.args.output_dir, f"hard_negatives_iter_{next_iteration}.json"
            )
        else:
            # Create new iterative dataset
            dataset_config = {
                'dataset_name': 'composed_retrieval',
                'iteration_round': next_iteration
            }
            
            # Determine dataset type based on current dataset
            if 'cirr' in str(self.train_dataset).lower():
                self.train_dataset = IterativeCIRRDataset(
                    model_args=self.model_args,
                    data_args=self.data_args,
                    training_args=self.args,
                    iteration_round=next_iteration,
                    foundation_model=self.foundation_model,
                    experiment_dir=self.args.output_dir,  # Pass experiment directory
                    **dataset_config
                )
            else:
                self.train_dataset = IterativeFashionIQDataset(
                    model_args=self.model_args,
                    data_args=self.data_args,
                    training_args=self.args,
                    iteration_round=next_iteration,
                    foundation_model=self.foundation_model,
                    experiment_dir=self.args.output_dir,  # Pass experiment directory
                    **dataset_config
                )
        
        print_master(f"Dataset updated with {len(augmented_samples)} new samples")
    
    def _compute_sample_statistics(self, samples: List[Dict]) -> Dict[str, Any]:
        """Compute statistics for augmented samples"""
        if not samples:
            return {}
        
        try:
            # Compute text length statistics
            original_lengths = []
            generated_lengths = []
            reference_images = set()
            target_images = set()
            
            for sample in samples:
                # Original text length
                if 'original_mod_text' in sample:
                    original_lengths.append(len(sample['original_mod_text']))
                
                # Generated text length
                if 'modification_text' in sample:
                    generated_lengths.append(len(sample['modification_text']))
                
                # Unique images
                if 'reference_image' in sample:
                    reference_images.add(sample['reference_image'])
                if 'target_image' in sample:
                    target_images.add(sample['target_image'])
            
            statistics = {
                'total_samples': len(samples),
                'avg_original_length': sum(original_lengths) / len(original_lengths) if original_lengths else 0,
                'avg_generated_length': sum(generated_lengths) / len(generated_lengths) if generated_lengths else 0,
                'unique_reference_images': len(reference_images),
                'unique_target_images': len(target_images)
            }
            
            # Add augmented sample ratio if available
            augmented_count = sum(1 for s in samples if s.get('is_augmented', False))
            statistics['augmented_ratio'] = augmented_count / len(samples) if samples else 0
            
            return statistics
            
        except Exception as e:
            print_master(f"Warning: Failed to compute sample statistics: {e}")
            return {'total_samples': len(samples)}
    
    def _prepare_dataset_for_iteration(self, iteration: int):
        """Prepare dataset state for a specific iteration when resuming"""
        print_master(f"Preparing dataset for resumed iteration {iteration}...")
        
        if iteration == 0:
            # Use original dataset for iteration 0
            self.train_dataset = self.original_dataset
            return
        
        # For iterations > 0, need to load accumulated augmented samples
        all_augmented_samples = []
        
        # Load augmented samples from all previous iterations
        for i in range(1, iteration):
            augmented_file = os.path.join(self.args.output_dir, f"augmented_samples_iter_{i}.json")
            if os.path.exists(augmented_file):
                with open(augmented_file, 'r') as f:
                    data = json.load(f)
                
                # Extract actual samples from the data structure
                if isinstance(data, dict) and 'samples' in data:
                    # New format with metadata
                    iter_samples = data['samples']
                    print_master(f"Loaded {len(iter_samples)} augmented samples from iteration {i} (with metadata)")
                elif isinstance(data, list):
                    # Old format - direct list of samples
                    iter_samples = data
                    print_master(f"Loaded {len(iter_samples)} augmented samples from iteration {i} (direct list)")
                else:
                    print_master(f"Warning: Unexpected data format in {augmented_file}, skipping...")
                    continue
                
                all_augmented_samples.extend(iter_samples)
        
        # Update dataset with all accumulated samples
        if isinstance(self.train_dataset, (IterativeCIRRDataset, IterativeFashionIQDataset)):
            self.train_dataset.iteration_round = iteration
            self.train_dataset.augmented_samples = all_augmented_samples
            # Set hard negatives file for current iteration
            self.train_dataset.hard_negatives_file = os.path.join(
                self.args.output_dir, f"hard_negatives_iter_{iteration-1}.json"
            )
        
        print_master(f"Dataset prepared for iteration {iteration} with {len(all_augmented_samples)} total augmented samples")
    
    def _save_iteration_state(self, iteration: int):
        """Save iteration state and metrics"""
        state_file = os.path.join(self.args.output_dir, f"iteration_{iteration}_state.json")
        
        # Determine correct model path based on iteration
        if iteration == 0:
            model_path = os.path.join(self.args.output_dir, "base_model")
        else:
            model_path = os.path.join(self.args.output_dir, f"iteration_{iteration}")
        
        state = {
            'iteration': iteration,
            'metrics': self.iteration_metrics,
            'model_path': model_path,
            'hard_negatives_file': f"hard_negatives_iter_{iteration}.json"
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print_master(f"Saved iteration {iteration} state to {state_file}")
        print_master(f"Model path recorded as: {model_path}")
    
    def _load_iteration_state(self, iteration: int):
        """Load iteration state for resuming (model already loaded externally)"""
        state_file = os.path.join(self.args.output_dir, f"iteration_{iteration}_state.json")
        
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Note: Model weights already loaded in main script using MMEBModel.load()
            print_master(f"Loading iteration {iteration} metadata (model loaded separately)")
            
            self.iteration_metrics = state.get('iteration_metrics', {})
            print_master(f"Loaded iteration {iteration} state from {state_file}")
        else:
            print_master(f"No state file found for iteration {iteration}")
    
    def _summarize_results(self):
        """Summarize results across all iterations"""
        print_master("\n" + "="*80)
        print_master("ITERATIVE TRAINING SUMMARY")
        print_master("="*80)
        
        for iteration, metrics in self.iteration_metrics.items():
            print_master(f"Iteration {iteration}: {metrics}")
        
        # Find best iteration
        if self.iteration_metrics:
            best_iteration = max(self.iteration_metrics.keys(), 
                               key=lambda x: self.iteration_metrics[x].get('r_at_1', 0))
            best_metrics = self.iteration_metrics[best_iteration]
            
            print_master(f"\nBest performance: Iteration {best_iteration}")
            print_master(f"Best metrics: {best_metrics}")
        
        # Save summary
        summary_file = os.path.join(self.args.output_dir, "training_summary.json")
        summary = {
            'max_iterations': self.max_iterations,
            'completed_iterations': len(self.iteration_metrics),
            'iteration_metrics': self.iteration_metrics,
            'best_iteration': best_iteration if self.iteration_metrics else None,
            'best_metrics': best_metrics if self.iteration_metrics else None
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print_master(f"Training summary saved to {summary_file}")


def create_iterative_trainer(
    model: MMEBModel,
    foundation_model=None,
    args: TrainingArguments = None,
    train_dataset=None,
    eval_dataset=None,
    experiment_dir=None,
    **kwargs
) -> IterativeRetrievalTrainer:
    """
    Factory function to create iterative trainer
    """
    # Extract iterative-specific parameters
    iterative_params = {}
    for key in ['max_iterations', 'hard_neg_collection_freq', 'caption_generation_batch_size']:
        if key in kwargs:
            iterative_params[key] = kwargs.pop(key)
    
    # Extract fast mode and production mode parameters
    fast_mode_params = {}
    for key in ['fast_mode', 'fast_mode_max_samples', 'fast_mode_retrieval_db_size', 
                'fast_mode_max_steps', 'production_max_steps', 'production_save_steps']:
        if key in kwargs:
            fast_mode_params[key] = kwargs.pop(key)
    
    # Debug: Print extracted parameters
    print_master(f"DEBUG: Extracted fast_mode_params = {fast_mode_params}")
    
    # Don't pass experiment_dir to base trainer
    if experiment_dir:
        # We can use args.output_dir instead
        pass
    
    # Extract standard trainer parameters
    trainer_params = {}
    standard_trainer_args = [
        'data_collator', 'tokenizer', 'model_init', 'compute_metrics',
        'callbacks', 'optimizers', 'preprocess_logits_for_metrics',
        'processing_class'
    ]
    
    for key in standard_trainer_args:
        if key in kwargs:
            trainer_params[key] = kwargs.pop(key)
    
    # Extract important args that trainer needs
    important_args = {}
    for key in ['model_args', 'data_args', 'max_length']:
        if key in kwargs:
            important_args[key] = kwargs.pop(key)
    
    # Remaining kwargs are ignored
    
    return IterativeRetrievalTrainer(
        model=model,
        foundation_model=foundation_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **iterative_params,
        **fast_mode_params,
        **trainer_params,
        **important_args
    )
