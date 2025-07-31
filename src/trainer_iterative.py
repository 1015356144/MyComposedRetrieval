"""
Iterative Trainer for Composed Image Retrieval
Implements iterative hard negative mining and knowledge distillation from foundation models
"""

import os
import json
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
                 **kwargs):
        super().__init__(**kwargs)
        
        self.foundation_model = foundation_model
        self.max_iterations = max_iterations
        self.hard_neg_collection_freq = hard_neg_collection_freq
        self.caption_generation_batch_size = caption_generation_batch_size
        
        # Keep track of iterations
        self.current_iteration = 0
        self.iteration_metrics = {}
        
        # Save original dataset for reference
        self.original_dataset = self.train_dataset
        
        # Try to resume from previous experiment
        self._try_resume_from_checkpoint()
        
        print_master(f"Initialized IterativeRetrievalTrainer with {max_iterations} max iterations")
    
    def _try_resume_from_checkpoint(self):
        """Try to resume from a previous checkpoint to avoid recomputation"""
        output_dir = self.args.output_dir
        
        # Look for the latest iteration state
        for i in range(self.max_iterations - 1, -1, -1):
            state_file = os.path.join(output_dir, f"iteration_{i}_state.json")
            if os.path.exists(state_file):
                print_master(f"Found checkpoint for iteration {i}, resuming...")
                
                # Load iteration state
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
                        self.train_dataset._load_hard_negatives()
                
                print_master(f"Resuming from iteration {self.current_iteration}")
                return True
        
        print_master("No previous checkpoint found, starting from scratch")
        return False
    
    def iterative_train(self, resume_from_iteration: int = 0):
        """
        Main iterative training loop
        """
        print_master("Starting iterative training process...")
        
        # Resume from specific iteration if specified
        if resume_from_iteration > 0:
            self.current_iteration = resume_from_iteration
            self._load_iteration_state(resume_from_iteration)
        
        for iteration in range(self.current_iteration, self.max_iterations):
            print_master(f"\n{'='*60}")
            print_master(f"Starting Iteration {iteration}")
            print_master(f"{'='*60}")
            
            self.current_iteration = iteration
            
            # Step 1: Train current model (or use base model for iteration 0)
            if iteration == 0:
                print_master("Iteration 0: Training base retrieval model...")
                self._train_base_model()
            else:
                print_master(f"Iteration {iteration}: Training with augmented data...")
                self._train_current_iteration()
            
            # Step 2: Evaluate current model performance
            eval_results = self._evaluate_current_model()
            self.iteration_metrics[iteration] = eval_results
            
            # Step 3: Collect hard negatives (if not last iteration)
            if iteration < self.max_iterations - 1:
                hard_negatives = self._collect_hard_negatives(iteration)
                
                # Step 4: Generate augmented captions using foundation model
                if len(hard_negatives) > 0:
                    augmented_samples = self._generate_augmented_captions(hard_negatives)
                    
                    # Step 5: Prepare dataset for next iteration
                    self._prepare_next_iteration_dataset(iteration + 1, augmented_samples)
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
        """Evaluate current model on validation set"""
        print_master(f"Evaluating iteration {self.current_iteration} model...")
        
        # This should call your evaluation pipeline
        # For now, return dummy metrics
        eval_results = {
            'r_at_1': 0.5,  # Placeholder
            'r_at_5': 0.7,  # Placeholder
            'r_at_10': 0.8  # Placeholder
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
            # Use dataset's built-in hard negative collection
            hard_negatives = self.train_dataset.collect_hard_negatives_batch(
                self.model,
                batch_size=8  # Fixed batch size for faster processing
            )
            
            # Cache the results
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
        """Fallback method for hard negative collection"""
        # This should implement retrieval evaluation and hard negative identification
        # For now, return empty list
        print_master("Using fallback hard negative collection")
        return []
    
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
            
            # Use dataset's built-in caption generation
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
    
    def _save_iteration_state(self, iteration: int):
        """Save iteration state and metrics"""
        state_file = os.path.join(self.args.output_dir, f"iteration_{iteration}_state.json")
        
        state = {
            'iteration': iteration,
            'metrics': self.iteration_metrics,
            'model_path': os.path.join(self.args.output_dir, f"iteration_{iteration}"),
            'hard_negatives_file': f"hard_negatives_iter_{iteration}.json"
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print_master(f"Saved iteration {iteration} state to {state_file}")
    
    def _load_iteration_state(self, iteration: int):
        """Load iteration state for resuming"""
        state_file = os.path.join(self.args.output_dir, f"iteration_{iteration}_state.json")
        
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.iteration_metrics = state.get('metrics', {})
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
    
    # Remaining kwargs are ignored (like model_args, data_args, etc.)
    
    return IterativeRetrievalTrainer(
        model=model,
        foundation_model=foundation_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **iterative_params,
        **trainer_params
    )
