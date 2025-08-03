#!/usr/bin/env python3
"""
Iterative Training Script for Composed Image Retrieval
Adapted from VLM2Vec training pipeline for iterative hard negative mining
"""

import logging
import os
import os.path
import sys

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

import torch
import wandb
import yaml
from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.model import MMEBModel
from src.trainer_iterative import IterativeRetrievalTrainer, create_iterative_trainer
from src.data.dataset.composed_retrieval_dataset import IterativeCIRRDataset, IterativeFashionIQDataset
from src.utils import print_rank, print_master, find_latest_checkpoint
from src.model.processor import load_processor, get_backbone_name


def load_foundation_model(model_args, data_args):
    """Load foundation model for caption generation"""
    foundation_model_name = getattr(model_args, 'foundation_model_name', None)
    
    if foundation_model_name:
        print_master(f"Loading foundation model: {foundation_model_name}")
        
        # Load foundation model directly from transformers (not wrapped by MMEBModel)
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        # Load the raw foundation model with generate capability
        foundation_model = AutoModelForVision2Seq.from_pretrained(
            foundation_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor
        foundation_processor = AutoProcessor.from_pretrained(
            foundation_model_name,
            trust_remote_code=True
        )
        
        # Attach processor to model for easy access
        setattr(foundation_model, 'processor', foundation_processor)
        
        print_master(f"Foundation model loaded: {foundation_model_name}")
        return foundation_model
    else:
        print_master("No foundation model specified")
        return None


def main():
    # Handle distributed training arguments
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Debug distributed setup
    print("Distributed init debug info:")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")

    # Check for existing checkpoints
    resume_checkpoint_dir = None
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        logger.info("No checkpoint found. Starting fresh training.")

    # Initialize WandB if enabled
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('Initializing wandb for iterative training')
            wandb.init(
                project=training_args.project_name or "iterative_composed_retrieval", 
                name=training_args.run_name or "iterative_training", 
                mode="online"
            )
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    # Load retrieval model with checkpoint resume support
    print_master("Loading retrieval model...")
    
    # Check for iterative training checkpoints
    resume_from_iteration = None
    if training_args.resume_from == 'auto':
        # Auto-detect latest iteration checkpoint
        for i in range(10, -1, -1):  # Check last 10 iterations
            iter_checkpoint = os.path.join(training_args.output_dir, f"iteration_{i}")
            if os.path.exists(iter_checkpoint):
                resume_from_iteration = i
                print_master(f"Found iteration checkpoint: iteration_{i}")
                break
    elif training_args.resume_from.startswith('iter_'):
        # Manual iteration specification: iter_2
        iter_num = training_args.resume_from.split('_')[1]
        if iter_num.isdigit():
            iter_checkpoint = os.path.join(training_args.output_dir, f"iteration_{iter_num}")
            if os.path.exists(iter_checkpoint):
                resume_from_iteration = int(iter_num)
                print_master(f"Manually resuming from iteration_{iter_num}")
    
    # Load model based on checkpoint availability
    if resume_from_iteration is not None:
        # Load from iteration checkpoint
        iter_checkpoint_path = os.path.join(training_args.output_dir, f"iteration_{resume_from_iteration}")
        print_master(f"Loading model from iteration checkpoint: {iter_checkpoint_path}")
        
        try:
            model = MMEBModel.load(iter_checkpoint_path, model_args)
            print_master(f"Successfully loaded model from iteration {resume_from_iteration}")
        except Exception as e:
            print_master(f"Failed to load iteration checkpoint: {e}")
            print_master("Falling back to base model...")
            model = MMEBModel.build(model_args)
            resume_from_iteration = None
    else:
        # Build new model or load from regular checkpoint
        if resume_checkpoint_dir:
            try:
                print_master(f"Loading model from checkpoint: {resume_checkpoint_dir}")
                model = MMEBModel.load(resume_checkpoint_dir, model_args)
            except Exception as e:
                print_master(f"Failed to load checkpoint: {e}")
                print_master("Building new model...")
                model = MMEBModel.build(model_args)
        else:
            model = MMEBModel.build(model_args)
    
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'Model backbone: {model_backbone}')
    
    # Load processor
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)

    # Load foundation model for caption generation
    foundation_model = load_foundation_model(model_args, data_args)

    # Load dataset configuration
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)
        
        # Check if this is an iterative training config
        is_iterative = any('iterative' in str(config).lower() for config in dataset_config.values())
        
        if is_iterative:
            print_master("Detected iterative training configuration")
            # For iterative training, we'll handle dataset loading in the trainer
            train_dataset = None
        else:
            # Standard dataset loading
            train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)

    # Create data collator
    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)

    # Create trainer
    if is_iterative:
        print_master("Creating iterative trainer...")
        
        # Extract iterative training parameters
        iterative_params = {}
        for config_name, config in dataset_config.items():
            if isinstance(config, dict):
                # Basic iterative parameters
                iterative_params.update({
                    'max_iterations': config.get('max_iterations', 3),
                    'hard_neg_collection_freq': config.get('hard_neg_collection_freq', 1),
                    'caption_generation_batch_size': config.get('caption_generation_batch_size', 8)
                })
                
                # Fast mode and production mode parameters
                fast_mode = config.get('fast_mode', False)
                iterative_params['fast_mode'] = fast_mode
                
                if fast_mode:
                    # Use fast mode settings
                    iterative_params.update({
                        'fast_mode_max_samples': config.get('fast_mode_max_samples', 100),
                        'fast_mode_retrieval_db_size': config.get('fast_mode_retrieval_db_size', 50),
                        'fast_mode_max_steps': config.get('fast_mode_max_steps', 5)
                    })
                    print_master(f"Fast mode enabled: {config.get('fast_mode_max_steps', 5)} steps per iteration")
                else:
                    # Use production mode settings
                    iterative_params.update({
                        'production_max_steps': config.get('production_max_steps', 1000),
                        'production_save_steps': config.get('production_save_steps', 100)
                    })
                    print_master(f"Production mode enabled: {config.get('production_max_steps', 1000)} steps per iteration")
                
                break
        
        # Create initial dataset for iteration 0
        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)
        
        # Debug: Print iterative_params to verify fast_mode is included
        print_master(f"DEBUG: iterative_params = {iterative_params}")
        
        trainer = create_iterative_trainer(
            model=model,
            foundation_model=foundation_model,
            processing_class=processor,
            args=training_args,
            model_args=model_args,
            train_dataset=train_dataset,
            data_collator=train_collator,
            max_length=data_args.max_len,
            **iterative_params
        )
        
        # Start iterative training with proper resume handling
        if resume_from_iteration is not None:
            print_master(f"Resuming iterative training from iteration {resume_from_iteration + 1}")
            trainer.iterative_train(resume_from_iteration=resume_from_iteration + 1)
        else:
            print_master("Starting iterative training from scratch")
            trainer.iterative_train(resume_from_iteration=0)
        
    else:
        print_master("Creating standard trainer...")
        trainer = IterativeRetrievalTrainer(
            model=model,
            foundation_model=foundation_model,
            processing_class=processor,
            args=training_args,
            model_args=model_args,
            data_args=data_args,
            train_dataset=train_dataset,
            data_collator=train_collator,
            max_length=data_args.max_len,
        )
        
        # Standard training
        trainer.train(resume_from_checkpoint=resume_checkpoint_dir)

    # Save final model
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)

    print_master("Training completed!")


if __name__ == "__main__":
    main()
