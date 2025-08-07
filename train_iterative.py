#!/usr/bin/env python3
"""
Iterative Training Script for Composed Image Retrieval
Adapted from VLM2Vec training pipeline for iterative hard negative mining
"""

import logging
import os
import os.path
import sys

# Enable wandb for production training monitoring
# os.environ['WANDB_DISABLED'] = 'true'  # Commented out to enable wandb

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

import torch
import wandb
import yaml
import json
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
        import torch.distributed as dist
        
        # Check if we're in distributed mode to avoid tensor parallel issues
        if dist.is_initialized():
            # Distributed mode: avoid device_map to prevent tensor parallel conflicts
            foundation_model = AutoModelForVision2Seq.from_pretrained(
                foundation_model_name,
                torch_dtype=torch.bfloat16,
                device_map=None,  # Avoid tensor parallel issues in PyTorch 2.4
                trust_remote_code=True
            )
        else:
            # Single GPU mode: use device_map="auto" for convenience
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
    
    # Disable tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
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

    # Set up logging_dir for train.log generation if not specified
    if not training_args.logging_dir:
        training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
        print_master(f"Setting logging_dir to: {training_args.logging_dir}")
    
    # Ensure logging directory exists
    os.makedirs(training_args.logging_dir, exist_ok=True)

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
    
    # 修复：按照VLM2Vec方式，先从原始模型获取正确的model_backbone
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=getattr(model_args, 'model_type', None))
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_master(f'Model backbone: {model_args.model_backbone}')
    
    # Check for iterative training checkpoints
    resume_from_iteration = None
    model_checkpoint_path = None
    
    def check_iteration_complete(output_dir, iteration, max_iterations):
        """Check if an iteration is completely finished"""
        state_file = os.path.join(output_dir, f"iteration_{iteration}_state.json")
        if not os.path.exists(state_file):
            return False
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            return state.get('iteration_complete', False)
        except:
            return False
    
    if training_args.resume_from == 'auto':
        # Auto-detect latest COMPLETE iteration checkpoint
        # First check for complete iterations (highest to lowest)
        for i in range(10, -1, -1):  # Check iterations 0-10
            if i == 0:
                # Special case for iteration 0 -> base_model
                model_path = os.path.join(training_args.output_dir, "base_model")
            else:
                model_path = os.path.join(training_args.output_dir, f"iteration_{i}")
            
            if os.path.exists(model_path) and check_iteration_complete(training_args.output_dir, i, 10):
                resume_from_iteration = i
                model_checkpoint_path = model_path
                print_master(f"Found COMPLETE iteration {i} checkpoint: {model_path}")
                break
        
        # If no complete iterations found, look for incomplete iterations
        if resume_from_iteration is None:
            for i in range(10, -1, -1):
                if i == 0:
                    model_path = os.path.join(training_args.output_dir, "base_model")
                else:
                    model_path = os.path.join(training_args.output_dir, f"iteration_{i}")
                
                if os.path.exists(model_path):
                    resume_from_iteration = i
                    model_checkpoint_path = model_path
                    print_master(f"Found INCOMPLETE iteration {i} checkpoint: {model_path}")
                    break
                
    elif training_args.resume_from.startswith('iter_'):
        # Manual iteration specification: iter_2
        iter_num = training_args.resume_from.split('_')[1]
        if iter_num.isdigit():
            iter_num = int(iter_num)
            if iter_num == 0:
                # Special case for iteration 0 -> base_model
                iter_checkpoint = os.path.join(training_args.output_dir, "base_model")
            else:
                iter_checkpoint = os.path.join(training_args.output_dir, f"iteration_{iter_num}")
            
            if os.path.exists(iter_checkpoint):
                resume_from_iteration = iter_num
                model_checkpoint_path = iter_checkpoint
                complete_status = "COMPLETE" if check_iteration_complete(training_args.output_dir, iter_num, 10) else "INCOMPLETE"
                print_master(f"Manually resuming from {complete_status} iteration {iter_num} at {iter_checkpoint}")
    
    # Unified model loading logic - load model only once
    model = None
    if model_checkpoint_path is not None:
        # Try to load from iteration/base checkpoint first
        print_master(f"Loading model from iteration checkpoint: {model_checkpoint_path}")
        try:
            model_args.checkpoint_path = model_checkpoint_path
            model = MMEBModel.load(model_args, is_trainable=True)  # 修复：保持LoRA可训练
            print_master(f"Successfully loaded model from iteration {resume_from_iteration}")
        except Exception as e:
            print_master(f"Failed to load iteration checkpoint: {e}")
            # Clear failed checkpoint path for fallback
            if hasattr(model_args, 'checkpoint_path'):
                delattr(model_args, 'checkpoint_path')
            model = None
    
    # Fallback to regular checkpoint if iteration checkpoint failed
    if model is None and resume_checkpoint_dir:
        print_master(f"Loading model from regular checkpoint: {resume_checkpoint_dir}")
        try:
            model_args.checkpoint_path = resume_checkpoint_dir
            model = MMEBModel.load(model_args, is_trainable=True)  # 修复：保持LoRA可训练
            print_master(f"Successfully loaded model from regular checkpoint")
        except Exception as e:
            print_master(f"Failed to load regular checkpoint: {e}")
            # Clear failed checkpoint path for building new model
            if hasattr(model_args, 'checkpoint_path'):
                delattr(model_args, 'checkpoint_path')
            model = None
    
    # Build new model if all checkpoint loading failed
    if model is None:
        print_master("Building new model...")
        model = MMEBModel.build(model_args)
    
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
            # Check if the found iteration is complete
            is_iteration_complete = check_iteration_complete(training_args.output_dir, resume_from_iteration, 10)
            
            if is_iteration_complete:
                # Complete iteration found - start from next iteration
                next_iteration = resume_from_iteration + 1
                print_master(f"Loaded COMPLETE iteration {resume_from_iteration}, starting from iteration {next_iteration}")
                trainer.iterative_train(resume_from_iteration=next_iteration)
            else:
                # Incomplete iteration found - resume from same iteration
                print_master(f"Loaded INCOMPLETE iteration {resume_from_iteration}, resuming from iteration {resume_from_iteration}")
                trainer.iterative_train(resume_from_iteration=resume_from_iteration)
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
