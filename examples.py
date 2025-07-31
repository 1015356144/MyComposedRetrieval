#!/usr/bin/env python3
"""
Example usage script for iterative training
This script shows how to use the iterative training system
"""

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def example_basic_training():
    """Example: Basic iterative training"""
    print("="*50)
    print("Example 1: Basic Iterative Training")
    print("="*50)
    
    cmd = """
python train_iterative.py \
    --model_backbone qwen2_vl \
    --dataset_name cirr \
    --num_iterations 3 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --fast_mode
"""
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- Uses Qwen2VL as backbone model")
    print("- Trains on CIRR dataset")
    print("- Runs 3 iterations of hard negative mining")
    print("- Uses fast mode for quick testing")
    print("")


def example_custom_foundation_model():
    """Example: Using custom foundation model"""
    print("="*50)
    print("Example 2: Custom Foundation Model")
    print("="*50)
    
    cmd = """
python train_iterative.py \
    --model_backbone qwen2_vl \
    --foundation_model_path ./models/Qwen2-VL-2B-Instruct \
    --dataset_name cirr \
    --num_iterations 5 \
    --hard_negative_k 10 \
    --caption_generation_batch_size 8
"""
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- Uses local foundation model for caption generation")
    print("- Mines top-10 hard negatives per iteration")
    print("- Processes captions in batches of 8")
    print("")


def example_resume_training():
    """Example: Resume from checkpoint"""
    print("="*50)
    print("Example 3: Resume Training")
    print("="*50)
    
    cmd = """
python train_iterative.py \
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \
    --resume_from_iteration 2 \
    --auto_resume
"""
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- Resumes training from iteration 2")
    print("- Automatically finds and loads checkpoint")
    print("- Continues with existing configuration")
    print("")


def example_evaluation():
    """Example: Evaluate trained models"""
    print("="*50)
    print("Example 4: Model Evaluation")
    print("="*50)
    
    cmd = """
python eval_iterative.py \
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \
    --iterations "1,3,5" \
    --eval_dataset cirr_test \
    --output_file results.json
"""
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- Evaluates models from iterations 1, 3, and 5")
    print("- Tests on CIRR test set")
    print("- Saves results to results.json")
    print("")


def example_fashioniq():
    """Example: FashionIQ dataset"""
    print("="*50)
    print("Example 5: FashionIQ Training")
    print("="*50)
    
    cmd = """
./run_iterative_training.sh fashioniq llava_next
"""
    print("Command:")
    print(cmd)
    print("\nDescription:")
    print("- Trains on FashionIQ dataset")
    print("- Uses LLaVA-Next as backbone")
    print("- Runs with shell script for convenience")
    print("")


def main():
    """Main function to run examples"""
    parser = argparse.ArgumentParser(description='Examples for iterative training')
    parser.add_argument('--example', type=int, choices=[1,2,3,4,5], 
                       help='Run specific example (1-5)')
    parser.add_argument('--all', action='store_true', 
                       help='Show all examples')
    
    args = parser.parse_args()
    
    examples = {
        1: example_basic_training,
        2: example_custom_foundation_model,
        3: example_resume_training,
        4: example_evaluation,
        5: example_fashioniq
    }
    
    if args.all:
        for i in range(1, 6):
            examples[i]()
    elif args.example:
        examples[args.example]()
    else:
        print("Iterative Composed Image Retrieval - Examples")
        print("="*50)
        print("Available examples:")
        print("1. Basic iterative training")
        print("2. Custom foundation model")
        print("3. Resume training")
        print("4. Model evaluation")
        print("5. FashionIQ training")
        print("\nUsage:")
        print("  python examples.py --example 1")
        print("  python examples.py --all")


if __name__ == '__main__':
    main()