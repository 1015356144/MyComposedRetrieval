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
python train_iterative.py \\
    --model_backbone qwen2_vl \\
    --dataset_name cirr \\
    --num_iterations 3 \\
    --batch_size 16 \\
    --learning_rate 1e-5 \\
    --fast_mode
"""
    print("Command:")
    print(cmd)
    print("\nThis will run a basic 3-iteration training with fast mode enabled.")
    print()

def example_with_foundation_model():
    """Example: Training with foundation model"""
    print("="*50)
    print("Example 2: Training with Foundation Model")
    print("="*50)
    
    cmd = """
python train_iterative.py \\
    --model_backbone qwen2_vl \\
    --dataset_name cirr \\
    --num_iterations 5 \\
    --foundation_model_path ./models/Qwen2-VL-2B-Instruct \\
    --foundation_model_backbone qwen2_vl \\
    --hard_negative_k 10 \\
    --batch_size 32 \\
    --learning_rate 1e-5 \\
    --exp_suffix "with_foundation"
"""
    print("Command:")
    print(cmd)
    print("\nThis will run training with foundation model for caption generation.")
    print()

def example_resume_training():
    """Example: Resume training from checkpoint"""
    print("="*50)
    print("Example 3: Resume Training")
    print("="*50)
    
    cmd = """
python train_iterative.py \\
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \\
    --auto_resume
"""
    print("Command:")
    print(cmd)
    print("\nThis will automatically resume from the latest checkpoint.")
    print()

def example_evaluation():
    """Example: Evaluate all iterations"""
    print("="*50)
    print("Example 4: Evaluation")
    print("="*50)
    
    cmd = """
python eval_iterative.py \\
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \\
    --eval_dataset cirr_test \\
    --output_file evaluation_results.json
"""
    print("Command:")
    print(cmd)
    print("\nThis will evaluate all iterations and save results.")
    print()

def example_using_config():
    """Example: Using configuration file"""
    print("="*50)
    print("Example 5: Using Configuration File")
    print("="*50)
    
    cmd = """
python train_iterative.py \\
    --config configs/cirr_iterative.yaml \\
    --foundation_model_path ./models/Qwen2-VL-2B-Instruct
"""
    print("Command:")
    print(cmd)
    print("\nThis will use the YAML configuration file with custom foundation model path.")
    print()

def show_project_structure():
    """Show expected project structure"""
    print("="*50)
    print("Expected Project Structure")
    print("="*50)
    
    structure = """
MyComposedRetrieval/
├── configs/
│   ├── cirr_iterative.yaml
│   ├── fashioniq_iterative.yaml
│   ├── test_iterative.yaml
│   └── data_config.json
├── src/
│   ├── trainer_iterative.py
│   ├── data/dataset/
│   │   └── composed_retrieval_dataset.py
│   └── ...
├── data/
│   ├── CIRR/
│   │   ├── cirr_captions_train.json
│   │   ├── cirr_image_splits.json
│   │   └── images/
│   └── FashionIQ/
│       ├── captions/
│       └── images/
├── models/
│   └── Qwen2-VL-2B-Instruct/  # Foundation model
├── experiments/               # Training outputs
├── train_iterative.py
├── eval_iterative.py
├── run_iterative_training.sh
└── README.md
"""
    print(structure)

def main():
    parser = argparse.ArgumentParser(description='Iterative Training Examples')
    parser.add_argument('--example', type=str, choices=[
        'basic', 'foundation', 'resume', 'eval', 'config', 'structure', 'all'
    ], default='all', help='Which example to show')
    
    args = parser.parse_args()
    
    if args.example == 'all':
        show_project_structure()
        example_basic_training()
        example_with_foundation_model()
        example_resume_training()
        example_evaluation()
        example_using_config()
    elif args.example == 'basic':
        example_basic_training()
    elif args.example == 'foundation':
        example_with_foundation_model()
    elif args.example == 'resume':
        example_resume_training()
    elif args.example == 'eval':
        example_evaluation()
    elif args.example == 'config':
        example_using_config()
    elif args.example == 'structure':
        show_project_structure()
    
    print("="*50)
    print("Quick Start:")
    print("1. Run: ./test_setup.sh")
    print("2. Run: ./run_iterative_training.sh --fast_mode")
    print("3. Check: ./experiments/ for outputs")
    print("="*50)

if __name__ == '__main__':
    main()
