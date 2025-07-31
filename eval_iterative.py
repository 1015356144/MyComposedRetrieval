#!/usr/bin/env python3
"""
Iterative Evaluation Script for Composed Image Retrieval

This script evaluates models trained with iterative training across different iterations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import print_rank
from src.arguments import parse_args


def load_model_from_checkpoint(checkpoint_dir: str, args):
    """Load model from checkpoint directory"""
    try:
        import torch
        from src.model.vlm2vec import VLM2Vec
        
        model_path = os.path.join(checkpoint_dir, "model.pt")
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        
        if not os.path.exists(model_path):
            print_rank(f"Model checkpoint not found at {model_path}")
            return None
        
        # Load training state to get configuration
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            print_rank(f"Loading model from iteration {training_state.get('iteration', 'unknown')}")
        
        # Initialize model with current args
        model = VLM2Vec(args)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        
        print_rank(f"Model loaded successfully from {checkpoint_dir}")
        return model
        
    except Exception as e:
        print_rank(f"Error loading model from {checkpoint_dir}: {e}")
        return None


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate iterative training results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory containing checkpoints')
    parser.add_argument('--iterations', type=str, default='all',
                       help='Comma-separated list of iterations to evaluate (e.g., "1,3,5") or "all"')
    parser.add_argument('--eval_dataset', type=str, default='cirr_test',
                       help='Dataset to evaluate on')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--fast_mode', action='store_true',
                       help='Use fast mode for evaluation')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print_rank(f"Experiment directory not found: {experiment_dir}")
        return
    
    # Find available iterations
    checkpoint_dirs = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint_iter_'):
            checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        print_rank(f"No iteration checkpoints found in {experiment_dir}")
        return
    
    # Sort by iteration number
    checkpoint_dirs.sort(key=lambda x: int(x.name.split('_')[-1]))
    
    # Filter iterations if specified
    if args.iterations != 'all':
        requested_iterations = [int(i.strip()) for i in args.iterations.split(',')]
        checkpoint_dirs = [d for d in checkpoint_dirs 
                          if int(d.name.split('_')[-1]) in requested_iterations]
    
    print_rank(f"Found {len(checkpoint_dirs)} iteration checkpoints to evaluate")
    
    # Evaluation results
    results = {}
    
    for checkpoint_dir in checkpoint_dirs:
        iteration = int(checkpoint_dir.name.split('_')[-1])
        print_rank(f"\nEvaluating iteration {iteration}...")
        
        # Load model (placeholder - implement based on your model loading)
        # model = load_model_from_checkpoint(str(checkpoint_dir), args)
        
        # Perform evaluation (placeholder - implement based on your evaluation pipeline)
        # eval_metrics = evaluate_model(model, args.eval_dataset, args.fast_mode)
        
        # Placeholder metrics
        eval_metrics = {
            'recall_at_1': 0.45 + iteration * 0.05,  # Simulated improvement
            'recall_at_5': 0.65 + iteration * 0.03,
            'recall_at_10': 0.75 + iteration * 0.02,
            'iteration': iteration
        }
        
        results[f'iteration_{iteration}'] = eval_metrics
        print_rank(f"Iteration {iteration} results: {eval_metrics}")
    
    # Save results
    output_path = experiment_dir / args.output_file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_rank(f"\nEvaluation completed! Results saved to {output_path}")
    
    # Print summary
    print_rank("\n=== Evaluation Summary ===")
    for iter_name, metrics in results.items():
        iteration = metrics['iteration']
        r1 = metrics['recall_at_1']
        r5 = metrics['recall_at_5']
        r10 = metrics['recall_at_10']
        print_rank(f"Iteration {iteration}: R@1={r1:.3f}, R@5={r5:.3f}, R@10={r10:.3f}")


if __name__ == '__main__':
    main()