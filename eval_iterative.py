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


def find_iteration_checkpoints(experiment_dir: str) -> List[Dict]:
    """Find all iteration checkpoints in experiment directory"""
    checkpoints = []
    
    if not os.path.exists(experiment_dir):
        print_rank(f"Experiment directory not found: {experiment_dir}")
        return checkpoints
    
    for item in os.listdir(experiment_dir):
        if item.startswith("checkpoint_iter_"):
            try:
                iter_num = int(item.split("_")[-1])
                checkpoint_dir = os.path.join(experiment_dir, item)
                
                # Check if checkpoint has required files
                model_path = os.path.join(checkpoint_dir, "model.pt")
                if os.path.exists(model_path):
                    checkpoints.append({
                        'iteration': iter_num,
                        'path': checkpoint_dir,
                        'model_path': model_path
                    })
            except ValueError:
                continue
    
    # Sort by iteration number
    checkpoints.sort(key=lambda x: x['iteration'])
    return checkpoints


def evaluate_checkpoint(checkpoint_info: Dict, args, eval_function) -> Dict:
    """Evaluate a single checkpoint"""
    print_rank(f"\n{'='*50}")
    print_rank(f"Evaluating Iteration {checkpoint_info['iteration']}")
    print_rank(f"Checkpoint: {checkpoint_info['path']}")
    print_rank(f"{'='*50}")
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(checkpoint_info['path'], args)
    if model is None:
        return {'error': 'Failed to load model'}
    
    try:
        # Run evaluation
        results = eval_function(model, args)
        results['iteration'] = checkpoint_info['iteration']
        results['checkpoint_path'] = checkpoint_info['path']
        
        print_rank(f"Iteration {checkpoint_info['iteration']} evaluation completed")
        return results
        
    except Exception as e:
        print_rank(f"Error evaluating iteration {checkpoint_info['iteration']}: {e}")
        return {'error': str(e), 'iteration': checkpoint_info['iteration']}


def run_evaluation_function(model, args):
    """Run the actual evaluation function"""
    # Import evaluation modules
    from eval import main as eval_main
    
    # Temporarily override the model in args
    original_model_path = getattr(args, 'model_path', None)
    
    try:
        # Set the loaded model
        args.model = model
        
        # Run evaluation
        results = eval_main(args)
        return results
        
    finally:
        # Restore original model path
        if original_model_path:
            args.model_path = original_model_path


def save_evaluation_results(results: List[Dict], output_file: str):
    """Save evaluation results to file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print_rank(f"Evaluation results saved to {output_file}")
    except Exception as e:
        print_rank(f"Error saving results: {e}")


def print_results_summary(results: List[Dict]):
    """Print summary of evaluation results"""
    print_rank(f"\n{'='*70}")
    print_rank(f"EVALUATION RESULTS SUMMARY")
    print_rank(f"{'='*70}")
    
    # Extract key metrics for comparison
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print_rank("No valid results to display")
        return
    
    # Print iteration comparison
    print_rank(f"{'Iteration':<10} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'mAP':<8}")
    print_rank(f"{'-'*50}")
    
    for result in valid_results:
        iteration = result.get('iteration', 'N/A')
        
        # Extract metrics (adjust based on your evaluation metrics)
        r1 = result.get('recall_at_1', result.get('R@1', 'N/A'))
        r5 = result.get('recall_at_5', result.get('R@5', 'N/A'))
        r10 = result.get('recall_at_10', result.get('R@10', 'N/A'))
        map_score = result.get('mAP', result.get('mean_ap', 'N/A'))
        
        # Format numbers
        def format_metric(metric):
            if isinstance(metric, (int, float)):
                return f"{metric:.3f}"
            return str(metric)
        
        print_rank(f"{iteration:<10} {format_metric(r1):<8} {format_metric(r5):<8} {format_metric(r10):<8} {format_metric(map_score):<8}")
    
    # Find best iteration
    if len(valid_results) > 1:
        # Find best by R@1 (adjust metric as needed)
        best_result = max(valid_results, key=lambda x: x.get('recall_at_1', x.get('R@1', 0)))
        print_rank(f"\nBest performing iteration: {best_result['iteration']}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Iterative Training Evaluation')
    
    parser.add_argument('--experiment_dir', type=str, required=True,
                      help='Directory containing iteration checkpoints')
    
    parser.add_argument('--iterations', type=str, default='all',
                      help='Iterations to evaluate (comma-separated numbers or "all")')
    
    parser.add_argument('--output_file', type=str, default=None,
                      help='File to save evaluation results')
    
    parser.add_argument('--eval_dataset', type=str, default='cirr_test',
                      help='Dataset to evaluate on')
    
    # Parse iterative-specific args first
    iterative_args, remaining_args = parser.parse_known_args()
    
    # Parse main evaluation args
    sys.argv = [sys.argv[0]] + remaining_args
    main_args = parse_args()
    
    # Merge arguments
    for key, value in vars(iterative_args).items():
        setattr(main_args, key, value)
    
    args = main_args
    
    print_rank("Starting iterative evaluation...")
    print_rank(f"Experiment directory: {args.experiment_dir}")
    print_rank(f"Evaluation dataset: {args.eval_dataset}")
    
    # Find available checkpoints
    checkpoints = find_iteration_checkpoints(args.experiment_dir)
    
    if not checkpoints:
        print_rank(f"No checkpoints found in {args.experiment_dir}")
        return
    
    print_rank(f"Found {len(checkpoints)} iteration checkpoints")
    
    # Filter checkpoints based on requested iterations
    if args.iterations != 'all':
        try:
            requested_iterations = [int(x.strip()) for x in args.iterations.split(',')]
            checkpoints = [cp for cp in checkpoints if cp['iteration'] in requested_iterations]
            print_rank(f"Evaluating {len(checkpoints)} selected iterations: {requested_iterations}")
        except ValueError:
            print_rank(f"Invalid iteration specification: {args.iterations}")
            return
    
    # Run evaluation for each checkpoint
    all_results = []
    
    for checkpoint_info in checkpoints:
        result = evaluate_checkpoint(checkpoint_info, args, run_evaluation_function)
        all_results.append(result)
    
    # Print summary
    print_results_summary(all_results)
    
    # Save results if output file specified
    if args.output_file:
        if not args.output_file.endswith('.json'):
            args.output_file += '.json'
        save_evaluation_results(all_results, args.output_file)
    else:
        # Default output file
        default_output = os.path.join(args.experiment_dir, 'evaluation_results.json')
        save_evaluation_results(all_results, default_output)
    
    print_rank("Iterative evaluation completed!")


if __name__ == '__main__':
    main()
