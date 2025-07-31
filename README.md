# Iterative Composed Image Retrieval

This project implements iterative training for composed image retrieval using VLM2Vec with hard negative mining and foundation model augmentation.

## Features

- 🔄 **Iterative Training**: Progressive hard negative mining across multiple training rounds
- 🎯 **Real Retrieval**: Actual VLM2Vec model inference instead of simulation
- 🤖 **Foundation Model Integration**: Qwen2VL for caption generation and data augmentation
- 💾 **Smart Caching**: Avoid repeated computations with checkpoint resumption
- 📊 **Progress Tracking**: Real-time progress display with ETA estimation
- 🔧 **Flexible Configuration**: Support for different model backbones and datasets

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Basic Training

```bash
# Run iterative training with default settings
./run_iterative_training.sh

# Or run directly with Python
python train_iterative.py \
    --model_backbone qwen2_vl \
    --dataset_name cirr \
    --num_iterations 3 \
    --foundation_model_path /path/to/qwen2-vl-model
```

### 3. Fast Mode (for testing)

```bash
# Quick test with subset of data
./run_iterative_training.sh --fast_mode
```

## Configuration

### Training Parameters

- `--num_iterations`: Number of iterative rounds (default: 3)
- `--hard_negative_k`: Top-k for hard negative mining (default: 5)
- `--foundation_model_path`: Path to foundation model for caption generation
- `--fast_mode`: Use subset of data for quick testing
- `--auto_resume`: Automatically resume from latest checkpoint

### Model Parameters

- `--model_backbone`: VLM backbone (qwen2_vl, llava, etc.)
- `--foundation_model_backbone`: Foundation model type (qwen2_vl, llava)
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate

### Example Commands

```bash
# Full training with Qwen2VL foundation model
python train_iterative.py \
    --model_backbone qwen2_vl \
    --dataset_name cirr \
    --num_iterations 5 \
    --hard_negative_k 10 \
    --foundation_model_path ./models/Qwen2-VL-2B-Instruct \
    --batch_size 32 \
    --learning_rate 1e-5

# Resume training from iteration 2
python train_iterative.py \
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \
    --resume_from_iteration 2

# Fast mode for debugging
python train_iterative.py \
    --fast_mode \
    --num_iterations 2
```

## Evaluation

### Evaluate All Iterations

```bash
python eval_iterative.py \
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \
    --eval_dataset cirr_test
```

### Evaluate Specific Iterations

```bash
python eval_iterative.py \
    --experiment_dir ./experiments/qwen2_vl_cirr_iterative \
    --iterations "1,3,5" \
    --output_file results.json
```

## Project Structure

```
├── src/
│   ├── data/dataset/
│   │   └── composed_retrieval_dataset.py  # Iterative CIRR dataset with real retrieval
│   ├── trainer_iterative.py              # Iterative trainer implementation
│   └── ...
├── train_iterative.py                    # Training script
├── eval_iterative.py                     # Evaluation script
├── run_iterative_training.sh             # Training shell script
└── experiments/                          # Experiment outputs
    └── {model}_{dataset}_iterative/
        ├── hard_negatives_iter_0.json    # Hard negatives per iteration
        ├── augmented_samples_iter_1.json # Generated augmented samples
        ├── checkpoint_iter_1/            # Model checkpoints
        └── evaluation_results.json       # Evaluation results
```

## How It Works

### 1. Iterative Training Loop

```
For each iteration:
1. Mine hard negatives using current model
2. Generate augmented captions with foundation model
3. Train on original + augmented data
4. Save checkpoint
```

### 2. Hard Negative Mining

- Uses real VLM2Vec retrieval on training data
- Identifies samples where ground truth is not top-1
- Collects top-ranked incorrect results as hard negatives

### 3. Caption Augmentation

- Foundation model (Qwen2VL) generates new modification texts
- Input: reference image + target image + original caption
- Output: similar but different modification text
- Creates positive samples from previous hard negatives

### 4. Smart Caching

- Caches hard negatives to avoid re-mining
- Saves augmented samples for checkpoint resumption
- Experiment directory tracks all intermediate results

## Advanced Features

### Real vs Simulated Retrieval

- **Real Retrieval**: Actual VLM2Vec model inference (default)
- **Simulated Retrieval**: Fast dummy results for testing
- Automatic fallback if real retrieval fails

### Progress Tracking

- Batch-level progress with ETA estimation
- Generation rate monitoring
- Comprehensive logging and statistics

### Foundation Model Support

- Qwen2VL: Multi-image conversation format
- LLaVA: Horizontal image concatenation
- Generic: Fallback for other models

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use fast mode
2. **Foundation Model Loading**: Check model path and permissions
3. **Checkpoint Resumption**: Verify experiment directory structure

### Debug Mode

```bash
# Enable detailed logging
PYTHONPATH=. python train_iterative.py --fast_mode --num_iterations 1
```

## Citation

If you use this work, please cite:

```bibtex
@article{iterative_cir_2025,
  title={Iterative Training for Composed Image Retrieval with Hard Negative Mining},
  author={Your Name},
  year={2025}
}
```