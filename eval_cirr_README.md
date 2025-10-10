# CIRR Evaluation System

This directory contains a complete evaluation system for CIRR (Composed Image Retrieval for Real-world scenes) dataset. The system is designed to work seamlessly with models trained using the iterative training pipeline.

## Files Overview

- `eval_cirr.py` - Main evaluation script with flexible model loading
- `eval_cirr.sh` - Shell script wrapper for easy command-line usage  
- `configs/cirr_eval_config.yaml` - Example configuration file
- `src/evaluation/cirr_evaluator.py` - Core CIRR evaluator implementation

## Quick Start

### 1. Basic Evaluation

```bash
# Evaluate a trained checkpoint
./eval_cirr.sh --model_path ./outputs/checkpoint-1000

# Evaluate an iteration model
./eval_cirr.sh --model_path ./outputs/iteration_2
```

### 2. Custom Configuration

```bash
# With custom model name and batch size
./eval_cirr.sh --model_path ./outputs/checkpoint-1000 \
               --model_name Qwen/Qwen2-VL-2B-Instruct \
               --batch_size 32

# With custom CIRR dataset paths
./eval_cirr.sh --model_path ./outputs/checkpoint-1000 \
               --cirr_data_dir /path/to/cirr \
               --cirr_image_dir /path/to/images

# Save results to file
./eval_cirr.sh --model_path ./outputs/checkpoint-1000 \
               --output_file ./results/eval_results.json
```

### 3. Using Python Directly

```bash
python eval_cirr.py --model_path ./outputs/checkpoint-1000 \
                    --batch_size 16 \
                    --device cuda \
                    --output_file ./results.json
```

## Command Line Options

### Required Arguments
- `--model_path`: Path to trained model checkpoint (checkpoint-xxx or iteration_x directory)

### Optional Arguments
- `--model_name`: Base model name (e.g., Qwen/Qwen2-VL-2B-Instruct). Auto-detected if not provided
- `--eval_config`: Path to evaluation config YAML file. Uses default if not provided
- `--batch_size`: Batch size for evaluation (default: 16)
- `--device`: Device to use - auto, cuda, cuda:0, etc. (default: auto)
- `--distributed`: Enable distributed evaluation across multiple GPUs
- `--output_file`: File to save results in JSON format
- `--cirr_data_dir`: Override default CIRR dataset directory
- `--cirr_image_dir`: Override default CIRR image directory
- `--verbose`: Print detailed results (default: true)

## Model Loading

The evaluation system supports multiple model loading scenarios:

### 1. Trainer Checkpoints
Standard training checkpoints saved by the trainer:
```
outputs/
├── checkpoint-1000/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── optimizer.pt
│   └── scheduler.pt
```

### 2. Iteration Models  
Models saved after each iteration of iterative training:
```
outputs/
├── base_model/
├── iteration_1/
├── iteration_2/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
```

### 3. Custom Model Paths
Any directory containing model weights and config files.

## Evaluation Metrics

The system computes two types of CIRR metrics:

### Global Recall@K
Standard retrieval evaluation across all candidate images:
- Recall@1, Recall@5, Recall@10, Recall@50

### Group Recall@K  
Evaluation within image groups (more realistic scenario):
- Group Recall@1, Group Recall@2, Group Recall@3

## Configuration

### Default Configuration
If no config file is provided, the system uses reasonable defaults:

```yaml
CIRR:
  data_dir: "/home/guohaiyun/yty_data/CIRR/cirr"
  image_base_dir: "/home/guohaiyun/yty_data/CIRR"
  validation:
    queries_file: "captions/cap.rc2.val.json"
    splits_file: "image_splits/split.rc2.val.json"
  evaluation:
    batch_size: 16
    global_recall_k: [1, 5, 10, 50]
    group_recall_k: [1, 2, 3]
```

### Custom Configuration
Create your own config file based on `configs/cirr_eval_config.yaml`:

```bash
# Use custom config
./eval_cirr.sh --model_path ./outputs/checkpoint-1000 \
               --eval_config ./my_eval_config.yaml
```

## Output Format

Results are printed to console and optionally saved to JSON file:

```json
{
  "model_path": "./outputs/checkpoint-1000",
  "model_name": "Qwen/Qwen2-VL-2B-Instruct", 
  "model_backbone": "qwen2_vl",
  "batch_size": 16,
  "distributed": false,
  "results": {
    "r_at_1": 0.2145,
    "r_at_5": 0.4523,
    "r_at_10": 0.5876,
    "r_at_50": 0.7654,
    "group_recall@1": 0.3456,
    "group_recall@2": 0.5123,
    "group_recall@3": 0.6234
  },
  "timestamp": "2025-09-09T10:30:45"
}
```

## Distributed Evaluation

For faster evaluation on multiple GPUs:

```bash
# Enable distributed evaluation
./eval_cirr.sh --model_path ./outputs/checkpoint-1000 --distributed

# Or with torchrun for explicit GPU control
torchrun --nproc_per_node=2 eval_cirr.py --model_path ./outputs/checkpoint-1000 --distributed
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure the model path exists and contains valid checkpoint files
   - Try specifying the base model name explicitly with `--model_name`

2. **CUDA Out of Memory**
   - Reduce batch size with `--batch_size 8` or smaller
   - Use `--device cpu` for CPU evaluation

3. **Dataset Not Found**
   - Verify CIRR dataset paths with `--cirr_data_dir` and `--cirr_image_dir`
   - Check that the dataset files exist at the specified locations

4. **Import Errors**
   - Ensure you're running from the project root directory
   - Check that all dependencies are installed

### Getting Help

```bash
# Show detailed usage information
./eval_cirr.sh --help

# Check if script can find eval_cirr.py
ls -la eval_cirr.py
```

## Integration with Training Pipeline

This evaluation system is designed to work seamlessly with the iterative training pipeline:

1. **During Training**: Models are saved as iteration checkpoints
2. **After Training**: Use this system to evaluate any saved model
3. **Comparison**: Compare different iterations or training configurations

Example workflow:
```bash
# Train model with iterative training
python train_iterative.py --config configs/training_config.yaml

# Evaluate base model
./eval_cirr.sh --model_path ./outputs/base_model

# Evaluate iteration 1
./eval_cirr.sh --model_path ./outputs/iteration_1

# Evaluate iteration 2 
./eval_cirr.sh --model_path ./outputs/iteration_2
```
