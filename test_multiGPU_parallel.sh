#!/bin/bash
# Quick Multi-GPU Parallel Test Script
# Tests the new distributed hard negative mining and caption generation

set -e

echo "🚀 Starting Multi-GPU Parallel Code Test"
echo "=================================="

# Configuration
NUM_GPUS=2  # Test with 2 GPUs
MODEL_TYPE="qwen2vl"
DATASET="cirr"

# Local model path
QWEN2VL_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-2B-Instruct"

# Environment setup
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export HF_HOME="$HOME/.cache/huggingface"
export WANDB_DISABLED=true
export WANDB_PROJECT="multiGPU_parallel_test"

# Experiment configuration
CONFIG_FILE="configs/cirr_iterative_multiGPU_test.yaml"
EXP_NAME="MultiGPU_Test_$(date +%Y%m%d_%H%M%S)"
MODEL_NAME="$QWEN2VL_PATH"
FOUNDATION_MODEL="$QWEN2VL_PATH"

# Output directory
export EXP_DIR="./experiments/$EXP_NAME"
export WANDB_NAME=$EXP_NAME
export WANDB_DIR=$EXP_DIR

echo "📋 Test Configuration:"
echo "  • GPUs: $NUM_GPUS"
echo "  • Model: $MODEL_TYPE"
echo "  • Dataset: $DATASET"  
echo "  • Config: $CONFIG_FILE"
echo "  • Output: $EXP_DIR"
echo ""

# Create experiment directory
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
echo ""

# Training command with torchrun for multi-GPU
echo "🏃 Starting multi-GPU training..."
cmd="CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_iterative.py \
    --model_name $MODEL_NAME \
    --foundation_model_name $FOUNDATION_MODEL \
    --lora \
    --lora_r 16 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataloader_num_workers 1 \
    --dataset_config $CONFIG_FILE \
    --run_name $EXP_NAME \
    --project_name $WANDB_PROJECT \
    --output_dir $EXP_DIR \
    --per_device_train_batch_size 1 \
    --lr_scheduler_type linear \
    --learning_rate 5e-5 \
    --max_steps 5 \
    --warmup_steps 1 \
    --save_steps 3 \
    --logging_steps 1 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from auto \
    --max_len 256 \
    --resize_use_processor True \
    --resize_min_pixels 3136 \
    --resize_max_pixels 35840 \
    2>&1 | tee $EXP_DIR/test.log"

echo "💻 Running command:"
echo $cmd
echo ""

# Execute training
start_time=$(date +%s)
eval $cmd
end_time=$(date +%s)

# Calculate duration
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "🎉 Multi-GPU Parallel Test Completed!"
echo "=================================="
echo "⏱️  Total time: ${minutes}m ${seconds}s"
echo "📁 Results saved to: $EXP_DIR"
echo "📜 Log file: $EXP_DIR/test.log"
echo ""

# Check if key files were created
echo "📋 Checking generated files:"
if [ -f "$EXP_DIR/hard_negatives_iter_0.json" ]; then
    hard_neg_count=$(jq length "$EXP_DIR/hard_negatives_iter_0.json" 2>/dev/null || echo "N/A")
    echo "  ✅ Hard negatives file: $hard_neg_count samples"
else
    echo "  ❌ Hard negatives file: Missing"
fi

if [ -f "$EXP_DIR/augmented_samples_iter_1.json" ]; then
    aug_count=$(jq length "$EXP_DIR/augmented_samples_iter_1.json" 2>/dev/null || echo "N/A")
    echo "  ✅ Augmented samples file: $aug_count samples"
else
    echo "  ❌ Augmented samples file: Missing"
fi

if [ -d "$EXP_DIR/cache" ]; then
    cache_files=$(ls -1 "$EXP_DIR/cache" | wc -l)
    echo "  ✅ Cache directory: $cache_files files"
else
    echo "  ❌ Cache directory: Missing"
fi

echo ""
echo "🔍 Check the log file for detailed multi-GPU execution information:"
echo "   tail -50 $EXP_DIR/test.log"
echo ""
echo "✨ Multi-GPU parallel test completed successfully!"
