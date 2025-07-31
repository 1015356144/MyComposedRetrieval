#!/bin/bash
# Iterative Training Script for Composed Image Retrieval
# Usage: ./run_iterative_training.sh [cirr|fashioniq] [qwen2vl|llava_next]

set -e

# Configuration
DATASET=${1:-"cirr"}  # cirr or fashioniq
MODEL_TYPE=${2:-"qwen2vl"}  # qwen2vl, llava_next, etc.

# Local model path
QWEN2VL_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-2B-Instruct"

echo "==> Starting iterative training for $DATASET with $MODEL_TYPE"
echo "==> Environment Setup"
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Environment variables - Use default locations for testing
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export HF_HOME="$HOME/.cache/huggingface"
export WANDB_DISABLED=true  # Disable wandb for testing
export WANDB_PROJECT="iterative_composed_retrieval"
# export WANDB_API_KEY="your_wandb_api_key"  # Not needed when disabled
# export HUGGING_FACE_HUB_TOKEN="your_hf_token"  # Optional

# Experiment configuration
if [ "$DATASET" == "cirr" ]; then
    CONFIG_FILE="configs/cirr_iterative.yaml"
    EXP_NAME="IterativeCIRR_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
elif [ "$DATASET" == "fashioniq" ]; then
    CONFIG_FILE="configs/fashioniq_iterative.yaml"
    EXP_NAME="IterativeFashionIQ_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
else
    echo "Error: Unknown dataset $DATASET. Use 'cirr' or 'fashioniq'"
    exit 1
fi

# Model configuration
if [ "$MODEL_TYPE" == "qwen2vl" ]; then
    MODEL_NAME="$QWEN2VL_PATH"
    FOUNDATION_MODEL="$QWEN2VL_PATH"
elif [ "$MODEL_TYPE" == "llava_next" ]; then
    MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
    FOUNDATION_MODEL="llava-hf/llava-v1.6-mistral-7b-hf"
else
    echo "Error: Unknown model type $MODEL_TYPE"
    exit 1
fi

# Output directory
export EXP_DIR="./experiments/$EXP_NAME"
export WANDB_NAME=$EXP_NAME
export WANDB_DIR=$EXP_DIR

echo "==> Experiment: $EXP_NAME"
echo "==> Output directory: $EXP_DIR"
echo "==> Config file: $CONFIG_FILE"

# Create experiment directory
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

# Training command - reduced for testing
cmd="CUDA_VISIBLE_DEVICES=0 python \
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
    --dataloader_num_workers 2 \
    --dataset_config $CONFIG_FILE \
    --run_name $EXP_NAME \
    --project_name $WANDB_PROJECT \
    --output_dir $EXP_DIR \
    --per_device_train_batch_size 2 \
    --lr_scheduler_type linear \
    --learning_rate 5e-5 \
    --max_steps 10 \
    --warmup_steps 2 \
    --save_steps 5 \
    --logging_steps 1 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from auto \
    --max_len 512 \
    --resize_use_processor True \
    --resize_min_pixels 3136 \
    --resize_max_pixels 35840 \
    2>&1 | tee $EXP_DIR/train.log"

echo "==> Running command:"
echo $cmd
echo ""

# Execute training
eval $cmd

echo ""
echo "==> Training completed!"
echo "==> Results saved to: $EXP_DIR"
echo "==> Log file: $EXP_DIR/train.log"