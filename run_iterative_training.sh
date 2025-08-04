#!/bin/bash
# Iterative Training Script for Composed Image Retrieval
# Usage: ./run_iterative_training.sh [cirr|fashioniq] [qwen2vl|llava_next] [num_gpus] [existing_exp_dir]
# Example: ./run_iterative_training.sh cirr qwen2vl 2  # Use 2 GPUs, create new experiment
# Example: ./run_iterative_training.sh cirr qwen2vl 2 ./experiments/IterativeCIRR_qwen2vl_20250805_000011  # Resume existing

set -e

# Configuration
DATASET=${1:-"cirr"}  # cirr or fashioniq
MODEL_TYPE=${2:-"qwen2vl"}  # qwen2vl, llava_next, etc.
EXISTING_EXP_DIR=${4:-""}  # Optional: existing experiment directory to resume

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
if [ -n "$EXISTING_EXP_DIR" ]; then
    # Resume existing experiment
    if [ ! -d "$EXISTING_EXP_DIR" ]; then
        echo "Error: Existing experiment directory $EXISTING_EXP_DIR does not exist"
        exit 1
    fi
    
    EXP_NAME=$(basename "$EXISTING_EXP_DIR")
    export EXP_DIR="$EXISTING_EXP_DIR"
    echo "==> Resuming existing experiment: $EXP_NAME"
    echo "==> Using existing directory: $EXP_DIR"
    
    # Determine dataset and config from experiment name or directory structure
    if [[ "$EXP_NAME" == *"CIRR"* ]] || [ -f "$EXP_DIR/cirr_iterative.yaml" ]; then
        CONFIG_FILE="configs/cirr_iterative.yaml"
        echo "==> Detected CIRR dataset"
    elif [[ "$EXP_NAME" == *"FashionIQ"* ]] || [ -f "$EXP_DIR/fashioniq_iterative.yaml" ]; then
        CONFIG_FILE="configs/fashioniq_iterative.yaml"
        echo "==> Detected FashionIQ dataset"
    else
        echo "==> Using provided dataset: $DATASET"
        if [ "$DATASET" == "cirr" ]; then
            CONFIG_FILE="configs/cirr_iterative.yaml"
        elif [ "$DATASET" == "fashioniq" ]; then
            CONFIG_FILE="configs/fashioniq_iterative.yaml"
        else
            echo "Error: Cannot determine dataset from experiment name and unknown dataset $DATASET"
            exit 1
        fi
    fi
else
    # Create new experiment
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
    
    # Output directory
    export EXP_DIR="./experiments/$EXP_NAME"
    echo "==> Creating new experiment: $EXP_NAME"
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
export WANDB_NAME=$EXP_NAME
export WANDB_DIR=$EXP_DIR

echo "==> Experiment: $EXP_NAME"
echo "==> Output directory: $EXP_DIR"
echo "==> Config file: $CONFIG_FILE"

# Create experiment directory if it doesn't exist
mkdir -p $EXP_DIR/wandb
if [ -z "$EXISTING_EXP_DIR" ]; then
    # Only clean wandb for new experiments
    rm -rf $EXP_DIR/wandb/*
fi

# Training command - Multi-GPU distributed training for testing
NUM_GPUS=${3:-2}  # Default to 2 GPUs, can be overridden as 3rd argument

if [ $NUM_GPUS -gt 1 ]; then
    echo "==> Using $NUM_GPUS GPUs for distributed training"
    cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
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
else
    echo "==> Using single GPU for training"
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
fi

echo "==> Running command:"
echo $cmd
echo ""

# Execute training
eval $cmd

echo ""
echo "==> Training completed!"
echo "==> Results saved to: $EXP_DIR"
echo "==> Log file: $EXP_DIR/train.log"