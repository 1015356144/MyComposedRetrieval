#!/bin/bash
# Iterative Training Script for Composed Image Retrieval - PRODUCTION VERSION
# Usage:
#   ./run_iterative_training.sh [cirr|fashioniq] [qwen2vl|qwen2vl_2b|llava_next] [num_gpus] [existing_exp_dir]
#
# Examples:
#   ./run_iterative_training.sh cirr qwen2vl 2
#   ./run_iterative_training.sh cirr qwen2vl 4
#   ./run_iterative_training.sh cirr qwen2vl 8
#   ./run_iterative_training.sh cirr qwen2vl_2b 1
#   ./run_iterative_training.sh cirr qwen2vl 2 ./experiments/IterativeCIRR_qwen2vl_20250805_000011

set -euo pipefail

#===============================#
#           Arguments           #
#===============================#
DATASET=${1:-"cirr"}           # cirr | fashioniq
MODEL_TYPE=${2:-"qwen2vl"}     # qwen2vl | qwen2vl_2b | llava_next
NUM_GPUS=${3:-2}               # 1 / 2 / 4 / 8
EXISTING_EXP_DIR=${4:-""}      # optional

#===============================#
#          Paths (HF)           #
#===============================#
QWEN2VL_2B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-2B-Instruct"
QWEN2VL_7B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct"
QWEN2_5VL_32B_PATH="/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2.5-VL-32B-Instruct"

echo "==> Starting iterative training for $DATASET with $MODEL_TYPE"
echo "==> Environment Setup"
echo "conda location: $(which conda || true)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

#===============================#
#       Environment Vars        #
#===============================#
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export HF_HOME="$HOME/.cache/huggingface"

# WandB (enabled by default)
export WANDB_DISABLED=false
export WANDB_PROJECT="iterative_composed_retrieval_production"
# export WANDB_API_KEY="your_wandb_api_key"   # (optional)
# export HUGGING_FACE_HUB_TOKEN="your_hf_token"  # (optional)

# NCCL / DDP stability
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=INFO

# Avoid tokenizer parallel spam
export TOKENIZERS_PARALLELISM=false

#===============================#
#    New / Resume Experiment    #
#===============================#
if [ -n "$EXISTING_EXP_DIR" ]; then
  if [ ! -d "$EXISTING_EXP_DIR" ]; then
    echo "Error: Existing experiment directory $EXISTING_EXP_DIR does not exist"
    exit 1
  fi
  EXP_DIR="$EXISTING_EXP_DIR"
  EXP_NAME=$(basename "$EXISTING_EXP_DIR")
  echo "==> Resuming existing experiment: $EXP_NAME"
  echo "==> Using existing directory: $EXP_DIR"

  if [[ "$EXP_NAME" == *"CIRR"* ]] || [ -f "$EXP_DIR/cirr_iterative.yaml" ]; then
    CONFIG_FILE="configs/cirr_iterative.yaml"
    echo "==> Detected CIRR dataset"
  elif [[ "$EXP_NAME" == *"FashionIQ"* ]] || [ -f "$EXP_DIR/fashioniq_iterative.yaml" ]; then
    CONFIG_FILE="configs/fashioniq_iterative.yaml"
    echo "==> Detected FashionIQ dataset"
  else
    echo "==> Using provided dataset arg: $DATASET"
    if [ "$DATASET" == "cirr" ]; then
      CONFIG_FILE="configs/cirr_iterative.yaml"
    elif [ "$DATASET" == "fashioniq" ]; then
      CONFIG_FILE="configs/fashioniq_iterative.yaml"
    else
      echo "Error: Unknown dataset $DATASET"
      exit 1
    fi
  fi
else
  if [ "$DATASET" == "cirr" ]; then
    CONFIG_FILE="configs/cirr_iterative.yaml"
    EXP_NAME="IterativeCIRR_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
  elif [ "$DATASET" == "fashioniq" ]; then
    CONFIG_FILE="configs/fashioniq_iterative.yaml"
    EXP_NAME="IterativeFashionIQ_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
  else
    echo "Error: Unknown dataset $DATASET"
    exit 1
  fi

  EXP_DIR="./experiments/$EXP_NAME"
  echo "==> Creating new experiment: $EXP_NAME"
fi

mkdir -p "$EXP_DIR/wandb" "$EXP_DIR/logs"
if [ -z "$EXISTING_EXP_DIR" ]; then
  rm -rf "$EXP_DIR/wandb"/*
fi

export WANDB_NAME="$EXP_NAME"
export WANDB_DIR="$EXP_DIR"

#===============================#
#        Model Selection        #
#===============================#
if [ "$MODEL_TYPE" == "qwen2vl" ]; then
  MODEL_NAME="$QWEN2VL_7B_PATH"
  FOUNDATION_MODEL="$QWEN2VL_7B_PATH"
elif [ "$MODEL_TYPE" == "qwen2vl_2b" ]; then
  MODEL_NAME="$QWEN2VL_2B_PATH"
  FOUNDATION_MODEL="$QWEN2VL_2B_PATH"
elif [ "$MODEL_TYPE" == "llava_next" ]; then
  MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
  FOUNDATION_MODEL="llava-hf/llava-v1.6-mistral-7b-hf"
else
  echo "Error: Unknown model type $MODEL_TYPE"
  exit 1
fi

echo "==> Experiment: $EXP_NAME"
echo "==> Output directory: $EXP_DIR"
echo "==> Config file: $CONFIG_FILE"

#===============================#
#     Per-GPU Batch/Workers     #
#===============================#
if [ "$NUM_GPUS" -gt 1 ]; then
  echo "==> Using $NUM_GPUS GPUs for distributed training"

  if [ "$NUM_GPUS" -eq 2 ]; then
    per_device_batch_size=64
    gradient_accumulation=4
    dataloader_workers=4
    cuda_devices="0,1"
  elif [ "$NUM_GPUS" -eq 4 ]; then
    per_device_batch_size=32
    gradient_accumulation=4
    dataloader_workers=6
    cuda_devices="0,1,2,3"
  elif [ "$NUM_GPUS" -eq 8 ]; then
    per_device_batch_size=48
    gradient_accumulation=1
    dataloader_workers=8
    cuda_devices="0,1,2,3,4,5,6,7"
  else
    echo "Unsupported GPU count: $NUM_GPUS. Falling back to 2-GPU defaults."
    per_device_batch_size=64
    gradient_accumulation=4
    dataloader_workers=4
    cuda_devices="0,1"
  fi

  #===========================#
  #   DDP (torchrun) Command  #
  #===========================#
  cmd="CUDA_VISIBLE_DEVICES=$cuda_devices torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_iterative.py \
    --model_name $MODEL_NAME \
    --foundation_model_name $FOUNDATION_MODEL \
    --lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataloader_num_workers $dataloader_workers \
    --dataset_config $CONFIG_FILE \
    --run_name $EXP_NAME \
    --project_name $WANDB_PROJECT \
    --output_dir $EXP_DIR \
    --per_device_train_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --warmup_steps 0.1 \
    --save_steps 3000 \
    --logging_steps 50 \
    --logging_dir $EXP_DIR/logs \
    --eval_steps 1000 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from none \
    --resume_from_iteration auto \
    --max_len 512 \
    --resize_use_processor True \
    --resize_min_pixels 3136 \
    --resize_max_pixels 35840 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --fp16 False \
    --bf16 True \
    --gradient_checkpointing False \
    --optim adamw_torch \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --report_to none \
    --group_by_reference_image \
    2>&1 | tee $EXP_DIR/training_output.log"

else
  echo "==> Using single GPU for training"

  cmd="CUDA_VISIBLE_DEVICES=7 python \
    train_iterative.py \
    --model_name $MODEL_NAME \
    --foundation_model_name $FOUNDATION_MODEL \
    --lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataloader_num_workers 4 \
    --dataset_config $CONFIG_FILE \
    --run_name $EXP_NAME \
    --project_name $WANDB_PROJECT \
    --output_dir $EXP_DIR \
    --per_device_train_batch_size 48 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --warmup_steps 250 \
    --save_steps 500 \
    --logging_steps 50 \
    --logging_dir $EXP_DIR/logs \
    --eval_steps 1000 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from none \
    --resume_from_iteration auto \
    --max_len 512 \
    --resize_use_processor True \
    --resize_min_pixels 3136 \
    --resize_max_pixels 35840 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --fp16 False \
    --bf16 True \
    --gradient_checkpointing False \
    --optim adamw_torch \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --report_to none \
    --group_by_reference_image \
    2>&1 | tee $EXP_DIR/training_output.log"
fi

echo "==> Running command:"
echo "$cmd"
echo ""

# Execute training
eval "$cmd"

echo ""
echo "==> Training completed!"
echo "==> Results saved to: $EXP_DIR"
echo "==> Log file: $EXP_DIR/train.log"
