#!/bin/bash

# 创建日志目录
LOG_DIR="/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/logs"
mkdir -p $LOG_DIR

# 设置日志文件名（包含时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/qwen25_vl_32b_rerank_$TIMESTAMP.log"

# 输出信息
echo "启动Qwen2.5-VL-32B重排序任务..."
echo "日志将保存到: $LOG_FILE"

# 后台运行命令并将输出重定向到日志文件
nohup accelerate launch --num_processes 8 /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/VQA_rerank_single_qwen25.py \
    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2.5-VL-32B-Instruct \
    --json_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/retrieval_results/checkpoint-1500_20250919_160740/cirr_retrieval_top10.json \
    --image_dir /home/guohaiyun/yty_data/CIRR/dev \
    --output_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/qwen25_32b/reranked_results_qwen25_32b.json \
    --batch_size 1 \
    --max_image_size 2048 > $LOG_FILE 2>&1 &

# 获取后台进程PID
PID=$!
echo "进程已启动，PID: $PID"
echo "可以使用以下命令检查运行状态:"
echo "  tail -f $LOG_FILE"
echo "  ps aux | grep $PID"
