#!/usr/bin/env python3
"""
CIRR Evaluation Script
使用优化后的CIRREvaluator进行CIRR数据集评估
"""

import os
import sys
import torch
import argparse
import yaml
from transformers import AutoProcessor
from src.model.model import MMEBModel
from src.evaluation.cirr_evaluator import CIRREvaluator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.utils import print_master


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CIRR Evaluation")
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--model_backbone', type=str, default='qwen2_vl',
                       help='Model backbone name')
    
    # 数据相关参数
    parser.add_argument('--eval_config', type=str, 
                       default='configs/cirr_eval_config.yaml',
                       help='Path to evaluation config file')
    parser.add_argument('--max_len', type=int, default=512,
                       help='Maximum sequence length')
    
    # 评估设置
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    
    return parser.parse_args()


def load_model_and_processor(args):
    """加载模型和处理器"""
    print_master(f"Loading model from {args.model_path}")
    
    # 创建模型参数
    model_args = ModelArguments()
    model_args.model_name = args.model_path
    model_args.model_backbone = args.model_backbone
    
    # 加载模型
    model = MMEBModel.build(model_args)
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    model = model.to(device)
    model.eval()
    
    print_master(f"Model loaded on device: {device}")
    return model, processor, device


def load_evaluation_config(config_path):
    """加载评估配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            eval_config = yaml.safe_load(f)
        print_master(f"Loaded evaluation config from {config_path}")
        return eval_config
    else:
        print_master(f"Config file not found: {config_path}, using defaults")
        return None


def create_data_args(args, eval_config):
    """创建数据参数"""
    data_args = DataArguments()
    data_args.max_len = args.max_len
    
    # 如果有评估配置，使用配置文件路径
    if eval_config:
        data_args.dataset_config = args.eval_config
    else:
        data_args.dataset_config = None
    
    return data_args


def main():
    args = parse_args()
    
    print_master("=" * 60)
    print_master("CIRR Evaluation Script")
    print_master("=" * 60)
    
    # 1. 加载评估配置
    eval_config = load_evaluation_config(args.eval_config)
    
    # 2. 加载模型和处理器
    model, processor, device = load_model_and_processor(args)
    
    # 3. 创建参数对象
    data_args = create_data_args(args, eval_config)
    model_args = ModelArguments()
    model_args.model_backbone = args.model_backbone
    
    # 4. 创建评估器
    print_master("Creating CIRR evaluator...")
    evaluator = CIRREvaluator(
        model=model,
        processor=processor,
        data_args=data_args,
        model_args=model_args,
        device=device,
        batch_size=args.batch_size
    )
    
    # 5. 运行评估
    print_master("Starting evaluation...")
    results = evaluator.evaluate()
    
    # 6. 显示结果
    print_master("=" * 60)
    print_master("Final Evaluation Results:")
    print_master("=" * 60)
    for metric, value in results.items():
        if isinstance(value, float):
            print_master(f"{metric}: {value:.4f} ({value*100:.2f}%)")
        else:
            print_master(f"{metric}: {value}")
    
    print_master("=" * 60)
    print_master("Evaluation completed!")


if __name__ == "__main__":
    main()
