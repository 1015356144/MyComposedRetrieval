#!/usr/bin/env python3
"""
简单的测试脚本，验证分解版本的重排序代码基本功能
"""

import json
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from VQA_rerank_decomposed import DecomposedRerankDataset

def test_dataset_loading():
    """测试数据集加载功能"""
    print("测试数据集加载功能...")
    
    json_file = '/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/decomposed_results_1.json'
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"错误：文件 {json_file} 不存在")
        return False
    
    try:
        # 加载JSON数据
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        print(f"原始数据包含 {len(json_data['queries'])} 个queries")
        
        # 测试加载前5个query
        dataset = DecomposedRerankDataset(json_data, '/dummy/path', max_queries=5)
        print(f"数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 检查第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"第一个样本的信息:")
            print(f"  Query ID: {sample['query_id']}")
            print(f"  子修改文本数量: {len(sample['sub_modifications'])}")
            print(f"  子修改文本: {sample['sub_modifications']}")
            print(f"  候选图像: {sample['candidate_image']}")
            print(f"  原始得分: {sample['original_score']}")
        
        return True
        
    except Exception as e:
        print(f"错误：{e}")
        return False


def test_sub_modifications_structure():
    """测试子修改文本的结构"""
    print("\n测试子修改文本的结构...")
    
    json_file = '/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/decomposed_results_1.json'
    
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        # 检查前几个query的子修改文本
        for i, query in enumerate(json_data['queries'][:3]):
            print(f"\nQuery {i}:")
            print(f"  原始修改文本: {query['modification_text']}")
            if 'sub_modifications' in query:
                print(f"  子修改文本: {query['sub_modifications']}")
                print(f"  子修改数量: {len(query['sub_modifications'])}")
            else:
                print("  警告：没有找到sub_modifications字段")
        
        return True
        
    except Exception as e:
        print(f"错误：{e}")
        return False


if __name__ == "__main__":
    print("开始测试分解版本的重排序代码...")
    
    success = True
    success &= test_dataset_loading()
    success &= test_sub_modifications_structure()
    
    if success:
        print("\n✓ 所有测试通过！代码基本功能正常。")
        print("\n下一步可以运行完整的重排序命令：")
        print("accelerate launch --num_processes 8 VQA_rerank_decomposed.py \\")
        print("    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \\")
        print("    --image_dir /home/guohaiyun/yty_data/CIRR/dev \\")
        print("    --output_file decomposed_reranked_results_test.json \\")
        print("    --batch_size 1 \\")
        print("    --max_image_size 512 \\")
        print("    --max_queries 10")
    else:
        print("\n✗ 测试失败，请检查代码和数据。")
        sys.exit(1)
