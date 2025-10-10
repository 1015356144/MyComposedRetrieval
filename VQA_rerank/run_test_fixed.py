#!/usr/bin/env python3
"""
测试修复后的去重功能
"""

import json
import os
from collections import defaultdict

def check_duplicates_in_results(json_file):
    """检查结果文件中是否还有重复项"""
    
    if not os.path.exists(json_file):
        print(f"文件 {json_file} 不存在")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_duplicates = 0
    total_queries = 0
    
    for query in data['queries']:
        if 'reranked_results' not in query:
            continue
            
        total_queries += 1
        candidates = query['reranked_results']
        
        # 检查当前query中的重复候选
        seen_candidates = set()
        duplicates_in_query = 0
        
        for candidate in candidates:
            candidate_image = candidate['candidate_image']
            if candidate_image in seen_candidates:
                duplicates_in_query += 1
                total_duplicates += 1
                print(f"Query {query['query_id']}: 发现重复候选 {candidate_image}")
            else:
                seen_candidates.add(candidate_image)
        
        if duplicates_in_query > 0:
            print(f"Query {query['query_id']}: 总共有 {duplicates_in_query} 个重复候选")
    
    print(f"\n总结:")
    print(f"处理的query数量: {total_queries}")
    print(f"发现的重复项总数: {total_duplicates}")
    
    if total_duplicates == 0:
        print("✓ 没有发现重复项！修复成功。")
    else:
        print(f"✗ 仍然存在 {total_duplicates} 个重复项。")

def compare_results(old_file, new_file):
    """比较修复前后的结果"""
    print("比较修复前后的结果...")
    
    if os.path.exists(old_file):
        print(f"\n检查原始文件: {old_file}")
        check_duplicates_in_results(old_file)
    
    if os.path.exists(new_file):
        print(f"\n检查修复后文件: {new_file}")
        check_duplicates_in_results(new_file)

if __name__ == "__main__":
    # 检查现有的结果文件
    results_dir = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results"
    
    old_result = os.path.join(results_dir, "decomposed_reranked_results_small_test.json")
    
    print("检查现有结果文件中的重复项...")
    check_duplicates_in_results(old_result)
    
    print("\n" + "="*60)
    print("建议运行以下命令来测试修复后的版本:")
    print("\n# 单进程版本（推荐，避免多进程重复问题）:")
    print("python VQA_rerank_decomposed.py \\")
    print("    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \\")
    print("    --image_dir /home/guohaiyun/yty_data/CIRR/dev \\")
    print(f"    --output_file {results_dir}/decomposed_reranked_results_fixed.json \\")
    print("    --batch_size 4 \\")
    print("    --max_image_size 768 \\")
    print("    --max_queries 10")
    print("\n# 多进程版本（现在有去重逻辑）:")
    print("accelerate launch --num_processes 4 VQA_rerank_decomposed.py \\")
    print("    --model_dir /home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct \\")
    print("    --image_dir /home/guohaiyun/yty_data/CIRR/dev \\")
    print(f"    --output_file {results_dir}/decomposed_reranked_results_multiproc_fixed.json \\")
    print("    --batch_size 2 \\")
    print("    --max_image_size 768 \\")
    print("    --max_queries 10")
