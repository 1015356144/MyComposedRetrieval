#!/usr/bin/env python3
import json

def analyze_missing_candidates():
    # 读取原始数据和重排序结果
    with open('/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/decomposed_results_1.json', 'r') as f:
        original_data = json.load(f)
    
    with open('/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/multi_reranked_results.json', 'r') as f:
        reranked_data = json.load(f)
    
    total_queries = len(original_data['queries'])
    total_missing_candidates = 0
    queries_with_missing_candidates = 0
    
    print(f"分析 {total_queries} 个查询的候选丢失情况...")
    print("="*60)
    
    # 统计各种情况
    candidate_loss_stats = {}
    
    for i, (orig_q, rerank_q) in enumerate(zip(original_data['queries'], reranked_data['queries'])):
        orig_candidates = len(orig_q.get('retrieval_results', []))
        rerank_candidates = len(rerank_q.get('reranked_results', []))
        
        if orig_candidates != rerank_candidates:
            queries_with_missing_candidates += 1
            missing = orig_candidates - rerank_candidates
            total_missing_candidates += missing
            
            # 统计丢失候选数的分布
            candidate_loss_stats[missing] = candidate_loss_stats.get(missing, 0) + 1
            
            # 显示前10个有问题的query的详细信息
            if queries_with_missing_candidates <= 10:
                print(f"Query {orig_q['query_id']}: {orig_candidates} -> {rerank_candidates} (丢失{missing}个)")
                
                # 显示丢失的具体候选
                orig_cand_set = set([r['candidate_image'] for r in orig_q['retrieval_results']])
                rerank_cand_set = set([r['candidate_image'] for r in rerank_q['reranked_results']])
                missing_cands = orig_cand_set - rerank_cand_set
                print(f"  丢失的候选: {list(missing_cands)[:3]}{'...' if len(missing_cands) > 3 else ''}")
                print()
    
    print("="*60)
    print(f"统计结果:")
    print(f"- 总查询数: {total_queries}")
    print(f"- 有候选丢失的查询数: {queries_with_missing_candidates} ({queries_with_missing_candidates/total_queries*100:.1f}%)")
    print(f"- 总丢失候选数: {total_missing_candidates}")
    print(f"- 平均每个查询丢失: {total_missing_candidates/total_queries:.2f} 个候选")
    
    print(f"\n候选丢失数分布:")
    for missing_count in sorted(candidate_loss_stats.keys()):
        count = candidate_loss_stats[missing_count]
        print(f"- 丢失{missing_count}个候选: {count}个查询 ({count/total_queries*100:.1f}%)")
    
    # 计算理论上的R@10（假设没有候选丢失）
    original_found_in_top10 = 0
    for q in original_data['queries']:
        target = q['target_hard']
        candidates = [r['candidate_image'] for r in q['retrieval_results'][:10]]
        if target in candidates:
            original_found_in_top10 += 1
    
    original_r10 = original_found_in_top10 / total_queries
    print(f"\n理论指标（基于原始数据）:")
    print(f"- 原始R@10: {original_r10:.4f} ({original_found_in_top10}/{total_queries})")
    print(f"- 元数据中的R@10: {original_data.get('metadata', {}).get('accuracy_at_k', 'N/A')}")

if __name__ == "__main__":
    analyze_missing_candidates()
