#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Asymmetric scaling for VQA yes/no logits to improve retrieval ranking.
Score = a * yes_logit - b * no_logit

Usage:
  python tune_asym_logits.py --input /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/R1_49/reranked_results_single.json \
      --Ks 1 5 10 \
      --a_values 0.5 0.75 1.0 1.25 1.5 2.0 \
      --b_values 0.5 0.75 1.0 1.25 1.5 2.0 \
      --use_original_rank_tiebreak
"""

"""
python tune_asym_logits.py \
--input /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/R1_49/reranked_results_single.json \
  --a_values $(python - <<'PY'
import numpy as np; print(*[f"{x:.2f}" for x in np.arange(0.30,1.61,0.05)])
PY
) \
  --b_values $(python - <<'PY'
import numpy as np; print(*[f"{x:.2f}" for x in np.arange(0.30,1.81,0.05)])
PY
) \
  --use_original_rank_tiebreak \
  --verbose
"""
import json
import argparse
from collections import defaultdict
import math


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to results JSON file")
    ap.add_argument("--Ks", type=int, nargs="+", default=[1], help="Recall@K list")
    ap.add_argument("--a_values", type=float, nargs="+",
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                    help="Grid for coefficient a (yes weight)")
    ap.add_argument("--b_values", type=float, nargs="+",
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                    help="Grid for coefficient b (no weight)")
    ap.add_argument("--use_original_rank_tiebreak", action="store_true",
                    help="Use original_rank as a secondary key to break ties")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def safe_get_positive_set(qobj):
    """Build the set of positives for this query."""
    pos = set()
    # 1) ground_truth.target_hard
    gt = qobj.get("ground_truth", {})
    th = gt.get("target_hard")
    if isinstance(th, str) and len(th) > 0:
        pos.add(th)

    # 2) target_soft (keys are also positives)
    ts = qobj.get("target_soft", {})
    if isinstance(ts, dict):
        for k in ts.keys():
            if isinstance(k, str) and len(k) > 0:
                pos.add(k)
    return pos


def recall_at_k_for_query(sorted_candidates, positives, K):
    """
    sorted_candidates: list of candidate_image (ordered by score desc)
    positives: set of positive candidate_image
    """
    topk = sorted_candidates[:K]
    return 1.0 if any(c in positives for c in topk) else 0.0


def recall_at_k_over_all(queries_sorted_lists, Ks=(1, 5, 10)):
    """
    queries_sorted_lists: list of (sorted_candidate_images, positives_set)
    return: dict {K: R@K}
    """
    res = {}
    n = len(queries_sorted_lists)
    for K in Ks:
        hit = 0.0
        for cand_list, pos_set in queries_sorted_lists:
            hit += recall_at_k_for_query(cand_list, pos_set, K)
        res[K] = hit / max(1, n)
    return res


def build_sorted_candidates_for_query(qobj, a, b, use_tiebreak=False):
    """
    From a single query object, build a sorted candidate list by score s = a*yes - b*no.
    Handle duplicates by keeping the max score for each candidate_image.

    Returns:
      sorted_candidate_images (list[str])
    """
    rer = qobj.get("reranked_results", [])
    # 如果没有 reranked_results, 尝试从 retrieval_results/某些字段兜底（可按需扩展）
    if not rer:
        return []

    best_score_by_cand = {}
    tiebreak_by_cand = {}

    for item in rer:
        ci = item.get("candidate_image")
        if not ci:
            continue
        y = item.get("yes_logit", None)
        n = item.get("no_logit", None)
        if y is None or n is None:
            # 若缺 logit，可以跳过或用原分数兜底，这里选择跳过
            # y = item.get("rerank_score", None)
            # n = 0.0
            # if y is None: continue
            continue

        s = a * float(y) - b * float(n)

        # 保留每个 candidate 的最大分
        if (ci not in best_score_by_cand) or (s > best_score_by_cand[ci]):
            best_score_by_cand[ci] = s
            # tiebreak: 优先使用 original_rank 更小者（原排序靠前）
            if use_tiebreak:
                orig_rank = item.get("original_rank", 10**9)
                tiebreak_by_cand[ci] = int(orig_rank)

    # 排序：主键是分数降序；若启用 tiebreak，则次键为 original_rank 升序
    if use_tiebreak:
        sorted_items = sorted(best_score_by_cand.items(),
                              key=lambda kv: (-kv[1], tiebreak_by_cand.get(kv[0], 10**9)))
    else:
        sorted_items = sorted(best_score_by_cand.items(), key=lambda kv: -kv[1])

    return [ci for ci, _ in sorted_items]


def evaluate_for_ab(data, a, b, Ks=(1, 5, 10), use_tiebreak=False):
    """
    data: loaded JSON dict.
    Returns: dict with recalls and summary score sum(R@K).
    """
    queries = data.get("queries", [])
    q_lists = []
    for q in queries:
        pos_set = safe_get_positive_set(q)
        cand_sorted = build_sorted_candidates_for_query(q, a, b, use_tiebreak=use_tiebreak)
        if not cand_sorted:
            # 若无候选（异常），给一个空列表（一定 miss）
            cand_sorted = []
        q_lists.append((cand_sorted, pos_set))

    recalls = recall_at_k_over_all(q_lists, Ks=Ks)
    score = sum(recalls[K] for K in Ks)
    return recalls, score


def main():
    args = parse_args()
    with open(args.input, "r") as f:
        data = json.load(f)

    Ks = tuple(args.Ks)
    best = {
        "a": None, "b": None, "score": -1.0, "recalls": None
    }

    print(f"Grid search over a x b: {len(args.a_values)} x {len(args.b_values)}")
    print(f"Ks = {Ks}, tie-break by original_rank = {args.use_original_rank_tiebreak}\n")

    for a in args.a_values:
        for b in args.b_values:
            recalls, score = evaluate_for_ab(
                data, a, b, Ks=Ks, use_tiebreak=args.use_original_rank_tiebreak
            )
            if args.verbose:
                rline = " ".join([f"R@{K}={recalls[K]:.4f}" for K in Ks])
                print(f"a={a:.3f}, b={b:.3f} -> {rline}, SUM={score:.4f}")

            if score > best["score"]:
                best.update(a=a, b=b, score=score, recalls=recalls)

    # 输出最优结果
    print("\n=== Best setting (maximize sum of R@K) ===")
    print(f"a = {best['a']}, b = {best['b']}")
    for K in Ks:
        print(f"R@{K}: {best['recalls'][K]:.6f}")
    print(f"Sum R@K: {best['score']:.6f}")

    # 对比原始（如果 metadata 里有）
    meta = data.get("metadata", {})
    r1o = meta.get("rerank_r1") or meta.get("original_r1")
    r5o = meta.get("rerank_r5") or meta.get("original_r5")
    r10o = meta.get("rerank_r10") or meta.get("original_r10")
    if (r1o is not None) and (r5o is not None) and (r10o is not None) and set(Ks) == {1,5,10}:
        print("\n(Reference from metadata)")
        print(f"Prev R@1={float(r1o):.6f}, R@5={float(r5o):.6f}, R@10={float(r10o):.6f}")


if __name__ == "__main__":
    main()
