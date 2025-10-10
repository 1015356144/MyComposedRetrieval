#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Recall@K (R@1, R@5, R@10) for different sub-modification score
aggregation strategies on decomposed reranked CIR results.

Usage:
    python test_multi.py \
        --json /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/decomposed_reranked_results_geometric.json \
        --methods mean,geometric,harmonic,min,max,median,pmean0.5,pmean2,lse
    python test_multi.py \
        --json /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/R1_49/2048_2048*28*28_decomposed_reranked_results.json \   

If no args are given, sensible defaults are used for the provided path and methods.

Notes:
- We only use scores in each candidate's `sub_modification_scores`.
- We re-rank per query by aggregated score (desc), tie-broken by original_rank (asc) if available.
- We then check the ground-truth `target_hard` image's rank and accumulate hits@K.
"""

import argparse
import json
import math
from statistics import median
from collections import defaultdict
from typing import List, Dict, Callable, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="Compute R@K for different aggregation methods.")
    parser.add_argument(
        "--json",
        type=str,
        default="/home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/results/qwen25_32b/R1_49/decomposed_reranked_results.json",
        help="Path to the decomposed reranked JSON."
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="mean,geometric,harmonic,min,max,median,pmean0.5,pmean2,lse",
        help="Comma-separated aggregation methods. "
             "Supported: mean, geometric, harmonic, min, max, median, pmean{p}, lse"
    )
    parser.add_argument(
        "--lse_tau",
        type=float,
        default=1.0,
        help="Temperature for Log-Sum-Exp aggregation (higher tau -> closer to max; tau->0 -> closer to mean)."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small epsilon to stabilize geometric/harmonic computations when scores may be zero."
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="1,5,10",
        help="Comma-separated K list for Recall@K."
    )
    parser.add_argument(
        "--verbose_missing",
        action="store_true",
        help="Print queries skipped due to missing fields."
    )
    return parser.parse_args()

# ---------------- Aggregation functions ---------------- #

def agg_mean(xs: List[float], **_kwargs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def agg_geometric(xs: List[float], eps: float = 1e-12, **_kwargs) -> float:
    # geometric mean on [0,1]; add eps to avoid log(0) / product 0
    prod = 1.0
    n = len(xs)
    if n == 0:
        return float("nan")
    for v in xs:
        prod *= max(v, eps)
    return prod ** (1.0 / n)

def agg_harmonic(xs: List[float], eps: float = 1e-12, **_kwargs) -> float:
    # harmonic mean on positives; clamp to eps to avoid div-by-zero
    n = len(xs)
    if n == 0:
        return float("nan")
    denom = 0.0
    for v in xs:
        denom += 1.0 / max(v, eps)
    return n / denom

def agg_min(xs: List[float], **_kwargs) -> float:
    return min(xs) if xs else float("nan")

def agg_max(xs: List[float], **_kwargs) -> float:
    return max(xs) if xs else float("nan")

def agg_median(xs: List[float], **_kwargs) -> float:
    return median(xs) if xs else float("nan")

def agg_pmean(xs: List[float], p: float, eps: float = 1e-12, **_kwargs) -> float:
    """
    Power mean (generalized mean):
        M_p(x) = ( (1/n) * sum( x_i^p ) )^(1/p)
    Special cases:
        p -> 1: arithmetic mean
        p -> 0: geometric mean (limit)
        p -> -1: harmonic mean
        p -> +inf: max
        p -> -inf: min
    """
    n = len(xs)
    if n == 0:
        return float("nan")
    # If p is very close to 0, fall back to geometric
    if abs(p) < 1e-9:
        return agg_geometric(xs, eps=eps)
    s = 0.0
    for v in xs:
        s += (max(v, eps)) ** p
    return (s / n) ** (1.0 / p)

def agg_lse(xs: List[float], tau: float = 1.0, eps: float = 1e-12, **_kwargs) -> float:
    """
    Log-Sum-Exp (with temperature tau) averaged back to [0,1]-ish scale:
        LSE_tau(x) = tau * log( (1/n) * sum( exp(x_i / tau) ) )
    Then pass through a simple squashing to [0,1] range:
        We use a logistic squashing on LSE centered at 0.5 to keep ordering stable.
    Note: This is mainly a smooth max; you can use raw LSE for ordering as well.
    """
    n = len(xs)
    if n == 0:
        return float("nan")
    # To maintain just ordering, raw LSE is sufficient; we keep it simple:
    # normalized by n to reduce dependence on number of subs.
    m = max(xs) if xs else 0.0
    # numerically stable LSE
    acc = 0.0
    for v in xs:
        acc += math.exp((v - m) / max(tau, eps))
    lse = m + max(tau, eps) * math.log(acc / n)
    # No strict need to squash; rankings rely on monotonicity. Return lse directly.
    return lse

def build_aggregators(method_names: List[str], tau: float, eps: float):
    """
    Parse method strings into callables.
    Supported tokens:
      - mean, geometric, harmonic, min, max, median, lse
      - pmean{p}   e.g., pmean0.5  pmean2  pmean-1
    """
    aggs: Dict[str, Callable[[List[float]], float]] = {}
    for name in method_names:
        name = name.strip()
        if not name:
            continue
        if name == "mean":
            aggs[name] = lambda xs, e=eps: agg_mean(xs)
        elif name == "geometric":
            aggs[name] = lambda xs, e=eps: agg_geometric(xs, eps=e)
        elif name == "harmonic":
            aggs[name] = lambda xs, e=eps: agg_harmonic(xs, eps=e)
        elif name == "min":
            aggs[name] = lambda xs, e=eps: agg_min(xs)
        elif name == "max":
            aggs[name] = lambda xs, e=eps: agg_max(xs)
        elif name == "median":
            aggs[name] = lambda xs, e=eps: agg_median(xs)
        elif name.startswith("pmean"):
            # pmean{p}
            p_str = name.replace("pmean", "", 1)
            if not p_str:
                raise ValueError("pmean requires a p value, e.g. pmean0.5 or pmean2")
            p = float(p_str)
            aggs[name] = lambda xs, e=eps, pp=p: agg_pmean(xs, p=pp, eps=e)
        elif name == "lse":
            aggs[name] = lambda xs, e=eps, t=tau: agg_lse(xs, tau=t, eps=e)
        else:
            raise ValueError(f"Unsupported aggregation method: {name}")
    return aggs

# ---------------- Metrics computation ---------------- #

def compute_recalls_for_method(
    data: dict,
    aggregator_name: str,
    aggregator_fn: Callable[[List[float]], float],
    k_list: List[int],
    verbose_missing: bool = False
) -> Tuple[Dict[int, int], int]:
    """
    Returns: (hits_at_k: dict(K->count), total_valid_queries)
    """
    hits_at_k = {k: 0 for k in k_list}
    total = 0

    for q in data.get("queries", []):
        target = q.get("ground_truth", {}).get("target_hard")
        if not target:
            if verbose_missing:
                print(f"[Skip] query_id={q.get('query_id')} missing ground_truth.target_hard")
            continue

        cand_list = q.get("reranked_results", [])
        if not cand_list:
            if verbose_missing:
                print(f"[Skip] query_id={q.get('query_id')} missing reranked_results")
            continue

        # Build (aggregated_score, tie_original_rank, candidate_image)
        scored = []
        for c in cand_list:
            subs = c.get("sub_modification_scores", {})
            if not subs:
                # If no sub scores, skip this candidate for this method
                continue
            # Extract scores from new structure: {"text": {"score": float, "yes_logit": float, "no_logit": float}}
            xs = []
            for sub_data in subs.values():
                if isinstance(sub_data, dict) and "score" in sub_data:
                    xs.append(sub_data["score"])
                elif isinstance(sub_data, (int, float)):
                    # Backward compatibility: if it's still the old format (direct score)
                    xs.append(float(sub_data))
            agg_score = aggregator_fn(xs)
            # Lower original_rank is better; use a large fallback for tie-breaking
            tie_rank = c.get("original_rank", 10**9)
            scored.append((agg_score, tie_rank, c.get("candidate_image")))

        if not scored:
            if verbose_missing:
                print(f"[Skip] query_id={q.get('query_id')} all candidates missing sub_modification_scores")
            continue

        # Sort by aggregated score DESC, tie by original_rank ASC
        scored.sort(key=lambda t: (-t[0], t[1]))

        # Find rank (1-based) of target
        total += 1
        found_rank = None
        for idx, (_, __, img) in enumerate(scored, start=1):
            if img == target:
                found_rank = idx
                break

        if found_rank is not None:
            for k in k_list:
                if found_rank <= k:
                    hits_at_k[k] += 1

    return hits_at_k, total

def format_ratio(n: int, d: int) -> str:
    if d == 0:
        return "N/A"
    return f"{n}/{d} = {n/d:.6f}"

def main():
    args = parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    aggregators = build_aggregators(methods, tau=args.lse_tau, eps=args.eps)

    print("\n=== Aggregation-based Recall Evaluation ===")
    print(f"File: {args.json}")
    print(f"Methods: {', '.join(methods)}")
    print(f"K list: {k_list}")
    print(f"LSE tau: {args.lse_tau}, eps: {args.eps}\n")

    # Also print original (if present in metadata) for quick comparison
    meta = data.get("metadata", {})
    if meta:
        print("Original (from metadata, if present):")
        for k in k_list:
            key = f"original_r{k}"
            if key in meta:
                print(f"  original_r{k}: {meta[key]:.6f}")
        for k in k_list:
            key = f"rerank_r{k}"
            if key in meta:
                print(f"  rerank_r{k}:   {meta[key]:.6f}")
        print()

    for name, fn in aggregators.items():
        hits_at_k, total = compute_recalls_for_method(
            data=data,
            aggregator_name=name,
            aggregator_fn=fn,
            k_list=k_list,
            verbose_missing=args.verbose_missing
        )
        print(f"[{name}] on {total} valid queries:")
        for k in k_list:
            print(f"  R@{k}: {format_ratio(hits_at_k[k], total)}")
        print()

if __name__ == "__main__":
    main()
