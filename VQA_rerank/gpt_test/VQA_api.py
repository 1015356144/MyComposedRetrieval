# file: vqa_rerank_gpt_subset.py (fixed)
import os
import io
import json
import base64
import argparse
from typing import List, Dict, Any, Set
from PIL import Image
from tqdm import tqdm
import numpy as np
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')


def b64_data_url(img: Image.Image, fmt="PNG", max_size=1024) -> str:
    w, h = img.size
    if max(w, h) > max_size:
        if w >= h:
            nw, nh = max_size, int(h * max_size / w)
        else:
            nh, nw = max_size, int(w * max_size / h)
        img = img.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def load_image(path: str, max_size=1024) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Load image failed: {path} ({e})，使用白图占位")
        return Image.new("RGB", (224, 224), color="white")


def create_messages_for_gpt(reference_data_url: str, candidate_data_url: str, modification_text: str) -> List[Dict[str, Any]]:
    """
    Chat Completions 多模态消息格式：
    - 文本片段使用 {"type":"text","text":...}
    - 图片片段使用 {"type":"image_url","image_url":{"url":...}}
    """
    instruction = (
        "You are a strict visual verifier. Output exactly one token: yes or no (lowercase). "
        "Do not add punctuation or explanations.\n"
        "Reference image: Picture1\n"
        "Candidate image: Picture2\n"
        f"Instruction:{modification_text}\n"
        "Decide if the candidate image matches the result of applying the instruction to the reference image.\n"
        "Return yes if all required elements implied by the instruction are satisfied (like counts, categories, attributes, spatial relations). "
        "If any required element is missing or contradicted, answer no.\n"
        "Answer:"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": reference_data_url}},
                {"type": "image_url", "image_url": {"url": candidate_data_url}},
                {"type": "text", "text": instruction},
            ],
        }
    ]

def call_gpt_yesno(
    client: OpenAI,
    model: str,
    ref_img: Image.Image,
    cand_img: Image.Image,
    modification_text: str,
    max_image_size: int = 1024,
) -> Dict[str, Any]:
    """
    使用 Chat Completions（支持多模态）并开启 logprobs。
    兼容解析：若 logprobs 不返回，则回退到文本判断。
    """
    ref_url = b64_data_url(ref_img, fmt="PNG", max_size=max_image_size)
    cand_url = b64_data_url(cand_img, fmt="PNG", max_size=max_image_size)
    messages = create_messages_for_gpt(ref_url, cand_url, modification_text)

    # --- Chat Completions ---
    # 注意：这里是 chat.completions，不是 responses.create
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1,        # 只生成1个token
        logprobs=True,       # 开启logprobs
        top_logprobs=5,      # 取top候选
    )

    # 文本输出
    out_txt = ""
    try:
        out_txt = resp.choices[0].message.content.strip().lower()
    except Exception:
        out_txt = ""

    # 解析 token 级 logprobs
    yes_lp, no_lp = None, None
    try:
        # choices[0].logprobs.content 是一个按 token 的列表（我们只生成1个token）
        token_items = resp.choices[0].logprobs.content  # list
        if token_items:
            # 第一个生成token
            first = token_items[0]
            # top_logprobs 是该 token 的备选列表
            tops = getattr(first, "top_logprobs", []) or []
            def pick_best(targets: List[str]):
                best = None
                for t in tops:
                    tok = t.token.lower()
                    if tok.strip() in targets:
                        v = float(t.logprob)
                        best = v if (best is None or v > best) else best
                return best
            yes_lp = pick_best(["yes"])
            no_lp  = pick_best(["no"])
    except Exception:
        pass

    # 归一化为 yes 概率
    yes_prob = None
    if yes_lp is not None and no_lp is not None:
        m = max(yes_lp, no_lp)
        yes_prob = np.exp(yes_lp - m) / (np.exp(yes_lp - m) + np.exp(no_lp - m))

    # 回退：只看字符串
    if yes_prob is None:
        if out_txt.startswith("yes"):
            yes_prob = 1.0
        elif out_txt.startswith("no"):
            yes_prob = 0.0

    return {
        "answer": out_txt,
        "yes_logprob": None if yes_lp is None else float(yes_lp),
        "no_logprob": None if no_lp is None else float(no_lp),
        "yes_prob": None if yes_prob is None else float(yes_prob),
    }

def calculate_original_metrics_subset(json_data: Dict[str, Any], allow_ids: Set[str]) -> Dict[str, Any]:
    found_at_1 = found_at_5 = found_at_10 = 0
    total = 0
    for q in json_data["queries"]:
        if str(q["query_id"]) not in allow_ids:
            continue
        total += 1
        tgt = q["target_hard"]
        results = q["retrieval_results"]
        top1 = results[:1]
        top5 = results[:5]
        top10 = results[:10]
        if any(r["candidate_image"] == tgt for r in top1):
            found_at_1 += 1
            found_at_5 += 1
            found_at_10 += 1
        elif any(r["candidate_image"] == tgt for r in top5):
            found_at_5 += 1
            found_at_10 += 1
        elif any(r["candidate_image"] == tgt for r in top10):
            found_at_10 += 1

    def safe(x): return x / total if total > 0 else 0.0
    return {
        "total_queries": total,
        "found_at_1": found_at_1,
        "found_at_5": found_at_5,
        "found_at_10": found_at_10,
        "r1": safe(found_at_1),
        "r5": safe(found_at_5),
        "r10": safe(found_at_10),
    }


def calculate_metrics_subset(reranked_results: Dict[str, List[Dict[str, Any]]],
                             json_data: Dict[str, Any],
                             allow_ids: Set[str]) -> Dict[str, Any]:
    found_at_1 = found_at_5 = found_at_10 = 0
    total = 0
    for q in json_data["queries"]:
        qid = str(q["query_id"])
        if qid not in allow_ids:
            continue
        total += 1
        tgt = q["target_hard"]
        if qid in reranked_results:
            cand = reranked_results[qid]
            top1 = cand[:1]
            top5 = cand[:5]
            top10 = cand[:10]
            if any(c["candidate_image"] == tgt for c in top1):
                found_at_1 += 1
                found_at_5 += 1
                found_at_10 += 1
            elif any(c["candidate_image"] == tgt for c in top5):
                found_at_5 += 1
                found_at_10 += 1
            elif any(c["candidate_image"] == tgt for c in top10):
                found_at_10 += 1

    def safe(x): return x / total if total > 0 else 0.0
    return {
        "total_queries": total,
        "found_at_1": found_at_1,
        "found_at_5": found_at_5,
        "found_at_10": found_at_10,
        "r1": safe(found_at_1),
        "r5": safe(found_at_5),
        "r10": safe(found_at_10),
    }


def main():
    parser = argparse.ArgumentParser(description="Rerank retrieval results using OpenAI GPT (image+text)")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o", help="OpenAI GPT multimodal model name")
    parser.add_argument("--json_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory (PNG files, no suffix in ids)")
    parser.add_argument("--output_file", type=str, default="reranked_results_gpt.json", help="Output JSON file")
    parser.add_argument("--max_image_size", type=int, default=1024, help="Max image dimension for base64")
    parser.add_argument("--query_ids", type=str, default="", help="Comma-separated query_id list, e.g., '0,5,17'")
    parser.add_argument("--query_ids_file", type=str, default="", help="A text file with one query_id per line")
    parser.add_argument("--dry_run", action="store_true", help="只做过滤和指标统计，不调用GPT（用于快速验证子集选择）")
    args = parser.parse_args()

    # 读取数据
    with open(args.json_file, "r") as f:
        data = json.load(f)

    # 读入 allowlist（统一转成字符串）
    allow_ids: Set[str] = set()
    if args.query_ids.strip():
        allow_ids.update([s.strip() for s in args.query_ids.split(",") if s.strip()])
    if args.query_ids_file.strip():
        with open(args.query_ids_file, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    allow_ids.add(s)
    # 如果未提供，则默认全部（也转字符串）
    if not allow_ids:
        allow_ids = {str(q["query_id"]) for q in data["queries"]}

    # 用字符串键构建子集
    queries_map: Dict[str, Dict[str, Any]] = {
        str(q["query_id"]): q for q in data["queries"] if str(q["query_id"]) in allow_ids
    }

    # 若为空，给出友好提示并展示可用ID的样例
    if len(queries_map) == 0:
        sample_ids = [str(q["query_id"]) for q in data["queries"][:20]]
        print("[ERROR] 子集为空。请检查 --query_ids / --query_ids_file 是否与 JSON 中的 query_id 一致（整数/字符串）。")
        print(f"        你的 allow_ids = {sorted(list(allow_ids))}")
        print(f"        JSON 前20个 query_id 示例 = {sample_ids}")
        return

    # 原始指标（仅子集）
    original_metrics = calculate_original_metrics_subset(data, allow_ids)
    print(f"[Original @subset] R@1={original_metrics['r1']:.4f} "
          f"({original_metrics['found_at_1']}/{original_metrics['total_queries']}), "
          f"R@5={original_metrics['r5']:.4f}, R@10={original_metrics['r10']:.4f}")

    # dry-run
    if args.dry_run:
        output_data = {
            "metadata": {
                **data.get("metadata", {}),
                "reranked": False,
                "rerank_model": args.gpt_model,
                "subset_only": True,
                "subset_size": len(queries_map),
                "original_r1": original_metrics["r1"],
                "original_r5": original_metrics["r5"],
                "original_r10": original_metrics["r10"],
                "original_found_at_1": original_metrics["found_at_1"],
                "original_found_at_5": original_metrics["found_at_5"],
                "original_found_at_10": original_metrics["found_at_10"],
            },
            "queries": [queries_map[qid] for qid in queries_map],
        }
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"[DRY RUN] Saved to {args.output_file}")
        return

    client = OpenAI()

    all_results: List[Dict[str, Any]] = []
    pbar = tqdm(total=sum(len(q["retrieval_results"]) for q in queries_map.values()),
                desc="GPT reranking (subset)")

    for qid, q in queries_map.items():
        ref_path = os.path.join(args.image_dir, f"{q['reference_image']}.png")
        ref_img = load_image(ref_path, max_size=args.max_image_size)

        for cand in q["retrieval_results"]:
            cand_path = os.path.join(args.image_dir, f"{cand['candidate_image']}.png")
            cand_img = load_image(cand_path, max_size=args.max_image_size)

            result = call_gpt_yesno(
                client=client,
                model=args.gpt_model,
                ref_img=ref_img,
                cand_img=cand_img,
                modification_text=q["modification_text"],
                max_image_size=args.max_image_size,
            )

            score = 1.0 if result["yes_prob"] is None and str(result["answer"]).startswith("y") else (
                0.0 if result["yes_prob"] is None else result["yes_prob"]
            )

            all_results.append({
                "query_id": qid,  # 字符串
                "candidate_image": cand["candidate_image"],
                "rank": cand["rank"],
                "original_score": cand.get("similarity_score", None),
                "candidate_index": cand.get("candidate_index", None),
                "rerank_score": float(score),
                "yes_logit": float(result["yes_logprob"]) if result["yes_logprob"] is not None else None,
                "no_logit": float(result["no_logprob"]) if result["no_logprob"] is not None else None,
            })
            pbar.update(1)

    pbar.close()

    # 汇总为每个 query 的重排（键为字符串 qid）
    reranked_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        reranked_by_qid.setdefault(r["query_id"], []).append(r)

    for qid, lst in reranked_by_qid.items():
        lst.sort(key=lambda x: x["rerank_score"], reverse=True)
        for i, item in enumerate(lst, start=1):
            item["new_rank"] = i

    reranked_metrics = calculate_metrics_subset(reranked_by_qid, data, allow_ids)
    print(f"[Reranked @subset] R@1={reranked_metrics['r1']:.4f} "
          f"({reranked_metrics['found_at_1']}/{reranked_metrics['total_queries']}), "
          f"R@5={reranked_metrics['r5']:.4f}, R@10={reranked_metrics['r10']:.4f}")

    # 仅输出子集
    output_queries = []
    for qid, q in queries_map.items():
        q_out = dict(q)
        if qid in reranked_by_qid:
            q_out["reranked_results"] = []
            for c in reranked_by_qid[qid][:10]:
                q_out["reranked_results"].append({
                    "rank": c["new_rank"],
                    "candidate_image": c["candidate_image"],
                    "rerank_score": c["rerank_score"],
                    "yes_logit": c["yes_logit"],
                    "no_logit": c["no_logit"],
                    "original_rank": c["rank"],
                    "original_score": c["original_score"],
                })
        output_queries.append(q_out)

    output = {
        "metadata": {
            **data.get("metadata", {}),
            "reranked": True,
            "rerank_model": args.gpt_model,
            "subset_only": True,
            "subset_size": len(queries_map),
            "original_r1": original_metrics["r1"],
            "original_r5": original_metrics["r5"],
            "original_r10": original_metrics["r10"],
            "original_found_at_1": original_metrics["found_at_1"],
            "original_found_at_5": original_metrics["found_at_5"],
            "original_found_at_10": original_metrics["found_at_10"],
            "rerank_r1": reranked_metrics["r1"],
            "rerank_r5": reranked_metrics["r5"],
            "rerank_r10": reranked_metrics["r10"],
            "rerank_found_at_1": reranked_metrics["found_at_1"],
            "rerank_found_at_5": reranked_metrics["found_at_5"],
            "rerank_found_at_10": reranked_metrics["found_at_10"],
        },
        "queries": output_queries,
    }

    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Results saved to {args.output_file}")


if __name__ == "__main__":
    main()

"""
python VQA_api.py \
  --gpt_model gpt-4o \
  --json_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/retrieval_results/checkpoint-1500_20250919_160740/cirr_retrieval_top10.json \
  --image_dir /home/guohaiyun/yty_data/CIRR/dev \
  --output_file /home/guohaiyun/yangtianyu/MyComposedRetrieval/VQA_rerank/gpt_test/results/subset_reranked3.json \
  --query_ids "111,230,284,290,326,380,902,1235,1314,1456,1732,1929,2007" \
  --max_image_size 1440
"""