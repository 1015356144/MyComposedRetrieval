import json
import os
import torch
import re
import ast
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
from tqdm import tqdm

# ==================== 日志 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 模型加载 ====================
def load_model_and_processor():
    logger.info("正在加载Qwen2VL-7B模型...")
    model_name = "/home/guohaiyun/yangtianyu/CPRCIR/checkpoints/hf_models/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    logger.info("模型加载完成")
    return model, processor

# ==================== 数据加载 ====================
def load_test_data(json_file_path):
    logger.info(f"正在加载测试数据: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"已加载 {len(data)} 个测试样例")
    return data

# ==================== 分段式 Prompt（无 JSON；强锚点） ====================
def construct_prompt(reference_image_path, hard_negative_image_path, modification_text):
    """
    说明：
    - 要求模型用明确的锚点输出：<<<STAGE k ...>>> 到 <<<END STAGE k>>>。
    - 每段内采用“竖线 | 分隔键值”的扁平格式，便于正则解析。
    - 强制 QA 与 intents 一一对应、同序同 id；极性问句；facts 去重且 ≤ 8。
    """
    tem_prompt = f"""You are a multimodal edit auditor. You get two images: FIRST=REFERENCE, SECOND=TARGET, and an input edit text.
Your only goal: minimally rewrite the edit text so it is true in the TARGET.

Input edit text: "{modification_text}"

HARD RULES:
- Absolute intents: use TARGET-only evidence. Relative intents: compare REFERENCE vs TARGET only for the relative part.
- EXACTLY one QA per intent (|qa| == |intents|), same order, same id.
- Target facts only for testing the intents. No duplicates; at most 8 facts. Use canonical predicate form: "<C#>.<property|relation>=<value|OtherC#>".
- Enumerate objects as C1..C6, then "… and others (not enumerated)" if needed.
- QA must be polar (yes/no). Forbidden tokens: how, how many, how much, how long, how far, how big, how tall, how wide, what, which, where, when, why, who, whom, whose.
- Use EXACT stage fences below. Do NOT use code fences or JSON. Do NOT add any extra commentary outside fences.

OUTPUT FORMAT (copy exactly these fences):

<<<STAGE 0 CAPTIONS>>>
[REFERENCE] <2–3 literal sentences; no speculation.>
[TARGET] <2–3 literal sentences; no speculation.>
<<<END STAGE 0>>>

<<<STAGE 1 INTENTS>>>
# One line per intent. Keep order I1, I2, ...
I1 | span="..." | intent="..." | type=absolute|relative | objects=[dog, color] | why="one line reason"
I2 | span="..." | intent="..." | type=absolute|relative | objects=[...] | why="..."
<<<END STAGE 1>>>

<<<STAGE 3 TARGET FACTS>>>
# ≤ 8 facts; canonical form; include minimal evidence_target
F1 | fact="C1.category=dog" | evidence_target="..."
F2 | fact="C1.pose=standing" | evidence_target="..."
# If more than 6 objects, you may append: C7..Ck summarized as "… and others (not enumerated)"
<<<END STAGE 3>>>

<<<STAGE 4 QA (ONE PER INTENT; POLAR)>>>
# EXACTLY one QA per intent; same id; scope by type (absolute→target_only; relative→compare_ref_target)
I1 | scope=target_only | Q="In the TARGET image, is the dog standing?" | A=Yes|No|Uncertain | fact_ids=[F1,F2] | evidence_target="..." | evidence_ref=""
I2 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, does the dog have the same color?" | A=Yes|No|Uncertain | fact_ids=[F3] | evidence_target="..." | evidence_ref="..."
<<<END STAGE 4>>>

<<<STAGE 6 LOCAL EDITS>>>
# Map answers to edits
I1 | action=keep|rewrite|generalize|delete | before="..." | after="..." | reason="..."
I2 | action=... | before="..." | after="..." | reason="..."
<<<END STAGE 6>>>

<<<STAGE 7 TEXT NEW>>>
text_new="..."
<<<END STAGE 7>>>
"""
    return tem_prompt

# ==================== 文本预处理 ====================
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
}
def _preclean_text(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.replace("\ufeff", "")
    s = _ZERO_WIDTH_RE.sub("", s)
    # 去掉 Markdown 代码围栏（如果模型仍然输出了）
    s = re.sub(r"```[\s\S]*?```", "", s)
    # 智能引号替换
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    return s.strip()

# ==================== 分段解析器 ====================
# 锚点正则（允许中间有额外空白/注释）
FENCE_PATTERNS = {
    "stage0": (re.compile(r"<<<STAGE\s*0\s*CAPTIONS>>>", re.I), re.compile(r"<<<END\s*STAGE\s*0>>>", re.I)),
    "stage1": (re.compile(r"<<<STAGE\s*1\s*INTENTS>>>", re.I), re.compile(r"<<<END\s*STAGE\s*1>>>", re.I)),
    "stage3": (re.compile(r"<<<STAGE\s*3\s*TARGET\s*FACTS>>>", re.I), re.compile(r"<<<END\s*STAGE\s*3>>>", re.I)),
    "stage4": (re.compile(r"<<<STAGE\s*4\s*QA.*?>>>", re.I), re.compile(r"<<<END\s*STAGE\s*4>>>", re.I)),
    "stage6": (re.compile(r"<<<STAGE\s*6\s*LOCAL\s*EDITS>>>", re.I), re.compile(r"<<<END\s*STAGE\s*6>>>", re.I)),
    "stage7": (re.compile(r"<<<STAGE\s*7\s*TEXT\s*NEW>>>", re.I), re.compile(r"<<<END\s*STAGE\s*7>>>", re.I)),
}

def _extract_block(text: str, start_re: re.Pattern, end_re: re.Pattern) -> str:
    start = start_re.search(text)
    end = end_re.search(text)
    if not start or not end or end.start() <= start.end():
        return ""
    return text[start.end():end.start()].strip()

def _parse_keyvals_line(line: str) -> dict:
    """
    解析类似：
    I1 | span="..." | intent="..." | type=absolute | objects=[dog, color] | why="..."
    F1 | fact="C1.category=dog" | evidence_target="..."
    I1 | scope=target_only | Q="..." | A=Yes | fact_ids=[F1,F2] | evidence_target="..." | evidence_ref=""
    I1 | action=keep | before="..." | after="..." | reason="..."
    """
    parts = [p.strip() for p in line.split("|")]
    out = {}
    if not parts:
        return out
    # 第一个片段通常以 I# 或 F# 开头
    id_m = re.match(r'^(I|F)\s*(\d+)', parts[0], flags=re.I)
    if id_m:
        out["id"] = f"{id_m.group(1).upper()}{id_m.group(2)}"
    else:
        # 允许 text_new="..." 行
        kv = parts[0]
        m2 = re.match(r'(\w+)\s*=\s*(.*)$', kv)
        if m2:
            out[m2.group(1)] = m2.group(2).strip()
    # 其余键值对
    for kv in parts[1:]:
        if "=" not in kv: 
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        # 去掉首尾引号
        if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
            v = v[1:-1]
        # 尝试把列表字面量转成 Python 列表
        if v.startswith("[") and v.endswith("]"):
            try:
                val = ast.literal_eval(v)
                # 统一转成字符串列表，去空白
                if isinstance(val, list):
                    val = [str(x).strip() for x in val]
                out[k] = val
                continue
            except Exception:
                # 回退：按逗号拆
                out[k] = [s.strip() for s in v.strip("[]").split(",") if s.strip()]
                continue
        out[k] = v
    return out

def parse_staged_response(response_text: str):
    """
    将分段文本解析为结构化 dict：
    {
      "captions": {"reference": "...", "target": "..."},
      "intents": [ {...}, ... ],
      "target_facts": [ {...}, ... ],
      "qa": [ {...}, ... ],
      "local_edits": [ {...}, ... ],
      "text_new": "..."
    }
    另外返回 meta 诊断。
    """
    raw = _preclean_text(response_text)

    blocks = {}
    for key, (sre, ere) in FENCE_PATTERNS.items():
        blocks[key] = _extract_block(raw, sre, ere)

    result = {
        "captions": {"reference": "", "target": ""},
        "intents": [],
        "target_facts": [],
        "qa": [],
        "local_edits": [],
        "text_new": ""
    }

    # 解析 STAGE 0
    b0 = blocks.get("stage0", "")
    if b0:
        # 查找 [REFERENCE] / [TARGET] 行
        ref_m = re.search(r'\[REFERENCE\]\s*(.+)', b0, flags=re.I)
        tgt_m = re.search(r'\[TARGET\]\s*(.+)', b0, flags=re.I)
        if ref_m:
            result["captions"]["reference"] = ref_m.group(1).strip()
        if tgt_m:
            result["captions"]["target"] = tgt_m.group(1).strip()

    # 解析 STAGE 1（多行 intents）
    b1 = blocks.get("stage1", "")
    if b1:
        for line in b1.splitlines():
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            obj = _parse_keyvals_line(line)
            # 规范字段名
            intent = {
                "id": obj.get("id", ""),
                "span": obj.get("span", ""),
                "intent": obj.get("intent", ""),
                "type": obj.get("type", ""),
                "objects": obj.get("objects", []),
                "why": obj.get("why", "")
            }
            result["intents"].append(intent)

    # 解析 STAGE 3（facts）
    b3 = blocks.get("stage3", "")
    if b3:
        seen_fact_texts = set()
        for line in b3.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = _parse_keyvals_line(line)
            fact_txt = obj.get("fact", "")
            # 去重：同样的 fact 文本跳过
            if fact_txt and fact_txt in seen_fact_texts:
                continue
            if fact_txt:
                seen_fact_texts.add(fact_txt)
            fact = {
                "id": obj.get("id", ""),
                "fact": fact_txt,
                "evidence_target": obj.get("evidence_target", "")
            }
            result["target_facts"].append(fact)
        # 限制 ≤ 8
        result["target_facts"] = result["target_facts"][:8]

    # 解析 STAGE 4（QA）
    b4 = blocks.get("stage4", "")
    if b4:
        for line in b4.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = _parse_keyvals_line(line)
            qa_item = {
                "id": obj.get("id", ""),
                "scope": obj.get("scope", ""),
                "question": obj.get("Q", obj.get("q", "")),
                "answer": obj.get("A", obj.get("a", "")),
                "evidence_target": obj.get("evidence_target", ""),
                "evidence_ref": obj.get("evidence_ref", ""),
                "fact_ids": obj.get("fact_ids", []),
            }
            # 统一 fact_ids 成列表
            if isinstance(qa_item["fact_ids"], str) and qa_item["fact_ids"]:
                qa_item["fact_ids"] = [s.strip() for s in qa_item["fact_ids"].strip("[]").split(",") if s.strip()]
            result["qa"].append(qa_item)

    # 解析 STAGE 6（Local edits）
    b6 = blocks.get("stage6", "")
    if b6:
        for line in b6.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = _parse_keyvals_line(line)
            edit = {
                "id": obj.get("id", ""),
                "action": obj.get("action", ""),
                "before": obj.get("before", ""),
                "after": obj.get("after", ""),
                "reason": obj.get("reason", "")
            }
            result["local_edits"].append(edit)

    # 解析 STAGE 7（text_new）
    b7 = blocks.get("stage7", "")
    if b7:
        # 兼容：既支持一整行 text_new="..."，也支持多行情况
        m = re.search(r'text_new\s*=\s*"(.*?)"', b7, flags=re.I|re.S)
        if m:
            result["text_new"] = m.group(1).strip()
        else:
            # 退化：取整段
            result["text_new"] = b7.strip()

    # ========== 诊断 meta ==========
    meta = _post_checks_staged(result)
    return {"parsed_stages": result, "raw_text": response_text, "parsed_meta": meta}

# ==================== 规范检查（分段版） ====================
_WH_WORDS_RE = re.compile(r'\b(how|what|which|where|when|why|who|whom|whose)\b', re.IGNORECASE)
_ABS_PREFIX_RE = re.compile(r'^\s*In the TARGET image,\s*(is|are|does|do|has|have|can|was|were|did)\b', re.IGNORECASE)
_REL_PREFIX_RE = re.compile(r'^\s*Comparing REFERENCE and TARGET,\s*(is|are|does|do|has|have|can|was|were|did)\b', re.IGNORECASE)

def _post_checks_staged(obj):
    intents = obj.get("intents", [])
    qa = obj.get("qa", [])
    tfacts = obj.get("target_facts", [])
    diag = {
        "counts": {"intents": len(intents), "qa": len(qa), "target_facts": len(tfacts)},
        "alignment": {"|qa|==|intents|": len(intents) == len(qa), "ids_match_and_order": True, "scope_matches_type": True},
        "question_polarity": {"wh_violations": [], "prefix_violations": []},
        "fact_refs": {"unknown_fact_ids": {}, "unused_facts": [], "duplicate_fact_texts": []},
        "id_uniqueness": {"intent_ids_unique": True, "qa_ids_unique": True, "fact_ids_unique": True}
    }

    intent_ids = [x.get("id") for x in intents]
    qa_ids = [x.get("id") for x in qa]
    fact_ids = [x.get("id") for x in tfacts]

    diag["id_uniqueness"]["intent_ids_unique"] = len(intent_ids) == len(set(intent_ids))
    diag["id_uniqueness"]["qa_ids_unique"] = len(qa_ids) == len(set(qa_ids))
    diag["id_uniqueness"]["fact_ids_unique"] = len(fact_ids) == len(set(fact_ids))

    if len(intents) == len(qa):
        for i, (ii, qq) in enumerate(zip(intents, qa)):
            if (ii or {}).get("id") != (qq or {}).get("id"):
                diag["alignment"]["ids_match_and_order"] = False
                break
            itype = (ii or {}).get("type")
            qscope = (qq or {}).get("scope")
            if itype == "absolute" and qscope != "target_only":
                diag["alignment"]["scope_matches_type"] = False
            if itype == "relative" and qscope != "compare_ref_target":
                diag["alignment"]["scope_matches_type"] = False
    else:
        diag["alignment"]["ids_match_and_order"] = False
        diag["alignment"]["scope_matches_type"] = False

    fact_id_set = set(fact_ids)
    used = set()
    for qq in qa:
        qid = qq.get("id")
        qtext = qq.get("question", "") or ""
        scope = qq.get("scope", "")
        if _WH_WORDS_RE.search(qtext):
            diag["question_polarity"]["wh_violations"].append(qid)
        if scope == "target_only":
            if not _ABS_PREFIX_RE.search(qtext):
                diag["question_polarity"]["prefix_violations"].append(qid)
        elif scope == "compare_ref_target":
            if not _REL_PREFIX_RE.search(qtext):
                diag["question_polarity"]["prefix_violations"].append(qid)
        missing = []
        for fid in (qq.get("fact_ids") or []):
            if fid not in fact_id_set:
                missing.append(fid)
            else:
                used.add(fid)
        if missing:
            diag["fact_refs"]["unknown_fact_ids"][qid] = missing

    diag["fact_refs"]["unused_facts"] = [fid for fid in fact_ids if fid not in used]

    # 重复 fact 文本
    seen_txt = set()
    dups = set()
    for f in tfacts:
        txt = (f or {}).get("fact", "")
        if not txt:
            continue
        if txt in seen_txt:
            dups.add(txt)
        seen_txt.add(txt)
    diag["fact_refs"]["duplicate_fact_texts"] = sorted(list(dups))
    return diag

# ==================== 生成 & 主流程 ====================
def generate_response(model, processor, reference_image_path, hard_negative_image_path, modification_text, base_path, user_prompt: str = ""):
    ref_img_full_path = os.path.join(base_path, reference_image_path.lstrip('./'))
    neg_img_full_path = os.path.join(base_path, hard_negative_image_path.lstrip('./'))

    if not os.path.exists(ref_img_full_path):
        logger.warning(f"参考图片不存在: {ref_img_full_path}")
        return {"error": f"参考图片不存在: {ref_img_full_path}"}
    if not os.path.exists(neg_img_full_path):
        logger.warning(f"困难负样本图片不存在: {neg_img_full_path}")
        return {"error": f"困难负样本图片不存在: {neg_img_full_path}"}

    try:
        prompt_text = user_prompt or construct_prompt(reference_image_path, hard_negative_image_path, modification_text)

        # 组装消息（先发两张图，再发文本）
        content = [
            {"type": "image", "image": ref_img_full_path},
            {"type": "image", "image": neg_img_full_path},
        ]
        if isinstance(prompt_text, str) and prompt_text.strip():
            content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        # 构造模型输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1200,  # 分段文本足够
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        # 解析分段文本
        parsed = parse_staged_response(output_text)
        if "parsed_stages" not in parsed:
            logger.warning(f"解析失败，原始输出片段: {output_text[:600]}")
        return parsed

    except Exception as e:
        logger.error(f"生成响应时出错: {str(e)}")
        return {"error": str(e)}

# ==================== 输出目录 ====================
def create_output_directory():
    base_output_dir = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/generate_modtext_test/output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"test_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"创建输出目录: {output_dir}")
    return output_dir

# ==================== 主函数 ====================
def main():
    logger.info("开始执行Qwen2VL7B模型测试")
    json_file_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/generate_modtext_test/hard_negatives_selected.json"
    base_image_path = "/home/guohaiyun/yty_data/CIRR"

    model, processor = load_model_and_processor()
    test_data = load_test_data(json_file_path)
    output_dir = create_output_directory()
    results = []

    logger.info(f"开始处理 {len(test_data)} 个测试样例")
    for idx, sample in enumerate(tqdm(test_data, desc="处理样例")):
        logger.info(f"处理样例 {idx + 1}/{len(test_data)}")
        reference_image = sample["reference_image"]
        hard_negative_image = sample["hard_negative_image"]
        modification_text = sample["modification_text"]
        user_prompt = sample.get("prompt", "")  # 可为空

        response = generate_response(
            model, processor, reference_image, hard_negative_image, modification_text, base_image_path, user_prompt=user_prompt
        )

        result = {
            "sample_id": idx,
            "original_data": sample,
            "input": {
                "reference_image": reference_image,
                "hard_negative_image": hard_negative_image,
                "original_modification_text": modification_text,
                "prompt_used": user_prompt
            },
            "output": response,  # 包含 parsed_stages / parsed_meta / raw_text
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)

        if (idx + 1) % 10 == 0:
            temp_output_file = os.path.join(output_dir, f"temp_results_{idx + 1}.json")
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存前 {idx + 1} 个样例的临时结果")

    final_output_file = os.path.join(output_dir, "final_results.json")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总
    summary_results = []
    for result in results:
        output = result["output"]
        summary = {
            "sample_id": result["sample_id"],
            "original_modification_text": result["input"]["original_modification_text"],
            "success": isinstance(output, dict) and ("parsed_stages" in output)
        }
        if summary["success"]:
            ps = output["parsed_stages"]
            meta = output.get("parsed_meta", {})
            summary["text_new"] = ps.get("text_new", "")
            summary["counts"] = meta.get("counts", {})
            summary["alignment"] = meta.get("alignment", {})
            summary["question_polarity"] = meta.get("question_polarity", {})
            summary["fact_refs"] = meta.get("fact_refs", {})
            summary["id_uniqueness"] = meta.get("id_uniqueness", {})
        else:
            summary["error"] = output.get("error", "Unknown error") if isinstance(output, dict) else "Output not dict"
        summary_results.append(summary)

    summary_file = os.path.join(output_dir, "results_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    stats = {
        "total_samples": len(test_data),
        "successful_generations": sum(1 for r in results if isinstance(r["output"], dict) and "parsed_stages" in r["output"]),
        "failed_generations": sum(1 for r in results if not (isinstance(r["output"], dict) and "parsed_stages" in r["output"])),
        "completion_time": datetime.now().isoformat(),
        "output_directory": output_dir
    }
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("测试完成！")
    logger.info(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
