import json
import os
import torch
import re
import ast
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_processor():
    """加载Qwen2VL7B模型和处理器"""
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

def load_test_data(json_file_path):
    """加载测试数据"""
    logger.info(f"正在加载测试数据: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"已加载 {len(data)} 个测试样例")
    return data

def construct_prompt(reference_image_path, hard_negative_image_path, modification_text):
    tem_prompt="""You are a rigorous multimodal editing assistant. You receive two images (reference image = first image; target image = second image) and an input edit text. Your job is to minimally revise the edit text so it matches the TARGET image.

Input edit text: "{modification_text}"

HARD GROUNDING RULES (read carefully):
- Final output MUST be derived only from the TARGET image. The reference image is used ONLY for relative intents (i.e., change-from-source requirements).
- If an absolute intent cites the reference or lacks target evidence, mark it as "Uncertain" and generalize/delete per policy.
- Do NOT infer occluded objects. Do NOT copy details that appear ONLY in the reference image.

=====================
IMAGE ANCHORS (read BEFORE stages)
=====================
- The FIRST image you receive is the REFERENCE image.
- The SECOND image you receive is the TARGET image.
- In "options_used", you MUST echo the anchors:
  "image_anchors": {"reference_seen_as":"first","target_seen_as":"second"}
- (Optional) Also include:
  "image_names": {"reference":"<name_or_id_if_known>","target":"<name_or_id_if_known>"}

=====================
STAGES (run in order)
=====================

Stage 0 — Dual Image Captioning (anchored; to stabilize perception)
- Produce concise, literal captions for BOTH images with EXPLICIT labels:
  - captions.reference: MUST start with "[REFERENCE]",only focus on the REFERENCE image, followed by 3-5 short sentences describing clearly visible objects, attributes (color/material), counts, spatial relations, and readable text. No speculation.
  - captions.target:    MUST start with "[TARGET]",only focus on the TARGET image, followed by 3-5 short sentences with the same constraints.
- If any detail is blurry or unreadable, write "unreadable" or "uncertain" — do NOT guess.
- Additionally, in "options_used", provide a short difference list that highlights the differences between the two images :
  "caption_diff": ["<difference cue 1>", "<difference cue 2>", "<difference cue 3>"]

Stage 1 — Text Parse & Intent Decomposition (from text only; coverage REQUIRED)
1A) Text Parse (write to options_used.parse)
- Tokenize and segment the edit text into minimal semantic units by punctuation (commas/semicolons), coordinators ("and/then"), and parentheses.
- Extract structured slots:
  - ACTION (ACT#): verbs like add/remove/change/place/move.
  - OBJECT (OBJ#): nouns with determiners/quantifiers (keep exact number words, e.g., "a/one/two/birds").
  - QUANTITY (Q#): normalized counts ("a/an/one" → 1; plural nouns without numerals → ≥2 unless restricted).
  - SPATIAL (S#): prepositions/relations (on/under/left-of/…).
  - BODY_PART (BP#): human/animal body parts (e.g., forearm).
  - GESTURE/POSE (G#): states like "fist closed", "arm extended".
  - ATTRIBUTE (A#): color/material/style/text/logo.
- Map each slot to its exact source span(s) in the text (character offsets or verbatim substrings).

1B) Coverage & Conflict Check (write to options_used.coverage)
- Build a coverage table: for every content-bearing phrase (from 1A), list which intent id(s) will cover it.
- "uncovered_spans": list all content words/phrases not covered by any intent (MUST be empty; if not, FIX decomposition).
- "conflicts": detect intra-text contradictions (e.g., "Add a bird" vs "Place birds ..."). Do NOT resolve with off-image facts; note the conflict and later let TARGET evidence drive the rewrite.
- If uncovered_spans ≠ empty, revise 1A and 1C until empty.

1C) Intent Decomposition (finalize)
- Create one atomic intent per slot or coupled slots when they form a single verifiable requirement.
- Each intent MUST have:
  id, span (verbatim substring), intent (paraphrase), objects/attributes.
- IMPORTANT: Parenthetical modifiers MUST become their own intents if they encode states/gestures (e.g., "(fist closed)").
- Examples for a composite text like: "Add a bird, Place birds on extended human forearm (fist closed)":
  - I# (ACTION+OBJECT+QUANTITY): "Add a bird" (add one bird).
  - I# (SPATIAL+OBJECT): "Place birds on ... forearm".
  - I# (BODY_PART): "human forearm".
  - I# (POSE): "extended forearm".
  - I# (GESTURE): "fist closed".
  (Your actual ids/spans must reflect the source text precisely; no overlaps; full coverage; no omissions.)

Stage 2 — Intent Typing
- For each intent, set type = "absolute" | "relative" and give a one-line reason.
  - absolute: can be verified from TARGET alone.
  - relative: requires comparing REFERENCE vs TARGET (typical for “add/remove/replace/swap/move”).
  (Tip: "Add/Remove/Replace/Move" are usually relative; spatial/pose/gesture may be absolute if they refer only to the outcome in TARGET.)

Stage 3 — Target Facts (grounding pass)
- From the TARGET only, list observable facts needed to test the intents.
- Counting Protocol: enumerate relevant objects as C1..Ck with coarse locations (left/center/right; front/back). Do NOT infer occluded items.
- Body-part grounding: if a human forearm is required, identify its visibility and orientation (extended vs bent) from TARGET only.
- Gesture grounding: if "fist closed" is required, cite visible finger/knuckle cues; otherwise mark Uncertain.
- Chair vs. stool rule: chair = has backrest; stool = no backrest (if furniture is involved).
- Color policy: if not clearly pure white, prefer “light gray” or “light-colored”.

Stage 4 — Sub-questions (anchor the wording)
- For each intent, write a concrete Yes/No/Uncertain question.
- Absolute: MUST begin with “In the TARGET image, …” and reference only TARGET evidence.
- Relative: MUST begin with “Comparing REFERENCE and TARGET, …” and explicitly require comparison.
- Include scope = "target_only" for absolute, scope = "compare_ref_target" for relative.

Stage 5 — Q&A with Evidence
- Answer each sub-question with Yes/No/Uncertain.
- Provide evidence_target (required) and evidence_ref (only if scope = compare_ref_target).
- Evidence must cite specific visible cues (color/shape/position/count/body-part/gesture/text).

Stage 6 — Local Edits
- Map answers to edits:
  - Yes → KEEP the corresponding requirement and (when possible) the original sentence pattern.
  - No → REWRITE that fragment to match the TARGET; do not introduce off-image facts.
  - Uncertain → generalize/delete/flag per uncertain_policy.
- Apply the minimum necessary changes. Resolve any textual conflicts (from Stage 1B) ONLY using TARGET evidence.

Stage 7 — Synthesize "text_new"
- Merge local edits into a coherent, conflict-free new edit text.
- Every requirement must be directly observable in the TARGET.
- Keep total length ≤ length_limit and use the same language as the input text.

=====================
OUTPUT JSON (exact)
=====================
Return EXACTLY ONE JSON object with the following schema (no extra text, no code fences):

{
  "options_used": {},
  "captions": {
    "reference": "",
    "target": ""
  },
  "intents": [
    {"id": "I1", "span": "...", "intent": "...", "type": "absolute|relative", "objects": ["..."]}
  ],
  "target_facts": [
    {"id": "F1", "fact": "...", "evidence_target": "..."}
  ],
  "qa": [
    {"id": "I1", "scope": "target_only|compare_ref_target", "question": "...", "answer": "Yes|No|Uncertain", "evidence_target": "...", "evidence_ref": ""}
  ],
  "local_edits": [
    {"id": "I1", "action": "keep|rewrite|generalize|delete", "before": "...", "after": "...", "reason": "..."}
  ],
  "text_new": "...",
  "self_check": {"consistency_ok": true, "visibility_ok": true, "notes": "..."}
}

FINAL RESPONSE CONTRACT:
- Output exactly one JSON object that matches the schema above.
- Do NOT include any prose, code fences, the FEWSHOT content, or a second JSON object.
- The first character of your message must be { and the last character must be }.
- If some field cannot be supported by TARGET evidence, set it to "Uncertain" and still return a single valid JSON object.

"""
    return tem_prompt.replace("{modification_text}", modification_text)

# ---------------- 解析工具：支持 fenced code、jsonc、单引号/True False、尾逗号 + 自动补全 ----------------

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
}

def _preclean_text(s: str) -> str:
    """基础预清洗：去 BOM/零宽字符，替换智能引号为普通引号"""
    if not isinstance(s, str):
        return s
    s = s.replace("\ufeff", "")
    s = _ZERO_WIDTH_RE.sub("", s)
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    return s

def _strip_jsonc(s: str) -> str:
    """去掉 // 和 /* */ 注释，并去掉对象/数组中的尾逗号"""
    s = _preclean_text(s)
    s = re.sub(r'//.*?(?=\n|\r|$)', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s

_PY_WORDS_MAP = [
    (re.compile(r'\btrue\b', flags=re.IGNORECASE), 'True'),
    (re.compile(r'\bfalse\b', flags=re.IGNORECASE), 'False'),
    (re.compile(r'\bnull\b', flags=re.IGNORECASE), 'None'),
]

def _json_like_to_python_literals(s: str) -> str:
    """将 json 风格关键字替换成 Python 字面量（AST 回退前使用）"""
    s = _preclean_text(s)
    for pat, rep in _PY_WORDS_MAP:
        s = pat.sub(rep, s)
    return s

def _try_parse_json(candidate: str):
    """优先用 json 解析；失败则尝试 jsonc 清洗后再解析"""
    cand = candidate.strip()
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        pass
    try:
        cleaned = _strip_jsonc(cand)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def _try_parse_ast(candidate: str):
    """AST 回退：把 json-like 替换为 Python 字面量，然后 literal_eval"""
    cand = candidate.strip()
    cand = _json_like_to_python_literals(cand)
    cand = _strip_jsonc(cand)
    try:
        obj = ast.literal_eval(cand)
        return obj
    except Exception:
        return None

def _extract_from_codeblocks(text: str):
    """
    从 ```json / ```jsonc / ``` 代码块中提取候选。优先取最后一个块里最后一个 JSON 对象。
    """
    blocks = re.findall(r"```(?:jsonc?|JSON|Json)?\s*([\s\S]*?)\s*```", text)
    for block in reversed(blocks):
        blk = block.strip()
        # 1) 块整体尝试
        obj = _try_parse_json(blk)
        if obj is not None:
            return obj
        # 2) 在块内部搜最后一个 {...}
        candidate = None
        for m in re.finditer(r"\{", blk):
            start = m.start()
            snippet = blk[start:]
            obj = _try_parse_json(snippet)
            if obj is not None:
                candidate = obj
        if candidate is not None:
            return candidate
        # 3) AST 回退（块整体）
        obj2 = _try_parse_ast(blk)
        if obj2 is not None:
            return obj2
        # 4) AST 回退（块内从每个 { 开始）
        for m in re.finditer(r"\{", blk):
            snippet = blk[m.start():]
            obj3 = _try_parse_ast(snippet)
            if obj3 is not None:
                return obj3
    return None

def _extract_last_json_object(text: str):
    """
    在整段文本中从每个左花括号出发尝试解析，取最后一个成功的对象。
    解析顺序：json -> jsonc清洗json -> AST 回退
    """
    t = _preclean_text(text)
    last_obj = None
    for m in re.finditer(r"\{", t):
        i = m.start()
        snippet = t[i:]
        obj = _try_parse_json(snippet)
        if obj is not None:
            last_obj = obj
            continue
        obj2 = _try_parse_ast(snippet)
        if obj2 is not None:
            last_obj = obj2
            continue
    if last_obj is None:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return last_obj

# ---------------- 自动补全（Coerce）与宽松校验 ----------------

def _coerce_schema_defaults(obj):
    """
    尝试将解析到的对象补全为目标schema需要的字段与类型。
    返回 (obj, changed: bool, notes: list[str])
    """
    notes = []
    changed = False
    if not isinstance(obj, dict):
        return obj, changed, ["root is not a dict"]

    def ensure_field(name, default):
        nonlocal changed
        if name not in obj or not isinstance(obj[name], type(default)):
            obj[name] = default
            changed = True
            notes.append(f"filled '{name}' with default")

    ensure_field("options_used", {})
    ensure_field("captions", {"reference": "", "target": ""})
    # 修 captions 子键
    if isinstance(obj.get("captions"), dict):
        if "reference" not in obj["captions"] or not isinstance(obj["captions"]["reference"], str):
            obj["captions"]["reference"] = ""
            changed = True
            notes.append("fixed captions.reference")
        if "target" not in obj["captions"] or not isinstance(obj["captions"]["target"], str):
            obj["captions"]["target"] = ""
            changed = True
            notes.append("fixed captions.target")
    else:
        obj["captions"] = {"reference": "", "target": ""}
        changed = True
        notes.append("reset captions to default dict")

    for name in ["intents", "target_facts", "qa", "local_edits"]:
        if name not in obj or not isinstance(obj[name], list):
            obj[name] = []
            changed = True
            notes.append(f"filled '{name}' as empty list")

    if "text_new" not in obj or not isinstance(obj["text_new"], str):
        obj["text_new"] = ""
        changed = True
        notes.append("filled 'text_new' as empty string")

    # self_check 补全
    default_self_check = {"consistency_ok": False, "visibility_ok": False, "notes": "filled by parser"}
    if "self_check" not in obj or not isinstance(obj["self_check"], dict):
        obj["self_check"] = default_self_check
        changed = True
        notes.append("filled 'self_check' with default")
    else:
        sc = obj["self_check"]
        if "consistency_ok" not in sc or not isinstance(sc["consistency_ok"], bool):
            sc["consistency_ok"] = False
            changed = True
            notes.append("filled self_check.consistency_ok")
        if "visibility_ok" not in sc or not isinstance(sc["visibility_ok"], bool):
            sc["visibility_ok"] = False
            changed = True
            notes.append("filled self_check.visibility_ok")
        if "notes" not in sc or not isinstance(sc["notes"], str):
            sc["notes"] = "filled by parser"
            changed = True
            notes.append("filled self_check.notes")

    return obj, changed, notes

def parse_json_response(response_text: str):
    """解析模型输出的JSON响应（预清洗 -> 直接JSON -> 代码块 -> 全文扫描 -> AST回退 -> 自动补全）"""
    response_text = _preclean_text(response_text.strip())

    # 1) 直接整体解析（json / jsonc）
    direct = _try_parse_json(response_text)
    if direct is not None:
        if validate_json_structure(direct):
            return {"parsed_json": direct, "raw_text": response_text}
        # 自动补全再验
        coerced, changed, notes = _coerce_schema_defaults(direct)
        if validate_json_structure(coerced):
            return {"parsed_json": coerced, "raw_text": response_text, "repaired": changed, "repair_notes": notes}
        else:
            return {"error": "JSON structure validation failed after coerce", "raw_text": response_text, "partial_json": coerced, "repair_notes": notes}

    # 2) 代码块中提取
    block_obj = _extract_from_codeblocks(response_text)
    if block_obj is not None:
        if validate_json_structure(block_obj):
            return {"parsed_json": block_obj, "raw_text": response_text}
        coerced, changed, notes = _coerce_schema_defaults(block_obj)
        if validate_json_structure(coerced):
            return {"parsed_json": coerced, "raw_text": response_text, "repaired": changed, "repair_notes": notes}
        else:
            return {"error": "JSON structure validation failed after coerce (codeblock)", "raw_text": response_text, "partial_json": coerced, "repair_notes": notes}

    # 3) 全文扫描最后一个 JSON 对象（含 AST 回退）
    try:
        last_obj = _extract_last_json_object(response_text)
        if validate_json_structure(last_obj):
            return {"parsed_json": last_obj, "raw_text": response_text}
        coerced, changed, notes = _coerce_schema_defaults(last_obj)
        if validate_json_structure(coerced):
            return {"parsed_json": coerced, "raw_text": response_text, "repaired": changed, "repair_notes": notes}
        else:
            return {"error": "JSON structure validation failed after coerce (scan)", "raw_text": response_text, "partial_json": coerced, "repair_notes": notes}
    except Exception as e:
        return {"error": f"JSON parsing failed: {str(e)}", "raw_text": response_text}

def validate_json_structure(parsed_json):
    """验证JSON结构是否符合预期格式（包含 captions 字段）"""
    required_fields = ["options_used", "captions", "intents", "target_facts", "qa", "local_edits", "text_new", "self_check"]
    if not isinstance(parsed_json, dict):
        return False
    for field in required_fields:
        if field not in parsed_json:
            logger.warning(f"缺少必要字段: {field}")
            return False
    if not isinstance(parsed_json["captions"], dict):
        logger.warning("字段 captions 应该是字典类型")
        return False
    for k in ["reference", "target"]:
        if k not in parsed_json["captions"] or not isinstance(parsed_json["captions"][k], str):
            logger.warning("captions.reference/target 缺失或类型错误")
            return False
    for field in ["intents", "target_facts", "qa", "local_edits"]:
        if not isinstance(parsed_json[field], list):
            logger.warning(f"字段 {field} 应该是列表类型")
            return False
    if not isinstance(parsed_json["self_check"], dict):
        logger.warning("字段 self_check 应该是字典类型")
        return False
    return True

def generate_response(model, processor, reference_image_path, hard_negative_image_path, modification_text, base_path, user_prompt: str = ""):
    """生成模型响应
    - user_prompt: 额外的文本提示（你自己写）。若为空，则只发送两张图片。
    """
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

        # 组装消息：如果 prompt 为空，则不加文本，只发两张图片
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

        # 非采样、低随机性（去掉temperature/top_p/top_k）
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # 解析 JSON
        parsed_result = parse_json_response(output_text)
        if "error" in parsed_result:
            logger.warning(f"解析失败，原始输出片段: {output_text[:600]}")
        return parsed_result

    except Exception as e:
        logger.error(f"生成响应时出错: {str(e)}")
        return {"error": str(e)}

def create_output_directory():
    """创建带时间戳的输出目录"""
    base_output_dir = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/generate_modtext_test/output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"test_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"创建输出目录: {output_dir}")
    return output_dir

def main():
    """主函数"""
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

        user_prompt = sample.get("prompt", "")  # 留空则只发图片

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
            "output": response,
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

    summary_results = []
    for result in results:
        output = result["output"]
        summary = {
            "sample_id": result["sample_id"],
            "original_modification_text": result["input"]["original_modification_text"],
            "success": "parsed_json" in output,
        }
        if "parsed_json" in output:
            parsed_data = output["parsed_json"]
            summary["new_text"] = parsed_data.get("text_new", "")
            summary["has_intents"] = len(parsed_data.get("intents", [])) > 0
            summary["has_target_facts"] = len(parsed_data.get("target_facts", [])) > 0
            summary["has_qa"] = len(parsed_data.get("qa", [])) > 0
            summary["has_local_edits"] = len(parsed_data.get("local_edits", [])) > 0
            summary["self_check"] = parsed_data.get("self_check", {})
        else:
            summary["error"] = output.get("error", "Unknown error")
            summary["new_text"] = ""
        summary_results.append(summary)

    summary_file = os.path.join(output_dir, "results_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    stats = {
        "total_samples": len(test_data),
        "successful_generations": sum(1 for r in results if "parsed_json" in r["output"]),
        "failed_generations": sum(1 for r in results if "error" in r["output"]),
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
