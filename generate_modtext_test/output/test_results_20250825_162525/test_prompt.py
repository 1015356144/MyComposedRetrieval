import json
import os
import torch
import re
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
    """构建输入文本prompt（避免 .format 花括号转义问题）"""
    prompt_template = """You are a rigorous multimodal editing assistant. You receive two images (reference image = first image; target image = second image) and an input edit text. Your job is to minimally revise the edit text so it matches the TARGET image.

Input edit text: "{modification_text}"

HARD GROUNDING RULES (read carefully):
- Final output MUST be derived only from the TARGET image. The reference image is used ONLY for relative intents (i.e., change-from-source requirements).
- If an absolute intent cites the reference or lacks target evidence, mark it as "Uncertain" and generalize/delete per policy.
- Do NOT infer occluded objects. Do NOT copy details that appear ONLY in the reference image.

=====================
STAGES (run in order)
=====================

Stage 0 — Dual Image Captioning (to stabilize perception)
- Produce concise, literal captions for BOTH images:
  - `captions.reference`: 1–3 short sentences describing clearly visible objects, attributes (color/material), counts, spatial relations, and text if clearly readable. No speculation.
  - `captions.target`: 1–3 short sentences with the same constraints.
- If a detail is blurry or unreadable, say "unreadable" or "uncertain"; do NOT guess.
- These captions are an observation step only; later reasoning must still obey the HARD GROUNDING RULES.

Stage 1 — Intent Decomposition (from text only)
- Split the input edit text into atomic sub-intents (color / count / add / remove / pose / layout / material / style / text / logo).
- For each sub-intent, output: `id`, `span` (the exact token/phrase from the text), `intent` (paraphrase), `objects/attributes`.

Stage 2 — Intent Typing
- For each sub-intent, set `type = "absolute" | "relative"` and give a one-line reason.
  - absolute: can be verified from TARGET alone.
  - relative: requires comparing REFERENCE vs TARGET (typical for “change/replace/swap” intents).

Stage 3 — Target Facts (grounding pass)
- From the TARGET only, list observable facts needed to test the intents.
- Counting Protocol: enumerate relevant objects as `C1..Ck` with coarse locations (left/center/right; front/back). Do not infer occluded items.
- Chair vs. stool rule: chair = has backrest; stool = no backrest.
- Color policy: if not clearly pure white, prefer “light gray” or “light-colored”.

Stage 4 — Sub-questions
- For each intent, write a concrete Yes/No/Uncertain question.
- Include `scope = "target_only"` for absolute intents; `scope = "compare_ref_target"` for relative intents.

Stage 5 — Q&A with Evidence
- Answer each sub-question with Yes/No/Uncertain.
- Provide `evidence_target` (required) and `evidence_ref` (only if `scope = compare_ref_target`).
- Evidence must cite specific visible cues (color/shape/position/count/text).

Stage 6 — Local Edits
- Map answers to edits:
  - Yes → KEEP the corresponding requirement and (when possible) the original sentence pattern.
  - No → REWRITE that fragment to match the TARGET; do not introduce off-image facts.
  - Uncertain → generalize/delete/flag per uncertain_policy.
- Apply the minimum necessary changes.

Stage 7 — Synthesize `text_new`
- Merge local edits into a coherent, conflict-free new edit text.
- Every requirement must be directly observable in the TARGET.
- Keep total length ≤ `length_limit` and use the same language as the input text.

Stage 8 — Verifier
- Every noun/attribute in `text_new` must be supported by a `TargetFact` id from Stage 3.
- If any token is unsupported, revise `text_new` to comply.

Stage 9 — Self-check
- Report consistency/visibility/safety notes. If issues remain, fix them and re-run the affected stages.

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

<FEWSHOT>  <!-- CONTEXT ONLY. DO NOT COPY ANY CONTENT BELOW INTO YOUR FINAL OUTPUT. -->
Example context:
- Reference image description: a brown dog outdoors among flowers.
- Target image description: a brown puppy lying inside a light-colored plush pet bed.
- Original edit text: "Brown dog sits on its bed"
- Expected new edit text: "Brown dog lies on its bed"

Example output (jsonc, not valid JSON — FOR REFERENCE ONLY):
```jsonc
{
  // example only
  "options_used": { "language": "en", "keep_sentence_patterns": true, "uncertain_policy": "generalize", "length_limit": 300 },
  "captions": {
    "reference": "A brown dog outdoors among flowers. The dog is in a garden setting.",
    "target": "A small brown puppy lying inside a light-colored plush pet bed."
  },
  "intents": [
    {"id":"I1","span":"Brown","intent":"dog color should be brown","type":"absolute","objects":["dog","color"]},
    {"id":"I2","span":"sits","intent":"dog pose should be sitting","type":"absolute","objects":["dog","pose"]},
    {"id":"I3","span":"on its bed","intent":"dog should be on its bed","type":"absolute","objects":["dog","bed","spatial_relation"]}
  ],
  "target_facts": [
    {"id":"F1","fact":"C1 is a brown dog near the center","evidence_target":"brown coat; inside a plush bed"},
    {"id":"F2","fact":"C2 is a soft pet bed","evidence_target":"cushioned, light-colored fabric surrounds the dog"},
    {"id":"F3","fact":"C1 pose is lying","evidence_target":"torso reclined, not upright"}
  ],
  "qa": [
    {"id":"I1","scope":"target_only","question":"Is the dog in the TARGET image brown?","answer":"Yes","evidence_target":"The dog's coat is uniformly brown.","evidence_ref":""},
    {"id":"I2","scope":"target_only","question":"Is the dog sitting in the TARGET image?","answer":"No","evidence_target":"The dog is lying in the bed; body reclined rather than upright.","evidence_ref":""},
    {"id":"I3","scope":"target_only","question":"Is the dog on its bed in the TARGET image?","answer":"Yes","evidence_target":"The dog is positioned inside a cushioned pet bed.","evidence_ref":""}
  ],
  "local_edits": [
    {"id":"I1","action":"keep","before":"Brown","after":"Brown","reason":"supported by F1"},
    {"id":"I2","action":"rewrite","before":"sits","after":"lies","reason":"supported by F3"},
    {"id":"I3","action":"keep","before":"on its bed","after":"on its bed","reason":"supported by F2"}
  ],
  "text_new":"Brown dog lies on its bed",
  "self_check":{"consistency_ok":true,"visibility_ok":true,"notes":"target-only facts used"}
}

"""
    # 只替换我们定义的占位符，避免 .format 与花括号冲突
    return prompt_template.replace("{modification_text}", modification_text)

def _extract_last_json_object(text: str):
    """
    使用 JSONDecoder.raw_decode 在文本中寻找最后一个完整 JSON 对象。
    能处理前后夹杂说明/示例/日志的情况。
    """
    decoder = json.JSONDecoder()
    last_obj = None
    last_span = None
    # 去掉代码块围栏，降低干扰
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # 逐个尝试从每个左花括号位置解析
    for m in re.finditer(r"\{", cleaned):
        i = m.start()
        try:
            obj, end = decoder.raw_decode(cleaned[i:])
            last_obj = obj
            last_span = (i, i + end)
        except json.JSONDecodeError:
            continue
    if last_obj is None:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return last_obj

def parse_json_response(response_text):
    """解析模型输出的JSON响应（更健壮）"""
    response_text = response_text.strip()
    # 第一招：完整解析
    try:
        parsed_json = json.loads(response_text)
        if validate_json_structure(parsed_json):
            return {"parsed_json": parsed_json, "raw_text": response_text}
    except Exception:
        pass

    # 第二招：取最后一个完整 JSON 对象
    try:
        parsed_json = _extract_last_json_object(response_text)
        if validate_json_structure(parsed_json):
            return {"parsed_json": parsed_json, "raw_text": response_text}
        else:
            return {"error": "JSON structure validation failed", "raw_text": response_text, "partial_json": parsed_json}
    except Exception as e:
        return {"error": f"JSON parsing failed: {str(e)}", "raw_text": response_text}

def validate_json_structure(parsed_json):
    """验证JSON结构是否符合预期格式"""
    required_fields = ["options_used", "intents", "target_facts", "qa", "local_edits", "text_new", "self_check"]
    if not isinstance(parsed_json, dict):
        return False
    for field in required_fields:
        if field not in parsed_json:
            logger.warning(f"缺少必要字段: {field}")
            return False
    array_fields = ["intents", "target_facts", "qa", "local_edits"]
    for field in array_fields:
        if not isinstance(parsed_json[field], list):
            logger.warning(f"字段 {field} 应该是列表类型")
            return False
    if not isinstance(parsed_json["self_check"], dict):
        logger.warning("字段 self_check 应该是字典类型")
        return False
    return True

def generate_response(model, processor, reference_image_path, hard_negative_image_path, modification_text, base_path):
    """生成模型响应"""
    ref_img_full_path = os.path.join(base_path, reference_image_path.lstrip('./'))
    neg_img_full_path = os.path.join(base_path, hard_negative_image_path.lstrip('./'))

    if not os.path.exists(ref_img_full_path):
        logger.warning(f"参考图片不存在: {ref_img_full_path}")
        return {"error": f"参考图片不存在: {ref_img_full_path}"}
    if not os.path.exists(neg_img_full_path):
        logger.warning(f"困难负样本图片不存在: {neg_img_full_path}")
        return {"error": f"困难负样本图片不存在: {neg_img_full_path}"}

    try:
        prompt_text = construct_prompt(reference_image_path, hard_negative_image_path, modification_text)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ref_img_full_path},
                    {"type": "image", "image": neg_img_full_path},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

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

        # 更稳的生成设置：不采样、低温度
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=0.1,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                repetition_penalty=1.05
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # 解析 JSON
        parsed_result = parse_json_response(output_text)
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

        response = generate_response(
            model, processor, reference_image, hard_negative_image, modification_text, base_image_path
        )

        result = {
            "sample_id": idx,
            "original_data": sample,
            "input": {
                "reference_image": reference_image,
                "hard_negative_image": hard_negative_image,
                "original_modification_text": modification_text
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
