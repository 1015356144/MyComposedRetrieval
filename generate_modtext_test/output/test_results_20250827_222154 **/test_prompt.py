import json
import os
import torch
import re
import ast
import gc
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

# ==================== 内存管理 ====================
def clear_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 获取并记录当前GPU内存使用情况
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.info(f"GPU内存使用情况 - 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ==================== 数据加载 ====================
def load_test_data(json_file_path):
    logger.info(f"正在加载测试数据: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"已加载 {len(data)} 个测试样例")
    return data

# ==================== 分段式 Prompt（已弃用） ====================
def construct_prompt(reference_image_path, hard_negative_image_path, modification_text):
    tem_prompt =' '
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

# ==================== 键值对解析工具 ====================
def _parse_keyvals_line(line: str) -> dict:
    """
    解析类似：
    I1 | span="..." | intent="..." | type=absolute | objects=[dog, color] | note="..."
    I1 | scope=target_only | Q="..." | A=Yes | evidence_target="..." | evidence_ref=""
    I1 | action=keep | before="..." | after="..." | reason="..."
    """
    parts = [p.strip() for p in line.split("|")]
    out = {}
    if not parts:
        return out
    # 第一个片段通常以 I# 开头
    id_m = re.match(r'^(I)\s*(\d+)', parts[0], flags=re.I)
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

# ==================== 新的解析器 - 适配当前prompt格式 ====================
def parse_staged_response(response_text: str):
    """
    解析基于阶段标题的响应：
    1. Generate captions
    2. Decompose intents
    3. Generate polar questions  
    4. Answer questions via image comparison
    5. Local edits
    6. Generate text_new (New modification text)
    
    返回结构化的 dict 和诊断信息
    """
    raw = _preclean_text(response_text)
    
    # 定义阶段标题的正则表达式（匹配实际输出格式）
    stage_patterns = {
        "captions": re.compile(r'Generate\s+Captions?\s*:?\s*$', re.I | re.M),
        "intents": re.compile(r'Decompose\s+Intents?\s*:?\s*$', re.I | re.M),
        "questions": re.compile(r'Generate\s+Polar\s+Questions?\s*:?\s*$', re.I | re.M),
        "answers": re.compile(r'Answer\s+Questions\s+via\s+Image\s+Comparison?\s*:?\s*$', re.I | re.M),
        "edits": re.compile(r'Local\s+Edits?\s*:?\s*$', re.I | re.M),
        "text_new": re.compile(r'(?:Generate\s+Text_new|New\s+modification\s+text)\s*:?\s*$', re.I | re.M)
    }
    
    result = {
        "captions": {"reference": "", "target": ""},
        "intents": [],
        "questions": [],
        "answers": [],
        "local_edits": [],
        "text_new": ""
    }
    
    # 将文本按阶段分割
    stage_blocks = _split_into_stages(raw, stage_patterns)
    
    # 解析各个阶段
    _parse_captions_block(stage_blocks.get("captions", ""), result)
    _parse_intents_block(stage_blocks.get("intents", ""), result)
    _parse_questions_block(stage_blocks.get("questions", ""), result)
    _parse_answers_block(stage_blocks.get("answers", ""), result)
    _parse_edits_block(stage_blocks.get("edits", ""), result)
    _parse_text_new_block(stage_blocks.get("text_new", ""), result)
    
    # 生成诊断信息
    meta = _generate_diagnostics(result)
    
    return {
        "parsed_stages": result,
        "raw_text": response_text,
        "parsed_meta": meta,
        "stage_blocks": stage_blocks  # 调试用
    }

def _split_into_stages(text: str, stage_patterns: dict) -> dict:
    """将文本按阶段标题分割成块"""
    stage_blocks = {}
    
    # 找到所有阶段标题的位置
    stage_positions = []
    for stage_name, pattern in stage_patterns.items():
        match = pattern.search(text)
        if match:
            stage_positions.append((match.start(), match.end(), stage_name))
    
    # 按位置排序
    stage_positions.sort()
    
    # 提取每个阶段的内容
    for i, (start_pos, end_pos, stage_name) in enumerate(stage_positions):
        # 确定当前阶段的结束位置
        if i + 1 < len(stage_positions):
            next_start = stage_positions[i + 1][0]
            content = text[end_pos:next_start].strip()
        else:
            content = text[end_pos:].strip()
        
        stage_blocks[stage_name] = content
    
    return stage_blocks

def _parse_captions_block(block: str, result: dict):
    """解析 Generate captions 阶段"""
    lines = block.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 查找 Reference: ... 或 Target: ...
        ref_match = re.match(r'(?:-\s*)?Reference:\s*(.+)', line, re.I)
        if ref_match:
            result["captions"]["reference"] = ref_match.group(1).strip()
            continue

        tgt_match = re.match(r'(?:-\s*)?Target:\s*(.+)', line, re.I)
        if tgt_match:
            result["captions"]["target"] = tgt_match.group(1).strip()
            continue
        
        # 也支持没有前缀的格式
        if "[REFERENCE]" in line:
            content = line.split("[REFERENCE]", 1)[1].strip()
            if content:
                result["captions"]["reference"] = content
        elif "[TARGET]" in line:
            content = line.split("[TARGET]", 1)[1].strip()
            if content:
                result["captions"]["target"] = content

def _parse_intents_block(block: str, result: dict):
    """解析 Decompose intents 阶段"""
    lines = block.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        
        # 解析格式：I# | span="..." | intent="..." | type=absolute|relative | objects=[...] | note="..."
        intent_data = _parse_keyvals_line(line)
        if intent_data.get("id") and intent_data.get("id").startswith("I"):
            intent = {
                "id": intent_data.get("id", ""),
                "span": intent_data.get("span", ""),
                "intent": intent_data.get("intent", ""),
                "type": intent_data.get("type", ""),
                "objects": intent_data.get("objects", []),
                "note": intent_data.get("note", "")
            }
            result["intents"].append(intent)

def _parse_questions_block(block: str, result: dict):
    """解析 Generate polar questions 阶段"""
    lines = block.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        
        # 解析格式：I# | scope=target_only|compare_ref_target | Q="...?"
        question_data = _parse_keyvals_line(line)
        if question_data.get("id") and question_data.get("id").startswith("I"):
            question = {
                "id": question_data.get("id", ""),
                "scope": question_data.get("scope", ""),
                "question": question_data.get("Q", question_data.get("q", ""))
            }
            result["questions"].append(question)

def _parse_answers_block(block: str, result: dict):
    """解析 Answer questions via image comparison 阶段"""
    lines = block.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        
        # 解析格式：I# | A=Yes|No|Uncertain | evidence_target="..." | evidence_ref="..."
        answer_data = _parse_keyvals_line(line)
        if answer_data.get("id") and answer_data.get("id").startswith("I"):
            answer = {
                "id": answer_data.get("id", ""),
                "answer": answer_data.get("A", answer_data.get("a", "")),
                "evidence_target": answer_data.get("evidence_target", ""),
                "evidence_ref": answer_data.get("evidence_ref", "")
            }
            result["answers"].append(answer)

def _parse_edits_block(block: str, result: dict):
    """解析 Local edits 阶段"""
    lines = block.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        
        # 解析格式：I# | action=keep|rewrite|generalize|delete | before="..." | after="..." | reason="..."
        edit_data = _parse_keyvals_line(line)
        if edit_data.get("id") and edit_data.get("id").startswith("I"):
            edit = {
                "id": edit_data.get("id", ""),
                "action": edit_data.get("action", ""),
                "before": edit_data.get("before", ""),
                "after": edit_data.get("after", ""),
                "reason": edit_data.get("reason", "")
            }
            result["local_edits"].append(edit)

def _parse_text_new_block(block: str, result: dict):
    """解析 Generate text_new 阶段"""
    lines = block.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        
        # 查找 text_new="..." 格式
        text_new_match = re.match(r'text_new\s*=\s*"([^"]*)"', line, re.I)
        if text_new_match:
            result["text_new"] = text_new_match.group(1).strip()
            break
        
        # 也支持不带引号的格式
        if line.lower().startswith('text_new'):
            content = re.sub(r'^text_new\s*=?\s*', '', line, flags=re.I).strip()
            if content:
                # 去掉可能的引号
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                result["text_new"] = content
                break
        
        # 如果没有找到text_new=格式，直接取第一行非空内容作为结果
        if line and not line.startswith('text_new'):
            result["text_new"] = line
            break

def _generate_diagnostics(result: dict) -> dict:
    """生成诊断信息"""
    intents = result.get("intents", [])
    questions = result.get("questions", [])
    answers = result.get("answers", [])
    edits = result.get("local_edits", [])
    
    diag = {
        "counts": {
            "intents": len(intents),
            "questions": len(questions), 
            "answers": len(answers),
            "edits": len(edits)
        },
        "alignment": {
            "all_counts_match": len(intents) == len(questions) == len(answers) == len(edits),
            "ids_sequence_correct": True,
            "scope_type_consistency": True
        },
        "question_format": {
            "wh_word_violations": [],
            "prefix_violations": []
        },
        "completeness": {
            "has_captions": bool(result["captions"]["reference"] and result["captions"]["target"]),
            "has_text_new": bool(result.get("text_new", "").strip()),
            "missing_stages": []
        }
    }
    
    # 检查ID序列的正确性
    intent_ids = [item.get("id", "") for item in intents]
    question_ids = [item.get("id", "") for item in questions]
    answer_ids = [item.get("id", "") for item in answers]
    edit_ids = [item.get("id", "") for item in edits]
    
    if not (intent_ids == question_ids == answer_ids == edit_ids):
        diag["alignment"]["ids_sequence_correct"] = False
    
    # 检查scope和type的一致性
    for i in range(min(len(intents), len(questions))):
        intent_type = intents[i].get("type", "")
        question_scope = questions[i].get("scope", "")
        
        if intent_type == "absolute" and question_scope != "target_only":
            diag["alignment"]["scope_type_consistency"] = False
        elif intent_type == "relative" and question_scope != "compare_ref_target":
            diag["alignment"]["scope_type_consistency"] = False
    
    # 检查问题格式
    wh_words_pattern = re.compile(r'\b(how|what|which|where|when|why|who|whom|whose)\b', re.I)
    abs_prefix_pattern = re.compile(r'^\s*In the TARGET image,\s*(is|are|does|do|has|have|can|was|were|did)\b', re.I)
    rel_prefix_pattern = re.compile(r'^\s*Comparing REFERENCE and TARGET,\s*(is|are|does|do|has|have|can|was|were|did)\b', re.I)
    
    for question in questions:
        q_id = question.get("id", "")
        q_text = question.get("question", "")
        q_scope = question.get("scope", "")
        
        if wh_words_pattern.search(q_text):
            diag["question_format"]["wh_word_violations"].append(q_id)
        
        if q_scope == "target_only" and not abs_prefix_pattern.search(q_text):
            diag["question_format"]["prefix_violations"].append(q_id)
        elif q_scope == "compare_ref_target" and not rel_prefix_pattern.search(q_text):
            diag["question_format"]["prefix_violations"].append(q_id)
    
    # 检查完整性
    if not result["captions"]["reference"]:
        diag["completeness"]["missing_stages"].append("captions_reference")
    if not result["captions"]["target"]:
        diag["completeness"]["missing_stages"].append("captions_target")
    if not intents:
        diag["completeness"]["missing_stages"].append("intents")
    if not questions:
        diag["completeness"]["missing_stages"].append("questions")
    if not answers:
        diag["completeness"]["missing_stages"].append("answers")
    if not edits:
        diag["completeness"]["missing_stages"].append("edits")
    if not result.get("text_new", "").strip():
        diag["completeness"]["missing_stages"].append("text_new")
    
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
        system_prompt = """
        You are a multimodal edit auditor. You receive TWO images (Picture 1 = REFERENCE, Picture 2 = TARGET) and ONE input edit text.
        Your only goal: minimally rewrite the edit text so it is true in the TARGET image.

        ##Output format contract (STRICT — no extra sections, no code fences, no JSON)
        - Output exactly these 6 sections(stages), in this order, with the exact section headers shown below.
        - Use ASCII quotes (") in key/value fields. Keep IDs unique (I1, I2, ...).
        - If any field is not applicable, write uncertain.

        ##Guidelines on determining the response <Response>
        - Responses include the Generate Captions, Decompose Intents, Generate Polar Questions, Answer Questions via Image Comparison, Local Edits, and Generate Text_new:
        - Every stage should be included in the response and can not be empty.
        Generate Captions:
        Reference: [REFERENCE] ...
        Target: [TARGET] ...

        Decompose Intents:
        I1 | span="..." | intent="..." | type=absolute|relative | objects=[...] | note="..."
        I2 | span="..." | intent="..." | type=absolute|relative | objects=[...] | note="..."

        Generate Polar Questions:
        I1 | scope=target_only|compare_ref_target | Q="In the TARGET image, ...?"
        I2 | scope=target_only|compare_ref_target | Q="Comparing REFERENCE and TARGET, ...?"

        Answer Questions via Image Comparison:
        I1 | A=Yes|No|Uncertain | evidence_target="..." | evidence_ref=""
        I2 | A=Yes|No|Uncertain | evidence_target="..." | evidence_ref="..."

        Local Edits:
        I1 | action=keep|rewrite|generalize|delete | before="..." | after="..." | reason="..."
        I2 | action=keep|rewrite|generalize|delete | before="..." | after="..." | reason="..."

        Generate Text_new:
        text_new="..."
        """
        #---Example 1---
        fs_input1 = """
        <Input>
            "REFERENCE": <image_path>,
            "TARGET": <image_path>,
            "Input edit text": "Shows a dog of a different breed of the same fur color standing on a white table."
            """
        fs_output1 = """
        Generate Captions:
            Reference: [REFERENCE] A short-haired brown dog with a collar sits outdoors on a paved area; tongue out; background is blurred urban/stone texture.
            Target: [TARGET] A long-haired cream/beige dog lies stretched out on a light-colored tabletop/bench indoors; floppy ears; long fur drapes over the edge.

        Decompose Intents:
            I1 | span="a dog of a different breed" | intent="target dog breed differs from REFERENCE" | type=relative | objects=["dog","breed_difference"] | note="requires comparing REFERENCE vs TARGET appearance"
            I2 | span="of the same fur color" | intent="target dog has same fur color as REFERENCE" | type=relative | objects=["dog","fur_color"] | note="color equality is a comparison claim"
            I3 | span="standing" | intent="the dog should be standing" | type=absolute | objects=["dog","pose"] | note="verify from target only"
            I4 | span="on a white table" | intent="the dog is on a white table" | type=absolute | objects=["dog","table","table_color=white"] | note="presence+color in target only"

        Generate Polar Questions:
            I1 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, is the dog in the TARGET a different breed than the dog in the REFERENCE?"
            I2 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, does the TARGET dog have the same fur color as the REFERENCE dog?"
            I3 | scope=target_only | Q="In the TARGET image, is the dog standing?"
            I4 | scope=target_only | Q="In the TARGET image, is the dog on a white table?"

        Answer Questions via Image Comparison:
            I1 | A=Yes | evidence_target="long hair, floppy ears, different head shape" | evidence_ref="short coat, upright ears"
            I2 | A=No | evidence_target="fur appears light cream/beige" | evidence_ref="fur appears medium brown"
            I3 | A=No | evidence_target="body and legs lying on the surface; not upright" | evidence_ref=""
            I4 | A=No | evidence_target="table surface is light-colored but not clearly white" | evidence_ref=""

        Local Edits:
            I1 | action=keep | before="a dog of a different breed" | after="a dog of a different breed" | reason="confirmed by comparison"
            I2 | action=rewrite | before="of the same fur color" | after="with light cream fur" | reason="target color differs from REFERENCE"
            I3 | action=rewrite | before="standing" | after="lying" | reason="target pose is lying"
            I4 | action=rewrite | before="on a white table" | after="on a light-colored table" | reason="color not clearly white"

        Generate Text_new:
            text_new="Shows a dog of a different breed with light cream fur lying on a light-colored table."
        """
        #---Example 2---
        fs_input2 = """
        <Input>
            "REFERENCE": <image_path>,
            "TARGET": <image_path>,
            "Input edit text": "Smaller dog, no background" 
        """

        fs_output2 = """
        Generate Captions:
            Reference: [REFERENCE] A slender grey dog stands on green grass in a yard. A wooden fence and trees are visible behind it. Full body is shown.
            Target: [TARGET] A close-up of a white-and-tan spaniel face with long ears. The background is pink like a poster, with white headline text at the top.

        Decompose Intents:
            I1 | span="Smaller dog" | intent="dog should appear smaller than in the REFERENCE" | type=relative | objects=["dog","size in frame"] | note="relative change of apparent size"
            I2 | span="no background" | intent="there should be no visible background (plain/transparent)" | type=absolute | objects=["background","emptiness/plain"] | note="absolute requirement about background presence"

        Generate Polar Questions:
            I1 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, does the dog appear smaller in the TARGET image than in the REFERENCE?"
            I2 | scope=target_only | Q="In the TARGET image, is there no background visible behind the dog?"

        Answer Questions via Image Comparison:
            I1 | A=No | evidence_target="the dog is a tight head close-up occupying most of the frame" | evidence_ref="full-body dog occupies a smaller portion of the frame"
            I2 | A=No | evidence_target="pink poster background with headline text is clearly visible" | evidence_ref=""

        Local Edits:
            I1 | action=rewrite | before="Smaller dog" | after="dog" | reason="target shows enlarged."
            I2 | action=rewrite | before="no background" | after="on a pink background with text" | reason="background is pink with headline text"

        Generate Text_new:
            text_new="dog on a pink background with text"
        """
        #---Example 3---
        fs_input3 = """
        <Input>
            "REFERENCE": <image_path>,
            "TARGET": <image_path>,
            "Input edit text": "Add two dogs."
        """

        fs_output3 = """
        Generate Captions:
            Reference: [REFERENCE] two white puppies are playing in the snow.
            Target: [TARGET] Three white dogs are playing on a snowy field.

        Decompose Intents:
            I1 | span="Add two dogs" | intent="add two dogs" | type=relative | objects=["dog"] | note="relative requirement"

        Generate Polar Questions:
            I1 | scope=compare_ref_target | Q="Comparing REFERENCE and TARGET, does the TARGET image have two more dogs than the REFERENCE?"

        Answer Questions via Image Comparison:
            I1 | A=No | evidence_target="three white dogs are playing on a snowy field" | evidence_ref="two white puppies are playing in the snow"

        Local Edits:
            I1 | action=rewrite | before="Add two dogs" | after="Add a dog" | reason="target has one more dogs than REFERENCE"

        Generate Text_new:
            text_new="Add a dog"
        """
        input_text=f"""
        ##Global rules (read carefully)
            -Image roles: The Picture 1 is REFERENCE (original), the Picture 2 is TARGET (to match).
            - Grounding:
                - Absolute requirements use TARGET-only evidence.
                - Relative requirements compare REFERENCE vs TARGET (e.g., “different breed”, “face the other direction”, “add/remove/replace/move”).
            - No speculation: If something is blurry/occluded/unreadable, say “uncertain” and do not guess.
            - One QA per intent: The number of QAs must equal the number of atomic intents, with the same IDs and order (|qa| == |intents|; I1→I1, I2→I2, …).
            - Polar questions only: QA must be yes/no questions (general questions). Banned words: how, how many, how much, how long, how far, how big, how tall, how wide, what, which, where, when, why, who, whom, whose.
            - Prefixes for QA:
                - Absolute: start with In the TARGET image, ...
                - Relative: start with Comparing REFERENCE and TARGET, ...
                - Allowed auxiliaries immediately after the prefix: is/are/does/do/has/have/can/was/were/did.
            - Keep it minimal: Keep the final rewrite concise, grammatical, and directly supported by TARGET evidence.

        <Input>
            "REFERENCE": Picture 1,
            "TARGET": Picture 2,
            "Input edit text": {modification_text}
        
        =====================
        STAGES (run in order)
        =====================
        ##Generate Captions:
        - You must not refer to the input edit text in this stage.
        - You must mention their unique features in their own captitons.
        - Do not reuse sentences across Reference and Target. If a sentence would be identical, rewrite it to include a detail unique to that image, or write "uncertain".
        - Tag mapping: "Reference" ALWAYS refers to Picture 1; "Target" ALWAYS refers to Picture 2.
        - Reference: [REFERENCE] only describe the reference image(REFERENCE, Picture 1) in 3-5 literal sentences. Be thorough and accurate; list visible objects, attributes (color/material), counts, spatial relations, and readable text. Do not mention TARGET here. If unreadable, say “uncertain”.
        - Target: [TARGET] only describe the target image(TARGET, Picture 2) in 3-5 literal sentences. Same constraints; do not mention REFERENCE here.

        ##Decompose Intents(only use edit text:{modification_text}):
        - One line per atomic intent, covering all content-bearing phrases in the input edit text (split by commas/“and/then”/parentheses).
        - you must only decompose the input edit text, do not refer to any information from the REFERENCE or TARGET.
        - Format (one per line):
            I# | span="..." | intent="..." | type=absolute|relative | objects=[... ] | note="one-line reason"
        - Hints:
            - Treat parenthetical modifiers as their own intents if they encode states/gestures/poses (e.g., “(fist closed)”).
            - Count words exactly (“a/one/two/…”) and keep them in span.
            - Max 6 intents; if more, merge logically but keep atomicity for verifiable units.

        ##Generate Polar Questions:
        - Create exactly one yes/no question per intent, same order and same ID.
        - Format (one per line):
            I# | scope=target_only|compare_ref_target | Q="...?"
        - Rules:
            - If type=absolute → scope=target_only and start with In the TARGET image, ...
            - If type=relative → scope=compare_ref_target and start with Comparing REFERENCE and TARGET, ...
            - Use only allowed auxiliaries after the prefix; no wh-words.

        ##Answer Questions via Image Comparison:
        - Answer each question (same IDs and order) with Yes/No/Uncertain and brief evidence based on the TARGET image(Picture 2).
        - Carefully consider the visibility of the object in the TARGET image(Picture 2),and only when the evidence is clear, you can answer "Yes".
        - Format (one per line):
            I# | A=Yes|No|Uncertain | evidence_target="..." | evidence_ref="..."
        - every question should be answered, and the answer should be based on the TARGET image only.
        - Notes:
            - If scope=target_only, fill evidence_ref="".
            - If scope=compare_ref_target, you can refer to the REFERENCE image for relative information, but the answer should be based on the TARGET image only.
            - If visibility is insufficient, choose Uncertain.

        ##Local Edits:
        - Map each intent to a local edit: Yes → keep, No → rewrite to match TARGET.
        - Format (one per line):
            I# | action=keep|rewrite|generalize|delete | before="(original span or clause)" | after="(new fragment)" | reason="(short why)"

        ##Generate Text_new:
        - Output a single fluent sentence (or two short clauses) by merging all local edits with action=keep or rewrite (skip delete; generalize if you marked generalize).
        - Format (one per line):
            text_new="..."
        - Rules:
            - Please keep all intents that you answered "Yes" in the local edits.
            - Use the TARGET image as the evidence source.
            - do not contain 'REFERENCE' or 'TARGET' in the text_new.
            - if rewrite, please rewrite it with target image's that feature.(e.g. "four dogs" but in target image only two dogs -> "two dogs")
            - try to keep the original format of the input edit text.(e.g. "a bird of a yellow color" -> "a bird of blue color")
        """
        content = [
            {"type": "image", "image": ref_img_full_path},
            {"type": "image", "image": neg_img_full_path},
            {"type": "text", "text": input_text}
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            #---Example 1---
            {"role": "user", "content": fs_input1},
            {"role": "assistant", "content": fs_output1},
            #---Example 2---
            {"role": "user", "content": fs_input2},
            {"role": "assistant", "content": fs_output2},
            #---Example 3---
            {"role": "user", "content": fs_input3},
            {"role": "assistant", "content": fs_output3},
            #---User Prompt---
            {"role": "user", "content": content}]

        # 构造模型输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,add_vision_id=True)
        # print(text)
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
                max_new_tokens=2048,  # 分段文本足够
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,  # 使用缓存提高效率
                temperature=1.0,  # 稳定参数
            )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        # 解析分段文本
        parsed = parse_staged_response(output_text)

        if "parsed_stages" not in parsed:
            logger.warning(f"解析失败，原始输出片段: {output_text[:600]}")
        
        # 清理GPU内存
        del inputs, generated_ids, generated_ids_trimmed
        clear_gpu_memory()
        
        return parsed

    except Exception as e:
        logger.error(f"生成响应时出错: {str(e)}")
        # 即使出错也要清理内存
        clear_gpu_memory()
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
    
    # 初始内存清理
    clear_gpu_memory()
    
    json_file_path = "/home/guohaiyun/yangtianyu/MyComposedRetrieval/generate_modtext_test/hard_negatives_selected.json"
    base_image_path = "/home/guohaiyun/yty_data/CIRR"

    model, processor = load_model_and_processor()
    
    # 模型加载后再次清理
    clear_gpu_memory()
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

        # 定期内存清理
        if (idx + 1) % 5 == 0:
            logger.info(f"执行定期内存清理 (样例 {idx + 1})")
            clear_gpu_memory()

        if (idx + 1) % 10 == 0:
            temp_output_file = os.path.join(output_dir, f"temp_results_{idx + 1}.json")
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存前 {idx + 1} 个样例的临时结果")
            # 在保存后进行更彻底的内存清理
            logger.info("执行深度内存清理...")
            clear_gpu_memory()

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
            summary["question_format"] = meta.get("question_format", {})
            summary["completeness"] = meta.get("completeness", {})
                
            # 统计解析到的各阶段数量
            summary["parsed_counts"] = {
                "has_captions": bool(ps.get("captions", {}).get("reference") and ps.get("captions", {}).get("target")),
                "intents_count": len(ps.get("intents", [])),
                "questions_count": len(ps.get("questions", [])),
                "answers_count": len(ps.get("answers", [])),
                "edits_count": len(ps.get("local_edits", [])),
                "text_new_present": bool(ps.get("text_new", "").strip())
            }
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
