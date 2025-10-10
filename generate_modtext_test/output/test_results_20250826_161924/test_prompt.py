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
- Do not enumerate more than 8 target facts (F1..F8). If more are possible, summarize.
- Limit intents ≤ 6, qa ≤ 8, local_edits ≤ 6.
- In counting protocol, enumerate objects up to C1..C6; then write ‘… and others (not enumerated)’.

Uniqueness & Linkage Rules:
- IDs must be unique: do not reuse F#, I#, or C#.
- No duplicate facts: do not state the same claim twice with paraphrases. Merge duplicates and keep the earliest F#.
- Normalize facts to the canonical pattern “<ObjectID>.<property|relation>=<value or OtherObjectID>”.
  Examples: C1.category=table; C1.material=wood; C1.count=4; C1.relation(front_of)=tree.
- QA must be intent-centered: generate exactly one QA item per intent (|qa| == |intents|).
- Each QA item must cite the minimal supporting facts via `fact_ids` (subset of target_facts). If none, still create QA and answer "Uncertain".

Polar Question Policy (Yes/No only):
- QA MUST use polar (yes/no) questions only. Banned tokens: how, how many, how much, how long, how far, how big, how tall, how wide, what, which, where, when, why, who, whom, whose.
- If you are about to write a wh-question, REWRITE it into a yes/no form.
- Counting: if the intent specifies an exact number N, ask “Are there exactly N <objects> … ?”. If it’s a comparison, ask “Does the TARGET have more/fewer <objects> than the REFERENCE?”.
- If evidence is insufficient for an exact count, answer “Uncertain” (do NOT switch to “how many”).


=====================
IMAGE ANCHORS (read BEFORE stages)
=====================
- The FIRST image you receive is the REFERENCE image.
- The SECOND image you receive is the TARGET image.
- In "options_used", you MUST echo the anchors:
  "image_anchors": {"reference_seen_as":"first","target_seen_as":"second"}

=====================
STAGES (run in order)
=====================

Stage 0 — Dual Image Captioning (anchored; to stabilize perception)
- Produce concise, literal captions for BOTH images with EXPLICIT labels:
  - captions.reference: MUST start with "[REFERENCE]",only focus on the REFERENCE image, followed by 3-5 short sentences describing clearly visible objects, attributes (color/material), counts, spatial relations, and readable text. No speculation.
  - captions.target:    MUST start with "[TARGET]",only focus on the TARGET image, followed by 3-5 short sentences describing clearly visible objects, attributes (color/material), counts, spatial relations, and readable text. No speculation.
- If any detail is blurry or unreadable, write "unreadable" or "uncertain" — do NOT guess.
- Additionally, in "options_used", provide a short difference list that highlights the differences between the two images :
  "caption_diff": ["<difference cue 1>", "<difference cue 2>", "<difference cue 3>"]

Stage 1 — Text Parse & Intent Decomposition (from Input edit text only; coverage REQUIRED)
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

Stage 3 — Target Facts (grounding + de-dup + index)
- From the TARGET only, list observable facts needed to test the intents.
- Counting Protocol: enumerate relevant objects as C1..Ck with coarse locations (left/center/right; front/back). Do NOT infer occluded items. Do not exceed C6; if more exist, add "… and others (not enumerated)".
- Write each fact using a canonical key-value form:
  fact: "<C#>.<property|relation>=<value or OtherC#>"
  e.g., "C1.category=table", "C1.material=wood", "C2.type=chair", "C2.backrest=true", "C1.front_of=C3".
- De-duplication: before emitting a fact, check if another fact with the same (C#, property, value) already exists; if yes, do NOT repeat—reuse its F# later via references.
- Limit target_facts to at most 8 items (F1..F8). If more apply, summarize and keep only the ones needed to answer the intents.

Stage 4 — Sub-questions (intent-anchored; yes/no only)
- Generate EXACTLY ONE sub-question per intent (|qa| == |intents|; same order). Set qa.id == intent.id.
- Question must be polar (yes/no). It MUST start with:
  • Absolute: “In the TARGET image, …”
  • Relative: “Comparing REFERENCE and TARGET, …”
- Allowed auxiliaries at the start (after the prefix): is/are/does/do/has/have/can/was/were/did.
- Disallow wh-words (how/what/which/where/when/why/who/whom/whose); if any would appear, rewrite to a yes/no form.
- Templates:
  • Existence: “In the TARGET image, is/are there <object>(s) … ?”
  • Attribute: “In the TARGET image, is the <object> <attribute> … ?”
  • Spatial: “In the TARGET image, is the <object> <relation> the <other object> … ?”
  • Pose/Action: “In the TARGET image, is the <object> <pose/action> … ?”
  • Exact count: “In the TARGET image, are there exactly N <objects> … ?”
  • Relative count: “Comparing REFERENCE and TARGET, does the TARGET have more/fewer <objects> than the REFERENCE?”
  • Direction/orientation: 
      - Absolute: “In the TARGET image, is the <object> facing left/right … ?”
      - Relative: “Comparing REFERENCE and TARGET, does the <object> face the opposite direction in the TARGET?”
- Each QA must include `fact_ids` (optional field) listing the minimal supporting F# set. If none, use [] and answer “Uncertain”.


Stage 5 — Q&A with Evidence
- Answer each sub-question with Yes/No/Uncertain.
- Provide evidence_target (required) and evidence_ref (only if scope = compare_ref_target).
- Evidence must cite specific visible cues (color/shape/position/count/body-part/gesture/text).
- Evidence must be consistent with `fact_ids` and should not introduce claims absent from the cited facts.

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
}

<FEWSHOT>  <!-- CONTEXT ONLY. DO NOT COPY ANY CONTENT BELOW INTO YOUR FINAL OUTPUT. -->
Example 1:
    Example context:
    - Reference image description: a wolf-like canine running through deep snow; snowy forest background.
    - Target image description: a medium-sized dog standing upright on a mound of snow with tree trunks directly behind it; no rock is visible.
    - Original edit text: "Dog stands in the snow in front of the rock."
    - Expected new edit text: "Dog stands in the snow in front of the tree"

    Example output (jsonc, not valid JSON — FOR REFERENCE ONLY):
    ```jsonc
    {
    "options_used": {
        "image_anchors": {"reference_seen_as": "first", "target_seen_as": "second"},
        "caption_diff": [
        "TARGET shows tree trunks directly behind the dog; no rock is visible."
        ]
    },
    "captions": {
        "reference": "[REFERENCE] A wolf-like canine plowing through deep snow with snow spray; tree trunks and a snowy forest background.",
        "target": "[TARGET] A medium-sized dog stands upright on a mound of snow with forelegs raised; several tree trunks are directly behind it; no rock is visible."
    },
    "intents": [
        {"id": "I1", "span": "Dog", "intent": "the subject is a dog", "type": "absolute", "objects": ["dog"]},
        {"id": "I2", "span": "stands", "intent": "the dog is standing", "type": "absolute", "objects": ["dog","pose"]},
        {"id": "I3", "span": "in the snow", "intent": "the dog is located in/on snow", "type": "absolute", "objects": ["snow","location"]},
        {"id": "I4", "span": "in front of the rock", "intent": "the dog is positioned in front of a rock", "type": "absolute", "objects": ["spatial_relation","rock"]}
    ],
    "target_facts": [
        {"id": "F1", "fact": "C1 is a dog near the center of the image", "evidence_target": "medium-sized dog with pointed ears centered on a snow mound"},
        {"id": "F2", "fact": "C1 is upright/standing on snow", "evidence_target": "dog on hind legs atop snow mound; torso vertical; forelegs raised"},
        {"id": "F3", "fact": "Snow covers the ground", "evidence_target": "white snow surface across foreground and around the dog"},
        {"id": "F4", "fact": "Tree trunks are directly behind C1; no rock is visible", "evidence_target": "vertical barked trunks behind dog; no distinct rock shape near the dog"}
    ],
    "qa": [
        {"id": "I1", "scope": "target_only", "question": "In the TARGET image, is the subject a dog?", "answer": "Yes", "evidence_target": "dog-like features (muzzle, ears) centered on snow", "evidence_ref": ""},
        {"id": "I2", "scope": "target_only", "question": "In the TARGET image, is the dog standing?", "answer": "Yes", "evidence_target": "upright on hind legs with body vertical", "evidence_ref": ""},
        {"id": "I3", "scope": "target_only", "question": "In the TARGET image, is the dog in the snow?", "answer": "Yes", "evidence_target": "dog positioned on a mound of snow; snow all around", "evidence_ref": ""},
        {"id": "I4", "scope": "target_only", "question": "In the TARGET image, is the dog in front of a rock?", "answer": "No", "evidence_target": "no rock visible; instead tree trunks directly behind the dog", "evidence_ref": ""}
    ],
    "local_edits": [
        {"id": "I1", "action": "keep", "before": "Dog", "after": "Dog", "reason": "supported by F1"},
        {"id": "I2", "action": "keep", "before": "stands", "after": "stands", "reason": "supported by F2"},
        {"id": "I3", "action": "keep", "before": "in the snow", "after": "in the snow", "reason": "supported by F3"},
        {"id": "I4", "action": "rewrite", "before": "in front of the rock", "after": "in front of the tree", "reason": "F4 shows tree trunks behind the dog and no rock present"}
    ],
    "text_new": "Dog stands in the snow in front of the tree",
    }

Example 2:
    Example context:
    - Reference image description: a short-haired brown dog (collar on) sitting outdoors.
    - Target image description: a long-haired cream/beige dog lying on a light-colored tabletop/bench indoors.
    - Original edit text: "Shows a dog of a different breed of the same fur color standing on a white table."
    - Expected new edit text: "Shows a dog of a different breed with light cream fur lying on a light-colored table."

    Example output (jsonc, not valid JSON — FOR REFERENCE ONLY):
    ```jsonc
    {
    // example only
    "options_used": {
        "image_anchors": {"reference_seen_as": "first", "target_seen_as": "second"},
        "caption_diff": [
        "REFERENCE: short-haired brown dog; TARGET: long-haired cream-colored dog.",
        "REFERENCE pose: sitting; TARGET pose: lying.",
        "TARGET surface is light-colored, not clearly white."
        ]
    },
    "captions": {
        "reference": "[REFERENCE] A short-haired brown dog with a collar sits outdoors; tongue out.",
        "target": "[TARGET] A long-haired cream/beige dog lies on a light-colored tabletop/bench indoors."
    },
    "intents": [
        {"id": "I1", "span": "a dog of a different breed", "intent": "the target should depict a dog of a different breed than the reference", "type": "relative", "objects": ["dog","breed_difference"]},
        {"id": "I2", "span": "of the same fur color", "intent": "the target dog should have the same fur color as the reference dog", "type": "relative", "objects": ["dog","fur color"]},
        {"id": "I3", "span": "standing", "intent": "the dog’s pose should be standing", "type": "absolute", "objects": ["dog","pose"]},
        {"id": "I4", "span": "on a white table", "intent": "the dog should be on a white table", "type": "absolute", "objects": ["table","color=white","spatial_relation"]}
    ],
    "target_facts": [
        {"id": "F1", "fact": "C1 is a long-haired dog lying on a tabletop/bench", "evidence_target": "dog stretched out with legs tucked; body on flat raised surface"},
        {"id": "F2", "fact": "C1 fur color is light cream/beige", "evidence_target": "pale cream coat across body and ears"},
        {"id": "F3", "fact": "The surface is light-colored but not clearly white", "evidence_target": "tabletop shows a faint green/gray tint"},
        {"id": "F4", "fact": "Reference dog differs in appearance from target dog", "evidence_target": "long hair, drooping ears", "evidence_ref": "short hair, small upright ears"},
        {"id": "F5", "fact": "Reference fur color is brown while target is cream", "evidence_target": "cream/beige coat", "evidence_ref": "medium brown coat"}
    ],
    "qa": [
        {"id": "I1", "scope": "compare_ref_target", "question": "Comparing REFERENCE and TARGET, is the target dog a different breed from the reference?", "answer": "Yes", "evidence_target": "long hair and drooping ears", "evidence_ref": "short hair and small ears"},
        {"id": "I2", "scope": "compare_ref_target", "question": "Comparing REFERENCE and TARGET, does the target dog have the same fur color as the reference?", "answer": "No", "evidence_target": "target is light cream/beige", "evidence_ref": "reference is brown"},
        {"id": "I3", "scope": "target_only", "question": "In the TARGET image, is the dog standing?", "answer": "No", "evidence_target": "the dog is lying on the surface"},
        {"id": "I4", "scope": "target_only", "question": "In the TARGET image, is the dog on a white table?", "answer": "No", "evidence_target": "table is light-colored but not clearly white"}
    ],
    "local_edits": [
        {"id": "I1", "action": "keep", "before": "a dog of a different breed", "after": "a dog of a different breed", "reason": "supported by F4"},
        {"id": "I2", "action": "rewrite", "before": "of the same fur color", "after": "with light cream fur", "reason": "target color per F2; differs from reference per F5"},
        {"id": "I3", "action": "rewrite", "before": "standing", "after": "lying", "reason": "pose per F1"},
        {"id": "I4", "action": "rewrite", "before": "on a white table", "after": "on a light-colored table", "reason": "surface is light but not clearly white per F3"}
    ],
    "text_new": "Shows a dog of a different breed with light cream fur lying on a light-colored table.",
    }

Example 3:
    Example context:
    - Reference image description: a small rough-coated brown dog stands on a rock in a field, facing the camera.
    - Target image description: a rough-coated brown dog stands on green grass in side profile, head facing left, tail raised.
    - Original edit text: "Make the dog lye down and look to right."
    - Expected new edit text: "Make the dog stand and look to the left."

    Example output (jsonc, not valid JSON — FOR REFERENCE ONLY):
    ```jsonc
    {
    // example only
    "options_used": {
        "image_anchors": {"reference_seen_as": "first", "target_seen_as": "second"},
        "caption_diff": [
        "REFERENCE: dog front-facing on rocks; TARGET: dog side profile on grass.",
        "REFERENCE pose: standing facing camera; TARGET pose: standing with head left.",
        "TARGET shows leftward gaze; not lying down."
        ]
    },
    "captions": {
        "reference": "[REFERENCE] A small brown terrier stands on a rock, facing the camera; fields in the background.",
        "target": "[TARGET] A brown terrier stands on grass in profile with its head facing left and tail raised."
    },
    "intents": [
        {"id": "I1", "span": "lye down", "intent": "the dog should be lying down", "type": "absolute", "objects": ["dog","pose=lying"]},
        {"id": "I2", "span": "look to right", "intent": "the dog should look to the right", "type": "absolute", "objects": ["dog","gaze=right"]},
        {"id": "I3", "span": "Make the dog", "intent": "the subject should be a dog", "type": "absolute", "objects": ["dog","presence"]}
    ],
    "target_facts": [
        {"id": "F1", "fact": "C1 is a dog standing on grass", "evidence_target": "all four legs extended; body upright"},
        {"id": "F2", "fact": "C1 faces left", "evidence_target": "head and muzzle oriented left; tail to the right"},
        {"id": "F3", "fact": "C1 is not lying down", "evidence_target": "no contact of torso with ground; stance posture"}
    ],
    "qa": [
        {"id": "I1", "scope": "target_only", "question": "In the TARGET image, is the dog lying down?", "answer": "No", "evidence_target": "standing posture with legs extended", "evidence_ref": ""},
        {"id": "I2", "scope": "target_only", "question": "In the TARGET image, is the dog looking to the right?", "answer": "No", "evidence_target": "head and muzzle oriented to the left", "evidence_ref": ""},
        {"id": "I3", "scope": "target_only", "question": "In the TARGET image, is the subject a dog?", "answer": "Yes", "evidence_target": "terrier-like features, tail and muzzle visible", "evidence_ref": ""}
    ],
    "local_edits": [
        {"id": "I1", "action": "rewrite", "before": "lye down", "after": "stand", "reason": "dog is standing per F1/F3"},
        {"id": "I2", "action": "rewrite", "before": "look to right", "after": "look to the left", "reason": "gaze is leftward per F2"},
        {"id": "I3", "action": "keep", "before": "dog", "after": "dog", "reason": "subject is a dog per F1"}
    ],
    "text_new": "Make the dog stand and look to the left.",
    }

Example 4:
    Example context:
    - Reference image description: a goose standing on grass, body and head oriented to the left.
    - Target image description: a goose in the water (swimming) with its head oriented to the left; beak open.
    - Original edit text: "make it swim and face the other direction"
    - Expected new edit text: "make it swim and face left"

    Example output (jsonc, not valid JSON — FOR REFERENCE ONLY):
    ```jsonc
    {
    // example only
    "options_used": {
        "language": "en",
        "keep_sentence_patterns": true,
        "uncertain_policy": "generalize",
        "length_limit": 180,
        "image_anchors": {"reference_seen_as": "first", "target_seen_as": "second"},
        "caption_diff": [
        "REFERENCE: goose stands on grass; TARGET: goose is in water (swimming).",
        "Both images face left (same direction).",
        "Environment changes from grass field to watery surface."
        ]
    },
    "captions": {
        "reference": "[REFERENCE] A goose stands on green grass, facing left.",
        "target": "[TARGET] A goose is in the water, swimming and facing left with its beak open."
    },
    "intents": [
        {"id":"I1","span":"make it swim","intent":"the subject should be swimming (in water)","type":"absolute","objects":["goose","action=swim","water"]},
        {"id":"I2","span":"face the other direction","intent":"the subject should face the opposite direction compared to the reference","type":"relative","objects":["goose","orientation"]}
    ],
    "target_facts": [
        {"id":"F1","fact":"C1 goose is in water and swimming","evidence_target":"body afloat; ripples around chest and neck"},
        {"id":"F2","fact":"C1 faces left in the target","evidence_target":"head and beak oriented left"},
        {"id":"F3","fact":"Reference goose also faces left","evidence_target":"—","evidence_ref":"head and body oriented left on grassy ground"}
    ],
    "qa": [
        {"id":"I1","scope":"target_only","question":"In the TARGET image, is the goose swimming (in water)?","answer":"Yes","evidence_target":"goose afloat in water with ripples"},
        {"id":"I2","scope":"compare_ref_target","question":"Comparing REFERENCE and TARGET, does the goose face the opposite direction in the TARGET?","answer":"No","evidence_target":"faces left","evidence_ref":"also faces left (same direction)"}
    ],
    "local_edits": [
        {"id":"I1","action":"keep","before":"make it swim","after":"make it swim","reason":"satisfied by F1"},
        {"id":"I2","action":"rewrite","before":"face the other direction","after":"face left","reason":"orientation matches left in TARGET; not opposite per F2 & F3"}
    ],
    "text_new":"make it swim and face left",
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
