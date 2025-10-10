# -*- coding: utf-8 -*-
"""
Prompt builder for generic foundation models (non-Qwen, non-LLaVA)
"""

from typing import Dict
from PIL import Image


def build_generic_inputs(processor, device, ref_image_path: str, tgt_image_path: str, original_text: str) -> Dict:
    """
    Build inputs for a generic VLM model.
    Returns a dict suitable for fm.generate(**inputs).
    """

    prompt = f"""
We are comparing two images: reference and target.
The original modification description is: "{original_text}".

Please rephrase or create a new modification text ("text_new") that
naturally describes the transformation from reference to target.
Keep it simple and retrieval-friendly.
    """.strip()

    ref_img = Image.open(ref_image_path).convert("RGB")
    tgt_img = Image.open(tgt_image_path).convert("RGB")

    inputs = processor(prompt, [ref_img, tgt_img], return_tensors="pt").to(device)
    return inputs
