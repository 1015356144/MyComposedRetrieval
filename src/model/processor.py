"""
Processor utilities for different model backbones
"""

from transformers import AutoProcessor
from typing import Any, Dict


def load_processor(model_args, data_args):
    """Load processor based on model arguments"""
    processor_name = model_args.processor_name or model_args.model_name
    
    processor = AutoProcessor.from_pretrained(
        processor_name,
        trust_remote_code=True
    )
    
    # Configure processor based on data args
    if hasattr(processor, 'image_processor') and data_args.resize_use_processor:
        if hasattr(processor.image_processor, 'min_pixels'):
            processor.image_processor.min_pixels = data_args.resize_min_pixels
        if hasattr(processor.image_processor, 'max_pixels'):
            processor.image_processor.max_pixels = data_args.resize_max_pixels
    
    return processor


def get_backbone_name(hf_config):
    """Extract backbone name from HuggingFace config"""
    if hasattr(hf_config, 'model_type'):
        return hf_config.model_type
    elif hasattr(hf_config, '_name_or_path'):
        name = hf_config._name_or_path.lower()
        if 'qwen' in name:
            return 'qwen2_vl'
        elif 'llava' in name:
            return 'llava'
        elif 'phi' in name:
            return 'phi3_v'
    return 'unknown'