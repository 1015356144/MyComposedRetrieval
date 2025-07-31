"""
Model definitions and utilities
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from typing import Dict, Any, Optional


class MMEBModel(nn.Module):
    """
    Multimodal Embedding Model for retrieval
    """
    
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        
        # Load backbone model
        self.backbone = AutoModel.from_pretrained(
            model_args.model_name,
            trust_remote_code=True
        )
        
        self.config = self.backbone.config
        
        # Pooling and normalization
        self.pooling = model_args.pooling
        self.normalize = model_args.normalize
        self.temperature = model_args.temperature
    
    @classmethod
    def build(cls, model_args):
        """Build model from arguments"""
        return cls(model_args)
    
    def forward(self, 
                input_ids=None,
                attention_mask=None,
                pixel_values=None,
                image_grid_thw=None,
                **kwargs):
        
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs
        )
        
        # Get pooled representation
        if self.pooling == 'eos':
            # Use EOS token representation
            last_hidden_state = outputs.last_hidden_state
            # Find EOS token position
            eos_positions = (input_ids == self.backbone.config.eos_token_id).nonzero(as_tuple=True)
            if len(eos_positions[0]) > 0:
                pooled_output = last_hidden_state[eos_positions[0], eos_positions[1]]
            else:
                pooled_output = last_hidden_state[:, -1]  # Use last token
        else:
            # Use mean pooling
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Normalize if specified
        if self.normalize:
            pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=-1)
        
        return {
            'pooled_output': pooled_output,
            'last_hidden_state': outputs.last_hidden_state
        }
    
    def encode(self, **inputs):
        """Encode inputs to embeddings"""
        return self.forward(**inputs)['pooled_output']


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