"""
Data collator for multimodal training
"""

import torch
from typing import Dict, List, Any
from transformers import DataCollatorMixin


class MultimodalDataCollator(DataCollatorMixin):
    """
    Data collator for multimodal composed retrieval training
    """
    
    def __init__(self, processor, model_args, data_args, training_args):
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of samples into a batch
        """
        batch_size = len(features)
        
        # Prepare images and texts
        images = []
        texts = []
        
        for feature in features:
            # For composed retrieval, we typically have reference image + text
            if 'reference_image' in feature:
                images.append(feature['reference_image'])
            
            if 'input_text' in feature:
                texts.append(feature['input_text'])
        
        # Process with the multimodal processor
        if images and texts:
            # Multimodal input (image + text)
            processed = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.data_args.max_len or 512
            )
        elif texts:
            # Text-only input
            processed = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.data_args.max_len or 512
            )
        else:
            # Empty batch
            processed = {}
        
        # Add metadata
        if 'query_id' in features[0]:
            processed['query_ids'] = [f['query_id'] for f in features]
        
        return processed