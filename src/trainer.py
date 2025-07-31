"""
Base trainer for multimodal embedding models
"""

import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict, Any, Optional


class MMEBTrainer(Trainer):
    """
    Base trainer for multimodal embedding models
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for retrieval training
        """
        outputs = model(**inputs)
        
        # Simple contrastive loss (placeholder)
        embeddings = outputs['pooled_output']
        
        # For simplicity, assume positive pairs
        batch_size = embeddings.size(0)
        if batch_size % 2 == 0:
            # Split into query and target embeddings
            query_emb = embeddings[:batch_size//2]
            target_emb = embeddings[batch_size//2:]
            
            # Compute similarity scores
            scores = torch.matmul(query_emb, target_emb.transpose(0, 1))
            
            # Apply temperature
            if hasattr(model, 'temperature'):
                scores = scores / model.temperature
            
            # Labels (diagonal is positive)
            labels = torch.arange(query_emb.size(0), device=scores.device)
            
            # Cross-entropy loss
            loss = nn.CrossEntropyLoss()(scores, labels)
        else:
            # Fallback loss
            loss = torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        
        return (loss, outputs) if return_outputs else loss