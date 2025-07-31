"""
Simple Composed Image Retrieval Dataset
A simplified version for testing and basic functionality
"""

import os
import json
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
from PIL import Image

from .base_pair_dataset import AutoPairDataset, add_metainfo_hook
from ...model.processor import VLM_IMAGE_TOKENS, process_input_text
from ...utils import print_rank


class SimpleCIRRDataset(Dataset):
    """
    Simple CIRR Dataset for basic composed image retrieval
    This is a simplified version without iterative training features
    """
    
    def __init__(self, 
                 model_args,
                 data_args, 
                 training_args,
                 **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # Store dataset configuration from kwargs
        self.dataset_config = kwargs
        
        # Load CIRR data
        self._load_cirr_data()
    
    def _load_cirr_data(self):
        """Load CIRR dataset"""
        print_rank(f"Loading simple CIRR dataset...")
        
        data_dir = self.dataset_config.get('data_dir', './data/CIRR')
        image_base_dir = self.dataset_config.get('image_base_dir', './data/CIRR')
        captions_file = self.dataset_config.get('captions_file')
        image_splits_file = self.dataset_config.get('image_splits_file')
        
        # Load captions
        with open(captions_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Load image paths mapping
        with open(image_splits_file, 'r') as f:
            self.image_splits = json.load(f)
        
        self.image_base_dir = image_base_dir
        print_rank(f"Loaded {len(self.annotations)} CIRR training samples")
        print_rank(f"Loaded {len(self.image_splits)} image path mappings")
        
        # Add num_rows property for VLM2Vec compatibility
        self.num_rows = len(self.annotations)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        sample = self.annotations[idx]
        
        # Get image paths from the mapping
        ref_image_path = self.image_splits.get(sample['reference'], sample['reference'])
        target_image_path = self.image_splits.get(sample['target_hard'], sample['target_hard'])
        
        # Get model backbone from model_args
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        
        # Use VLM2Vec's unified text processing for query (reference image + modification text)
        query_text = process_input_text(
            instruction="",  # No specific instruction for CIRR
            model_backbone=model_backbone,
            text=sample['caption'],
            add_image_token=True  # Add image token for reference image
        )
        
        # For positive (target), just add image token
        pos_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",  # Empty text for target image
            add_image_token=True
        )
        
        # For negative, also just add image token  
        neg_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True
        )
        
        return {
            'query_text': query_text,
            'query_image': self._load_image(ref_image_path),
            'pos_text': pos_text,
            'pos_image': self._load_image(target_image_path),
            'neg_text': neg_text,
            'neg_image': self._load_image(ref_image_path),  # Use reference as negative for now
            'global_dataset_name': 'CIRR'
        }
    
    def _load_image(self, image_path):
        """Load and process image, return in VLM2Vec format"""
        if isinstance(image_path, str):
            # Handle relative paths from image_splits
            if image_path.startswith('./'):
                full_path = os.path.join(self.image_base_dir, image_path[2:])  # Remove './'
            else:
                full_path = os.path.join(self.image_base_dir, image_path)
            
            if not os.path.exists(full_path):
                print_rank(f"Warning: Image not found at {full_path}")
                # Use a placeholder path for missing images
                full_path = "dummy_image"
        else:
            full_path = str(image_path)
        
        # Return in VLM2Vec expected format - collator will handle actual image loading
        return {
            "paths": [full_path],
            "bytes": [None],  # Let collator handle image loading from path
            "resolutions": [None]  # Let processor handle resizing
        }


class SimpleFashionIQDataset(SimpleCIRRDataset):
    """
    Simple FashionIQ Dataset - similar structure to CIRR but for fashion domain
    """
    
    def _load_cirr_data(self):
        """Override to load FashionIQ data"""
        print_rank(f"Loading simple FashionIQ dataset...")
        
        # FashionIQ specific loading logic
        data_dir = self.dataset_config.get('data_dir', './data/FashionIQ')
        
        # Load different categories (dress, shirt, toptee)
        categories = ['dress', 'shirt', 'toptee']
        all_annotations = []
        
        for category in categories:
            cat_file = os.path.join(data_dir, 'captions', f'cap.{category}.train.json')
            if os.path.exists(cat_file):
                with open(cat_file, 'r') as f:
                    cat_data = json.load(f)
                    # Add category info
                    for item in cat_data:
                        item['category'] = category
                    all_annotations.extend(cat_data)
        
        self.annotations = all_annotations
        self.image_dir = os.path.join(data_dir, 'images')
        print_rank(f"Loaded {len(self.annotations)} FashionIQ training samples")


# Register the dataset classes
AutoPairDataset.registry["SimpleCIRRDataset"] = SimpleCIRRDataset
AutoPairDataset.registry["SimpleFashionIQDataset"] = SimpleFashionIQDataset
