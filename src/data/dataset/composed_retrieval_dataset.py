"""
Iterative CIRR Dataset for Composed Image Retrieval
Supports hard negative mining and iterative training
"""

import os
import json
import torch
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
from PIL import Image


class IterativeCIRRDataset(Dataset):
    """
    Dataset for iterative training on CIRR data
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_base_dir: str,
                 captions_file: str,
                 image_splits_file: str,
                 processor=None,
                 num_sample_per_subset: int = None,
                 fast_mode: bool = False,
                 **kwargs):
        
        self.data_dir = data_dir
        self.image_base_dir = image_base_dir
        self.captions_file = captions_file
        self.image_splits_file = image_splits_file
        self.processor = processor
        self.num_sample_per_subset = num_sample_per_subset
        self.fast_mode = fast_mode
        
        # Load data
        self._load_data()
        
        # Hard negatives (updated during iterative training)
        self.hard_negatives = {}
        self.augmented_samples = []
        
    def _load_data(self):
        """Load CIRR dataset"""
        # Load captions
        with open(self.captions_file, 'r') as f:
            self.captions_data = json.load(f)
        
        # Load image splits
        with open(self.image_splits_file, 'r') as f:
            self.image_splits = json.load(f)
        
        # Create samples list
        self.samples = []
        for caption_data in self.captions_data:
            sample = {
                'query_id': caption_data['pairid'],
                'reference_image': caption_data['reference'],
                'target_image': caption_data['target_hard'],
                'caption': caption_data['caption'],
                'target_soft': caption_data.get('target_soft', caption_data['target_hard'])
            }
            self.samples.append(sample)
        
        # Limit samples if specified
        if self.num_sample_per_subset:
            self.samples = self.samples[:self.num_sample_per_subset]
        
        if self.fast_mode:
            self.samples = self.samples[:100]  # Very small for testing
        
        print(f"Loaded {len(self.samples)} CIRR samples")
    
    def add_hard_negatives(self, hard_negatives: Dict):
        """Add hard negatives for iterative training"""
        self.hard_negatives = hard_negatives
        print(f"Added hard negatives for {len(hard_negatives)} queries")
    
    def add_augmented_samples(self, augmented_samples: List):
        """Add augmented samples from foundation model"""
        self.augmented_samples.extend(augmented_samples)
        print(f"Added {len(augmented_samples)} augmented samples")
    
    def __len__(self):
        return len(self.samples) + len(self.augmented_samples)
    
    def __getitem__(self, idx):
        # Get sample (original or augmented)
        if idx < len(self.samples):
            sample = self.samples[idx]
        else:
            sample = self.augmented_samples[idx - len(self.samples)]
        
        # Load images
        ref_image_path = os.path.join(self.image_base_dir, sample['reference_image'] + '.png')
        target_image_path = os.path.join(self.image_base_dir, sample['target_image'] + '.png')
        
        try:
            ref_image = Image.open(ref_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
        except:
            # Fallback for missing images
            ref_image = Image.new('RGB', (224, 224), color='white')
            target_image = Image.new('RGB', (224, 224), color='white')
        
        # Create input text
        input_text = f"Find an image of {sample['caption']}"
        
        return {
            'query_id': sample['query_id'],
            'reference_image': ref_image,
            'target_image': target_image,
            'input_text': input_text,
            'caption': sample['caption']
        }


class IterativeFashionIQDataset(Dataset):
    """
    Dataset for iterative training on FashionIQ data
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_base_dir: str,
                 categories: List[str] = None,
                 processor=None,
                 **kwargs):
        
        self.data_dir = data_dir
        self.image_base_dir = image_base_dir
        self.categories = categories or ['dress', 'shirt', 'toptee']
        self.processor = processor
        
        # Load data
        self._load_data()
        
        # Hard negatives and augmented samples
        self.hard_negatives = {}
        self.augmented_samples = []
    
    def _load_data(self):
        """Load FashionIQ dataset"""
        self.samples = []
        
        for category in self.categories:
            # Load category data (placeholder)
            # In real implementation, load from actual FashionIQ files
            category_samples = [
                {
                    'query_id': f'{category}_{i}',
                    'reference_image': f'{category}_ref_{i}',
                    'target_image': f'{category}_target_{i}',
                    'captions': [f'Make it more {category}-like', f'Change the color'],
                    'category': category
                }
                for i in range(10)  # Placeholder data
            ]
            self.samples.extend(category_samples)
        
        print(f"Loaded {len(self.samples)} FashionIQ samples")
    
    def add_hard_negatives(self, hard_negatives: Dict):
        """Add hard negatives for iterative training"""
        self.hard_negatives = hard_negatives
    
    def add_augmented_samples(self, augmented_samples: List):
        """Add augmented samples from foundation model"""
        self.augmented_samples.extend(augmented_samples)
    
    def __len__(self):
        return len(self.samples) + len(self.augmented_samples)
    
    def __getitem__(self, idx):
        # Get sample
        if idx < len(self.samples):
            sample = self.samples[idx]
        else:
            sample = self.augmented_samples[idx - len(self.samples)]
        
        # Create placeholder images (in real implementation, load actual images)
        ref_image = Image.new('RGB', (224, 224), color='white')
        target_image = Image.new('RGB', (224, 224), color='gray')
        
        # Combine captions
        input_text = ' and '.join(sample['captions'])
        
        return {
            'query_id': sample['query_id'],
            'reference_image': ref_image,
            'target_image': target_image,
            'input_text': input_text,
            'category': sample['category']
        }