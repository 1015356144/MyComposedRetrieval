import os
import json
import torch
import hashlib
import time
from typing import List, Dict, Any, Optional, Union
from PIL import Image
from torch.utils.data import Dataset
import torch.distributed as dist

from ...utils import print_rank
from .composed_retrieval_dataset import IterativeCIRRDataset


class IterativeNHREditDataset(IterativeCIRRDataset):
    """
    Iterative NHR-Edit Dataset for composed image retrieval with hard negative mining
    Inherits from IterativeCIRRDataset to reuse hard negative mining infrastructure
    """
    
    def __init__(self, 
                 model_args,
                 data_args, 
                 training_args,
                 iteration_round: int = 0,
                 foundation_model=None,
                 experiment_dir=None,
                 **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # Store dataset configuration from kwargs
        self.dataset_config = kwargs
        
        self.iteration_round = iteration_round
        self.foundation_model = foundation_model
        
        # Set experiment directory for storing hard negatives
        self.experiment_dir = experiment_dir or getattr(training_args, 'output_dir', "./experiments/default")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Hard negatives file based on experiment directory
        self.hard_negatives_file = os.path.join(self.experiment_dir, f"hard_negatives_iter_{iteration_round}.json")
        
        self.hard_negatives_cache = {}
        self.augmented_samples = []
        
        # Load original NHR-Edit data
        self._load_cirr_data()  # Keep method name for compatibility with parent class
        
        # Load hard negatives from previous iterations if exists
        if iteration_round > 0:
            self._load_hard_negatives(iteration_round)
    
    def _load_cirr_data(self):
        """Load NHR-Edit dataset (keep method name for parent class compatibility)"""
        print_rank(f"Loading NHR-Edit dataset...")
        self._load_nhr_edit_local()
    
    def _load_nhr_edit_local(self):
        """Load NHR-Edit from local files"""
        data_dir = self.dataset_config.get('data_dir', '/home/share/yty_data/iitolstykh/NHR-Edit')
        image_base_dir = self.dataset_config.get('image_base_dir', '/home/share/yty_data/iitolstykh/NHR-Edit')
        metadata_file = self.dataset_config.get('metadata_file', 'train/metadata.jsonl')
        
        # Load metadata
        metadata_path = os.path.join(data_dir, metadata_file)
        self.annotations = []
        
        print_rank(f"Loading metadata from: {metadata_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    
                    # Convert to CIRR-like format for compatibility
                    converted_sample = self._convert_nhr_sample(sample, image_base_dir)
                    if converted_sample:
                        self.annotations.append(converted_sample)
                        
                except json.JSONDecodeError as e:
                    print_rank(f"Warning: Invalid JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print_rank(f"Warning: Error processing sample at line {line_num}: {e}")
                    continue
        
        self.image_base_dir = image_base_dir
        print_rank(f"Loaded {len(self.annotations)} NHR-Edit training samples")
        
        # Extract all possible target candidates for retrieval
        self._build_retrieval_candidates()
        
        # Add num_rows property for VLM2Vec compatibility
        self.num_rows = len(self.annotations)
    
    def _convert_nhr_sample(self, sample: Dict, image_base_dir: str) -> Optional[Dict]:
        """Convert NHR-Edit sample to CIRR-like format"""
        try:
            # Extract required fields
            source_file = sample.get('source_file_name', '')
            edited_file = sample.get('edited_file_name', '')
            augmented_instructions = sample.get('augmented_instructions', [])
            edit_instruction = sample.get('edit_instruction', '')
            
            # Validate required fields
            if not source_file or not edited_file:
                return None
            
            # Use edit_instruction as the modification text
            if edit_instruction:
                instruction_text = edit_instruction
            else:
                return None
            
            # Build full image paths and verify they exist
            source_path = os.path.join(image_base_dir, source_file)
            edited_path = os.path.join(image_base_dir, edited_file)
            
            # Verify images exist
            if not os.path.exists(source_path) or not os.path.exists(edited_path):
                return None
            
            # Convert to CIRR-like format
            converted = {
                'sample_id': sample.get('sample_id', f"nhr_{len(self.annotations)}"),
                'reference': source_file,  # Use relative path as key (compatible with CIRR format)
                'target_hard': edited_file,  # Use relative path as key
                'caption': instruction_text,
                'img_width': sample.get('img_width'),
                'img_height': sample.get('img_height'),
                'category': sample.get('category'),
                'style': sample.get('style'),
                'kind': sample.get('kind'),
                # Store original NHR-Edit fields for potential future use
                'original_nhr_sample': sample
            }
            
            return converted
            
        except Exception as e:
            print_rank(f"Error converting NHR-Edit sample: {e}")
            return None
    
    def _build_retrieval_candidates(self):
        """
        Build the complete set of retrieval candidates from NHR-Edit dataset.
        
        Unlike CIRR which has a separate image_splits file, for NHR-Edit we need to
        collect all unique images (both source and edited) from the metadata.
        """
        print_rank("Building NHR-Edit retrieval candidate set...")
        
        # Collect all unique images from annotations
        all_images = set()
        
        for ann in self.annotations:
            all_images.add(ann['reference'])  # source image
            all_images.add(ann['target_hard'])  # edited image
        
        print_rank(f"Found {len(all_images)} unique images in NHR-Edit dataset")
        
        # Validate that images exist and build retrieval candidates
        retrieval_candidates = []
        valid_candidates = 0
        missing_images = []
        
        for img_path in sorted(all_images):  # Sort for consistency
            if self._image_exists(img_path):
                retrieval_candidates.append(img_path)
                valid_candidates += 1
            else:
                missing_images.append(img_path)
                if len(missing_images) <= 5:  # Only log first few to avoid spam
                    print_rank(f"Warning: Image not found: {img_path}")
        
        # Store as list for easier indexing during retrieval
        self.retrieval_candidates = retrieval_candidates
        
        # Create image_splits dictionary for compatibility with parent class methods
        # This maps image basename to full relative path
        self.image_splits = {}
        for img_path in retrieval_candidates:
            basename = os.path.basename(img_path)
            self.image_splits[basename] = img_path
            # Also map the full relative path to itself
            self.image_splits[img_path] = img_path
        
        print_rank(f"Built NHR-Edit retrieval candidate set:")
        print_rank(f"  • Total unique images found: {len(all_images)}")
        print_rank(f"  • Valid candidates (files exist): {valid_candidates}")
        print_rank(f"  • Missing files: {len(missing_images)}")
        
        # Log some statistics
        if len(missing_images) > 5:
            print_rank(f"  • Total missing files: {len(missing_images)} (only first 5 logged)")
        
        # Verify we have sufficient candidates for hard negative mining
        if len(self.retrieval_candidates) < 1000:
            print_rank(f"⚠️  Warning: Only {len(self.retrieval_candidates)} retrieval candidates found.")
            print_rank(f"    This might be insufficient for high-quality hard negative mining.")
            print_rank(f"    Consider using a larger subset of the dataset.")
        else:
            print_rank(f"✅ Good! {len(self.retrieval_candidates)} candidates available for hard negative mining.")
        
        # Verify that training samples are covered by retrieval candidates
        self._validate_candidate_coverage()
        
        return self.retrieval_candidates
    
    def _validate_candidate_coverage(self):
        """Validate that all reference and target images from training data are in candidate set"""
        print_rank("Validating NHR-Edit candidate set coverage...")
        
        # Create a set of candidate paths for fast lookup
        candidate_paths_set = set(self.retrieval_candidates)
        
        # Check coverage of training samples
        missing_refs = 0
        missing_targets = 0
        total_refs = set()
        total_targets = set()
        
        for ann in self.annotations:
            # Get reference and target paths (already in correct format)
            ref_path = ann['reference']
            target_path = ann['target_hard']
            
            total_refs.add(ref_path)
            total_targets.add(target_path)
            
            if ref_path not in candidate_paths_set:
                missing_refs += 1
            if target_path not in candidate_paths_set:
                missing_targets += 1
        
        print_rank(f"NHR-Edit coverage validation results:")
        print_rank(f"  • Unique reference images: {len(total_refs)}")
        print_rank(f"  • Unique target images: {len(total_targets)}")
        print_rank(f"  • Missing reference images: {missing_refs}")
        print_rank(f"  • Missing target images: {missing_targets}")
        
        if missing_refs == 0 and missing_targets == 0:
            print_rank(f"✅ Perfect coverage! All training images are in the candidate set.")
        else:
            print_rank(f"⚠️  Coverage issues detected. This may affect hard negative mining quality.")
        
        total_unique_images = len(total_refs | total_targets)
        if total_unique_images > 0:
            coverage_rate = (total_unique_images - missing_refs - missing_targets) / total_unique_images * 100
            print_rank(f"  • Overall coverage rate: {coverage_rate:.1f}%")
    
    def __len__(self):
        """Return total dataset size including augmented samples"""
        return len(self.annotations) + len(self.augmented_samples)
    
    def _get_original_sample(self, idx):
        """Get original NHR-Edit sample - override parent class method"""
        from ...model.processor import process_input_text
        
        sample = self.annotations[idx]
        
        # Get image paths (already in correct format from NHR-Edit)
        ref_image_path = sample['reference']  # relative path like 'images/00/image_xxx.jpg'
        target_image_path = sample['target_hard']  # relative path like 'images/00/image_yyy.jpg'
        
        # Get model backbone from model_args
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        
        # Use VLM2Vec's unified text processing for query (reference image + modification text)
        # Use empty instruction to match CIRR's approach
        query_text = process_input_text(
            instruction="",  # Empty instruction like CIRR
            model_backbone=model_backbone,
            text=sample['caption'],  # The edit instruction text
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
            'query_image': self._load_image_data(ref_image_path),
            'pos_text': pos_text,
            'pos_image': self._load_image_data(target_image_path),
            'neg_text': neg_text,
            'neg_image': None,  # Will be handled by trainer
            'global_dataset_name': 'NHR-Edit'
        }
    
    def _get_augmented_sample(self, idx):
        """Get augmented NHR-Edit sample - override parent class method"""
        from ...model.processor import process_input_text
        
        if idx >= len(self.augmented_samples):
            # Fallback to first original sample
            return self._get_original_sample(0)
        
        aug_sample = self.augmented_samples[idx]
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        
        # Use VLM2Vec's unified text processing for augmented query
        query_text = process_input_text(
            instruction="",  # Empty instruction like CIRR
            model_backbone=model_backbone,
            text=aug_sample['modification_text'],  # The augmented modification text
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
            'query_image': self._load_image_data(aug_sample['reference_image']),
            'pos_text': pos_text,
            'pos_image': self._load_image_data(aug_sample['target_image']),
            'neg_text': neg_text,
            'neg_image': None,  # Will be handled by trainer
            'global_dataset_name': 'NHR-Edit-Augmented'
        }
    
    def _load_image_data(self, image_path: str) -> Dict:
        """Load image data in the format expected by the collator"""
        try:
            full_path = self._get_full_image_path(image_path)
            
            # Return in the format expected by MultimodalDataCollator
            return {
                'paths': [full_path],
                'resolutions': [None],  # Let processor handle resizing
                'bytes': [None]  # Use path instead of bytes
            }
        except Exception as e:
            print_rank(f"Error loading image {image_path}: {e}")
            # Return dummy data
            return {
                'paths': [None],
                'resolutions': [None],
                'bytes': [None]
            }
    
    def get_dataset_statistics(self):
        """Get statistics about the dataset"""
        if not hasattr(self, 'annotations') or len(self.annotations) == 0:
            return {}
        
        stats = {
            'total_samples': len(self.annotations),
            'total_images': len(self.retrieval_candidates),
            'categories': {},
            'styles': {},
            'kinds': {},
            'image_resolutions': {},
        }
        
        for ann in self.annotations:
            # Count categories
            category = ann.get('category', 'Unknown')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Count styles
            style = ann.get('style', 'Unknown')
            stats['styles'][style] = stats['styles'].get(style, 0) + 1
            
            # Count kinds
            kind = ann.get('kind', 'Unknown')
            stats['kinds'][kind] = stats['kinds'].get(kind, 0) + 1
            
            # Count image resolutions
            if ann.get('img_width') and ann.get('img_height'):
                resolution = f"{ann['img_width']}x{ann['img_height']}"
                stats['image_resolutions'][resolution] = stats['image_resolutions'].get(resolution, 0) + 1
        
        return stats


# Register the dataset class manually
from .base_pair_dataset import AutoPairDataset
AutoPairDataset.registry["IterativeNHREditDataset"] = IterativeNHREditDataset


# Register the dataset for import
__all__ = ['IterativeNHREditDataset']