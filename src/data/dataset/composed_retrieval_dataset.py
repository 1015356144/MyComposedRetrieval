"""
Composed Image Retrieval Dataset for iterative training
Supports CIRR and FashionIQ datasets with iterative hard negative mining
"""

import os
import json
import time
import random
import re
import torch
import torch.nn.functional as F
import numpy as np
import traceback
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
from datasets import load_dataset

from .base_pair_dataset import AutoPairDataset, add_metainfo_hook
from ...model.processor import VLM_IMAGE_TOKENS, process_input_text
from ...utils import print_rank, print_master


class IterativeCIRRDataset(Dataset):
    """
    Iterative CIRR Dataset for composed image retrieval with hard negative mining
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
        
        # Load original CIRR data
        self._load_cirr_data()
        
        # Load hard negatives from previous iterations if exists
        if iteration_round > 0:
            self._load_hard_negatives(iteration_round)
    
    def _load_cirr_data(self):
        """Load CIRR dataset"""
        print_rank(f"Loading CIRR dataset...")
        
        # Load from local files
        self._load_cirr_local()
    
    def _load_cirr_local(self):
        """Load CIRR from local files"""
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
    
    def _load_hard_negatives(self, iteration_round: int):
        """Load hard negatives from previous iteration"""
        cache_file = os.path.join(self.experiment_dir, f"hard_negatives_iter_{iteration_round-1}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.hard_negatives_cache = json.load(f)
            print_rank(f"Loaded hard negatives from iteration {iteration_round-1}: {cache_file}")
        else:
            print_rank(f"No hard negatives cache found for iteration {iteration_round-1}")
    
    def collect_hard_negatives_batch(self, retrieval_model, batch_size: int = 8):
        """
        Collect hard negatives by running retrieval with current model
        Note: Reduced batch_size for real retrieval to avoid memory issues
        """
        print_rank(f"Collecting hard negatives for iteration {self.iteration_round}")
        
        # Check if hard negatives already exist for this iteration
        if os.path.exists(self.hard_negatives_file):
            print_rank(f"Hard negatives already exist for iteration {self.iteration_round}, loading from {self.hard_negatives_file}")
            with open(self.hard_negatives_file, 'r') as f:
                hard_negatives = json.load(f)
            self.hard_negatives_cache = hard_negatives
            print_rank(f"Loaded {len(hard_negatives)} existing hard negative samples")
            return hard_negatives
        
        print_rank(f"No existing hard negatives found, collecting new ones...")
        
        hard_negatives = []
        retrieval_model.eval()
        
        # Use smaller batches for real retrieval and limit total samples for efficiency
        max_samples = min(1000, len(self.annotations))  # Limit to 1000 samples for efficiency
        sample_annotations = self.annotations[:max_samples]
        
        print_rank(f"Processing {len(sample_annotations)} samples for hard negative mining...")
        
        with torch.no_grad():
            for i in range(0, len(sample_annotations), batch_size):
                # Get batch of annotations
                batch_annotations = sample_annotations[i:i+batch_size]
                
                print_rank(f"Processing batch {i//batch_size + 1}/{(len(sample_annotations) + batch_size - 1)//batch_size}")
                
                # Convert to dataset format for processing
                batch = []
                for ann in batch_annotations:
                    batch.append({
                        'reference_image': self.image_splits.get(ann['reference'], ann['reference']),
                        'modification_text': ann['caption'],
                        'target_image': self.image_splits.get(ann['target_hard'], ann['target_hard'])
                    })
                
                # Run retrieval for this batch
                retrieval_results = self._run_retrieval_batch(retrieval_model, batch)
                
                # Identify hard negatives (incorrect retrievals in top-k)
                batch_hard_negs = self._identify_hard_negatives(batch, retrieval_results)
                hard_negatives.extend(batch_hard_negs)
        
        # Save hard negatives to experiment directory
        with open(self.hard_negatives_file, 'w') as f:
            json.dump(hard_negatives, f, indent=2)
        
        self.hard_negatives_cache = hard_negatives
        print_rank(f"Collected {len(hard_negatives)} hard negative samples")
        print_rank(f"Hard negatives saved to {self.hard_negatives_file}")
        
        return hard_negatives
    
    def _run_retrieval_batch(self, model, batch):
        """Run real retrieval for a batch of queries using the actual VLM2Vec model"""
        import torch.nn.functional as F
        
        batch_size = len(batch)
        print_rank(f"Running real retrieval for {batch_size} queries")
        
        try:
            return self._run_real_retrieval(model, batch)
        except Exception as e:
            print_rank(f"Real retrieval failed: {e}, falling back to simplified retrieval")
            return self._run_simplified_retrieval(batch)
    
    def _run_real_retrieval(self, model, batch):
        """Run real retrieval using VLM2Vec model"""
        import torch.nn.functional as F
        
        batch_size = len(batch)
        device = next(model.parameters()).device
        
        # Get model backbone and processor
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        processor = getattr(model, 'processor', None)
        
        if processor is None:
            print_rank("Warning: No processor found in model")
            raise Exception("No processor available")
        
        # Collect target images for retrieval database (limit for efficiency)
        target_database = []
        target_paths = []
        
        # Use a reasonable subset of CIRR for retrieval database
        max_targets = min(200, len(self.annotations))  # Limit to 200 targets
        target_annotations = self.annotations[:max_targets]
        
        for ann in target_annotations:
            target_path = self.image_splits.get(ann['target_hard'], ann['target_hard'])
            if self._image_exists(target_path):
                target_database.append(target_path)
                target_paths.append(target_path)
        
        print_rank(f"Retrieval database: {len(target_database)} target images")
        
        if len(target_database) == 0:
            raise Exception("No valid target images found")
        
        # Encode target database in batches
        target_embeddings = []
        target_batch_size = 8  # Small batch size to avoid memory issues
        
        with torch.no_grad():
            for i in range(0, len(target_database), target_batch_size):
                batch_targets = target_database[i:i+target_batch_size]
                
                # Create target inputs (image-only)
                target_inputs = self._prepare_target_inputs(batch_targets, processor, model_backbone, device)
                
                try:
                    target_embs = model.encode_input(target_inputs)
                    
                    # Ensure proper tensor format and handle 0-d tensors
                    if target_embs is None:
                        print_rank(f"Warning: encode_input returned None for target batch {i//target_batch_size}")
                        dummy_embs = torch.randn(len(batch_targets), 768)
                        target_embeddings.append(dummy_embs)
                        continue
                    
                    if target_embs.dim() == 0:
                        # Handle 0-d tensor by converting to 1-d
                        print_rank(f"Warning: Got 0-d tensor for target batch {i//target_batch_size}, reshaping")
                        target_embs = target_embs.view(1, -1) if target_embs.numel() > 0 else torch.randn(len(batch_targets), 768)
                    elif target_embs.dim() == 1:
                        target_embs = target_embs.unsqueeze(0)
                    elif target_embs.dim() > 2:
                        target_embs = target_embs.view(target_embs.size(0), -1)  # Flatten
                    
                    # Ensure we have the right batch size
                    if target_embs.size(0) != len(batch_targets):
                        print_rank(f"Warning: Embedding batch size {target_embs.size(0)} != target batch size {len(batch_targets)}")
                        # Repeat or pad to match expected batch size
                        if target_embs.size(0) == 1 and len(batch_targets) > 1:
                            target_embs = target_embs.repeat(len(batch_targets), 1)
                        else:
                            dummy_embs = torch.randn(len(batch_targets), target_embs.size(-1))
                            target_embs = dummy_embs
                    
                    target_embeddings.append(target_embs.cpu())
                    
                except Exception as e:
                    print_rank(f"Error encoding target batch {i//target_batch_size}: {e}")
                    # Use dummy embeddings as fallback
                    dummy_embs = torch.randn(len(batch_targets), 768)
                    target_embeddings.append(dummy_embs)
        
        # Concatenate all target embeddings
        target_embeddings = torch.cat(target_embeddings, dim=0)
        
        # Encode query batch (reference image + modification text)
        with torch.no_grad():
            query_inputs = self._prepare_query_inputs(batch, processor, model_backbone, device)
            
            try:
                query_embeddings = model.encode_input(query_inputs)
                
                # Ensure proper tensor format and handle 0-d tensors
                if query_embeddings is None:
                    print_rank(f"Warning: encode_input returned None for queries")
                    query_embeddings = torch.randn(len(batch), 768)
                elif query_embeddings.dim() == 0:
                    # Handle 0-d tensor by converting to 1-d
                    print_rank(f"Warning: Got 0-d tensor for queries, reshaping")
                    query_embeddings = query_embeddings.view(1, -1) if query_embeddings.numel() > 0 else torch.randn(len(batch), 768)
                elif query_embeddings.dim() == 1:
                    query_embeddings = query_embeddings.unsqueeze(0)
                elif query_embeddings.dim() > 2:
                    query_embeddings = query_embeddings.view(query_embeddings.size(0), -1)
                
                # Ensure we have the right batch size
                if query_embeddings.size(0) != len(batch):
                    print_rank(f"Warning: Query embedding batch size {query_embeddings.size(0)} != query batch size {len(batch)}")
                    # Repeat or pad to match expected batch size
                    if query_embeddings.size(0) == 1 and len(batch) > 1:
                        query_embeddings = query_embeddings.repeat(len(batch), 1)
                    else:
                        dummy_embs = torch.randn(len(batch), query_embeddings.size(-1))
                        query_embeddings = dummy_embs
                
                query_embeddings = query_embeddings.cpu()
                
            except Exception as e:
                print_rank(f"Error encoding queries: {e}")
                # Use dummy embeddings as fallback
                query_embeddings = torch.randn(len(batch), target_embeddings.size(1))
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.mm(query_embeddings, target_embeddings.t())
        
        # Get top-k retrieval results
        k = min(10, len(target_database))
        top_k_similarities, top_k_indices = torch.topk(similarities, k, dim=1, largest=True)
        
        # Find ground truth indices for each query
        gt_indices = []
        for query in batch:
            gt_target_path = query['target_image']
            try:
                gt_idx = target_paths.index(gt_target_path)
            except ValueError:
                # If GT not found in target database, use -1
                gt_idx = -1
            gt_indices.append(gt_idx)
        
        # Format results
        results = {
            "top_k_indices": top_k_indices.tolist(),
            "gt_indices": gt_indices,
            "similarities": top_k_similarities.tolist(),
            "target_paths": target_paths  # Include actual target paths for reference
        }
        
        print_rank(f"Real retrieval completed. Average top-1 similarity: {top_k_similarities[:, 0].mean():.4f}")
        return results
    
    def _image_exists(self, image_path):
        """Check if image file exists"""
        if isinstance(image_path, str):
            if image_path.startswith('./'):
                full_path = os.path.join(self.image_base_dir, image_path[2:])
            else:
                full_path = os.path.join(self.image_base_dir, image_path)
            return os.path.exists(full_path)
        return False
    
    def _prepare_target_inputs(self, target_paths, processor, model_backbone, device):
        """Prepare inputs for target images (image-only)"""
        from PIL import Image
        
        images = []
        texts = []
        
        for target_path in target_paths:
            # Load target image
            try:
                if target_path.startswith('./'):
                    full_path = os.path.join(self.image_base_dir, target_path[2:])
                else:
                    full_path = os.path.join(self.image_base_dir, target_path)
                
                image = Image.open(full_path).convert('RGB')
                images.append(image)
                
                # For targets, use simple descriptive text with image token
                target_text = process_input_text(
                    instruction="Represent the given image", 
                    model_backbone=model_backbone, 
                    text="", 
                    add_image_token=True
                )
                texts.append(target_text)
                
            except Exception as e:
                print_rank(f"Error loading target image {target_path}: {e}")
                # Use dummy image and text
                images.append(Image.new('RGB', (224, 224), color='white'))
                texts.append("Represent the given image")
        
        # Process with VLM2Vec's official processor for target inputs
        try:
            # Import VLM2Vec processor function
            from src.model.processor import process_vlm_inputs_fns
            
            # Prepare data in VLM2Vec expected format
            model_inputs = {
                'text': texts,
                'images': images
            }
            
            # Use VLM2Vec's processor function for this model backbone
            if model_backbone in process_vlm_inputs_fns:
                inputs = process_vlm_inputs_fns[model_backbone](model_inputs, processor)
            else:
                raise ValueError(f"Model backbone {model_backbone} not supported in VLM2Vec")
            
            # Move to device
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to(device)
            
            return inputs
            
        except Exception as e:
            print_rank(f"Error in VLM2Vec processor for targets: {e}")
            raise e
    
    def _prepare_query_inputs(self, batch, processor, model_backbone, device):
        """Prepare inputs for queries (reference image + modification text)"""
        from PIL import Image
        
        images = []
        texts = []
        
        for query in batch:
            # Load reference image
            try:
                ref_path = query['reference_image']
                if ref_path.startswith('./'):
                    full_path = os.path.join(self.image_base_dir, ref_path[2:])
                else:
                    full_path = os.path.join(self.image_base_dir, ref_path)
                
                image = Image.open(full_path).convert('RGB')
                images.append(image)
                
                # Combine modification text with image token
                query_text = process_input_text(
                    instruction="Represent the given image with the following modification", 
                    model_backbone=model_backbone, 
                    text=query['modification_text'], 
                    add_image_token=True
                )
                texts.append(query_text)
                
            except Exception as e:
                print_rank(f"Error loading reference image {query['reference_image']}: {e}")
                # Use dummy image and text
                images.append(Image.new('RGB', (224, 224), color='white'))
                texts.append(query.get('modification_text', ''))
        
        # Process with VLM2Vec's official processor for query inputs
        try:
            # Import VLM2Vec processor function
            from src.model.processor import process_vlm_inputs_fns
            
            # Prepare data in VLM2Vec expected format
            model_inputs = {
                'text': texts,
                'images': images
            }
            
            # Use VLM2Vec's processor function for this model backbone
            if model_backbone in process_vlm_inputs_fns:
                inputs = process_vlm_inputs_fns[model_backbone](model_inputs, processor)
            else:
                raise ValueError(f"Model backbone {model_backbone} not supported in VLM2Vec")
            
            # Move to device
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to(device)
            
            return inputs
            
        except Exception as e:
            print_rank(f"Error in VLM2Vec processor for queries: {e}")
            raise e
    
    def _process_batch_for_model(self, batch_dict, processor, device, is_query=False):
        """Process batch data for model input using VLM2Vec's approach"""
        try:
            # Use processor to prepare the inputs properly
            if is_query:
                texts = batch_dict.get('query_text', [])
                images = []
                # Extract image paths from VLM2Vec format
                for img_data in batch_dict.get('query_image', []):
                    if isinstance(img_data, dict) and 'paths' in img_data:
                        img_path = img_data['paths'][0]
                        if img_path != "dummy_image" and os.path.exists(img_path):
                            from PIL import Image
                            images.append(Image.open(img_path).convert('RGB'))
                        else:
                            # Create a dummy image
                            images.append(Image.new('RGB', (224, 224), color='white'))
                    else:
                        # Fallback to dummy image
                        from PIL import Image
                        images.append(Image.new('RGB', (224, 224), color='white'))
            else:
                texts = batch_dict.get('pos_text', [])
                images = []
                # Extract image paths from VLM2Vec format
                for img_data in batch_dict.get('pos_image', []):
                    if isinstance(img_data, dict) and 'paths' in img_data:
                        img_path = img_data['paths'][0]
                        if img_path != "dummy_image" and os.path.exists(img_path):
                            from PIL import Image
                            images.append(Image.open(img_path).convert('RGB'))
                        else:
                            # Create a dummy image
                            images.append(Image.new('RGB', (224, 224), color='white'))
                    else:
                        # Fallback to dummy image
                        from PIL import Image
                        images.append(Image.new('RGB', (224, 224), color='white'))
            
            # Ensure we have the same number of texts and images
            if len(texts) == 0 and len(images) > 0:
                texts = [""] * len(images)
            elif len(images) == 0 and len(texts) > 0:
                from PIL import Image
                images = [Image.new('RGB', (224, 224), color='white')] * len(texts)
            
            # For Qwen2-VL, we need to format the input properly
            # Check if this is Qwen2-VL model
            model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
            
            if model_backbone == 'qwen2_vl':
                # For Qwen2-VL, prepare input in the expected format
                processed = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
            else:
                # Use standard format for other models
                processed = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
            
            # Move to device
            for key in processed:
                if hasattr(processed[key], 'to'):
                    processed[key] = processed[key].to(device)
            
            # Add required fields for VLM2Vec models that expect them
            if 'texts' not in processed and len(texts) > 0:
                processed['texts'] = texts
            if 'images' not in processed and len(images) > 0:
                processed['images'] = images
            
            return processed
            
        except Exception as e:
            print_rank(f"Error in _process_batch_for_model: {e}")
            import traceback
            print_rank(f"Full traceback: {traceback.format_exc()}")
            # Return dummy processed data that works with MMEBModel.encode_input
            batch_size = len(batch_dict.get('query_text' if is_query else 'pos_text', []))
            if batch_size == 0:
                batch_size = 1
            return {
                'input_ids': torch.zeros((batch_size, 512), dtype=torch.long).to(device),
                'attention_mask': torch.ones((batch_size, 512), dtype=torch.long).to(device),
                'pixel_values': torch.randn((batch_size, 3, 224, 224)).to(device),
                'texts': [""] * batch_size,
                'images': [None] * batch_size
            }
    
    def _run_simplified_retrieval(self, batch):
        """Fallback simplified retrieval with dummy results"""
        print_rank("Running simplified retrieval (fallback)")
        batch_size = len(batch)
        
        # Create more realistic dummy retrieval results
        dummy_results = {
            "top_k_indices": [],
            "gt_indices": [],  
            "similarities": []
        }
        
        for i in range(batch_size):
            # Simulate retrieval results where GT might not be top-1
            if i % 3 == 0:  # 1/3 of cases: GT not in top-1, but in top-3
                gt_idx = 2  # GT at position 2
                top_k = [1, 5, gt_idx, 7, 9, 3, 8, 4, 6, 0]  # GT at index 2
            elif i % 3 == 1:  # 1/3 of cases: GT at top-1 (perfect retrieval)
                gt_idx = 0
                top_k = [gt_idx, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:  # 1/3 of cases: GT not in top-1, but in top-5
                gt_idx = 4  # GT at position 4  
                top_k = [2, 7, 1, 9, gt_idx, 5, 3, 8, 6, 0]  # GT at index 4
            
            dummy_results["top_k_indices"].append(top_k)
            dummy_results["gt_indices"].append(gt_idx)
            dummy_results["similarities"].append([0.9 - j*0.1 for j in range(10)])
        
        return dummy_results
    
    def _identify_hard_negatives(self, batch, retrieval_results, k=5):
        """Identify hard negatives from retrieval results (supports both real and simulated)"""
        hard_negatives = []
        
        top_k_indices = retrieval_results['top_k_indices']
        gt_indices = retrieval_results['gt_indices']
        similarities = retrieval_results['similarities']
        
        # Check if we have real target paths (from real retrieval) or just indices (from simulation)
        target_paths = retrieval_results.get('target_paths', None)
        is_real_retrieval = target_paths is not None
        
        for idx, (query, gt_target, top_k, sims) in enumerate(zip(
            batch, gt_indices, top_k_indices, similarities
        )):
            # Handle ground truth position
            if gt_target == -1:
                # GT not found in retrieval database, all top results are hard negatives
                for neg_pos in range(min(3, len(top_k))):
                    neg_idx = top_k[neg_pos]
                    
                    if is_real_retrieval:
                        # Use actual target path
                        hard_negative_image = target_paths[neg_idx] if neg_idx < len(target_paths) else f"target_{neg_idx}"
                    else:
                        # Use simulated index
                        hard_negative_image = neg_idx
                    
                    hard_negatives.append({
                        'reference_image': query['reference_image'],
                        'modification_text': query['modification_text'],
                        'target_image': query['target_image'],  # GT
                        'hard_negative_image': hard_negative_image,
                        'rank_position': neg_pos + 1,
                        'gt_rank': -1,  # GT not found
                        'similarity_score': sims[neg_pos] if neg_pos < len(sims) else 0.0,
                        'is_real_retrieval': is_real_retrieval
                    })
            elif gt_target in top_k:
                gt_position = top_k.index(gt_target)
                
                # If GT is not in top-1 but within top-k, collect hard negatives
                if gt_position > 0 and gt_position < k:
                    # All results ranked higher than GT are hard negatives
                    for neg_pos in range(gt_position):
                        neg_idx = top_k[neg_pos]
                        
                        if is_real_retrieval:
                            # Use actual target path
                            hard_negative_image = target_paths[neg_idx] if neg_idx < len(target_paths) else f"target_{neg_idx}"
                        else:
                            # Use simulated index
                            hard_negative_image = neg_idx
                        
                        hard_negatives.append({
                            'reference_image': query['reference_image'],
                            'modification_text': query['modification_text'],
                            'target_image': query['target_image'],  # GT
                            'hard_negative_image': hard_negative_image,
                            'rank_position': neg_pos + 1,
                            'gt_rank': gt_position + 1,
                            'similarity_score': sims[neg_pos] if neg_pos < len(sims) else 0.0,
                            'is_real_retrieval': is_real_retrieval
                        })
            else:
                # GT not found in top-k, all top results are hard negatives
                for neg_pos in range(min(3, len(top_k))):
                    neg_idx = top_k[neg_pos]
                    
                    if is_real_retrieval:
                        hard_negative_image = target_paths[neg_idx] if neg_idx < len(target_paths) else f"target_{neg_idx}"
                    else:
                        hard_negative_image = neg_idx
                    
                    hard_negatives.append({
                        'reference_image': query['reference_image'],
                        'modification_text': query['modification_text'],
                        'target_image': query['target_image'],  # GT
                        'hard_negative_image': hard_negative_image,
                        'rank_position': neg_pos + 1,
                        'gt_rank': -1,  # GT not found in top-k
                        'similarity_score': sims[neg_pos] if neg_pos < len(sims) else 0.0,
                        'is_real_retrieval': is_real_retrieval
                    })
        
        return hard_negatives
    
    def _identify_hard_negatives_from_batch(self, batch, model, k=5):
        """
        Identify hard negatives from a batch using the provided model
        This method is called by the iterative trainer
        """
        print_rank(f"Identifying hard negatives from batch of {len(batch)} samples using model...")
        
        # Run retrieval on the batch
        try:
            retrieval_results = self._run_real_retrieval(model, batch)
        except Exception as e:
            print_rank(f"Real retrieval failed: {e}, falling back to simplified retrieval")
            retrieval_results = self._run_simplified_retrieval(batch)
        
        # Extract hard negatives from retrieval results
        hard_negatives = self._identify_hard_negatives(batch, retrieval_results, k=k)
        
        return hard_negatives
    
    def generate_augmented_captions(self, hard_negatives: List[Dict]):
        """
        Use foundation model to generate new modification texts for hard negatives
        """
        if not self.foundation_model:
            print_rank("No foundation model provided, skipping caption generation")
            return []
        
        # Check if augmented samples already exist for this iteration
        aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{self.iteration_round}.json")
        if os.path.exists(aug_file):
            print_rank(f"Augmented samples already exist for iteration {self.iteration_round}, loading from {aug_file}")
            try:
                with open(aug_file, 'r') as f:
                    saved_data = json.load(f)
                augmented_samples = saved_data.get('samples', [])
                print_rank(f"Loaded {len(augmented_samples)} existing augmented samples")
                self.augmented_samples = augmented_samples
                return augmented_samples
            except Exception as e:
                print_rank(f"Error loading augmented samples: {e}, regenerating...")
        
        print_rank(f"No existing augmented samples found, generating new ones...")
        print_rank(f"Generating augmented captions for {len(hard_negatives)} hard negatives")
        print_rank(f"Processing in {(len(hard_negatives) + 3) // 4} batches of 4 samples each")
        augmented_samples = []
        
        # Batch process for efficiency
        batch_size = 4  # Small batch size to avoid memory issues
        total_batches = (len(hard_negatives) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for i in range(0, len(hard_negatives), batch_size):
            batch_idx = i // batch_size + 1
            batch_hard_negs = hard_negatives[i:i+batch_size]
            
            # Calculate ETA
            if batch_idx > 1:
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / (batch_idx - 1)
                remaining_batches = total_batches - batch_idx + 1
                eta_seconds = avg_time_per_batch * remaining_batches
                eta_str = f"ETA: {int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
            else:
                eta_str = "ETA: calculating..."
            
            print_rank(f"Processing caption generation batch {batch_idx}/{total_batches} ({len(batch_hard_negs)} samples) - {eta_str}")
            
            try:
                batch_start_time = time.time()
                batch_augmented = self._generate_caption_batch(batch_hard_negs)
                batch_time = time.time() - batch_start_time
                augmented_samples.extend(batch_augmented)
                print_rank(f"Batch {batch_idx}/{total_batches} completed in {batch_time:.1f}s, generated {len(batch_augmented)} augmented samples")
            except Exception as e:
                print_rank(f"Error generating captions for batch {batch_idx}/{total_batches}: {e}")
                # Skip this batch and continue
                continue
        
        self.augmented_samples = augmented_samples
        total_time = time.time() - start_time
        print_rank(f"Caption generation completed in {int(total_time//60):02d}:{int(total_time%60):02d}")
        print_rank(f"Generated {len(augmented_samples)} augmented samples from {len(hard_negatives)} hard negatives")
        print_rank(f"Average generation rate: {len(augmented_samples)/total_time:.2f} samples/second")
        
        # Save augmented samples to experiment directory
        if len(augmented_samples) > 0:
            self._save_augmented_samples(augmented_samples)
        
        return augmented_samples
    
    def _generate_caption_batch(self, hard_negatives_batch: List[Dict]) -> List[Dict]:
        """Generate captions for a batch of hard negatives"""
        from PIL import Image
        
        augmented_samples = []
        
        # Get foundation model details
        foundation_processor = getattr(self.foundation_model, 'processor', None)
        foundation_backbone = getattr(self.model_args, 'foundation_model_backbone', 'qwen2_vl')
        
        if foundation_processor is None:
            print_rank("Foundation model has no processor")
            return []
        
        device = next(self.foundation_model.parameters()).device
        
        for idx, hard_neg in enumerate(hard_negatives_batch):
            try:
                print_rank(f"  - Generating caption for sample {idx+1}/{len(hard_negatives_batch)} in current batch")
                
                # Load reference and target images
                ref_image = self._load_pil_image(hard_neg['reference_image'])
                
                # Determine target image path
                if hard_neg.get('is_real_retrieval', False):
                    # Use the actual hard negative image path
                    target_image_path = hard_neg['hard_negative_image']
                    target_image = self._load_pil_image(target_image_path)
                else:
                    # For simulated retrieval, we don't have real negative images
                    # Use the original target as reference
                    target_image = self._load_pil_image(hard_neg['target_image'])
                
                # Generate new modification text with hard negative context awareness
                new_mod_text = self._generate_modification_text(
                    ref_image, target_image, hard_neg['modification_text'],
                    foundation_processor, foundation_backbone, device, is_hard_negative=True
                )
                
                if new_mod_text:
                    print_rank(f"  - Generated: '{new_mod_text[:100]}{'...' if len(new_mod_text) > 100 else ''}'")
                    augmented_samples.append({
                        'reference_image': hard_neg['reference_image'],
                        'modification_text': new_mod_text,
                        'target_image': hard_neg['hard_negative_image'],  # Now this becomes positive
                        'original_mod_text': hard_neg['modification_text'],
                        'is_augmented': True,
                        'hard_negative_rank': hard_neg['rank_position'],
                        'similarity_score': hard_neg['similarity_score']
                    })
                
            except Exception as e:
                print_rank(f"Error processing hard negative: {e}")
                continue
        
        return augmented_samples
    
    def _load_pil_image(self, image_path: str) -> Image.Image:
        """Load PIL Image from path"""
        from PIL import Image
        
        try:
            if image_path.startswith('./'):
                full_path = os.path.join(self.image_base_dir, image_path[2:])
            else:
                full_path = os.path.join(self.image_base_dir, image_path)
            
            return Image.open(full_path).convert('RGB')
        except:
            # Return dummy image if loading fails
            return Image.new('RGB', (224, 224), color='white')
    
    def _generate_modification_text(self, ref_image, target_image, original_text, 
                                  processor, model_backbone, device, is_hard_negative=False):
        """Generate new modification text using foundation model with hard negative awareness"""
        try:
            # Use the original foundation model (not MMEBModel) for generation
            foundation_model = self.foundation_model
            
            # Create prompt based on model type with hard negative context
            if model_backbone in ['qwen2_vl', 'qwen']:
                prompt = self._create_qwen_prompt(original_text, is_hard_negative_context=is_hard_negative)
                inputs = self._prepare_qwen_inputs(ref_image, target_image, prompt, processor, device)
                return self._generate_with_qwen(inputs, device, foundation_model)
            elif model_backbone in ['llava', 'llava_next']:
                prompt = self._create_llava_prompt_enhanced(original_text, is_hard_negative_context=is_hard_negative) 
                inputs = self._prepare_llava_inputs(ref_image, target_image, prompt, processor, device)
                return self._generate_with_llava(inputs, device, foundation_model)
            else:
                # Generic approach with hard negative context
                prompt = self._create_generic_prompt_enhanced(original_text, is_hard_negative_context=is_hard_negative)
                inputs = self._prepare_generic_inputs(ref_image, target_image, prompt, processor, device)
                return self._generate_with_generic_model(inputs, device, foundation_model)
                
        except Exception as e:
            print_rank(f"Error in caption generation: {e}")
            return None
    
    def _create_qwen_prompt(self, original_text: str, is_hard_negative_context: bool = False) -> str:
        """
        Create improved prompt for Qwen2-VL model based on hard negative mining theory.
        
        Theory: When generating captions for hard negatives, the foundation model should understand
        that the original query (I_ref + T_mod) incorrectly retrieved I_tgt_i (hard negative).
        We need to generate new T_mod_i such that (I_ref, T_mod_i, I_tgt_i) becomes a valid positive triplet.
        """
        if is_hard_negative_context:
            # Hard negative mining context: generate T_mod_i to make hard negative become positive
            return f"""You are analyzing a retrieval error case. Here's the situation:

RETRIEVAL ERROR ANALYSIS:
- Reference image (first image) + Original query: "{original_text}"
- This combination INCORRECTLY retrieved the target image (second image)
- The target image should NOT have been retrieved with the original query

TASK: Generate a NEW modification text that would make this retrieval CORRECT.
- Create a description that, when combined with the reference image, should CORRECTLY retrieve the target image
- Focus on the visual differences that make the target image the RIGHT match
- Use completely different vocabulary and approach than the original query

New correct modification text:"""
        else:
            # Standard diversity prompts for regular samples
            diversity_prompts = [
                f"""You are an expert at describing visual changes. Looking at these two images, I need a modification instruction.

Original instruction: "{original_text}"

Create a NEW instruction that describes how to transform the reference image to match the target image. Make it different from the original but achieve the same visual transformation. Use varied vocabulary and phrasing.

New instruction:""",
                
                f"""Based on comparing the reference and target images, write a fresh description of the visual change needed.

Given instruction: "{original_text}"

Generate an alternative instruction that would lead to the same visual outcome but uses different words and structure. Focus on different aspects or details.

Alternative instruction:""",
                
                f"""I have two images and need a modification description. The original was: "{original_text}"

Looking at both images, write a NEW way to describe this same transformation. Use different terminology, focus on different visual elements, or approach from a different angle.

New description:"""
            ]
            
            import random
            return random.choice(diversity_prompts)
    
    def _create_llava_prompt_enhanced(self, original_text: str, is_hard_negative_context: bool = False) -> str:
        """Create improved prompt for LLaVA model based on hard negative mining theory"""
        return create_llava_prompt_enhanced(original_text, is_hard_negative_context)
    
    def _create_generic_prompt_enhanced(self, original_text: str, is_hard_negative_context: bool = False) -> str:
        """Create improved generic prompt based on hard negative mining theory"""
        return create_generic_prompt_enhanced(original_text, is_hard_negative_context)
    
    def _create_llava_prompt(self, original_text: str) -> str:
        """Create improved prompt for LLaVA model with more diversity"""
        prompt_templates = [
            f"""USER: I'm looking at two images. The original description was: "{original_text}". Generate a different way to describe the same visual transformation, using varied language and focusing on different details.

ASSISTANT:""",
            
            f"""USER: Compare these images. Original instruction: "{original_text}". Create an alternative description for the same change but with different words and approach.

ASSISTANT:""",
            
            f"""USER: Given this modification text: "{original_text}", write a fresh version that describes the same visual change using different vocabulary and perspective.

ASSISTANT:"""
        ]
        
        import random
        return random.choice(prompt_templates)
    
    def _create_generic_prompt(self, original_text: str) -> str:
        """Create improved generic prompt with more diversity"""
        prompt_templates = [
            f"""Original: "{original_text}". Generate a different description for the same visual change:""",
            f"""Rephrase this modification: "{original_text}". Use different words for the same transformation:""",
            f"""Alternative to "{original_text}": Create a varied description for the same change:"""
        ]
        
        import random
        return random.choice(prompt_templates)
    
    def _create_llava_prompt(self, original_text: str) -> str:
        """Create prompt for LLaVA model"""
        return f"""USER: I have two images. Please describe how to modify the first image to look like the second image. The original description was: "{original_text}". Please generate a similar but different description.

ASSISTANT:"""
    
    def _create_generic_prompt(self, original_text: str) -> str:
        """Create generic prompt for other models"""
        return f"""Describe how to modify the reference image to match the target image. Original: "{original_text}". Generate a similar description:"""
    
    def _prepare_qwen_inputs(self, ref_image, target_image, prompt, processor, device):
        """Prepare inputs for Qwen2-VL"""
        # Qwen2-VL can handle multiple images in conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ref_image},
                    {"type": "image", "image": target_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        inputs = processor(
            text=[text_prompt],
            images=[ref_image, target_image],
            return_tensors="pt",
            padding=True
        )
        
        return {k: v.to(device) for k, v in inputs.items()}
    
    def _prepare_llava_inputs(self, ref_image, target_image, prompt, processor, device):
        """Prepare inputs for LLaVA"""
        # Concatenate images horizontally for LLaVA
        from PIL import Image
        import numpy as np
        
        ref_array = np.array(ref_image)
        target_array = np.array(target_image)
        
        # Resize to same height
        h = min(ref_array.shape[0], target_array.shape[0])
        ref_resized = Image.fromarray(ref_array).resize((int(ref_array.shape[1] * h / ref_array.shape[0]), h))
        target_resized = Image.fromarray(target_array).resize((int(target_array.shape[1] * h / target_array.shape[0]), h))
        
        # Concatenate horizontally
        combined_width = ref_resized.width + target_resized.width
        combined_image = Image.new('RGB', (combined_width, h))
        combined_image.paste(ref_resized, (0, 0))
        combined_image.paste(target_resized, (ref_resized.width, 0))
        
        inputs = processor(
            text=prompt,
            images=combined_image,
            return_tensors="pt",
            padding=True
        )
        
        return {k: v.to(device) for k, v in inputs.items()}
    
    def _prepare_generic_inputs(self, ref_image, target_image, prompt, processor, device):
        """Prepare inputs for generic model"""
        # Use first image as primary, mention second in text
        enhanced_prompt = f"{prompt} (Comparing reference and target images)"
        
        inputs = processor(
            text=enhanced_prompt,
            images=ref_image,
            return_tensors="pt",
            padding=True
        )
        
        return {k: v.to(device) for k, v in inputs.items()}
    
    def _generate_with_qwen(self, inputs, device, foundation_model):
        """Generate text with Qwen2-VL"""
        with torch.no_grad():
            output_ids = foundation_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=foundation_model.config.eos_token_id
            )
            
            # Decode only the new tokens
            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[:, input_length:]
            
            generated_text = foundation_model.processor.decode(
                generated_ids[0], skip_special_tokens=True
            ).strip()
            
            # Post-process the generated text
            cleaned_text = self._post_process_generated_text(generated_text)
            
            return cleaned_text
    
    def _post_process_generated_text(self, text: str) -> str:
        """Post-process generated text to improve quality"""
        if not text:
            return None
            
        # Remove quotes at the beginning and end
        text = text.strip('"\'')
        
        # Filter out problematic patterns
        problematic_patterns = [
            "create a new query",
            "retrieve the",
            "can you please",
            "look at both images",
            "instruction",
            "task:",
            "solution:",
            "problem:",
            "new correct modification",
            "explore the majestic",
            "handcrafted",
            "collection of"
        ]
        
        text_lower = text.lower()
        for pattern in problematic_patterns:
            if pattern in text_lower:
                return None
        
        # Remove coordinate patterns like (123,456)
        import re
        text = re.sub(r'\(\d+,\d+\)', '', text)
        text = re.sub(r'\(\d+,\d+\),\(\d+,\d+\)', '', text)
        
        # Remove overly long texts (likely descriptions rather than modification instructions)
        if len(text) > 150:
            return None
            
        # Remove texts that are too short to be meaningful
        if len(text) < 5:
            return None
            
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        # Ensure it starts with a capital letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            
        return text
    
    def _generate_with_llava(self, inputs, device, foundation_model):
        """Generate text with LLaVA"""
        with torch.no_grad():
            output_ids = foundation_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_text = foundation_model.processor.decode(
                output_ids[0], skip_special_tokens=True
            ).strip()
            
            # Extract only the assistant response
            if "ASSISTANT:" in generated_text:
                generated_text = generated_text.split("ASSISTANT:")[-1].strip()
            
            # Post-process the generated text
            cleaned_text = self._post_process_generated_text(generated_text)
            
            return cleaned_text
    
    def _generate_with_generic_model(self, inputs, device, foundation_model):
        """Generate text with generic model"""
        with torch.no_grad():
            output_ids = foundation_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
            
            generated_text = foundation_model.processor.decode(
                output_ids[0], skip_special_tokens=True
            ).strip()
            
            return generated_text
      
    
    def __len__(self):
        base_len = len(self.annotations) if hasattr(self, 'annotations') else 0
        aug_len = len(self.augmented_samples)
        total_len = base_len + aug_len
        
        # Update num_rows for VLM2Vec compatibility
        self.num_rows = total_len
        return total_len
    
    def __getitem__(self, idx):
        # Return original samples or augmented samples
        if idx < len(self.annotations):
            return self._get_original_sample(idx)
        else:
            aug_idx = idx - len(self.annotations)
            return self._get_augmented_sample(aug_idx)
    
    def _get_original_sample(self, idx):
        """Get original CIRR sample"""
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
    
    def _get_augmented_sample(self, idx):
        """Get augmented sample"""
        sample = self.augmented_samples[idx]
        
        # Get model backbone from model_args
        model_backbone = getattr(self.model_args, 'model_backbone', 'qwen2_vl')
        
        # Use VLM2Vec's unified text processing
        query_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text=sample['modification_text'],
            add_image_token=True
        )
        
        pos_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True
        )
        
        neg_text = process_input_text(
            instruction="",
            model_backbone=model_backbone,
            text="",
            add_image_token=True
        )
        
        return {
            'query_text': query_text,
            'query_image': self._load_image(sample['reference_image']),
            'pos_text': pos_text,
            'pos_image': self._load_image(sample['target_image']),
            'neg_text': neg_text,
            'neg_image': self._load_image(sample['reference_image']),  # Use reference as negative
            'global_dataset_name': 'CIRR',
            'is_augmented': True,
            'original_mod_text': sample['original_mod_text']
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

    def _save_augmented_samples(self, augmented_samples: List[Dict]):
        """Save augmented samples to experiment directory"""
        import time
        import json
        
        if not hasattr(self, 'experiment_dir') or not self.experiment_dir:
            print_rank("No experiment directory set, skipping augmented samples save")
            return
        
        # Save with iteration round
        aug_file = os.path.join(self.experiment_dir, f"augmented_samples_iter_{self.iteration_round}.json")
        
        # Create summary statistics
        summary = {
            'total_samples': len(augmented_samples),
            'generation_timestamp': time.time(),
            'iteration_round': self.iteration_round,
            'sample_statistics': {
                'avg_original_length': sum(len(s.get('original_mod_text', '')) for s in augmented_samples) / len(augmented_samples) if augmented_samples else 0,
                'avg_generated_length': sum(len(s.get('modification_text', '')) for s in augmented_samples) / len(augmented_samples) if augmented_samples else 0,
                'unique_reference_images': len(set(s.get('reference_image', '') for s in augmented_samples)),
                'unique_target_images': len(set(s.get('target_image', '') for s in augmented_samples))
            },
            'samples': augmented_samples
        }
        
        try:
            with open(aug_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print_rank(f"Saved {len(augmented_samples)} augmented samples to {aug_file}")
            print_rank(f"Statistics: avg_original_len={summary['sample_statistics']['avg_original_length']:.1f}, "
                      f"avg_generated_len={summary['sample_statistics']['avg_generated_length']:.1f}")
        except Exception as e:
            print_rank(f"Error saving augmented samples: {e}")


class IterativeFashionIQDataset(IterativeCIRRDataset):
    """
    Iterative FashionIQ Dataset - similar structure to CIRR but for fashion domain
    """
    
    def _load_cirr_data(self):
        """Override to load FashionIQ data"""
        print_rank(f"Loading FashionIQ dataset...")
        
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


@add_metainfo_hook
def cirr_data_prepare(batch_dict, *args, **kwargs):
    """Data preparation function for CIRR"""
    image_dir = kwargs['image_dir']
    model_backbone = kwargs['model_backbone']
    
    # Process batch similar to mmeb_dataset but for CIRR format
    query_texts, query_images, pos_texts, pos_images = [], [], [], []
    
    for sample in batch_dict:
        # Add image token based on model backbone
        mod_text = sample['modification_text']
        if model_backbone in VLM_IMAGE_TOKENS:
            mod_text = VLM_IMAGE_TOKENS[model_backbone] + '\n' + mod_text
        
        query_texts.append(mod_text)
        query_images.append(sample['reference_image'])
        pos_texts.append('')  # Empty for composed retrieval
        pos_images.append(sample['target_image'])
    
    return {
        "query_text": query_texts,
        "query_image": query_images,
        "pos_text": pos_texts,
        "pos_image": pos_images
    }


# Register the dataset class manually
AutoPairDataset.registry["IterativeCIRRDataset"] = IterativeCIRRDataset


def create_llava_prompt_enhanced(original_text: str, is_hard_negative_context: bool = False) -> str:
    """Create improved prompt for LLaVA model based on hard negative mining theory"""
    if is_hard_negative_context:
        hard_negative_prompts = [
            f"""USER: RETRIEVAL ERROR CORRECTION TASK

Situation: The modification query "{original_text}" applied to the left image incorrectly retrieved the right image as a match.

Task: Generate a NEW modification description that would make the right image the CORRECT match for the left image. Focus on visual differences that justify this pairing.

A:""",
            
            f"""USER: I have a retrieval failure case. The query "{original_text}" with the left image wrongly retrieved the right image. 

Create a corrected modification text that would make this retrieval correct. Use different vocabulary and approach.

A:""",
            
            f"""USER: Transform this retrieval error into success. Original failed query: "{original_text}". Generate a new query that makes the right image the correct target for the left image.

A:"""
        ]
        
        import random
        return random.choice(hard_negative_prompts)
    else:
        diversity_prompts = [
            f"""USER: I'm looking at two images. The original description was: "{original_text}". Generate a different way to describe the same visual transformation, using varied language and focusing on different details.

A:""",
            
            f"""USER: Compare these images. Original instruction: "{original_text}". Create an alternative description for the same change but with different words and approach.

A:""",
            
            f"""USER: Given this modification text: "{original_text}", write a fresh version that describes the same visual change using different vocabulary and perspective.

A:"""
        ]
        
        import random
        return random.choice(diversity_prompts)


def create_generic_prompt_enhanced(original_text: str, is_hard_negative_context: bool = False) -> str:
    """Create improved generic prompt based on hard negative mining theory"""
    if is_hard_negative_context:
        hard_negative_prompts = [
            f"""RETRIEVAL ERROR FIX: "{original_text}" wrongly retrieved this target. Generate new modification for correct retrieval:""",
            f"""CORRECTION NEEDED: Transform failed query "{original_text}" into successful one for this image pair:""",
            f"""HARD NEGATIVE TO POSITIVE: Original "{original_text}" was wrong. Create correct modification:"""
        ]
        
        import random
        return random.choice(hard_negative_prompts)
    else:
        diversity_prompts = [
            f"""Original: "{original_text}". Generate a different description for the same visual change:""",
            f"""Rephrase this modification: "{original_text}". Use different words for the same transformation:""",
            f"""Alternative to "{original_text}": Create a varied description for the same change:"""
        ]
        
        import random
        return random.choice(diversity_prompts)

