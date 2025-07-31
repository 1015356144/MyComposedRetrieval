"""
Mixed dataset loader for combining multiple datasets
"""

import yaml
from typing import Dict, Any
from torch.utils.data import ConcatDataset

from ..dataset.composed_retrieval_dataset import IterativeCIRRDataset, IterativeFashionIQDataset


def init_mixed_dataset(dataset_config: Dict[str, Any], model_args, data_args, training_args):
    """
    Initialize mixed dataset from configuration
    """
    datasets = []
    
    for dataset_name, config in dataset_config.items():
        if not isinstance(config, dict):
            continue
            
        dataset_parser = config.get('dataset_parser')
        
        if dataset_parser == 'IterativeCIRRDataset':
            dataset = IterativeCIRRDataset(
                data_dir=config['data_dir'],
                image_base_dir=config['image_base_dir'],
                captions_file=config['captions_file'],
                image_splits_file=config['image_splits_file'],
                num_sample_per_subset=config.get('num_sample_per_subset'),
                fast_mode=config.get('fast_mode', False)
            )
        elif dataset_parser == 'IterativeFashionIQDataset':
            dataset = IterativeFashionIQDataset(
                data_dir=config.get('data_dir'),
                image_base_dir=config.get('image_base_dir'),
                categories=config.get('categories', ['dress', 'shirt', 'toptee'])
            )
        else:
            print(f"Unknown dataset parser: {dataset_parser}")
            continue
        
        datasets.append(dataset)
    
    if len(datasets) == 1:
        return datasets[0]
    elif len(datasets) > 1:
        return ConcatDataset(datasets)
    else:
        raise ValueError("No valid datasets found in configuration")