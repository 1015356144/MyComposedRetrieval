"""
MyComposedRetrieval: Iterative Training for Composed Image Retrieval

This package implements iterative training with hard negative mining and foundation model augmentation
for composed image retrieval tasks, built on top of VLM2Vec.
"""

__version__ = "1.0.0"

# Core modules
from . import data
from . import model
from . import utils
from . import trainer_iterative

# Main classes
from .trainer_iterative import IterativeRetrievalTrainer
from .data.dataset.composed_retrieval_dataset import IterativeCIRRDataset, IterativeFashionIQDataset

__all__ = [
    'IterativeRetrievalTrainer',
    'IterativeCIRRDataset',
    'IterativeFashionIQDataset',
    'data',
    'model',
    'utils'
]