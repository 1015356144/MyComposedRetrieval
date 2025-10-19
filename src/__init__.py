"""
MyComposedRetrieval: Iterative Training for Composed Image Retrieval

This package implements iterative training with hard negative mining and foundation model augmentation
for composed image retrieval tasks, built on top of VLM2Vec.
"""

__version__ = "1.0.0"

# Core modules (lazily import to avoid circular deps)
from . import data  # re-exports datasets/collators
from . import model
from . import utils

# Main classes
from .trainer_iterative_ import IterativeRetrievalTrainer
from .data.dataset.cirr import IterativeCIRRDataset
from .data.dataset.fashioniq import IterativeFashionIQDataset

__all__ = [
    'IterativeRetrievalTrainer',
    'IterativeCIRRDataset',
    'IterativeFashionIQDataset',
    'data',
    'model',
    'utils'
]
