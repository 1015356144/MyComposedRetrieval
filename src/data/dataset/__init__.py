# Re-export datasets and (optionally) register to your AutoPairDataset registry.

from .base_iterative_dataset import IterativeRetrievalDataset
from .cirr import IterativeCIRRDataset
from .fashioniq import IterativeFashionIQDataset

# Optional: keep backward compatibility with your old registry mechanism
try:
    # If your project has this registry (as in your original file)
    from .base_pair_dataset import AutoPairDataset  # adjust path if needed
    AutoPairDataset.registry["IterativeCIRRDataset"] = IterativeCIRRDataset
    AutoPairDataset.registry["IterativeFashionIQDataset"] = IterativeFashionIQDataset
except Exception:
    # If registry is not available at import time, just ignore—caller can register later.
    pass

__all__ = [
    "IterativeRetrievalDataset",
    "IterativeCIRRDataset",
    "IterativeFashionIQDataset",
]
