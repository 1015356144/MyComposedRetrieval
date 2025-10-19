# -*- coding: utf-8 -*-
"""
Base classes for iterative composed-image-retrieval datasets.
This file only handles:
- common wiring (__len__/__getitem__/shard)
- path normalization and lightweight image I/O helpers
- abstract hooks for dataset-specific loading and sample building

Other concerns (retrieval candidate building, embeddings cache, hard-negative mining,
caption generation, file I/O sync, prompts) are intentionally moved to dedicated modules.
"""

import os
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from torch.utils.data import Dataset

# Keep using your existing utilities to minimize behavior drift
from ...utils import print_rank  # same name as before
from ...model.processor import process_input_text  # keep same import path as your current project
from ...model.processor import VLM_IMAGE_TOKENS     # used by some collators/pipelines


class IterativeRetrievalDataset(Dataset, ABC):
    """
    Abstract base for iterative retrieval datasets (CIRR/FashionIQ, etc.).
    This class does NOT perform retrieval/embedding/captioning itself anymore.
    It focuses on data loading, indexing, and returning samples, while exposing
    stable hooks for higher-level pipelines to call into dedicated modules.
    """

    # --- Public constructor (stable signature) ---
    def __init__(
        self,
        model_args,
        data_args,
        training_args,
        iteration_round: int = 0,
        foundation_model=None,
        experiment_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.iteration_round = iteration_round
        self.foundation_model = foundation_model

        self.dataset_config: Dict[str, Any] = dict(kwargs)  # keep raw config

        # Where downstream modules (HN cache / augmented samples / embeddings cache) will write
        self.experiment_dir = experiment_dir or getattr(training_args, "output_dir", "./experiments/default")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Primary storage, filled by _load_data()
        self.annotations: List[Dict[str, Any]] = []
        self.image_splits: Dict[str, str] = {}
        self.image_base_dir: str = self.dataset_config.get("image_base_dir", "")

        # In-memory augmentation stash (may be filled by caption generator step)
        self.augmented_samples: List[Dict[str, Any]] = []

        # Reference-id bookkeeping so grouped samplers and losses can map
        # paths to stable integer buckets.
        self._reference_id_lookup: Dict[str, int] = {}

        # Optional: where retrieval candidates will be stored by a future step
        self.retrieval_candidates: List[str] = []

        # Load dataset-specific metadata/annotations
        self._load_data()

        # Handy for HF Trainer integrations that query num_rows
        self.num_rows: int = len(self.annotations) + len(self.augmented_samples)

    # ---------- Abstract hooks to implement in subclasses ----------
    @abstractmethod
    def _load_data(self) -> None:
        """Load self.annotations / self.image_splits / self.image_base_dir."""
        raise NotImplementedError

    @abstractmethod
    def _get_original_sample(self, idx: int) -> Dict[str, Any]:
        """
        Return one ORIGINAL sample (NOT augmented) at index `idx`.
        The returned dict should follow your VLM2Vec training format:
            {
              'query_text', 'query_image',
              'pos_text', 'pos_image',
              'neg_text', 'neg_image',
              'global_dataset_name',
              'reference_image',  # (optional) path hint for grouped sampler
              ...
            }
        """
        raise NotImplementedError

    @abstractmethod
    def _get_augmented_sample(self, idx: int) -> Dict[str, Any]:
        """
        Return one AUGMENTED sample at index `idx`.
        Same format as _get_original_sample, but 'is_augmented': True and
        'original_mod_text' are recommended for bookkeeping.
        """
        raise NotImplementedError

    # ---------- Public Dataset interface ----------
    def __len__(self) -> int:
        base_len = len(self.annotations)
        aug_len = len(self.augmented_samples)
        total = base_len + aug_len
        self.num_rows = total  # keep parity with your previous code
        return total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < len(self.annotations):
            return self._get_original_sample(idx)
        else:
            aug_idx = idx - len(self.annotations)
            return self._get_augmented_sample(aug_idx)

    # ---------- Sharding for HF Trainer ----------
    def shard(self, num_shards: int, index: int):
        """
        Consistent with your previous implementation: shallow copy with sliced annotations.
        Augmented samples are NOT sharded here by default—if needed, do it upstream.
        """
        total_samples = len(self.annotations)
        per_shard = (total_samples + num_shards - 1) // num_shards
        start = index * per_shard
        end = min(start + per_shard, total_samples)

        sharded = copy.copy(self)
        sharded.annotations = self.annotations[start:end]
        sharded.num_rows = len(sharded.annotations) + len(sharded.augmented_samples)

        print_rank(
            f"Sharded dataset: GPU {index}/{num_shards} gets samples {start}-{end} "
            f"({len(sharded.annotations)} samples)"
        )
        return sharded

    # ---------- Path & image helpers (shared by subclasses) ----------
    def _get_full_image_path(self, image_path: str) -> str:
        """
        Normalize relative/absolute paths using dataset's base dir.
        Mirrors the logic in your original _get_full_image_path().
        """
        if not isinstance(image_path, str):
            return str(image_path)

        if image_path.startswith("./"):
            return os.path.join(self.image_base_dir, image_path[2:])
        elif os.path.isabs(image_path):
            return image_path
        else:
            return os.path.join(self.image_base_dir, image_path)

    def _load_image(self, image_path: str) -> Dict[str, Any]:
        """
        Return image handle in VLM2Vec-collator friendly dict format.
        Defers actual PIL loading to the collator/processor.
        """
        if isinstance(image_path, str):
            full_path = self._get_full_image_path(image_path)
            if not os.path.exists(full_path):
                print_rank(f"Warning: Image not found at {full_path}")
                full_path = "dummy_image"
        else:
            full_path = str(image_path)

        return {
            "paths": [full_path],
            "bytes": [None],
            "resolutions": [None],
        }

    # ---------- Reference grouping helpers ----------
    def _get_reference_id(self, reference_image: str) -> int:
        """
        Map a reference image path (relative or absolute) to a stable integer id.
        This enables downstream components (samplers, triplet loss) to quickly
        group all samples that share the same reference.
        """
        full_path = self._get_full_image_path(reference_image)
        ref_id = self._reference_id_lookup.get(full_path)
        if ref_id is None:
            ref_id = len(self._reference_id_lookup)
            self._reference_id_lookup[full_path] = ref_id
        return ref_id
