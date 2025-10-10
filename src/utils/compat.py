# src/utils/compat.py
import logging as _logging
_logging.basicConfig(level=_logging.DEBUG,
                     format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = _logging.getLogger(__name__)

import os
import torch

def _is_dist_initialized() -> bool:
    try:
        return torch.distributed.is_available() and torch.distributed.is_initialized()
    except Exception:
        return False

def print_rank(message: str):
    """If distributed is initialized, print the rank."""
    if _is_dist_initialized():
        logger.info(f'rank{torch.distributed.get_rank()}: ' + str(message))
    else:
        logger.info(str(message))

def print_master(message: str):
    """If distributed is initialized print only on rank 0."""
    if _is_dist_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(str(message))
    else:
        logger.info(str(message))

def find_latest_checkpoint(output_dir: str):
    """Scan the output directory and return the latest checkpoint path."""
    if not os.path.exists(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None

    # Sort by checkpoint number and return the latest one
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest_checkpoint

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch
