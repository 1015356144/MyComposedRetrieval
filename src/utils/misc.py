# utils/misc.py
from __future__ import annotations
from typing import Iterable, Iterator, List, TypeVar
import random
import numpy as np
import torch

T = TypeVar("T")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def chunked(it: Iterable[T], size: int) -> Iterator[List[T]]:
    buf: List[T] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
