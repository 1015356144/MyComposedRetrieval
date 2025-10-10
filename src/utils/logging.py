# utils/logging.py
from __future__ import annotations
import os
import sys
import time

def is_dist_initialized():
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

def get_rank() -> int:
    if is_dist_initialized():
        import torch.distributed as dist
        try:
            return dist.get_rank()
        except Exception:
            return 0
    # 环境变量兜底（有些启动器会注入）
    return int(os.environ.get("RANK", "0"))

def is_main_process() -> bool:
    return get_rank() == 0

def _ts() -> str:
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return t

def print_rank(msg: str) -> None:
    """
    分布式与单机统一打印：
    - 自动带时间戳与 rank 前缀
    - 不抛异常，确保训练不中断
    """
    try:
        rank = get_rank()
        sys.stdout.write(f"[{_ts()}][rank {rank}] {msg}\n")
        sys.stdout.flush()
    except Exception:
        # 最保守的兜底
        print(msg)
