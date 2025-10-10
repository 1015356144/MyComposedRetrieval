# -*- coding: utf-8 -*-
import os, json, tempfile

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _atomic_replace(tmp_path: str, dst_path: str):
    if os.path.exists(dst_path):
        os.remove(dst_path)
    os.replace(tmp_path, dst_path)

def write_json_atomic(obj, dst_path: str, *, mkdir: bool = True, fsync: bool = True, indent: int = 2):
    """以原子方式写 JSON（先写临时文件，再替换），可选 fsync 落盘。"""
    if mkdir:
        ensure_dir(os.path.dirname(dst_path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(dst_path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        _atomic_replace(tmp, dst_path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def write_bytes_atomic(data: bytes, dst_path: str, *, mkdir: bool = True, fsync: bool = True):
    if mkdir:
        ensure_dir(os.path.dirname(dst_path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(dst_path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        _atomic_replace(tmp, dst_path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
