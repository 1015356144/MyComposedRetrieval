# -*- coding: utf-8 -*-
import os
from typing import Iterable

def safe_realpath(p: str) -> str:
    try:
        return os.path.normpath(os.path.realpath(p))
    except Exception:
        return os.path.normpath(p)

def is_image_path(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def join(*xs) -> str:
    return os.path.join(*xs)

def exists(p: str) -> bool:
    return os.path.exists(p)

def norm(p: str) -> str:
    return os.path.normpath(p)

def list_files(root: str, exts: Iterable[str] = None):
    exts = {e.lower() for e in (exts or [])}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not exts or os.path.splitext(fn)[1].lower() in exts:
                yield os.path.join(dirpath, fn)
