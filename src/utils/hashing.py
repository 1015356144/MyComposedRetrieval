# -*- coding: utf-8 -*-
import hashlib
from typing import Iterable

def md5_of_list(items: Iterable[str]) -> str:
    m = hashlib.md5()
    for it in sorted(map(str, items)):
        m.update(it.encode("utf-8"))
    return m.hexdigest()

def small_hash(items: Iterable[str], n: int = 8) -> str:
    return md5_of_list(items)[:n]
