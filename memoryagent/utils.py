from __future__ import annotations

import re
from typing import Iterable, List, Set


_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def unique_tokens(text: str) -> Set[str]:
    return set(tokenize(text))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def hash_embed(text: str, dim: int) -> List[float]:
    tokens = tokenize(text)
    if dim <= 0:
        raise ValueError("dim must be positive")
    vector = [0.0] * dim
    for token in tokens:
        idx = hash(token) % dim
        vector[idx] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]
