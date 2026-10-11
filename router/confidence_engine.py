"""Combined confidence: C_total = w1*C_text_logprob + w2*C_vision + w3*C_rag_relevance.

When a component is unavailable (no attachment, or no retrieval in direct mode) its weight is
redistributed proportionally over the available components, so C_total stays in [0, 1].
A model output that is not valid JSON always scores 0 and therefore always falls back.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import config


@dataclass
class ConfidenceBreakdown:
    c_total: float
    c_text: float
    c_vision: float | None
    c_rag: float | None
    weights: dict[str, float]
    valid: bool


def text_confidence(logprobs: Sequence[float]) -> float:
    """Geometric-mean token probability of the generated JSON, exp(mean logprob)."""
    if not logprobs:
        return 0.0
    return float(min(1.0, max(0.0, math.exp(sum(logprobs) / len(logprobs)))))


def compute_confidence(logprobs: Sequence[float], vision_confidence: float | None,
                       rag_relevance: float | None, valid: bool = True,
                       weights: dict[str, float] | None = None) -> ConfidenceBreakdown:
    base = weights or config.ROUTER_WEIGHTS
    c_text = text_confidence(logprobs)
    parts = {"text": c_text, "vision": vision_confidence, "rag": rag_relevance}
    active = {k: base[k] for k, v in parts.items() if v is not None}
    norm = sum(active.values()) or 1.0
    used = {k: w / norm for k, w in active.items()}
    total = sum(used[k] * min(1.0, max(0.0, parts[k])) for k in used)
    return ConfidenceBreakdown(
        c_total=float(total) if valid else 0.0, c_text=c_text,
        c_vision=vision_confidence, c_rag=rag_relevance, weights=used, valid=valid)
