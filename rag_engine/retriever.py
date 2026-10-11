"""Hybrid retriever: dense vectors + BM25 keyword search, fused with a weighted sum."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "it",
         "we", "our", "this", "that", "with", "be", "as", "at", "by", "from", "i", "my", "you", "your"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class RetrievedChunk:
    chunk_id: str
    kb_id: str
    title: str
    text: str
    dense_score: float
    bm25_score: float
    score: float


@dataclass
class RetrievalResult:
    query: str
    hits: list[RetrievedChunk] = field(default_factory=list)

    @property
    def relevance(self) -> float:
        """C_rag: fused score of the best hit, clipped to [0, 1]."""
        return float(min(1.0, max(0.0, self.hits[0].score))) if self.hits else 0.0

    def kb_ids(self) -> list[str]:
        return [h.kb_id for h in self.hits]

    def context(self, k: int = config.RAG_CONTEXT_K) -> str:
        """Prompt context lines in the same format used for SFT data."""
        return "\n".join(f"- [{h.kb_id}] {h.title}" for h in self.hits[:k]) or "- (none retrieved)"


class HybridRetriever:
    def __init__(self, dense: Any | None = None, chunks: list[dict[str, Any]] | None = None,
                 alpha: float = config.HYBRID_ALPHA, use_dense: bool = True) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks if chunks is not None else config.read_jsonl(config.CHUNKS_PATH)
        self._ids = [c["chunk_id"] for c in self.chunks]
        self._pos = {cid: i for i, cid in enumerate(self._ids)}
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        if use_dense and dense is None:
            from rag_engine.vector_db import DenseIndex

            dense = DenseIndex()
        self.dense = dense if use_dense else None
        self.alpha = alpha if self.dense is not None else 0.0

    def search(self, query: str, k: int = config.RAG_TOP_K) -> RetrievalResult:
        pool = max(3 * k, 10)
        raw = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        peak = float(raw.max()) if raw.size else 0.0
        bm25 = raw / peak if peak > 0 else np.zeros_like(raw)

        dense_scores: dict[str, float] = {}
        if self.dense is not None:
            dense_scores = {cid: max(0.0, sim) for cid, sim in self.dense.search(query, pool)}
        candidates = set(dense_scores)
        candidates.update(self._ids[i] for i in np.argsort(-bm25)[:pool] if bm25[i] > 0)

        scored: list[RetrievedChunk] = []
        for cid in candidates:
            chunk = self.chunks[self._pos[cid]]
            d, b = dense_scores.get(cid, 0.0), float(bm25[self._pos[cid]])
            scored.append(RetrievedChunk(cid, chunk["kb_id"], chunk["title"], chunk["text"], d, b,
                                         self.alpha * d + (1.0 - self.alpha) * b))
        scored.sort(key=lambda h: h.score, reverse=True)

        hits: list[RetrievedChunk] = []
        seen: set[str] = set()
        for hit in scored:                       # best chunk per article
            if hit.kb_id not in seen:
                seen.add(hit.kb_id)
                hits.append(hit)
            if len(hits) == k:
                break
        return RetrievalResult(query=query, hits=hits)
