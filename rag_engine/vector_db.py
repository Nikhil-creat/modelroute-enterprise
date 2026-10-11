"""Dense vector search over the knowledge base (ChromaDB or FAISS)."""
from __future__ import annotations

import threading

import numpy as np

import config

_EMBEDDER = None
_LOCK = threading.Lock()


def get_embedder(model_name: str | None = None):
    """Process-wide cached sentence-transformer."""
    global _EMBEDDER
    with _LOCK:
        if _EMBEDDER is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _EMBEDDER = SentenceTransformer(model_name or config.EMBEDDING_MODEL, device=device)
    return _EMBEDDER


def embed_query(text: str) -> np.ndarray:
    vec = get_embedder().encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32")


class DenseIndex:
    """Cosine-similarity search returning ``(chunk_id, similarity)`` pairs."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or config.RAG_BACKEND
        self._collection = None
        self._faiss = None
        self._chunk_ids: list[str] = []
        if self.backend == "chroma":
            from data_pipeline.build_vector_store import get_chroma_client

            self._collection = get_chroma_client().get_collection(config.CHROMA_COLLECTION)
        elif self.backend == "faiss":
            import faiss

            self._faiss = faiss.read_index(str(config.FAISS_PATH))
            self._chunk_ids = [c["chunk_id"] for c in config.read_jsonl(config.CHUNKS_PATH)]
        else:
            raise ValueError(f"unknown RAG backend: {self.backend}")

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        vec = embed_query(query)
        if self._collection is not None:
            res = self._collection.query(query_embeddings=vec.tolist(), n_results=k,
                                         include=["distances"])
            return [(cid, 1.0 - float(dist)) for cid, dist in zip(res["ids"][0], res["distances"][0])]
        scores, idx = self._faiss.search(vec, k)
        return [(self._chunk_ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
