"""Chunk the knowledge base, embed it and persist a vector index.

Backends:
    chroma (default)  ChromaDB collection (embedded persistent client, or HTTP client when
                      CHROMA_HOST is set, e.g. the docker-compose ``chroma`` service)
    faiss             FAISS inner-product index stored at config.FAISS_PATH
    both              build both

A flat ``chunks.jsonl`` is always written; the BM25 sparse retriever reads it.

Usage:
    python -m data_pipeline.build_vector_store --backend chroma
"""
from __future__ import annotations

import argparse
from typing import Any

import config


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows of ``size`` words."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(articles: list[dict[str, Any]], size: int = config.CHUNK_WORDS,
                 overlap: int = config.CHUNK_OVERLAP_WORDS) -> list[dict[str, Any]]:
    """Turn KB articles into retrievable chunks. Each chunk is prefixed with its title."""
    chunks: list[dict[str, Any]] = []
    for article in articles:
        for index, piece in enumerate(chunk_words(article["content"], size, overlap)):
            chunks.append({
                "chunk_id": f"{article['kb_id']}::c{index}",
                "kb_id": article["kb_id"],
                "title": article["title"],
                "intent": article["intent"],
                "department": article["department"],
                "chunk_index": index,
                "text": f"{article['title']}. {piece}",
            })
    return chunks


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64):
    """Return L2-normalised sentence-transformer embeddings as a float32 numpy array."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype("float32")


def get_chroma_client():
    import chromadb

    if config.CHROMA_HOST:
        return chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    path = config.VECTOR_DIR / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def index_chroma(chunks: list[dict[str, Any]], embeddings, batch: int = 256) -> int:
    client = get_chroma_client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:  # collection may not exist yet
        pass
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION,
                                                 metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.add(
            ids=[c["chunk_id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=embeddings[start:start + batch].tolist(),
            metadatas=[{"kb_id": c["kb_id"], "title": c["title"], "intent": c["intent"],
                        "department": c["department"], "chunk_index": c["chunk_index"]} for c in part],
        )
    return collection.count()


def index_faiss(embeddings) -> int:
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    config.FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_PATH))
    return index.ntotal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["chroma", "faiss", "both"], default="chroma")
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--chunk-words", type=int, default=config.CHUNK_WORDS)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP_WORDS)
    args = parser.parse_args()

    config.ensure_dirs()
    articles = config.read_jsonl(config.KB_PATH)
    chunks = build_chunks(articles, args.chunk_words, args.overlap)
    config.write_jsonl(config.CHUNKS_PATH, chunks)
    print(f"articles: {len(articles)}  chunks: {len(chunks)} -> {config.CHUNKS_PATH}")

    embeddings = embed_texts([c["text"] for c in chunks], args.model)
    if args.backend in ("chroma", "both"):
        print(f"chroma collection '{config.CHROMA_COLLECTION}' size: {index_chroma(chunks, embeddings)}")
    if args.backend in ("faiss", "both"):
        print(f"faiss index vectors: {index_faiss(embeddings)} -> {config.FAISS_PATH}")

    # Sanity check: a known symptom should retrieve its own article.
    probe = "the device keeps rebooting and shows a kernel panic"
    probe_vec = embed_texts([probe], args.model)
    if args.backend in ("chroma", "both"):
        collection = get_chroma_client().get_collection(config.CHROMA_COLLECTION)
        result = collection.query(query_embeddings=probe_vec.tolist(), n_results=3)
        print("probe top-3:", [m["kb_id"] + " " + m["title"] for m in result["metadatas"][0]])


if __name__ == "__main__":
    main()
