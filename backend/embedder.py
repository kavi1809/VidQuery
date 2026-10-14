"""
embedder.py
Dense embeddings and the Chroma vector store.

Two interchangeable backends, selected by EMBEDDING_BACKEND:

    sbert   paraphrase-multilingual-MiniLM-L12-v2, local, offline, no quota
    gemini  hosted embedding API, no torch install

Both index the transcript in its ORIGINAL language. The document's design
translated every caption segment to English before embedding, which for a
one-hour video means well over a thousand calls to a free translation
endpoint — slow, rate-limited, and lossy. A multilingual encoder puts a Tamil
question and a Hindi transcript in the same vector space directly, which is
the entire reason for choosing one.

All videos share a single collection, filtered by a video_id metadata field.
"""

import logging
import threading
from typing import Dict, List, Optional

import chromadb

import config
import gemini

log = logging.getLogger("vidquery.embedder")

_model = None
_model_lock = threading.Lock()
_client = None
_collection = None


# ── Backends ───────────────────────────────────────────────────────────────
def _get_sbert():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                log.info("Loading %s (first run downloads ~470 MB)", config.SBERT_MODEL)
                _model = SentenceTransformer(config.SBERT_MODEL)
                log.info("Embedding model ready")
    return _model


def embed(texts: List[str], is_query: bool = False) -> List[List[float]]:
    if not texts:
        return []

    if config.EMBEDDING_BACKEND == "gemini":
        return gemini.embed_texts(
            texts, task="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        )

    model = _get_sbert()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def embedding_backend() -> str:
    return "gemini" if config.EMBEDDING_BACKEND == "gemini" else "sbert"


def model_ready() -> bool:
    """
    False while the local encoder still has to be loaded.

    Loading takes ten to thirty seconds and happens lazily inside the first
    encode, so without this the job sits at "Indexing chunks" for half a
    minute doing something else entirely.
    """
    return config.EMBEDDING_BACKEND == "gemini" or _model is not None


def warm() -> None:
    """Forces the lazy load now, so the caller can label the wait correctly."""
    if config.EMBEDDING_BACKEND != "gemini":
        _get_sbert()


# ── Chroma ─────────────────────────────────────────────────────────────────
def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _doc_id(video_id: str, chunk_id: str) -> str:
    """Chunk ids are per-video, so they need namespacing in a shared collection."""
    return f"{video_id}::{chunk_id}"


def index_chunks(video_id: str, chunks: List[Dict],
                 language: Optional[str] = None,
                 progress=None) -> int:
    """
    Embeds and stores chunks, replacing anything already held for this video.

    Args:
        progress: optional callable(fraction: float) for job status reporting.
    """
    if not chunks:
        return 0

    collection = get_collection()
    drop_video(video_id)

    total = len(chunks)

    # Sized for progress reporting, not throughput. A fixed large batch means
    # a typical video is a single batch, so the bar jumps once and then sits
    # still through the slowest part of ingest. Aiming for ~8 batches keeps it
    # moving without meaningful encoding overhead.
    ceiling = 64 if config.EMBEDDING_BACKEND == "gemini" else 128
    batch_size = max(8, min(ceiling, -(-total // 8)))
    stored = 0

    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        vectors = embed([c["text"] for c in batch])

        collection.add(
            ids=[_doc_id(video_id, c["chunk_id"]) for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "video_id":  video_id,
                    "chunk_id":  c["chunk_id"],
                    "start":     float(c["start"]),
                    "end":       float(c["end"]),
                    "timestamp": c["timestamp"],
                    "language":  language or "unknown",
                }
                for c in batch
            ],
        )
        stored += len(batch)
        if progress:
            progress(stored / total)

    log.info("Indexed %d chunks for %s", stored, video_id)
    return stored


def drop_video(video_id: str) -> None:
    """Removes every vector belonging to one video."""
    try:
        get_collection().delete(where={"video_id": video_id})
    except Exception as exc:
        log.warning("Could not clear vectors for %s: %s", video_id, exc)


def count_for_video(video_id: str) -> int:
    try:
        got = get_collection().get(where={"video_id": video_id}, include=[])
        return len(got.get("ids", []))
    except Exception:
        return 0


def vector_search(video_id: str, query: str, top_k: int) -> List[Dict]:
    """Cosine search restricted to one video."""
    collection = get_collection()
    query_vec = embed([query], is_query=True)
    if not query_vec:
        return []

    try:
        res = collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            where={"video_id": video_id},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        log.warning("Vector search failed for %s: %s", video_id, exc)
        return []

    if not res.get("ids") or not res["ids"][0]:
        return []

    out = []
    for i, _ in enumerate(res["ids"][0]):
        meta = res["metadatas"][0][i] or {}
        out.append({
            "chunk_id":  meta.get("chunk_id", res["ids"][0][i].split("::")[-1]),
            "text":      res["documents"][0][i],
            "start":     meta.get("start", 0.0),
            "end":       meta.get("end", 0.0),
            "timestamp": meta.get("timestamp", "0:00"),
            "score":     1.0 - float(res["distances"][0][i]),
            "source":    "vector",
        })
    return out
