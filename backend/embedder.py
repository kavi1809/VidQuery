"""
embedder.py
Embeds transcript chunks using Sentence-BERT and stores them in ChromaDB.
Also handles BM25 index for hybrid search.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import os


# ── Singleton model loader (load once, reuse) ──────────────────────────────
_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    """Loads the Sentence-BERT model once and caches it."""
    global _model
    if _model is None:
        print("Loading embedding model (first time only, ~30 seconds)...")
        # all-MiniLM-L6-v2: fast, lightweight, great for semantic search
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded.")
    return _model


# ── ChromaDB client ────────────────────────────────────────────────────────
def get_chroma_client() -> chromadb.Client:
    """Returns a persistent ChromaDB client that saves data to disk."""
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def get_or_create_collection(video_id: str) -> chromadb.Collection:
    """
    Returns an existing ChromaDB collection for a video, or creates a new one.
    Each video gets its own collection named by its video ID.
    """
    client = get_chroma_client()
    # Collection names must be alphanumeric + underscores
    safe_name = f"vid_{video_id.replace('-', '_')}"
    collection = client.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for semantic search
    )
    return collection


# ── Main embedding pipeline ────────────────────────────────────────────────
def embed_and_store(video_id: str, chunks: List[Dict]) -> int:
    """
    Embeds all chunks and stores them in ChromaDB.

    Args:
        video_id: YouTube video ID (used as collection name)
        chunks:   List of chunk dicts from transcript.chunk_transcript()

    Returns:
        Number of chunks stored
    """
    if not chunks:
        return 0

    model = get_embedding_model()
    collection = get_or_create_collection(video_id)

    # If collection already has data for this video, skip re-embedding
    if collection.count() > 0:
        print(f"Collection for {video_id} already exists ({collection.count()} chunks). Skipping embedding.")
        return collection.count()

    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    # Store in ChromaDB
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "start":     chunk["start"],
                "end":       chunk["end"],
                "timestamp": chunk["timestamp"],
                "video_id":  video_id,
            }
            for chunk in chunks
        ]
    )

    print(f"Stored {len(chunks)} chunks for video {video_id}.")
    return len(chunks)


def vector_search(video_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """
    Performs semantic (vector) search against a video's chunks.

    Returns top_k most semantically similar chunks.
    """
    model = get_embedding_model()
    collection = get_or_create_collection(video_id)

    if collection.count() == 0:
        return []

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "chunk_id":  results["ids"][0][i],
            "text":      results["documents"][0][i],
            "metadata":  results["metadatas"][0][i],
            "score":     1 - results["distances"][0][i],  # cosine similarity
            "source":    "vector"
        })

    return output


# ── BM25 Keyword Search ────────────────────────────────────────────────────
def build_bm25_index(chunks: List[Dict]) -> tuple:
    """
    Builds a BM25 index from chunks for keyword search.

    Returns:
        (bm25_model, tokenized_corpus, chunks)
    """
    corpus = [chunk["text"] for chunk in chunks]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def bm25_search(bm25, chunks: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
    """
    Performs BM25 keyword search.

    Returns top_k chunks with highest BM25 scores.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair chunks with scores and sort
    scored = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    output = []
    for score, chunk in scored:
        if score > 0:  # only include chunks with some keyword match
            output.append({
                "chunk_id": chunk["chunk_id"],
                "text":     chunk["text"],
                "metadata": {
                    "start":     chunk["start"],
                    "end":       chunk["end"],
                    "timestamp": chunk["timestamp"],
                    "video_id":  chunk.get("video_id", ""),
                },
                "score":  float(score),
                "source": "bm25"
            })

    return output


def collection_exists(video_id: str) -> bool:
    """Check if a video's chunks are already stored in ChromaDB."""
    collection = get_or_create_collection(video_id)
    return collection.count() > 0
