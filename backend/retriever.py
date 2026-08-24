"""
retriever.py
Hybrid search: combines BM25 keyword + vector semantic search,
then re-ranks results using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict
from embedder import vector_search, bm25_search, build_bm25_index


def reciprocal_rank_fusion(
    rankings: List[List[Dict]],
    k: int = 60
) -> List[Dict]:
    """
    Merges multiple ranked lists into one using Reciprocal Rank Fusion.

    RRF score for a document = sum of 1/(rank + k) across all lists.
    This ensures a document ranked highly in multiple lists rises to the top,
    even if it's not #1 in any single list.

    Args:
        rankings: A list of ranked result lists (e.g. [vector_results, bm25_results])
        k:        Smoothing constant (default 60, from original RRF paper)

    Returns:
        Merged and re-ranked list of chunks
    """
    score_map: Dict[str, float] = {}
    doc_map:   Dict[str, Dict]  = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            cid = doc["chunk_id"]
            if cid not in score_map:
                score_map[cid] = 0.0
                doc_map[cid]   = doc
            score_map[cid] += 1.0 / (rank + k)

    # Sort by descending RRF score
    sorted_ids = sorted(score_map, key=lambda x: score_map[x], reverse=True)

    results = []
    for cid in sorted_ids:
        doc = doc_map[cid].copy()
        doc["rrf_score"] = round(score_map[cid], 6)
        results.append(doc)

    return results


def hybrid_search(
    video_id: str,
    query: str,
    chunks: List[Dict],
    top_k: int = 8,
    final_k: int = 4
) -> List[Dict]:
    """
    Full hybrid search pipeline:
        1. BM25 keyword search  → top_k results
        2. Vector semantic search → top_k results
        3. RRF fusion           → merged + re-ranked
        4. Return final_k best

    Args:
        video_id: YouTube video ID
        query:    User's natural language question
        chunks:   All transcript chunks (needed for BM25)
        top_k:    How many results to fetch from each method
        final_k:  How many final results to return after fusion

    Returns:
        List of the best matching chunks, each with timestamp metadata
    """
    # Step 1: BM25 keyword search
    bm25_model, indexed_chunks = build_bm25_index(chunks)
    bm25_results = bm25_search(bm25_model, indexed_chunks, query, top_k=top_k)

    # Step 2: Vector semantic search
    vector_results = vector_search(video_id, query, top_k=top_k)

    # If one method returns nothing, fall back to the other
    if not bm25_results and not vector_results:
        return []
    if not bm25_results:
        return vector_results[:final_k]
    if not vector_results:
        return bm25_results[:final_k]

    # Step 3: Fuse rankings
    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    # Step 4: Return top final_k
    return fused[:final_k]


def format_context(results: List[Dict]) -> str:
    """
    Formats retrieved chunks into a clean context string for the LLM prompt.
    Includes timestamp labels so the LLM can reference them.

    Example output:
        [0:45] The transformer architecture uses self-attention to ...
        [1:30] Each attention head learns a different representation ...
    """
    if not results:
        return "No relevant content found in the video transcript."

    lines = []
    for r in results:
        meta = r.get("metadata", {})
        timestamp = meta.get("timestamp", "?:??")
        text = r.get("text", "")
        lines.append(f"[{timestamp}] {text}")

    return "\n\n".join(lines)


def extract_timestamps(results: List[Dict]) -> List[Dict]:
    """
    Extracts timestamp info from results for the frontend to use
    to create clickable jump links in the extension sidebar.

    Returns:
        List of {timestamp, start_seconds, text_preview}
    """
    timestamps = []
    for r in results:
        meta = r.get("metadata", {})
        timestamps.append({
            "timestamp":     meta.get("timestamp", "0:00"),
            "start_seconds": meta.get("start", 0),
            "text_preview":  r.get("text", "")[:80] + "...",
        })
    return timestamps
