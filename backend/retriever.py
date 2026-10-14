"""
retriever.py
Hybrid retrieval: BM25 over tokens, cosine over vectors, merged by Reciprocal
Rank Fusion, with a bounded click-through boost.

Two things here differ from the original design.

BM25 indexes the CHUNKER's output, not the embedder's. BM25 scores term
frequencies over tokenised text; a 384-dimension vector is not something it
can consume. Drawing it downstream of the embedder implied a dependency that
cannot exist.

RRF is the consumer of both searches, not a third search alongside them. The
original figure drew BM25, vector search and fusion as three peers all feeding
the generator, which leaves fusion with no inputs.
"""

import logging
import math
import threading
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

import config
import embedder
import store

log = logging.getLogger("vidquery.retriever")

# BM25 indexes are rebuilt from SQLite on demand and cached per video, rather
# than rebuilt on every single question.
_bm25_cache: Dict[str, Tuple[BM25Okapi, List[Dict], List[set]]] = {}
_cache_lock = threading.Lock()

def tokenize(text: str) -> List[str]:
    """
    Splits on anything that is not a letter, digit, or combining mark.

    The obvious `\\w+` is wrong for this project's target languages. Python's
    `\\w` covers letters and digits but NOT Unicode marks (category M), so the
    Tamil word கவனம் loses its virama and splits into fragments, and the same
    happens across Devanagari, Bengali, Telugu and Kannada. Keeping marks
    attached to their base character is what lets BM25 match an Indic term at
    all.
    """
    tokens: List[str] = []
    current: List[str] = []

    for char in text.lower():
        if unicodedata.category(char)[0] in ("L", "N", "M"):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))
    return tokens


def invalidate(video_id: str) -> None:
    with _cache_lock:
        _bm25_cache.pop(video_id, None)


def _get_bm25(video_id: str) -> Optional[Tuple[BM25Okapi, List[Dict], List[set]]]:
    """Builds (and caches) the index, the chunks, and per-chunk token sets."""
    with _cache_lock:
        if video_id in _bm25_cache:
            return _bm25_cache[video_id]

    chunks = store.get_chunks(video_id)
    if not chunks:
        return None

    corpus = [tokenize(c["text"]) for c in chunks]
    if not any(corpus):
        return None

    bm25 = BM25Okapi(corpus)
    _use_nonnegative_idf(bm25, corpus)

    built = (bm25, chunks, [set(tokens) for tokens in corpus])
    with _cache_lock:
        _bm25_cache[video_id] = built
    return built


def _use_nonnegative_idf(bm25: BM25Okapi, corpus: List[List[str]]) -> None:
    """
    Replaces Okapi's IDF with the Lucene variant, log(1 + (N-df+0.5)/(df+0.5)).

    Textbook Okapi IDF, log((N-df+0.5)/(df+0.5)), goes NEGATIVE for any term
    in more than half the documents. rank_bm25 tries to patch that by flooring
    negative values at epsilon * average_idf, but on a single video's chunks
    the average is itself negative, so the floor is negative too.

    The consequence is worse than it sounds. Scoring a query like "what does
    she say about failure" then means the common words contribute large
    negative amounts proportional to how often they occur, so chunks are
    effectively ranked by how FEW ordinary words they contain rather than how
    many rare query terms they match — and the one chunk actually containing
    "failure" loses to chunks that merely say "she" less often. BM25 stops
    being a keyword search and becomes noise, which quietly removes half of
    the hybrid retrieval the evaluation is meant to be measuring.

    The Lucene form is always positive: common terms earn a little, rare terms
    earn a lot, and nothing is ever penalised for appearing.
    """
    doc_freq = Counter()
    for document in corpus:
        for term in set(document):
            doc_freq[term] += 1

    total = len(corpus)
    bm25.idf = {
        term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }
    bm25.average_idf = sum(bm25.idf.values()) / len(bm25.idf)


# ── The two searches ───────────────────────────────────────────────────────
def bm25_search(video_id: str, query: str, top_k: int) -> List[Dict]:
    built = _get_bm25(video_id)
    if not built:
        return []

    bm25, chunks, chunk_tokens = built
    tokens = tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    wanted = set(tokens)

    # Relevance is decided by term overlap, not by the sign of the score.
    #
    # Okapi BM25's IDF term, log((N - df + 0.5) / (df + 0.5)), goes NEGATIVE
    # for any term appearing in more than half the documents. A short video
    # chunks into a handful of documents, so ordinary query words land in most
    # of them and every score comes back negative — rank_bm25 floors negative
    # IDF at epsilon * average_idf, which is itself negative once enough terms
    # are common. Filtering on `score > 0` therefore drops every result on
    # exactly the videos most likely to be tested, leaving fusion with one
    # input and quietly reducing hybrid retrieval to vector-only.
    #
    # RRF consumes ranks rather than scores, so the absolute values never
    # reach it. All this needs to do is order the chunks that share a term.
    candidates = [
        (float(score), chunk)
        for score, chunk, present in zip(scores, chunks, chunk_tokens)
        if wanted & present
    ]
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {
            "chunk_id":  c["chunk_id"],
            "text":      c["text"],
            "start":     c["start"],
            "end":       c["end"],
            "timestamp": c["timestamp"],
            "score":     s,
            "source":    "bm25",
        }
        for s, c in candidates[:top_k]
    ]


# ── Fusion ─────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(rankings: List[List[Dict]], k: int = None,
                           boosts: Optional[Dict[str, float]] = None) -> List[Dict]:
    """
    RRF score for a chunk = Σ 1/(rank + k) over the lists it appears in.

    The optional boost is added afterwards and is deliberately tiny. A rank-0
    contribution is 1/60 ≈ 0.0167 and a two-list maximum is about 0.033, so a
    boost of 0.1 — six times the whole range — would let a single click
    override retrieval entirely. config.FEEDBACK_MAX_BOOST caps it at 0.002.
    """
    k = config.RRF_K if k is None else k
    boosts = boosts or {}

    scores: Dict[str, float] = {}
    docs: Dict[str, Dict] = {}
    seen_in: Dict[str, set] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            cid = doc["chunk_id"]
            if cid not in docs:
                docs[cid] = doc
                scores[cid] = 0.0
                seen_in[cid] = set()
            scores[cid] += 1.0 / (rank + k)
            seen_in[cid].add(doc.get("source", "?"))

    fused = []
    for cid, score in scores.items():
        boost = boosts.get(cid, 0.0)
        doc = dict(docs[cid])
        doc["rrf_score"] = round(score, 6)
        doc["boost"] = round(boost, 6)
        doc["final_score"] = round(score + boost, 6)
        doc["matched_by"] = sorted(seen_in[cid])
        fused.append(doc)

    fused.sort(key=lambda d: d["final_score"], reverse=True)
    return fused


def hybrid_search(video_id: str, question: str,
                  keyword_query: Optional[str] = None,
                  top_k: int = None, final_k: int = None) -> List[Dict]:
    """
    Args:
        question:      the user's question, unchanged — this goes to the
                       multilingual encoder, which handles the cross-lingual
                       match natively.
        keyword_query: the question rendered in the transcript's language, for
                       BM25. Defaults to the question itself.
    """
    top_k = top_k or config.RETRIEVE_PER_METHOD
    final_k = final_k or config.RETRIEVE_FINAL

    lexical = bm25_search(video_id, keyword_query or question, top_k)
    semantic = embedder.vector_search(video_id, question, top_k)

    if not lexical and not semantic:
        return []

    boosts = store.chunk_boosts(video_id)
    fused = reciprocal_rank_fusion([semantic, lexical], boosts=boosts)
    return fused[:final_k]


# ── Output shaping ─────────────────────────────────────────────────────────
def timestamps_for(results: List[Dict]) -> List[Dict]:
    """Chip data for the sidebar: where to seek, and what's there."""
    out = []
    for r in results:
        preview = (r.get("text") or "").strip()
        if len(preview) > 110:
            preview = preview[:107].rstrip() + "…"
        out.append({
            "chunk_id":      r["chunk_id"],
            "timestamp":     r.get("timestamp", "0:00"),
            "start_seconds": float(r.get("start", 0.0)),
            "text_preview":  preview,
        })
    return out
