"""
chunker.py
Groups raw transcript segments into overlapping, timestamp-tagged chunks.

Every pipeline — captions, ASR, visual OCR — converges here, so the input is
a plain list of {text, start, duration} segments regardless of where it came
from.

Chunking is by TIME, not by line count. Caption lines run from two to fifteen
words, so a fixed "5 lines per chunk" produces chunks that vary about
sevenfold in length; BM25's document-length normalisation then fights the
chunker, and retrieval quality — and therefore every number in the evaluation
— moves for reasons that have nothing to do with the retrieval method.
"""

from typing import Dict, List

import config


def format_timestamp(seconds: float) -> str:
    """Seconds → 'M:SS', or 'H:MM:SS' past the hour."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chunk_segments(
    segments: List[Dict],
    window: float = None,
    overlap: float = None,
) -> List[Dict]:
    """
    Args:
        segments: [{"text": str, "start": float, "duration": float}, ...]
        window:   chunk length in seconds  (default config.CHUNK_SECONDS)
        overlap:  seconds shared with the next chunk (default config.CHUNK_OVERLAP_S)

    Returns:
        [{chunk_id, text, start, end, timestamp}, ...]
    """
    window = window or config.CHUNK_SECONDS
    overlap = config.CHUNK_OVERLAP_S if overlap is None else overlap

    segments = [s for s in segments if (s.get("text") or "").strip()]
    if not segments:
        return []

    if overlap >= window:                      # guard against a zero/negative stride
        overlap = window / 3.0
    step = window - overlap

    segments.sort(key=lambda s: s["start"])
    last_end = max(s["start"] + s.get("duration", 0.0) for s in segments)

    chunks: List[Dict] = []
    cursor = segments[0]["start"]
    idx = 0
    seen: set = set()

    while cursor < last_end:
        win_end = cursor + window
        window_segs = [
            s for s in segments
            if s["start"] < win_end and (s["start"] + s.get("duration", 0.0)) > cursor
        ]

        if window_segs:
            text = " ".join(s["text"].strip() for s in window_segs)
            start = window_segs[0]["start"]
            end = max(s["start"] + s.get("duration", 0.0) for s in window_segs)

            # A slow stretch can produce a window identical to the previous one.
            key = (round(start, 2), text)
            if key not in seen:
                seen.add(key)
                chunks.append({
                    "chunk_id":  f"chunk_{idx}",
                    "text":      text,
                    "start":     round(start, 2),
                    "end":       round(end, 2),
                    "timestamp": format_timestamp(start),
                })
                idx += 1

        cursor += step

    return chunks
