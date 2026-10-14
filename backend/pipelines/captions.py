"""
pipelines/captions.py
Pulls YouTube's own caption track.

The transcript is indexed in its ORIGINAL language. The original design ran
every segment through a translation service before chunking; for a one-hour
video that is well over a thousand calls to a free endpoint, which is slow,
rate-limited, and loses information before it is ever indexed. The
multilingual encoder is there precisely so that step is unnecessary — only the
question gets translated, and only for BM25's benefit.

Supports youtube-transcript-api 1.x (instance API, returns objects) and the
pre-1.0 static API (returns dicts).
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from . import ExtractionError

log = logging.getLogger("vidquery.captions")

# Sound cues carry no searchable content: "[Music]", "[Applause]", "[♪♪♪]",
# and bare note runs used to mark instrumental passages.
_SOUND_CUE = re.compile(r"\[[^\]]{0,30}\]")
_NOTES = re.compile(r"[♪♫]+")
_EMPTY = re.compile(r"^[\s\-_~*·.]*$")


def _normalise(raw) -> List[Dict]:
    """Both API generations, and both item shapes, into one flat form."""
    segments = []
    for item in raw:
        if isinstance(item, dict):
            text = item.get("text", "")
            start = item.get("start", 0.0)
            duration = item.get("duration", 0.0)
        else:                                   # FetchedTranscriptSnippet (1.x)
            text = getattr(item, "text", "")
            start = getattr(item, "start", 0.0)
            duration = getattr(item, "duration", 0.0)

        text = (text or "").replace("\n", " ")
        text = _SOUND_CUE.sub(" ", text)
        text = _NOTES.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text or _EMPTY.match(text):
            continue

        segments.append({
            "text": text,
            "start": float(start or 0.0),
            "duration": float(duration or 0.0),
        })
    return segments


def extract(video_id: str, language: Optional[str] = None,
            progress=None) -> Tuple[List[Dict], str]:
    """
    Returns (segments, language_code).

    Raises ExtractionError when no usable caption track exists — the caller
    then falls through to the next pipeline.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise ExtractionError(
            "youtube-transcript-api is not installed"
        ) from exc

    if progress:
        progress("Fetching captions")

    wanted = [language] if language else None
    detected = language or "unknown"

    try:
        if hasattr(YouTubeTranscriptApi, "fetch"):           # 1.x
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=wanted) if wanted else api.fetch(video_id)
            detected = getattr(fetched, "language_code", detected)
            raw = list(fetched)
        else:                                                # 0.6.x
            raw = YouTubeTranscriptApi.get_transcript(
                video_id, languages=wanted or ["en"]
            )
    except Exception as exc:
        raise ExtractionError(f"Could not fetch captions: {exc}") from exc

    segments = _normalise(raw)
    if not segments:
        raise ExtractionError("Caption track is present but empty")

    words = sum(len(s["text"].split()) for s in segments)
    if words < 20:
        raise ExtractionError(
            f"Caption track has only {words} words — not enough to index"
        )

    log.info("Captions for %s: %d segments, %d words, lang=%s",
             video_id, len(segments), words, detected)

    if progress:
        progress("Captions ready", 1.0)

    return segments, detected
