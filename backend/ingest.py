"""
ingest.py
Runs the detector, picks a pipeline, and drives it through to an indexed video.

This is where the fallback chain lives. Auto-captions can be present but
empty, and a video can have an audio stream carrying nothing but room tone, so
"the detector said captions" is a starting point rather than a verdict. Each
pipeline that yields nothing hands off to the next, and when they are all
exhausted the result is an explicit "nothing to extract" — never a fabricated
transcript.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import chunker
import detector
import embedder
import retriever
import store
from pipelines import ExtractionError, asr, captions, visual

log = logging.getLogger("vidquery.ingest")

# Order matters: each entry is tried in turn from the detector's choice onward.
CHAIN = [
    (detector.PIPELINE_CAPTIONS, captions.extract),
    (detector.PIPELINE_ASR,      asr.extract),
    (detector.PIPELINE_VISUAL,   visual.extract),
]


def _chain_from(pipeline: str) -> List[Tuple[str, Callable]]:
    names = [name for name, _ in CHAIN]
    start = names.index(pipeline) if pipeline in names else 0
    return CHAIN[start:]


def extract_with_fallback(video_id: str, pipeline: str,
                          language: Optional[str] = None,
                          report: Callable = None) -> Tuple[List[Dict], str, str]:
    """
    Returns (segments, language, pipeline_actually_used).
    Raises ExtractionError with every attempt's reason when all of them fail.
    """
    attempts = []

    for name, extract_fn in _chain_from(pipeline):
        try:
            if report:
                report(f"Trying {name}", None)
            segments, detected = extract_fn(
                video_id,
                language=language if name == pipeline else None,
                progress=report,
            )
            if segments:
                if attempts:
                    log.info("Fell back to %s after: %s", name, "; ".join(attempts))
                return segments, detected, name
            attempts.append(f"{name}: produced nothing")
        except ExtractionError as exc:
            attempts.append(f"{name}: {exc}")
            log.info("Pipeline %s unavailable for %s — %s", name, video_id, exc)
        except Exception as exc:
            attempts.append(f"{name}: {exc}")
            log.warning("Pipeline %s errored for %s: %s", name, video_id, exc)

    raise ExtractionError(
        "No content could be extracted from this video. " + " | ".join(attempts)
    )


def process_video(video_id: str, report: Callable, force: bool = False) -> Dict:
    """
    Full ingest. Designed to be called from a background job; `report` is the
    job's progress callback.
    """
    if not force and store.has_chunks(video_id) and embedder.count_for_video(video_id):
        existing = store.get_video(video_id) or {}
        report("Already indexed", 1.0)
        return {
            "pipeline": existing.get("pipeline", "cached"),
            "language": existing.get("language"),
            "chunk_count": existing.get("chunk_count", 0),
            "cached": True,
        }

    report("Checking video type", 0.05)
    decision = detector.detect(video_id)

    if decision["pipeline"] == detector.PIPELINE_NONE:
        raise ExtractionError(decision["reason"])

    report(decision["reason"], 0.10)

    def extraction_progress(stage: str, fraction: float = None) -> None:
        # Extraction owns 10% to 70% of the overall bar.
        report(stage, None if fraction is None else 0.10 + 0.60 * fraction)

    segments, language, used = extract_with_fallback(
        video_id,
        decision["pipeline"],
        decision.get("language"),
        extraction_progress,
    )

    report("Splitting into chunks", 0.72)
    chunks = chunker.chunk_segments(segments)
    if not chunks:
        raise ExtractionError("Transcript was too short to index")

    store.save_chunks(video_id, chunks)
    retriever.invalidate(video_id)

    if not embedder.model_ready():
        report("Loading the embedding model (first run)", 0.76)
        embedder.warm()          # load here, while the stage says so

    report(f"Indexing {len(chunks)} chunks", 0.78)

    def index_progress(fraction: float) -> None:
        report(f"Indexing {len(chunks)} chunks", 0.78 + 0.20 * fraction)

    embedder.index_chunks(video_id, chunks, language=language, progress=index_progress)
    store.save_video(video_id, used, language, len(chunks))

    report("Ready", 1.0)
    return {
        "pipeline": used,
        "language": language,
        "chunk_count": len(chunks),
        "cached": False,
    }
