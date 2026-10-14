"""
detector.py
Decides which extraction pipeline a video needs, cheapest check first.

    1. captions        one API call, well under a second
    2. audio streams   yt-dlp metadata only, no download, 2-3 s
    3. silence probe   first 30 s of audio, 5-8 s
    4. frame variance  60 s of 360p video, 15-20 s

The expensive layers only run when the cheap ones were inconclusive, so an
ordinary captioned video costs one HTTP request and nothing else.

One correction against the original routing: a video that is silent but not a
slideshow used to fall through to Whisper. Whisper hallucinates on silence —
it emits subtitle credits and "thank you for watching" learned from its
training data — so silence now routes to the visual pipeline unconditionally,
and a video with neither audio nor readable frames returns an explicit
"nothing to extract" rather than a fabricated transcript.
"""

import logging
import subprocess
from typing import Dict, Optional, Tuple

import config

log = logging.getLogger("vidquery.detector")

PIPELINE_CAPTIONS = "captions"
PIPELINE_ASR      = "asr"
PIPELINE_VISUAL   = "visual"
PIPELINE_NONE     = "none"


# ── Layer 1: captions ──────────────────────────────────────────────────────
def has_captions(video_id: str) -> Tuple[bool, Optional[str], bool]:
    """
    Returns (found, language_code, is_generated).

    Works against youtube-transcript-api 1.x (instance API) and falls back to
    the pre-1.0 static API so an older pin still runs.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        log.warning("youtube-transcript-api not installed")
        return False, None, False

    try:
        if hasattr(YouTubeTranscriptApi, "list"):            # 1.x
            listing = YouTubeTranscriptApi().list(video_id)
        else:                                                # 0.6.x
            listing = YouTubeTranscriptApi.list_transcripts(video_id)

        manual, generated = None, None
        for t in listing:
            if getattr(t, "is_generated", False):
                generated = generated or t
            else:
                manual = manual or t

        chosen = manual or generated
        if chosen is None:
            return False, None, False
        return True, chosen.language_code, bool(getattr(chosen, "is_generated", False))

    except Exception as exc:
        log.info("No captions for %s (%s)", video_id, type(exc).__name__)
        return False, None, False


# ── yt-dlp helpers ─────────────────────────────────────────────────────────
def _ytdlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def has_audio(video_id: str) -> Optional[bool]:
    """
    True / False, or None when it could not be determined (yt-dlp missing or
    extraction blocked). None means "don't trust this layer", not "no audio".
    """
    if not _ytdlp_available():
        return None

    import yt_dlp

    try:
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
        formats = info.get("formats") or []
        return any(f.get("acodec") not in (None, "none") for f in formats)
    except Exception as exc:
        log.warning("Audio probe failed for %s: %s", video_id, exc)
        return None


def is_silent(video_id: str) -> Optional[bool]:
    """
    Downloads a short audio head and checks the encoded size. Silence
    compresses to almost nothing; anything with speech will not.
    """
    if not _ytdlp_available():
        return None

    import yt_dlp

    out = config.MEDIA_DIR / f"{video_id}.probe"
    try:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "worstaudio/bestaudio",
            "outtmpl": str(out) + ".%(ext)s",
            "download_ranges": lambda *_: [
                {"start_time": 0, "end_time": config.SILENCE_PROBE_SECONDS}
            ],
            "force_keyframes_at_cuts": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        produced = list(config.MEDIA_DIR.glob(f"{video_id}.probe*"))
        if not produced:
            return None
        size = max(p.stat().st_size for p in produced)
        for p in produced:
            p.unlink(missing_ok=True)

        log.info("Silence probe for %s: %d bytes", video_id, size)
        return size < config.SILENCE_MAX_BYTES

    except Exception as exc:
        log.warning("Silence probe failed for %s: %s", video_id, exc)
        for p in config.MEDIA_DIR.glob(f"{video_id}.probe*"):
            p.unlink(missing_ok=True)
        return None


# ── The decision ───────────────────────────────────────────────────────────
def detect(video_id: str) -> Dict:
    """
    Returns:
        {pipeline, language, reason, is_generated, checks: {...}}
    """
    checks: Dict[str, object] = {}

    found, language, generated = has_captions(video_id)
    checks["captions"] = found
    if found:
        return {
            "pipeline": PIPELINE_CAPTIONS,
            "language": language,
            "is_generated": generated,
            "reason": f"{'Auto-generated' if generated else 'Manual'} captions ({language})",
            "checks": checks,
        }

    audio = has_audio(video_id)
    checks["has_audio"] = audio

    if audio is False:
        return {
            "pipeline": PIPELINE_VISUAL,
            "language": None,
            "is_generated": False,
            "reason": "No audio stream — reading frames",
            "checks": checks,
        }

    if audio is None and not _ytdlp_available():
        # Without yt-dlp there is no path past captions; say so plainly rather
        # than starting a pipeline that cannot possibly produce anything.
        return {
            "pipeline": PIPELINE_NONE,
            "language": None,
            "is_generated": False,
            "reason": "This video has no captions, and yt-dlp is not installed "
                      "(pip install -r requirements-asr.txt)",
            "checks": checks,
        }

    silent = is_silent(video_id)
    checks["silent"] = silent

    if silent:
        # Silent, whatever the frames look like. Never hand silence to an ASR
        # model — it will invent speech rather than return nothing.
        return {
            "pipeline": PIPELINE_VISUAL,
            "language": None,
            "is_generated": False,
            "reason": "Audio track is silent — reading frames",
            "checks": checks,
        }

    return {
        "pipeline": PIPELINE_ASR,
        "language": None,
        "is_generated": False,
        "reason": "Audio present, no captions — transcribing",
        "checks": checks,
    }
