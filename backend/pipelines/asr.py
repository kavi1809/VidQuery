"""
pipelines/asr.py
Speech recognition for videos with audio but no captions.

Uses faster-whisper rather than openai-whisper: identical model weights, run
through CTranslate2, roughly four to eight times quicker on CPU at int8 with
equivalent accuracy. On a laptop that is the difference between a one-hour
lecture taking forty minutes and taking six — which is the difference between
the pipeline being usable and being a demo that never finishes.

Install with:  pip install -r requirements-asr.txt
"""

import logging
import shutil
from typing import Dict, List, Optional, Tuple

import config
from . import ExtractionError

log = logging.getLogger("vidquery.asr")

_model = None


def available() -> Tuple[bool, str]:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False, "faster-whisper is not installed (pip install -r requirements-asr.txt)"
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return False, "yt-dlp is not installed (pip install -r requirements-asr.txt)"
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg is not on PATH — required to decode downloaded audio"
    return True, ""


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        size = config.__dict__.get("WHISPER_SIZE", "base")
        log.info("Loading faster-whisper '%s' (int8, CPU)", size)
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def _download_audio(video_id: str, progress=None) -> str:
    import yt_dlp

    target = config.MEDIA_DIR / video_id

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                progress("Downloading audio", 0.05 + 0.20 * (done / total))

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(target) + ".%(ext)s",
        "progress_hooks": [hook],
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    produced = sorted(config.MEDIA_DIR.glob(f"{video_id}.*"))
    audio = next((p for p in produced if p.suffix == ".mp3"), None) or \
            (produced[0] if produced else None)

    if audio is None:
        raise ExtractionError(
            "yt-dlp produced no audio file. YouTube changes extraction "
            "regularly — try upgrading yt-dlp first."
        )
    return str(audio)


def extract(video_id: str, language: Optional[str] = None,
            progress=None) -> Tuple[List[Dict], str]:
    """Returns (segments, detected_language_code)."""
    ok, reason = available()
    if not ok:
        raise ExtractionError(reason)

    if progress:
        progress("Downloading audio", 0.05)

    audio_path = _download_audio(video_id, progress)

    try:
        if progress:
            progress("Transcribing", 0.30)

        model = _get_model()
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,          # drops silence, which is where ASR invents speech
            beam_size=5,
        )

        detected = getattr(info, "language", None) or language or "unknown"
        total = getattr(info, "duration", 0.0) or 0.0

        segments: List[Dict] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if text:
                segments.append({
                    "text": text,
                    "start": float(seg.start),
                    "duration": float(seg.end - seg.start),
                })
            if progress and total:
                progress("Transcribing", 0.30 + 0.65 * min(1.0, seg.end / total))

        if not segments:
            raise ExtractionError(
                "No speech detected in the audio track"
            )

        log.info("ASR for %s: %d segments, lang=%s", video_id, len(segments), detected)
        return segments, detected

    finally:
        try:
            from pathlib import Path
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass
