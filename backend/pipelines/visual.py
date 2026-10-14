"""
pipelines/visual.py
Reads slide-only videos: no captions, no speech, everything on screen.

Six layers, in order:

    A  frame extraction     one frame every 5 s, via OpenCV
    B  pHash dedup          720 raw frames down to 30-80 real slides
    C  adaptive preprocess  invert dark backgrounds, CLAHE low contrast
    D  parallel OCR         PaddleOCR and EasyOCR, in processes not threads
    E  normalised voting    agreement score after aggressive normalisation
    F  confidence gate      Gemini vision on frames OCR could not read

Two departures from the original design.

TrOCR is gone. `trocr-base-printed` is a single-LINE recognizer with no text
detection stage — handed a full slide it returns a short hallucinated
fragment, because it was never built to find text, only to read a crop it was
given. PaddleOCR and EasyOCR are both complete detect-then-recognise
pipelines, so the ensemble is two engines that can actually do the job rather
than three where one cannot.

Voting normalises before comparing. Raw Levenshtein similarity between two
CORRECT full-page OCR outputs is often 0.5-0.6, because the engines return
regions in different reading orders with different line breaks. Comparing raw
strings would push almost every frame below the 0.7 gate and send the whole
video to the vision model, which defeats the point of having a gate.

The gate's fallback is Gemini vision rather than a local LLaVA — one hosted
dependency instead of an Ollama install and a GPU.

Install with:  pip install -r requirements-visual.txt
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import config
from . import ExtractionError

log = logging.getLogger("vidquery.visual")

FRAME_INTERVAL_S = 5
PHASH_THRESHOLD  = 8
CONF_HIGH = 0.70          # OCR trusted outright
CONF_LOW  = 0.40          # below this, the frame is probably not text


def available() -> Tuple[bool, str]:
    missing = []
    for mod, label in (("cv2", "opencv-python"), ("imagehash", "imagehash"),
                       ("PIL", "pillow")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(label)

    engines = []
    for mod in ("paddleocr", "easyocr"):
        try:
            __import__(mod)
            engines.append(mod)
        except ImportError:
            pass

    if missing:
        return False, f"missing: {', '.join(missing)} (pip install -r requirements-visual.txt)"
    if not engines:
        return False, "no OCR engine installed (pip install -r requirements-visual.txt)"
    return True, ""


# ── Layer A: frames ────────────────────────────────────────────────────────
def _download_video(video_id: str, progress=None) -> str:
    import yt_dlp

    target = config.MEDIA_DIR / f"{video_id}_v"
    opts = {
        "quiet": True, "no_warnings": True,
        "format": "worst[height>=360]/worst",
        "outtmpl": str(target) + ".%(ext)s",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    produced = sorted(config.MEDIA_DIR.glob(f"{video_id}_v.*"))
    if not produced:
        raise ExtractionError("Could not download video for frame extraction")
    return str(produced[0])


def extract_frames(video_path: str, interval: int = FRAME_INTERVAL_S) -> List[Dict]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(fps * interval))

    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append({"image": frame, "start": idx / fps})
        idx += 1
    cap.release()

    log.info("Extracted %d frames at %ds intervals", len(frames), interval)
    return frames


# ── Layer B: perceptual hash dedup ─────────────────────────────────────────
def dedup_frames(frames: List[Dict], threshold: int = PHASH_THRESHOLD) -> List[Dict]:
    """
    Average pixel difference is fragile against compression artifacts and
    brightness drift. A perceptual hash fingerprints visual structure, so two
    renderings of the same slide collapse together.
    """
    import cv2
    import imagehash
    from PIL import Image

    if not frames:
        return []

    def phash(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return imagehash.phash(Image.fromarray(rgb))

    unique = [frames[0]]
    prev = phash(frames[0]["image"])

    for frame in frames[1:]:
        current = phash(frame["image"])
        if (current - prev) > threshold:      # Hamming distance
            unique.append(frame)
            prev = current

    log.info("pHash dedup: %d frames to %d slides", len(frames), len(unique))
    return unique


# ── Layer C: adaptive preprocessing ────────────────────────────────────────
def preprocess(image):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 80:                                  # dark slide
        gray = cv2.bitwise_not(gray)
    if gray.std() < 40:                                   # flat contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return cv2.fastNlMeansDenoising(gray, h=10)


# ── Layer D: parallel OCR ──────────────────────────────────────────────────
# Threads buy almost nothing here: these are compute-bound models contending
# for the same cores, so a ThreadPoolExecutor yields perhaps 1.2x rather than
# the 3x a naive reading suggests. Processes give real parallelism.

def _run_paddle(image_path: str) -> str:
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image_path, cls=True)
        lines = []
        for page in (result or []):
            for entry in (page or []):
                if entry and len(entry) > 1:
                    lines.append(entry[1][0])
        return "\n".join(lines)
    except Exception as exc:
        log.warning("PaddleOCR failed: %s", exc)
        return ""


def _run_easyocr(image_path: str) -> str:
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return "\n".join(reader.readtext(image_path, detail=0))
    except Exception as exc:
        log.warning("EasyOCR failed: %s", exc)
        return ""


def ensemble_ocr(image_path: str) -> Dict[str, str]:
    with ProcessPoolExecutor(max_workers=2) as pool:
        paddle = pool.submit(_run_paddle, image_path)
        easy = pool.submit(_run_easyocr, image_path)
        return {"paddle": paddle.result(), "easyocr": easy.result()}


# ── Layer E: normalised voting ─────────────────────────────────────────────
def _normalise(text: str) -> str:
    """
    Strips everything the engines legitimately disagree about: case, spacing,
    punctuation, and line order. What remains is the content they either agree
    on or do not.
    """
    import re
    lines = [
        re.sub(r"[^\w\s]", "", ln.lower()).strip()
        for ln in text.splitlines()
    ]
    lines = [re.sub(r"\s+", " ", ln) for ln in lines if ln.strip()]
    return "\n".join(sorted(lines))


def vote(candidates: Dict[str, str]) -> Tuple[str, float]:
    """
    Returns (best_text, agreement).

    Agreement measures CONSENSUS, not correctness — two engines with similar
    training data can agree confidently on the same misreading of a stylised
    font. It gates how much work the frame gets, nothing more.
    """
    texts = [t for t in candidates.values() if t.strip()]
    if not texts:
        return "", 0.0
    if len(texts) == 1:
        return texts[0], 0.5          # nothing to agree with; treat as uncertain

    try:
        from Levenshtein import ratio
    except ImportError:
        from difflib import SequenceMatcher

        def ratio(a, b):
            return SequenceMatcher(None, a, b).ratio()

    scored = []
    for i, cand in enumerate(texts):
        others = [t for j, t in enumerate(texts) if j != i]
        agreement = sum(
            ratio(_normalise(cand), _normalise(o)) for o in others
        ) / len(others)
        scored.append((agreement, cand))

    scored.sort(key=lambda p: p[0], reverse=True)
    return scored[0][1], scored[0][0]


# ── Layer F: confidence gate ───────────────────────────────────────────────
def _encode_jpeg(image) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def resolve_frame(image, ocr_text: str, confidence: float) -> str:
    """
    >= 0.70  trust the OCR outright — this is a text-heavy slide
    0.40-0.70  merge OCR with a vision description — mixed content
    <  0.40  the frame is probably a diagram; let the vision model read it
    """
    import gemini

    if confidence >= CONF_HIGH:
        return ocr_text

    try:
        description = gemini.describe_frame(_encode_jpeg(image))
    except Exception as exc:
        log.warning("Vision fallback failed: %s", exc)
        return ocr_text

    if confidence >= CONF_LOW and ocr_text.strip():
        return f"{ocr_text}\n\n[Figure] {description}"
    return f"[Figure] {description}"


# ── Orchestration ──────────────────────────────────────────────────────────
def extract(video_id: str, language: Optional[str] = None,
            progress=None) -> Tuple[List[Dict], str]:
    ok, reason = available()
    if not ok:
        raise ExtractionError(reason)

    import cv2

    if progress:
        progress("Downloading video", 0.05)
    video_path = _download_video(video_id, progress)

    try:
        if progress:
            progress("Extracting frames", 0.15)
        frames = extract_frames(video_path)
        if not frames:
            raise ExtractionError("No frames could be read from this video")

        if progress:
            progress("Removing duplicate slides", 0.25)
        slides = dedup_frames(frames)

        segments: List[Dict] = []
        scratch = config.MEDIA_DIR / f"{video_id}_frame.png"
        total = len(slides)

        for i, slide in enumerate(slides):
            cv2.imwrite(str(scratch), preprocess(slide["image"]))
            text, confidence = vote(ensemble_ocr(str(scratch)))
            resolved = resolve_frame(slide["image"], text, confidence).strip()

            if resolved:
                next_start = slides[i + 1]["start"] if i + 1 < total else slide["start"] + FRAME_INTERVAL_S
                segments.append({
                    "text": resolved,
                    "start": slide["start"],
                    "duration": max(1.0, next_start - slide["start"]),
                })

            if progress:
                progress(f"Reading slide {i + 1} of {total}", 0.30 + 0.65 * ((i + 1) / total))

        scratch.unlink(missing_ok=True)

        if not segments:
            raise ExtractionError("No readable text or figures found in this video")

        log.info("Visual pipeline for %s: %d slides read", video_id, len(segments))
        return segments, language or "unknown"

    finally:
        try:
            os.unlink(video_path)
        except Exception:
            pass
