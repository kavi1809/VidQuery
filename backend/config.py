"""
config.py
Single source of truth for paths, model names, and tunable constants.

Everything here can be overridden from backend/.env — nothing is hardcoded
in the modules that use it.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


# ── TLS on an intercepting network ─────────────────────────────────────────
# Corporate proxies terminate TLS and re-sign with a private root. That root
# is installed in the OS certificate store, but Python ships its own CA bundle
# and never looks there, so YouTube and Gemini calls fail with
# CERTIFICATE_VERIFY_FAILED. truststore points Python at the system store,
# which is where the proxy's root already is.
#
# This verifies certificates normally — it does not disable checking.
if os.getenv("VIDQUERY_SYSTEM_CERTS", "true").lower() != "false":
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:                       # not installed, or not supported
        pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ── Storage ────────────────────────────────────────────────────────────────
DATA_DIR   = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "vidquery.db"
MEDIA_DIR  = DATA_DIR / "media"          # downloaded audio / frames, transient

for _d in (DATA_DIR, CHROMA_DIR, MEDIA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# One Chroma collection for every video, filtered by a video_id metadata field.
# Per-video collections would mean one HNSW index per video and would block
# the playlist-level retrieval in the future-work section.
CHROMA_COLLECTION = "vidquery"


# ── Gemini ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# gemini-1.5-flash is retired, and 2.5-flash is closed to new API keys — it
# still appears in models.list() but returns 404 on the first real call.
#
# Pinned rather than tracking gemini-flash-latest on purpose: the evaluation
# numbers are model-dependent, so the model has to be a fixed, quotable part
# of the setup. Bump it deliberately and re-run the evaluation.
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_MODEL)


# ── Embeddings ─────────────────────────────────────────────────────────────
# "sbert"  → local sentence-transformers, offline, no quota
# "gemini" → hosted embedding API, no torch install required
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sbert").lower()

SBERT_MODEL = os.getenv(
    "SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")


# ── Chunking ───────────────────────────────────────────────────────────────
# Time windows, not line counts. Caption lines run from two to fifteen words,
# so a "5-line chunk" varies about sevenfold in length and BM25's length
# normalisation fights it.
CHUNK_SECONDS   = _float("CHUNK_SECONDS", 45.0)
CHUNK_OVERLAP_S = _float("CHUNK_OVERLAP_S", 10.0)


# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVE_PER_METHOD = _int("RETRIEVE_PER_METHOD", 8)   # top-k from BM25 and vector each
RETRIEVE_FINAL      = _int("RETRIEVE_FINAL", 4)        # chunks handed to the model
RRF_K               = _int("RRF_K", 60)

# A rank-0 RRF score is 1/60 ≈ 0.0167, so a two-list fused score tops out near
# 0.033. The boost has to sit well inside that range or one click pins a chunk
# to rank 1 forever.
FEEDBACK_MAX_BOOST   = _float("FEEDBACK_MAX_BOOST", 0.002)
FEEDBACK_MIN_EVENTS  = _int("FEEDBACK_MIN_EVENTS", 3)
FEEDBACK_ENABLED     = os.getenv("FEEDBACK_ENABLED", "true").lower() != "false"


# ── Faithfulness ───────────────────────────────────────────────────────────
# Threshold is a starting point, not a finding. Calibrate it on the dev set
# before quoting a hallucination rate.
FAITHFULNESS_THRESHOLD = _float("FAITHFULNESS_THRESHOLD", 0.75)
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "distilbert-base-multilingual-cased")

# Thresholds for the sbert-cosine fallback used when bert-score is absent.
# Provisional: set from a handful of hand-checked answers, not calibrated.
# Replace both from labelled dev-set data before quoting any figure.
SEMANTIC_HIGH   = _float("SEMANTIC_HIGH", 0.45)
SEMANTIC_MEDIUM = _float("SEMANTIC_MEDIUM", 0.30)


# ── Detector ───────────────────────────────────────────────────────────────
SILENCE_PROBE_SECONDS = _int("SILENCE_PROBE_SECONDS", 30)
SILENCE_MAX_BYTES     = _int("SILENCE_MAX_BYTES", 15_000)
SLIDESHOW_SAMPLE_SECS = _int("SLIDESHOW_SAMPLE_SECS", 60)
SLIDESHOW_MAX_VARIANCE = _float("SLIDESHOW_MAX_VARIANCE", 5.0)


# ── Jobs ───────────────────────────────────────────────────────────────────
JOB_WORKERS      = _int("JOB_WORKERS", 2)
JOB_RETENTION_S  = _int("JOB_RETENTION_S", 3600)


# ── Server ─────────────────────────────────────────────────────────────────
# The extension runs on https://www.youtube.com and calls http://127.0.0.1.
# Chrome treats loopback as a trustworthy origin, so this is allowed, but the
# origin still has to be named explicitly — "*" with credentials is rejected.
ALLOWED_ORIGINS = [
    o.strip() for o in
    os.getenv("ALLOWED_ORIGINS", "https://www.youtube.com,https://youtube.com").split(",")
    if o.strip()
]
