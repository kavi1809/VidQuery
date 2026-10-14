"""
main.py
VidQuery API.

    GET  /health
    POST /detect      which pipeline this video needs, without processing it
    POST /process     enqueue ingest, returns a job id immediately
    GET  /status/{id} stage and progress for a running job
    POST /ask         hybrid retrieval, grounded answer, faithfulness check
    POST /summarize   whole-video summary
    POST /feedback    timestamp chip click, feeding the retrieval boost

Run:  uvicorn main:app --reload --port 8000
"""

import logging
import re
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
import embedder
import faithfulness
import gemini
import ingest
import jobs
import retriever
import schemas
import store
from pipelines import ExtractionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vidquery")

# The Windows console is cp1252 by default, which cannot encode the em dashes
# and arrows in these messages — they arrive as replacement characters, and a
# stray Unicode name in a video title raises UnicodeEncodeError mid-log.
for _handler in logging.getLogger().handlers:
    stream = getattr(_handler, "stream", None)
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    log.info("VidQuery ready — model=%s, embeddings=%s",
             config.GEMINI_MODEL, embedder.embedding_backend())
    if not config.GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY is not set — /ask and /summarize will fail")
    yield
    jobs.registry.shutdown()


app = FastAPI(title="VidQuery API", version="2.0.0", lifespan=lifespan)


# ── CORS ───────────────────────────────────────────────────────────────────
# Named origins, not "*". A wildcard origin combined with credentials is
# rejected outright by browsers, and the extension calls from a real origin.
#
# allow_private_network is what lets a page on https://www.youtube.com reach
# http://127.0.0.1. Chrome sends an extra Access-Control-Request-Private-
# Network header on the preflight for public → loopback requests, and
# Starlette rejects the preflight outright unless this is set.
#
# Note that this is CORSMiddleware's job, not something to bolt on in a
# separate middleware: an outer middleware that answers OPTIONS itself never
# reaches CORSMiddleware, so the preflight comes back with the private-network
# header but no Access-Control-Allow-Origin. The browser then rejects it and
# the fetch surfaces as a bare network error — while /health keeps answering
# fine from the extension popup, which runs on a chrome-extension:// origin
# and is exempt from CORS entirely.
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_private_network=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("No video supplied")

    if _ID_RE.match(value):
        return value

    parsed = urlparse(value if "//" in value else f"https://{value}")
    host = (parsed.hostname or "").replace("www.", "")

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path.startswith(("/embed/", "/v/", "/shorts/")):
            candidate = parsed.path.split("/")[2]
        else:
            candidate = (parse_qs(parsed.query).get("v") or [""])[0]
    else:
        candidate = value

    if not _ID_RE.match(candidate):
        raise ValueError(f"Could not read a video id from {value!r}")
    return candidate


def require_indexed(video_id: str) -> dict:
    record = store.get_video(video_id)
    if not record or not store.has_chunks(video_id):
        raise HTTPException(
            status_code=409,
            detail="This video has not been indexed yet. Call /process first.",
        )
    return record


@app.exception_handler(ExtractionError)
async def extraction_error_handler(request: Request, exc: ExtractionError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    from pipelines import asr, visual

    asr_ok, asr_reason = asr.available()
    visual_ok, visual_reason = visual.available()

    return {
        "status": "ok",
        "gemini": gemini.health(),
        "embeddings": embedder.embedding_backend(),
        "pipelines": {
            "captions": {"available": True},
            "asr": {"available": asr_ok, "reason": asr_reason or None},
            "visual": {"available": visual_ok, "reason": visual_reason or None},
        },
        "feedback_enabled": config.FEEDBACK_ENABLED,
    }


# ── Detect ─────────────────────────────────────────────────────────────────
@app.post("/detect", response_model=schemas.DetectResponse)
def detect_video(req: schemas.DetectRequest):
    import detector

    try:
        video_id = extract_video_id(req.video)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record = store.get_video(video_id)
    if record and store.has_chunks(video_id):
        return schemas.DetectResponse(
            video_id=video_id,
            pipeline=record["pipeline"],
            language=record["language"],
            reason=f"Already indexed — {record['chunk_count']} chunks",
            indexed=True,
        )

    decision = detector.detect(video_id)
    return schemas.DetectResponse(
        video_id=video_id,
        pipeline=decision["pipeline"],
        language=decision.get("language"),
        reason=decision["reason"],
        indexed=False,
    )


# ── Process ────────────────────────────────────────────────────────────────
@app.post("/process", response_model=schemas.ProcessResponse)
def process_video(req: schemas.ProcessRequest):
    try:
        video_id = extract_video_id(req.video)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not req.force and store.has_chunks(video_id) and embedder.count_for_video(video_id):
        record = store.get_video(video_id) or {}
        return schemas.ProcessResponse(
            job_id="cached",
            video_id=video_id,
            state=jobs.DONE,
            stage=f"Ready — {record.get('chunk_count', 0)} chunks indexed",
            progress=1.0,
            cached=True,
        )

    def work(job: jobs.Job, report):
        result = ingest.process_video(video_id, report, force=req.force)
        job.pipeline = result["pipeline"]
        job.language = result.get("language")
        job.chunk_count = result["chunk_count"]

    job = jobs.registry.submit(video_id, work)
    return schemas.ProcessResponse(
        job_id=job.job_id,
        video_id=video_id,
        state=job.state,
        stage=job.stage,
        progress=job.progress,
    )


@app.get("/status/{job_id}", response_model=schemas.StatusResponse)
def job_status(job_id: str):
    job = jobs.registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return schemas.StatusResponse(**job.as_dict())


# ── Ask ────────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=schemas.AskResponse)
def ask(req: schemas.AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    record = require_indexed(req.video_id)
    transcript_language = record.get("language") or "unknown"

    # BM25 needs terms that can literally match the transcript, so the question
    # is translated into its language. The vector side keeps the question as
    # asked — the multilingual encoder handles that direction itself.
    keyword_query = question
    if transcript_language not in ("unknown", "en") and config.GEMINI_API_KEY:
        keyword_query = gemini.translate_question(question, transcript_language)

    results = retriever.hybrid_search(
        req.video_id, question, keyword_query=keyword_query
    )

    if not results:
        return schemas.AskResponse(
            answer="I couldn't find that in this video.",
            timestamps=[],
            sources=[],
            confidence=schemas.Confidence(
                confidence="high", threshold=config.FAITHFULNESS_THRESHOLD,
                ungrounded=[], method="no-retrieval", flagged=False,
            ),
            retrieval={"matched": 0},
        )

    try:
        answer = gemini.answer_question(
            question, results,
            chat_history=[t.model_dump() for t in req.chat_history],
        )
    except gemini.GeminiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except gemini.GeminiBusy as exc:
        # Retries are already exhausted inside gemini._generate.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("Generation failed")
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {exc}")

    verdict = faithfulness.check(answer, results)

    return schemas.AskResponse(
        answer=answer,
        timestamps=[schemas.TimestampChip(**t) for t in retriever.timestamps_for(results)],
        sources=[r["chunk_id"] for r in results],
        confidence=schemas.Confidence(**verdict),
        retrieval={
            "matched": len(results),
            "keyword_query": keyword_query if keyword_query != question else None,
            "matched_by": {r["chunk_id"]: r.get("matched_by", []) for r in results},
        },
    )


# ── Summarize ──────────────────────────────────────────────────────────────
@app.post("/summarize", response_model=schemas.SummarizeResponse)
def summarize(req: schemas.SummarizeRequest):
    require_indexed(req.video_id)
    chunks = store.get_chunks(req.video_id)
    full_text = " ".join(c["text"] for c in chunks)

    try:
        return schemas.SummarizeResponse(summary=gemini.summarize(full_text))
    except gemini.GeminiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except gemini.GeminiBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("Summarisation failed")
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {exc}")


# ── Feedback ───────────────────────────────────────────────────────────────
@app.post("/feedback", response_model=schemas.FeedbackResponse)
def feedback(req: schemas.FeedbackRequest):
    """
    A timestamp chip click is an implicit positive signal. The boost it earns
    is capped at config.FEEDBACK_MAX_BOOST and applied after fusion — see
    retriever.reciprocal_rank_fusion for why the cap has to be small.
    """
    store.log_feedback(req.video_id, req.chunk_id, req.clicked, req.question)
    return schemas.FeedbackResponse(recorded=True)
