"""
schemas.py
Request and response models for the API.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── /detect ────────────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    video: str = Field(..., description="YouTube URL or bare video id")


class DetectResponse(BaseModel):
    video_id: str
    pipeline: str
    language: Optional[str] = None
    reason: str
    indexed: bool = False


# ── /process ───────────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    video: str
    force: bool = False


class ProcessResponse(BaseModel):
    job_id: str
    video_id: str
    state: str
    stage: str
    progress: float
    cached: bool = False


# ── /status ────────────────────────────────────────────────────────────────
class StatusResponse(BaseModel):
    job_id: str
    video_id: str
    state: Literal["queued", "running", "done", "error"]
    stage: str
    progress: float
    pipeline: Optional[str] = None
    language: Optional[str] = None
    chunk_count: int = 0
    error: Optional[str] = None


# ── /ask ───────────────────────────────────────────────────────────────────
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    video_id: str
    question: str
    chat_history: List[ChatTurn] = []


class TimestampChip(BaseModel):
    chunk_id: str
    timestamp: str
    start_seconds: float
    text_preview: str


class Confidence(BaseModel):
    confidence: Literal["high", "medium", "low", "unknown"]
    bertscore: Optional[float] = None
    semantic: Optional[float] = None
    grounding: Optional[float] = None
    threshold: float
    ungrounded: List[str] = []
    method: str
    flagged: bool


class AskResponse(BaseModel):
    answer: str
    timestamps: List[TimestampChip]
    sources: List[str]
    confidence: Confidence
    retrieval: Dict = {}


# ── /summarize ─────────────────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    video_id: str


class SummarizeResponse(BaseModel):
    summary: str


# ── /feedback ──────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    video_id: str
    chunk_id: str
    clicked: bool = True
    question: Optional[str] = None


class FeedbackResponse(BaseModel):
    recorded: bool
