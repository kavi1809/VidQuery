"""
main.py
VidQuery FastAPI backend — all API endpoints.
 
Run with:
    uvicorn main:app --reload --port 8000
"""
 
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import List, Optional
import traceback
 
from transcript import extract_video_id, fetch_transcript, chunk_transcript, get_full_text
from embedder  import embed_and_store, collection_exists
from retriever import hybrid_search, format_context, extract_timestamps
from llm       import generate_answer, generate_summary
 
 
# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="VidQuery API",
    description="AI-powered YouTube video Q&A backend",
    version="1.0.0"
)
 
# ── Private Network Access middleware ──────────────────────────────────────
# Fixes Chrome's block on loopback (127.0.0.1) from public origins like youtube.com
class PrivateNetworkMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            response = JSONResponse(content={}, status_code=200)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            return response
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
 
app.add_middleware(PrivateNetworkMiddleware)
 
# Standard CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# In-memory chunk cache
_chunks_cache: dict = {}
 
 
# ── Request / Response models ──────────────────────────────────────────────
class ProcessRequest(BaseModel):
    video_url: str
 
class ProcessResponse(BaseModel):
    video_id:    str
    chunk_count: int
    message:     str
 
class AskRequest(BaseModel):
    video_id:     str
    question:     str
    chat_history: Optional[List[dict]] = []
 
class AskResponse(BaseModel):
    answer:     str
    timestamps: List[dict]
    sources:    List[str]
 
class SummarizeRequest(BaseModel):
    video_id: str
 
class SummarizeResponse(BaseModel):
    summary: str
 
 
# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "VidQuery API"}
 
 
@app.post("/process", response_model=ProcessResponse)
def process_video(req: ProcessRequest):
    try:
        video_id = extract_video_id(req.video_url)
 
        if video_id in _chunks_cache:
            chunks = _chunks_cache[video_id]
            return ProcessResponse(
                video_id=video_id,
                chunk_count=len(chunks),
                message=f"Already processed. {len(chunks)} chunks ready."
            )
 
        print(f"Fetching transcript for {video_id}...")
        raw_transcript = fetch_transcript(video_id)
        chunks = chunk_transcript(raw_transcript, chunk_size=5, overlap=1)
        _chunks_cache[video_id] = chunks
        stored = embed_and_store(video_id, chunks)
 
        return ProcessResponse(
            video_id=video_id,
            chunk_count=len(chunks),
            message=f"Video processed. {stored} chunks embedded and ready."
        )
 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
 
 
@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    try:
        video_id = req.video_id
        question = req.question.strip()
 
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
 
        if video_id not in _chunks_cache:
            raise HTTPException(
                status_code=400,
                detail="Video not processed yet. Call /process first."
            )
 
        chunks = _chunks_cache[video_id]
 
        results = hybrid_search(
            video_id=video_id,
            query=question,
            chunks=chunks,
            top_k=8,
            final_k=4
        )
 
        if not results:
            return AskResponse(
                answer="I couldn't find relevant information in this video for your question.",
                timestamps=[],
                sources=[]
            )
 
        context    = format_context(results)
        answer     = generate_answer(question=question, context=context, chat_history=req.chat_history)
        timestamps = extract_timestamps(results)
        sources    = [r["chunk_id"] for r in results]
 
        return AskResponse(answer=answer, timestamps=timestamps, sources=sources)
 
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to answer: {str(e)}")
 
 
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_video(req: SummarizeRequest):
    try:
        video_id = req.video_id
 
        if video_id not in _chunks_cache:
            raise HTTPException(
                status_code=400,
                detail="Video not processed yet. Call /process first."
            )
 
        chunks    = _chunks_cache[video_id]
        full_text = " ".join(chunk["text"] for chunk in chunks)
        summary   = generate_summary(full_text)
 
        return SummarizeResponse(summary=summary)
 
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")