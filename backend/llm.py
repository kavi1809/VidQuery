"""
llm.py
Sends retrieved context + user question to Gemini API and returns a grounded answer.
"""
 
import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict
 
load_dotenv()
 
# ── Gemini client setup ────────────────────────────────────────────────────
def get_model(model_name: str = "gemini-1.5-flash") -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Make sure your .env file has GEMINI_API_KEY=your_key_here"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)
 
 
# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are VidQuery, an AI assistant that answers questions about YouTube videos.
 
You are given:
1. Relevant excerpts from the video transcript (with timestamps like [2:15])
2. A user question about the video
 
Rules you MUST follow:
- Answer ONLY based on the provided transcript excerpts. Do not use outside knowledge.
- If the answer is not in the excerpts, say: "I couldn't find that in this video."
- Always mention the timestamp(s) where the answer can be found, like: "At 2:15, ..."
- Keep answers concise and clear — the user wants a quick answer, not an essay.
- If the question asks for a summary, summarize based on the excerpts provided.
- Be conversational and helpful."""
 
 
# ── Answer generation ──────────────────────────────────────────────────────
def generate_answer(
    question: str,
    context: str,
    chat_history: List[Dict] = None
) -> str:
    """
    Sends question + transcript context to Gemini and returns the answer.
 
    Args:
        question:     The user's question
        context:      Formatted transcript excerpts with timestamps
        chat_history: Previous Q&A turns (kept for future multi-turn support)
 
    Returns:
        Gemini's answer as a string
    """
    model = get_model()
 
    # Build full prompt — Gemini takes system + user together
    full_prompt = f"""{SYSTEM_PROMPT}
 
Here are the relevant excerpts from the video transcript:
 
{context}
 
---
 
User question: {question}"""
 
    # Add chat history context if available
    if chat_history:
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in chat_history[-4:]  # last 2 exchanges
        )
        full_prompt = f"Previous conversation:\n{history_text}\n\n{full_prompt}"
 
    response = model.generate_content(full_prompt)
    return response.text
 
 
def generate_summary(full_transcript_text: str) -> str:
    """
    Generates a structured summary of the entire video.
    Called when user clicks 'Summarize' in the extension.
    """
    model = get_model()
 
    # Truncate if very long
    max_chars = 15000
    if len(full_transcript_text) > max_chars:
        full_transcript_text = full_transcript_text[:max_chars] + "\n...[transcript truncated]"
 
    prompt = f"""Please summarize this YouTube video transcript.
 
Transcript:
{full_transcript_text}
 
Provide:
1. A 2-3 sentence overview of what the video is about
2. 4-6 key points covered (as bullet points)
3. The main takeaway in one sentence
 
Keep it concise and useful."""
 
    response = model.generate_content(prompt)
    return response.text