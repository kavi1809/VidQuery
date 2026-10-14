"""
gemini.py
Every call to the Gemini API lives here.

Gemini is the only hosted service this project uses. It covers four jobs that
would otherwise need four separate dependencies:

    generation      answering from the retrieved excerpts
    translation     the question into the transcript's language, for BM25
    vision          reading diagram frames (replaces a local LLaVA / Ollama)
    embeddings      optional alternative to local Sentence-BERT

Everything else — Sentence-BERT, faster-whisper, the OCR engines, BERTScore —
runs locally and needs no key.
"""

import logging
import time
from typing import Dict, List, Optional

import config

log = logging.getLogger("vidquery.gemini")

_client = None


class GeminiUnavailable(RuntimeError):
    """Raised when the key is missing or the SDK is not installed."""


def get_client():
    """Lazily builds the Gemini client so import never fails without a key."""
    global _client
    if _client is not None:
        return _client

    if not config.GEMINI_API_KEY:
        raise GeminiUnavailable(
            "GEMINI_API_KEY is not set. Copy backend/.env.example to backend/.env "
            "and add your key from https://aistudio.google.com/apikey"
        )
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiUnavailable(
            "google-genai is not installed. Run: pip install google-genai"
        ) from exc

    _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


class GeminiBusy(RuntimeError):
    """The model is rate-limited or overloaded. Retrying later may work."""


# Flash models are shared capacity and return 503 in bursts, especially on the
# free tier. Without a retry a perfectly good question fails for a reason that
# has nothing to do with the question.
_RETRY_STATUSES = (429, 503)
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.5, 4.0)


def _generate(prompt: str, system: Optional[str] = None,
              temperature: float = 0.2, model: Optional[str] = None) -> str:
    from google.genai import types

    client = get_client()
    cfg = types.GenerateContentConfig(temperature=temperature)
    if system:
        cfg.system_instruction = system

    last_error = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = client.models.generate_content(
                model=model or config.GEMINI_MODEL,
                contents=prompt,
                config=cfg,
            )
            return (resp.text or "").strip()
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if status not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS - 1:
                if status in _RETRY_STATUSES:
                    raise GeminiBusy(
                        "Gemini is overloaded right now. Wait a moment and ask again."
                    ) from exc
                raise
            last_error = exc
            delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
            log.warning("Gemini %s, retrying in %.1fs (attempt %d/%d)",
                        status, delay, attempt + 1, _MAX_ATTEMPTS)
            time.sleep(delay)

    raise GeminiBusy("Gemini is overloaded right now. Wait a moment and ask again.") from last_error


# ── Answering ──────────────────────────────────────────────────────────────
# Grounding has to forbid outside knowledge without also forbidding synthesis.
# An earlier version said "never infer details the excerpts do not state", and
# the model read summarising across several fragments as inference and refused
# questions the excerpts plainly answered. Note that refusing is invisible in a
# hallucination-rate metric — it scores as a perfectly faithful answer — so
# over-refusal quietly destroys answer quality while the faithfulness numbers
# look excellent.
ANSWER_SYSTEM = """You are VidQuery, answering questions about one specific YouTube video.

You will be given excerpts from that video's transcript, each tagged with the \
timestamp where it occurs.

WHAT THE EXCERPTS LOOK LIKE
They are captions of spoken conversation, often auto-generated. Expect missing \
punctuation, filler words, false starts, repetition, and transcription errors. \
">>" marks a change of speaker. Read through that messiness. Never refuse because \
the wording is rough, a sentence is cut off, or the phrasing is informal.

HOW TO ANSWER
- Use only what the excerpts contain. Never add facts from outside the video.
- Reading ACROSS excerpts is expected. Gather what is said in several places and \
put it in your own words. Summarising the excerpts is the task, not inference.
- Cite the timestamp for each point, in square brackets, like [2:15].
- Answer the question that was asked, directly, in two to four sentences unless it \
genuinely needs more.
- Reply in the SAME LANGUAGE the question was asked in, whatever language the \
excerpts are in.

WHEN YOU CANNOT ANSWER
Only when the excerpts are about a genuinely different subject from the question. \
Then reply exactly: "I couldn't find that in this video."
If the excerpts cover the topic only partly, answer from what they do support and \
briefly note what is missing. A partial answer is better than a refusal."""


def answer_question(question: str, excerpts: List[Dict],
                    chat_history: Optional[List[Dict]] = None) -> str:
    """
    Args:
        question: the user's question, in their own language
        excerpts: retrieved chunks — each needs "timestamp" and "text"
        chat_history: prior [{role, content}] turns, oldest first
    """
    context = "\n\n".join(
        f"[{c.get('timestamp', '?:??')}] {c.get('text', '')}" for c in excerpts
    )

    parts = []
    if chat_history:
        recent = chat_history[-4:]
        turns = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in recent
        )
        parts.append(f"Earlier in this conversation:\n{turns}")

    parts.append(f"Transcript excerpts:\n\n{context}")
    parts.append(f"Question: {question}")

    return _generate("\n\n---\n\n".join(parts), system=ANSWER_SYSTEM)


def summarize(full_text: str, max_chars: int = 40_000) -> str:
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n…[transcript truncated]"

    prompt = f"""Summarise this video transcript.

Transcript:
{full_text}

Give:
1. A two or three sentence overview of what the video covers
2. Four to six key points, as bullets
3. The single main takeaway

Reply in the same language as the transcript."""
    return _generate(prompt, temperature=0.3)


# ── Translation (for BM25 only) ────────────────────────────────────────────
def translate_question(question: str, target_language: str) -> str:
    """
    Translates the question into the transcript's language so BM25 has terms
    that can actually match. The vector side keeps the original question — the
    multilingual encoder handles that direction natively.

    Falls back to the original question on any failure; a missed translation
    degrades keyword recall but must never fail the request.
    """
    try:
        out = _generate(
            f"Translate this search query into {target_language}. "
            f"Return ONLY the translation, no quotes, no explanation.\n\n{question}",
            temperature=0.0,
        )
        return out or question
    except Exception as exc:
        log.warning("Question translation failed, using original: %s", exc)
        return question


# ── Vision (replaces LLaVA) ────────────────────────────────────────────────
def describe_frame(image_bytes: bytes, mime_type: str = "image/jpeg",
                   hint: Optional[str] = None) -> str:
    """
    Describes a slide, diagram, or chart frame in natural language.

    This is the confidence-gated fallback from the visual pipeline: it runs
    only when the OCR ensemble disagrees badly enough that the frame is
    probably not text-heavy.
    """
    from google.genai import types

    client = get_client()
    instruction = hint or (
        "This is a frame from a lecture video. Describe its content so it can be "
        "searched later: read any text verbatim, and describe diagrams, charts, or "
        "flowcharts in specific terms — what the parts are and how they connect. "
        "No preamble."
    )
    resp = client.models.generate_content(
        model=config.GEMINI_VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            instruction,
        ],
    )
    return (resp.text or "").strip()


def extract_code_from_frame(image_bytes: bytes, mime_type: str = "image/jpeg",
                            language: Optional[str] = None) -> Optional[str]:
    """
    Reads source code off a frame of a programming tutorial.

    This reads the SCREEN, not the transcript. Spoken narration does not
    contain syntax — nobody dictates their brackets — so extracting code from
    transcript text returns nothing on real tutorials, and asking a model to
    reconstruct it from a verbal description produces plausible invented code,
    which is exactly what the grounding rules exist to prevent.
    """
    from google.genai import types

    lang = f"{language} " if language else ""
    prompt = (
        f"If this frame shows {lang}source code in an editor or terminal, transcribe it "
        f"EXACTLY as written — preserve indentation, spelling, and line breaks. "
        f"Return only the code, with no fences and no commentary. "
        f"If the frame shows no code, reply with exactly: NO_CODE_FOUND"
    )
    client = get_client()
    resp = client.models.generate_content(
        model=config.GEMINI_VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    text = (resp.text or "").strip()
    if not text or "NO_CODE_FOUND" in text:
        return None
    return text.strip("`").strip()


# ── Embeddings (optional backend) ──────────────────────────────────────────
def embed_texts(texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Used only when EMBEDDING_BACKEND=gemini."""
    from google.genai import types

    client = get_client()
    resp = client.models.embed_content(
        model=config.GEMINI_EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    )
    return [e.values for e in resp.embeddings]


def health() -> Dict:
    """Reports configuration without making a billable call."""
    return {
        "configured": bool(config.GEMINI_API_KEY),
        "model": config.GEMINI_MODEL,
        "vision_model": config.GEMINI_VISION_MODEL,
    }
