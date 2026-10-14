"""
faithfulness.py
Post-generation check: did the answer stay inside the retrieved excerpts?

This is a stage with its own output — the confidence state the sidebar shows —
not a footnote on the generation step.

Two layers, both degrading gracefully:

    semantic   BERTScore F1 between the answer and the excerpts it came from
    lexical    content terms in the answer that appear in no excerpt

bert-score pulls a large model and is in requirements-eval.txt, not the core
install. When it is absent the lexical layer still runs and the method is
reported honestly rather than silently downgraded.

The threshold is a STARTING POINT, not a finding. Calibrate it against
human-judged labels on a dev set before quoting a hallucination rate —
"percentage of answers our own detector flagged" measures the detector, not
the system.
"""

import logging
import re
from typing import Dict, List, Optional

import config

log = logging.getLogger("vidquery.faithfulness")

_scorer = None
_scorer_tried = False

_WORD = re.compile(r"\w+", re.UNICODE)

# Function words carry no grounding signal; flagging them produces noise.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "with", "as", "by", "from", "it", "its",
    "he", "she", "they", "we", "you", "i", "his", "her", "their", "our", "your",
    "not", "no", "can", "will", "would", "should", "could", "may", "might",
    "do", "does", "did", "has", "have", "had", "about", "into", "over", "also",
    "there", "here", "what", "which", "who", "when", "where", "how", "why",
    "video", "transcript", "says", "said", "mentions", "explains", "discusses",
    "at", "around", "timestamp",
}


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# Timestamp citations are ours, not the model's content; they drag similarity
# down without saying anything about faithfulness.
_CITATION = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def semantic_similarity(answer: str, excerpts: List[Dict]) -> Optional[float]:
    """
    Mean over answer sentences of the best cosine against any single excerpt.

    Alignment matters more than it sounds. Comparing the whole answer against
    all four excerpts concatenated scores a correct answer badly, because the
    reference is then ~1200 characters of conversation of which any one
    sentence addresses a small part — the irrelevant remainder dilutes the
    similarity, and faithful answers land near 0.2 alongside invented ones.
    Scoring each sentence against its best-matching chunk asks the question
    that actually matters: is this claim supported SOMEWHERE in what we
    retrieved? It is the same reason BERTScore aligns tokens rather than
    averaging a document.

    Uses the encoder already loaded for retrieval, so it costs no extra
    dependency and no download, and being multilingual it works on a Tamil
    answer over a Hindi transcript.
    """
    body = _CITATION.sub(" ", answer).strip()
    if not body:
        return None

    sentences = [s.strip() for s in _SENTENCE.split(body) if len(s.strip()) > 15]
    if not sentences:
        sentences = [body]

    sources = [c.get("text", "").strip() for c in excerpts if c.get("text", "").strip()]
    if not sources:
        return None

    try:
        import embedder
        if not embedder.model_ready():
            return None

        vectors = embedder.embed(sentences + sources)
        if len(vectors) != len(sentences) + len(sources):
            return None

        sentence_vecs = vectors[:len(sentences)]
        source_vecs = vectors[len(sentences):]

        best = [
            max(_cosine(sv, src) for src in source_vecs)
            for sv in sentence_vecs
        ]
        return sum(best) / len(best)
    except Exception as exc:
        log.warning("Semantic similarity failed: %s", exc)
        return None


def _get_scorer():
    """Loads BERTScore once, or gives up permanently and says so."""
    global _scorer, _scorer_tried
    if _scorer_tried:
        return _scorer
    _scorer_tried = True
    try:
        from bert_score import BERTScorer
        log.info("Loading BERTScore (%s)", config.BERTSCORE_MODEL)
        _scorer = BERTScorer(
            model_type=config.BERTSCORE_MODEL,
            num_layers=6,
            rescale_with_baseline=False,   # no baseline file for this model
            idf=False,
        )
    except Exception as exc:
        log.info("BERTScore unavailable (%s) — using lexical grounding only", exc)
        _scorer = None
    return _scorer


# ── Layer 2: semantic similarity ───────────────────────────────────────────
def bertscore_f1(answer: str, excerpts: List[Dict]) -> Optional[float]:
    scorer = _get_scorer()
    if scorer is None or not answer.strip():
        return None

    reference = " ".join(c.get("text", "") for c in excerpts).strip()
    if not reference:
        return None

    try:
        _, _, f1 = scorer.score([answer], [reference])
        return float(f1.item())
    except Exception as exc:
        log.warning("BERTScore failed: %s", exc)
        return None


# ── Layer 3: lexical grounding ─────────────────────────────────────────────
def grounding_ratio(answer: str, excerpts: List[Dict]) -> Optional[float]:
    """
    Fraction of the answer's content words that appear in the excerpts.

    Reported alongside the raw count because an absolute count cannot be
    thresholded: a faithful sixty-word answer paraphrases more than a faithful
    ten-word one, so "more than three unmatched terms" flags every long answer
    and almost no short one. The ratio is comparable across lengths.

    None when the answer has no content words to judge.
    """
    source = " ".join(c.get("text", "") for c in excerpts).lower()
    source_terms = set(_WORD.findall(source))

    content = [
        term for term in _WORD.findall(answer.lower())
        if len(term) >= 4 and term not in _STOPWORDS
    ]
    if not content:
        return None

    matched = sum(1 for term in content if term in source_terms)
    return matched / len(content)


def ungrounded_terms(answer: str, excerpts: List[Dict],
                     min_length: int = 4) -> List[str]:
    """
    Content terms in the answer that appear nowhere in the excerpts.

    Deliberately conservative. Exact string matching over raw text flags every
    paraphrase, plural and inflection, so this normalises case, strips the
    model's own framing vocabulary, and ignores short tokens and anything
    numeric-with-a-match. It is a hint for the confidence indicator, not a
    verdict.
    """
    source = " ".join(c.get("text", "") for c in excerpts).lower()
    source_terms = set(_WORD.findall(source))

    flagged = []
    for term in _WORD.findall(answer.lower()):
        if len(term) < min_length or term in _STOPWORDS or term in source_terms:
            continue
        # Tolerate simple morphology before flagging.
        if any(term.startswith(s) or s.startswith(term) for s in source_terms
               if abs(len(s) - len(term)) <= 3):
            continue
        if term not in flagged:
            flagged.append(term)

    return flagged[:12]


# ── Combined verdict ───────────────────────────────────────────────────────
def check(answer: str, excerpts: List[Dict]) -> Dict:
    """
    Returns:
        {
          "confidence": "high" | "medium" | "low" | "unknown",
          "bertscore":  float | None,
          "threshold":  float,
          "ungrounded": [str, ...],
          "method":     "bertscore+lexical" | "lexical",
          "flagged":    bool
        }
    """
    refusal = "couldn't find that in this video" in answer.lower()

    score = bertscore_f1(answer, excerpts)
    ungrounded = ungrounded_terms(answer, excerpts)
    ratio = grounding_ratio(answer, excerpts)

    # Best available signal wins: BERTScore if it is installed, otherwise the
    # retrieval encoder's cosine, and word overlap only as a last resort.
    semantic = None if score is not None else semantic_similarity(answer, excerpts)

    if score is not None:
        method = "bertscore+lexical"
    elif semantic is not None:
        method = "sbert-cosine+lexical"
    else:
        method = "lexical"

    if refusal:
        # An honest refusal is the grounded outcome, not a low-confidence one.
        confidence, flagged = "high", False
    elif score is not None:
        if score >= config.FAITHFULNESS_THRESHOLD and len(ungrounded) <= 2:
            confidence, flagged = "high", False
        elif score >= config.FAITHFULNESS_THRESHOLD - 0.1:
            confidence, flagged = "medium", False
        else:
            confidence, flagged = "low", True
    elif semantic is not None:
        # PROVISIONAL. Chosen to separate a handful of hand-checked answers on
        # one video — grounded ones scored 0.37 to 0.64, an invented control
        # 0.20 — which is nowhere near enough data to call these calibrated.
        # Tuning a threshold on five examples is exactly the mistake the
        # evaluation chapter has to avoid, so treat these as defaults that get
        # replaced: label a dev set, plot the two distributions, and put the
        # boundary where it actually belongs. Installing bert-score supersedes
        # this path entirely.
        if semantic >= config.SEMANTIC_HIGH:
            confidence, flagged = "high", False
        elif semantic >= config.SEMANTIC_MEDIUM:
            confidence, flagged = "medium", False
        else:
            confidence, flagged = "low", True
    elif ratio is None:
        confidence, flagged = "unknown", False
    else:
        # Lexical-only thresholds. A grounded abstractive answer over
        # conversational captions typically lands around 0.7-0.85: the model
        # is summarising, not quoting, so perfect overlap is neither expected
        # nor desirable. These are deliberately loose because this layer is a
        # fallback — install bert-score for the semantic check, and calibrate
        # both against human labels before quoting a hallucination rate.
        if ratio >= 0.70:
            confidence, flagged = "high", False
        elif ratio >= 0.50:
            confidence, flagged = "medium", False
        else:
            confidence, flagged = "low", True

    return {
        "confidence": confidence,
        "bertscore":  round(score, 4) if score is not None else None,
        "semantic":   round(semantic, 4) if semantic is not None else None,
        "grounding":  round(ratio, 3) if ratio is not None else None,
        "threshold":  config.FAITHFULNESS_THRESHOLD,
        "ungrounded": ungrounded,
        "method":     method,
        "flagged":    flagged,
    }
