"""
evaluate.py
The evaluation harness: retrieval ablation, answer quality, timestamp accuracy.

Reports Precision@k, Recall@k and MRR for BM25 alone, vector alone, and the
RRF hybrid, so the claim that hybrid beats either single method is measured
rather than asserted. Answer quality is ROUGE-L against your reference answers
plus the faithfulness score, and Timestamp Accuracy is the share of answers
whose top chip lands within a tolerance of where the answer is really
discussed.

Two things this does deliberately:

  * Click feedback is FORCED OFF. Leaving it on means retrieval changes as the
    harness runs and the run is not reproducible.
  * Bootstrap confidence intervals are printed alongside every mean. With a
    test set of tens of questions, a gap of a few points is inside the noise,
    and a bare "0.71 vs 0.57" invites exactly the question you cannot answer
    without them.

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --no-answers      # retrieval only, no API calls
    python eval/evaluate.py --k 4 --tolerance 15
"""

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Must precede the backend import: config reads it at module load.
os.environ["FEEDBACK_ENABLED"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import config                                    # noqa: E402
import embedder                                  # noqa: E402
import faithfulness                              # noqa: E402
import retriever                                 # noqa: E402
import store                                     # noqa: E402

EVAL_DIR = Path(__file__).parent
TESTSET = EVAL_DIR / "testset.json"


# ── Retrieval metrics ──────────────────────────────────────────────────────
def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> Optional[float]:
    """
    None when nothing was marked relevant — averaging a zero there would
    punish the system for a gap in the annotation.

    Note the ceiling: with k=4 and six relevant chunks, recall cannot exceed
    0.67. Report the denominator alongside the figure or it is not readable.
    """
    if not relevant:
        return None
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


# ── ROUGE-L ────────────────────────────────────────────────────────────────
def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            current.append(
                previous[j] + 1 if token_a == token_b
                else max(current[j], previous[j + 1])
            )
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """F-measure over the longest common subsequence. Implemented here so the
    harness needs no extra dependency."""
    cand = retriever.tokenize(candidate)
    ref = retriever.tokenize(reference)
    if not cand or not ref:
        return 0.0

    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0

    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


# ── Statistics ─────────────────────────────────────────────────────────────
def bootstrap_ci(values: List[float], iterations: int = 2000,
                 alpha: float = 0.05) -> tuple:
    """Percentile bootstrap 95% CI for the mean."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return (float("nan"), float("nan"))

    rng = random.Random(12345)          # fixed seed: reruns agree
    means = []
    n = len(clean)
    for _ in range(iterations):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(iterations * alpha / 2)]
    hi = means[int(iterations * (1 - alpha / 2))]
    return (lo, hi)


def mean_of(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def fmt(value: Optional[float], ci: tuple = None) -> str:
    if value is None:
        return "     —"
    text = f"{value:.3f}"
    if ci and not any(x != x for x in ci):      # not NaN
        text += f"  [{ci[0]:.2f}–{ci[1]:.2f}]"
    return text


# ── Retrieval methods under test ───────────────────────────────────────────
def run_method(method: str, video_id: str, question: str, k: int) -> List[Dict]:
    per_method = config.RETRIEVE_PER_METHOD

    if method == "bm25":
        return retriever.bm25_search(video_id, question, per_method)[:k]
    if method == "vector":
        return embedder.vector_search(video_id, question, per_method)[:k]
    return retriever.hybrid_search(video_id, question, top_k=per_method, final_k=k)


METHODS = ("bm25", "vector", "hybrid")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VidQuery retrieval and answers")
    parser.add_argument("--testset", default=str(TESTSET))
    parser.add_argument("--k", type=int, default=config.RETRIEVE_FINAL)
    parser.add_argument("--tolerance", type=float, default=15.0,
                        help="seconds a chip may be off and still count as correct")
    parser.add_argument("--no-answers", action="store_true",
                        help="skip generation; retrieval metrics only, no API calls")
    args = parser.parse_args()

    path = Path(args.testset)
    if not path.exists():
        print(f"No test set at {path}.\nRun: python eval/build_testset.py <video_id>")
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    questions = [q for q in data.get("questions", []) if q.get("question")]
    if not questions:
        print("Test set has no questions yet. Annotate it first.")
        sys.exit(1)

    store.init_db()

    # Every question must be answerable from an indexed video, or the run is
    # silently measuring missing data rather than retrieval quality.
    missing = sorted({
        q["video_id"] for q in questions if not store.has_chunks(q["video_id"])
    })
    if missing:
        print("These videos are not indexed:", ", ".join(missing))
        print("Run: python eval/build_testset.py " + " ".join(missing))
        sys.exit(1)

    print(f"Test set : {len(questions)} questions over "
          f"{len({q['video_id'] for q in questions})} videos")
    print(f"k        : {args.k}")
    print(f"Model    : {config.GEMINI_MODEL}")
    print(f"Feedback : {'ON — NOT REPRODUCIBLE' if config.FEEDBACK_ENABLED else 'off (forced)'}")
    print()

    scores = {m: {"p": [], "r": [], "mrr": []} for m in METHODS}
    timestamp_hits: List[float] = []
    rouge_scores: List[float] = []
    faith_scores: List[Optional[float]] = []
    refusals = 0
    flagged = 0

    for i, q in enumerate(questions, start=1):
        video_id = q["video_id"]
        question = q["question"]
        relevant = q.get("relevant_chunks", [])

        for method in METHODS:
            results = run_method(method, video_id, question, args.k)
            ids = [r["chunk_id"] for r in results]
            scores[method]["p"].append(precision_at_k(ids, relevant, args.k))
            scores[method]["r"].append(recall_at_k(ids, relevant, args.k))
            scores[method]["mrr"].append(reciprocal_rank(ids, relevant))

        hybrid = run_method("hybrid", video_id, question, args.k)

        # Timestamp Accuracy — does the top chip land where the answer is?
        target = q.get("answer_time_s")
        if target is not None and hybrid:
            delta = min(abs(r["start"] - target) for r in hybrid)
            timestamp_hits.append(1.0 if delta <= args.tolerance else 0.0)

        if not args.no_answers:
            import gemini
            try:
                answer = gemini.answer_question(question, hybrid)
            except Exception as exc:
                print(f"  [{i}] generation failed: {str(exc)[:70]}")
                continue

            if "couldn't find that in this video" in answer.lower():
                refusals += 1

            reference = q.get("reference_answer")
            if reference:
                rouge_scores.append(rouge_l(answer, reference))

            verdict = faithfulness.check(answer, hybrid)
            faith_scores.append(verdict.get("bertscore") or verdict.get("semantic"))
            if verdict["flagged"]:
                flagged += 1

        if i % 10 == 0:
            print(f"  {i}/{len(questions)} …")

    # ── Retrieval table ────────────────────────────────────────────────
    print("\nRETRIEVAL" + " " * 22 + "(95% bootstrap CI)")
    print(f"  {'method':8} {'P@' + str(args.k):22} {'R@' + str(args.k):22} {'MRR':22}")
    for method in METHODS:
        row = scores[method]
        p, r, m = mean_of(row["p"]), mean_of(row["r"]), mean_of(row["mrr"])
        print(f"  {method:8} {fmt(p, bootstrap_ci(row['p'])):22} "
              f"{fmt(r, bootstrap_ci(row['r'])):22} {fmt(m, bootstrap_ci(row['mrr'])):22}")

    hybrid_p = mean_of(scores["hybrid"]["p"]) or 0
    best_single = max(mean_of(scores["bm25"]["p"]) or 0, mean_of(scores["vector"]["p"]) or 0)
    lo, hi = bootstrap_ci(scores["hybrid"]["p"])
    print(f"\n  hybrid − best single: {hybrid_p - best_single:+.3f}")
    if not (lo != lo) and lo <= best_single <= hi:
        print("  NOTE: the best single method sits inside the hybrid CI — this")
        print("        test set is too small to claim hybrid wins. Add questions.")

    # ── Answer quality ─────────────────────────────────────────────────
    if timestamp_hits:
        ta = mean_of(timestamp_hits)
        print(f"\nTIMESTAMP ACCURACY (±{args.tolerance:.0f}s)")
        print(f"  {fmt(ta, bootstrap_ci(timestamp_hits))}   n={len(timestamp_hits)}")

    if not args.no_answers:
        print("\nANSWER QUALITY")
        if rouge_scores:
            print(f"  ROUGE-L      {fmt(mean_of(rouge_scores), bootstrap_ci(rouge_scores))}")
        if faith_scores:
            print(f"  faithfulness {fmt(mean_of(faith_scores), bootstrap_ci(faith_scores))}")
        answered = len(questions) - refusals
        print(f"  answered     {answered}/{len(questions)}   refusals {refusals}")
        print(f"  flagged      {flagged}/{len(questions)}")
        print()
        print("  'flagged' is the detector's firing rate, NOT a hallucination rate.")
        print("  For a hallucination rate, hand-label a sample and score the")
        print("  detector against those labels.")


if __name__ == "__main__":
    main()
