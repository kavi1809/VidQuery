"""
build_testset.py
Scaffolds a test set for annotation.

Indexes each video, then writes every chunk out with its id and timestamp so
you can read through and mark which ones answer each question. Annotating
against real chunk ids is the only way Precision@k and Recall@k mean anything
— guessing relevance after seeing what the system retrieved measures nothing.

Usage:
    python eval/build_testset.py 46P1rL0rzPE aircAruvnKk
    python eval/build_testset.py --chunks 46P1rL0rzPE > chunks.txt

Writes eval/testset.json with an empty question list, and eval/chunks/<id>.txt
with the readable chunk dump.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import ingest                                    # noqa: E402
import store                                     # noqa: E402

EVAL_DIR = Path(__file__).parent
CHUNK_DIR = EVAL_DIR / "chunks"
TESTSET = EVAL_DIR / "testset.json"


def report(stage: str, fraction: float = None) -> None:
    pct = f" {fraction * 100:5.1f}%" if fraction is not None else ""
    print(f"    {stage}{pct}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a VidQuery test set")
    parser.add_argument("videos", nargs="+", help="YouTube video ids")
    parser.add_argument("--chunks-only", action="store_true",
                        help="dump chunks for already-indexed videos, skip processing")
    args = parser.parse_args()

    store.init_db()
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for video_id in args.videos:
        print(f"\n{video_id}")

        if not args.chunks_only:
            try:
                result = ingest.process_video(video_id, report)
                print(f"    -> {result['pipeline']}, {result['chunk_count']} chunks")
            except Exception as exc:
                print(f"    FAILED: {exc}")
                continue

        chunks = store.get_chunks(video_id)
        if not chunks:
            print("    no chunks stored")
            continue

        out = CHUNK_DIR / f"{video_id}.txt"
        with out.open("w", encoding="utf-8") as fh:
            fh.write(f"# {video_id} — {len(chunks)} chunks\n")
            fh.write("# Mark the chunk_ids that answer each of your questions.\n\n")
            for c in chunks:
                fh.write(f"[{c['chunk_id']}] {c['timestamp']} ({c['start']:.0f}s)\n")
                fh.write(f"    {c['text']}\n\n")

        record = store.get_video(video_id) or {}
        summary.append({
            "video_id": video_id,
            "pipeline": record.get("pipeline"),
            "language": record.get("language"),
            "chunks": len(chunks),
        })
        print(f"    chunks written to {out.relative_to(EVAL_DIR.parent)}")

    if not TESTSET.exists():
        TESTSET.write_text(json.dumps({
            "_readme": [
                "One entry per question. relevant_chunks are the chunk ids that",
                "genuinely answer it — read eval/chunks/<video_id>.txt to find them.",
                "answer_time_s is where the answer is actually discussed, used for",
                "Timestamp Accuracy. reference_answer is your own wording of the",
                "correct answer, used for ROUGE-L.",
                "Aim for 80-120 questions. Thirty cannot separate 0.71 from 0.57."
            ],
            "videos": summary,
            "questions": [
                {
                    "id": "q1",
                    "video_id": summary[0]["video_id"] if summary else "VIDEO_ID",
                    "question": "What does she say about failure?",
                    "relevant_chunks": ["chunk_0"],
                    "answer_time_s": 12.0,
                    "reference_answer": "She got out for zero in the 2017 World Cup final.",
                }
            ],
        }, indent=2), encoding="utf-8")
        print(f"\nTemplate written to {TESTSET.relative_to(EVAL_DIR.parent)}")
    else:
        print(f"\n{TESTSET.name} already exists — left untouched")


if __name__ == "__main__":
    main()
