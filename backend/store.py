"""
store.py
Durable state: chunk text, video metadata, and click feedback in SQLite.

The vectors live in Chroma, but BM25 needs the raw text at query time and a
process restart must not lose it. Keeping chunks only in a module-level dict
is what makes a restarted server answer "video not processed yet" for a video
whose embeddings are sitting on disk.
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional

import config

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    pipeline     TEXT NOT NULL,
    language     TEXT,
    title        TEXT,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    video_id   TEXT NOT NULL,
    chunk_id   TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    start      REAL NOT NULL,
    end        REAL NOT NULL,
    timestamp  TEXT NOT NULL,
    PRIMARY KEY (video_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id, idx);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   TEXT NOT NULL,
    question   TEXT,
    chunk_id   TEXT NOT NULL,
    clicked    INTEGER NOT NULL,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_chunk ON feedback(video_id, chunk_id);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(config.SQLITE_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _lock, _conn() as con:
        con.executescript(SCHEMA)


# ── Videos ─────────────────────────────────────────────────────────────────
def save_video(video_id: str, pipeline: str, language: Optional[str],
               chunk_count: int, title: Optional[str] = None) -> None:
    with _lock, _conn() as con:
        con.execute(
            """INSERT INTO videos (video_id, pipeline, language, title, chunk_count)
               VALUES (?,?,?,?,?)
               ON CONFLICT(video_id) DO UPDATE SET
                   pipeline=excluded.pipeline,
                   language=excluded.language,
                   title=COALESCE(excluded.title, videos.title),
                   chunk_count=excluded.chunk_count""",
            (video_id, pipeline, language, title, chunk_count),
        )


def get_video(video_id: str) -> Optional[Dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM videos WHERE video_id=?", (video_id,)
        ).fetchone()
    return dict(row) if row else None


# ── Chunks ─────────────────────────────────────────────────────────────────
def save_chunks(video_id: str, chunks: List[Dict]) -> None:
    """Replaces any existing chunks for this video."""
    with _lock, _conn() as con:
        con.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
        con.executemany(
            """INSERT INTO chunks (video_id, chunk_id, idx, text, start, end, timestamp)
               VALUES (?,?,?,?,?,?,?)""",
            [
                (video_id, c["chunk_id"], i, c["text"],
                 c["start"], c["end"], c["timestamp"])
                for i, c in enumerate(chunks)
            ],
        )


def get_chunks(video_id: str) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT chunk_id, text, start, end, timestamp "
            "FROM chunks WHERE video_id=? ORDER BY idx",
            (video_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def has_chunks(video_id: str) -> bool:
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM chunks WHERE video_id=? LIMIT 1", (video_id,)
        ).fetchone()
    return row is not None


def delete_video(video_id: str) -> None:
    with _lock, _conn() as con:
        con.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
        con.execute("DELETE FROM videos WHERE video_id=?", (video_id,))


# ── Feedback ───────────────────────────────────────────────────────────────
def log_feedback(video_id: str, chunk_id: str, clicked: bool,
                 question: Optional[str] = None) -> None:
    with _lock, _conn() as con:
        con.execute(
            "INSERT INTO feedback (video_id, question, chunk_id, clicked) VALUES (?,?,?,?)",
            (video_id, question, chunk_id, int(clicked)),
        )


def chunk_boosts(video_id: str) -> Dict[str, float]:
    """
    Click-through rate per chunk, scaled into RRF's range.

    Returns {chunk_id: boost}. Chunks below the interaction floor are omitted
    rather than boosted by zero, so the caller can tell "no data" from "no clicks".
    """
    if not config.FEEDBACK_ENABLED:
        return {}

    with _conn() as con:
        rows = con.execute(
            """SELECT chunk_id, SUM(clicked) AS hits, COUNT(*) AS shown
               FROM feedback WHERE video_id=?
               GROUP BY chunk_id HAVING shown >= ?""",
            (video_id, config.FEEDBACK_MIN_EVENTS),
        ).fetchall()

    return {
        r["chunk_id"]: config.FEEDBACK_MAX_BOOST * (r["hits"] / r["shown"])
        for r in rows if r["shown"]
    }
