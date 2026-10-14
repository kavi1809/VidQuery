"""
jobs.py
Background processing with pollable status.

This is the change that most alters the shape of the system. An ASR run on a
one-hour lecture takes tens of minutes and a visual run takes several; a
blocking POST hits the browser's fetch timeout and uvicorn's worker timeout
long before either finishes. /process therefore returns a job id immediately
and the extension polls /status, which is also what finally gives the
sidebar's progress bar something real to report.

Jobs live in memory — they describe work in flight, not durable state. What
matters (chunks, vectors, feedback) is written to SQLite and Chroma as each
stage completes, so a restart mid-job loses the job, not the video.
"""

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

import config

log = logging.getLogger("vidquery.jobs")

QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"


@dataclass
class Job:
    job_id: str
    video_id: str
    state: str = QUEUED
    stage: str = "Queued"
    progress: float = 0.0
    pipeline: Optional[str] = None
    language: Optional[str] = None
    chunk_count: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["progress"] = round(self.progress, 3)
        return d


class JobRegistry:
    def __init__(self, workers: int):
        self._jobs: Dict[str, Job] = {}
        self._by_video: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="vq-job")

    # ── lookup ─────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for_video(self, video_id: str) -> Optional[Job]:
        with self._lock:
            job_id = self._by_video.get(video_id)
            job = self._jobs.get(job_id) if job_id else None
        if job and job.state in (QUEUED, RUNNING):
            return job
        return None

    # ── submission ─────────────────────────────────────────────────────
    def submit(self, video_id: str, work: Callable[[Job, Callable], None]) -> Job:
        """
        `work` receives (job, report) where report(stage, fraction=None)
        updates the job's visible progress.
        """
        existing = self.active_for_video(video_id)
        if existing:
            return existing

        job = Job(job_id=uuid.uuid4().hex[:12], video_id=video_id)
        with self._lock:
            self._jobs[job.job_id] = job
            self._by_video[video_id] = job.job_id

        self._pool.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Callable) -> None:
        def report(stage: str, fraction: Optional[float] = None) -> None:
            job.stage = stage
            if fraction is not None:
                job.progress = max(0.0, min(1.0, float(fraction)))
            job.updated_at = time.time()
            log.debug("[%s] %s %.0f%%", job.job_id, stage, job.progress * 100)

        job.state = RUNNING
        report("Starting", 0.01)

        try:
            work(job, report)
            job.state = DONE
            job.progress = 1.0
            if job.stage in ("Starting", "Queued"):
                job.stage = "Ready"
        except Exception as exc:
            job.state = ERROR
            job.error = str(exc) or exc.__class__.__name__
            job.stage = "Failed"
            log.error("Job %s failed: %s\n%s", job.job_id, exc, traceback.format_exc())
        finally:
            job.updated_at = time.time()
            self._sweep()

    # ── housekeeping ───────────────────────────────────────────────────
    def _sweep(self) -> None:
        cutoff = time.time() - config.JOB_RETENTION_S
        with self._lock:
            stale = [
                jid for jid, j in self._jobs.items()
                if j.state in (DONE, ERROR) and j.updated_at < cutoff
            ]
            for jid in stale:
                video_id = self._jobs[jid].video_id
                del self._jobs[jid]
                if self._by_video.get(video_id) == jid:
                    del self._by_video[video_id]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


registry = JobRegistry(config.JOB_WORKERS)
