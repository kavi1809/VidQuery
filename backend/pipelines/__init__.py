"""
pipelines
Content extraction. Every pipeline returns the same shape so the chunker
downstream never needs to know which one ran:

    [{"text": str, "start": float, "duration": float}, ...]

Each module guards its own heavy imports, so a missing OCR stack degrades that
one pipeline into a clear error message instead of taking the API down at
import time.
"""

from typing import Dict, List, Protocol


class ExtractionError(RuntimeError):
    """Raised when a pipeline cannot produce a transcript."""


class Progress(Protocol):
    def __call__(self, stage: str, fraction: float = None) -> None: ...


__all__ = ["ExtractionError", "Progress"]
