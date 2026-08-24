"""
transcript.py
Fetches YouTube transcript and splits it into timestamp-aware chunks.
"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from typing import List, Dict


def extract_video_id(url_or_id: str) -> str:
    """
    Extracts the video ID from a full YouTube URL or returns it directly
    if already an ID.

    Examples:
        https://www.youtube.com/watch?v=dQw4w9WgXcQ  → dQw4w9WgXcQ
        https://youtu.be/dQw4w9WgXcQ                 → dQw4w9WgXcQ
        dQw4w9WgXcQ                                  → dQw4w9WgXcQ
    """
    if "youtube.com/watch" in url_or_id:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url_or_id)
        return parse_qs(parsed.query)["v"][0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    else:
        # assume it's already a video ID
        return url_or_id.strip()


def fetch_transcript(video_id: str) -> List[Dict]:
    """
    Fetches the raw transcript from YouTube.

    Returns a list of dicts like:
        [{ "text": "Hello everyone", "start": 0.0, "duration": 2.5 }, ...]

    Raises:
        ValueError if transcript is unavailable or disabled.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except TranscriptsDisabled:
        raise ValueError(
            "Transcripts are disabled for this video. "
            "Try a video with captions/subtitles enabled."
        )
    except NoTranscriptFound:
        raise ValueError(
            "No transcript found for this video. "
            "The video may not have captions available."
        )
    except Exception as e:
        raise ValueError(f"Failed to fetch transcript: {str(e)}")


def chunk_transcript(
    transcript: List[Dict],
    chunk_size: int = 5,
    overlap: int = 1
) -> List[Dict]:
    """
    Groups raw transcript entries into overlapping chunks.

    Each chunk contains `chunk_size` transcript lines, with `overlap`
    lines shared with the previous chunk to preserve context at boundaries.

    Args:
        transcript:  Raw list of transcript dicts from fetch_transcript()
        chunk_size:  Number of transcript lines per chunk (default 5)
        overlap:     Number of lines to overlap between chunks (default 1)

    Returns:
        List of chunk dicts:
        [
            {
                "chunk_id":   "chunk_0",
                "text":       "combined text of the chunk",
                "start":      0.0,        ← start time in seconds
                "end":        15.3,       ← end time in seconds
                "timestamp":  "0:00"      ← human readable start
            },
            ...
        ]
    """
    if not transcript:
        return []

    chunks = []
    step = chunk_size - overlap
    i = 0
    chunk_index = 0

    while i < len(transcript):
        window = transcript[i: i + chunk_size]

        # Combine text of all entries in this window
        combined_text = " ".join(entry["text"].strip() for entry in window)

        start_sec = window[0]["start"]
        last_entry = window[-1]
        end_sec = last_entry["start"] + last_entry.get("duration", 0)

        # Format start time as MM:SS
        minutes = int(start_sec // 60)
        seconds = int(start_sec % 60)
        timestamp_str = f"{minutes}:{seconds:02d}"

        chunks.append({
            "chunk_id":  f"chunk_{chunk_index}",
            "text":      combined_text,
            "start":     round(start_sec, 2),
            "end":       round(end_sec, 2),
            "timestamp": timestamp_str,
        })

        i += step
        chunk_index += 1

    return chunks


def get_full_text(transcript: List[Dict]) -> str:
    """Returns the complete transcript as a single string (for summarization)."""
    return " ".join(entry["text"].strip() for entry in transcript)


# ── Quick test (run: python transcript.py) ──────────────────────────────────
if __name__ == "__main__":
    # Rick Astley - Never Gonna Give You Up (has captions, safe to test)
    TEST_ID = "dQw4w9WgXcQ"

    print(f"Fetching transcript for video: {TEST_ID}")
    raw = fetch_transcript(TEST_ID)
    print(f"Raw entries: {len(raw)}")
    print(f"First entry: {raw[0]}")

    chunks = chunk_transcript(raw, chunk_size=5, overlap=1)
    print(f"\nChunks created: {len(chunks)}")
    print(f"First chunk:\n{chunks[0]}")
    print(f"\nSecond chunk:\n{chunks[1]}")
