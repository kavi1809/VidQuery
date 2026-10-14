# VidQuery

Ask questions about any YouTube video and jump straight to the moment that
answers them. A Chrome extension over a local FastAPI backend, using
retrieval-augmented generation with hybrid search.

Gemini is the only hosted service. Everything else — embeddings, speech
recognition, OCR, the faithfulness check — runs locally.

> Architecture: see [`docs/architecture-revised.html`](docs/architecture-revised.html)
> (open in a browser). It supersedes Figure 1 in the project document.

---

## How it works

```
Chrome extension  ──POST /process──▶  FastAPI  ──▶  job queue
       ▲                                                │
       └──────GET /status/{id}───────────────────────────┘
                                                        │
                                          ┌─────────────┴─────────────┐
                                     smart detector: captions? audio? silent?
                                          │             │            │
                                     captions         ASR         visual
                                    (yt caption)  (faster-whisper) (OCR + Gemini vision)
                                          └─────────────┬────────────┘
                                                   fallback chain
                                                        ▼
                                     chunker — 45 s windows, 10 s overlap
                                          ┌─────────────┴─────────────┐
                                     BM25 index                SBERT ▶ ChromaDB
                                          │                           │
                                     BM25 search                vector search
                                          └─────────────┬─────────────┘
                                              RRF fusion (+ click boost)
                                                        ▼
                                          Gemini — grounded answer
                                                        ▼
                                     faithfulness gate ▶ confidence state
```

---

## Setup

### 1. Backend

Requires Python 3.11 or newer (tested on 3.14).

```bash
cd backend
py -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Installing CPU-only torch first keeps the download near 200 MB instead of
pulling ~2.5 GB of CUDA wheels you will not use.

### 2. API key

```bash
copy .env.example .env          # cp on macOS / Linux
```

Put your key in `backend/.env`. Get one free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```
GEMINI_API_KEY=...
```

### 3. Run

```bash
.venv\Scripts\uvicorn main:app --reload --port 8000
```

Check it: <http://127.0.0.1:8000/health> lists the model, the embedding
backend, and which pipelines are installed. Interactive docs are at
<http://127.0.0.1:8000/docs>.

### 4. Extension

1. Open `chrome://extensions`
2. Turn on **Developer mode**
3. **Load unpacked** → select the `extension/` folder
4. Open any YouTube video

The first video after each server start takes an extra 10–30 seconds while the
embedding model loads. The panel reports this as its own stage.

---

## Optional pipelines

The core install handles videos that have captions — the large majority. The
other two pipelines are separate so a failed install cannot break the API.

**Videos with speech but no captions**

```bash
pip install -r requirements-asr.txt
winget install Gyan.FFmpeg          # ffmpeg must be on PATH
```

**Slide-only videos with no audio**

Install this in its **own** virtualenv. PaddlePaddle pins numpy and protobuf
versions that conflict with ChromaDB, and on Windows it is the most fragile
dependency in the project.

```bash
py -m venv .venv-visual
.venv-visual\Scripts\activate
pip install -r requirements-visual.txt
```

**Evaluation and BERTScore**

```bash
pip install -r requirements-eval.txt
```

Without it, the faithfulness gate falls back to lexical grounding and reports
`"method": "lexical"` rather than silently pretending it ran BERTScore.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model, embedding backend, installed pipelines |
| `/detect` | POST | Which pipeline this video needs, without processing it |
| `/process` | POST | Enqueue ingest; returns a `job_id` immediately |
| `/status/{job_id}` | GET | Stage and progress of a running job |
| `/ask` | POST | Hybrid retrieval, grounded answer, confidence |
| `/summarize` | POST | Whole-video summary |
| `/feedback` | POST | Chip click, feeding the retrieval boost |

`/process` never blocks. An ASR run on a one-hour lecture takes tens of
minutes; a synchronous request would time out in the browser long before it
finished.

---

## Configuration

Every constant lives in `config.py` and can be overridden in `.env`. The ones
that matter:

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.6-flash` | 1.5 is retired; 2.5 is closed to new keys |
| `EMBEDDING_BACKEND` | `sbert` | `gemini` skips the torch install entirely |
| `CHUNK_SECONDS` | `45` | Time windows, not line counts |
| `RETRIEVE_FINAL` | `4` | Chunks passed to the model |
| `RRF_K` | `60` | Fusion smoothing constant |
| `FEEDBACK_MAX_BOOST` | `0.002` | Must stay well under the ~0.033 RRF ceiling |
| `FEEDBACK_ENABLED` | `true` | **Set to `false` when running evaluation** |
| `FAITHFULNESS_THRESHOLD` | `0.75` | Calibrate on your own dev set |

### On a corporate network

Intercepting proxies re-sign TLS with a private root that Python's bundled CA
bundle does not know, which surfaces as `CERTIFICATE_VERIFY_FAILED` on YouTube
and Gemini calls. `truststore` is installed by default and points Python at
the OS certificate store, where that root already is. Certificates are still
verified. Disable with `VIDQUERY_SYSTEM_CERTS=false`.

---

## Layout

```
backend/
  main.py            FastAPI app and endpoints
  config.py          all settings, .env overrides
  schemas.py         request/response models
  jobs.py            background jobs with pollable status
  ingest.py          detector → pipeline → chunk → index, with fallback
  detector.py        captions? audio? silent? slideshow?
  chunker.py         time-windowed, timestamp-tagged chunks
  embedder.py        SBERT or Gemini embeddings, ChromaDB
  retriever.py       BM25 + vector + RRF + click boost
  faithfulness.py    BERTScore and lexical grounding
  gemini.py          every Gemini call: answer, translate, vision, embed
  store.py           SQLite: chunks, videos, feedback
  pipelines/
    captions.py      YouTube caption track
    asr.py           faster-whisper
    visual.py        frames → pHash → OCR → Gemini vision

extension/
  manifest.json      Manifest V3
  src/content.js     sidebar UI
  src/api.js         backend client, including status polling
  src/sidebar.css    panel styles
  src/background.js  service worker
  popup.html/js      toolbar popup

docs/
  architecture-revised.html
```

---

## Notes on the design

A few decisions that differ from the original project document, and why:

- **BM25 indexes the chunker's output, not the embedder's.** BM25 scores
  tokens; it cannot consume a 384-dimension vector.
- **RRF consumes both searches.** It is not a third search running alongside
  them.
- **The click boost is capped at 0.002.** A rank-0 RRF contribution is
  1/60 ≈ 0.0167, so the two-list maximum is about 0.033. A boost of 0.1 would
  be six times the whole score range and one click would pin a chunk to the
  top permanently.
- **Transcripts are indexed in their source language.** Translating every
  caption segment first means 1,500+ calls per video to a rate-limited
  endpoint, and loses information before indexing. The multilingual encoder
  removes the need; only the question is translated, for BM25's benefit.
- **Silent videos never go to ASR.** Whisper hallucinates subtitle credits on
  silence, so silence routes to the visual pipeline unconditionally.
- **Code extraction reads frames, not the transcript.** Nobody dictates their
  brackets aloud; the code is on screen.
- **TrOCR is not in the OCR ensemble.** `trocr-base-printed` is a single-line
  recognizer with no detection stage and cannot read a full slide.

---

MIT License — Kaviya S, Executive M.Tech (Applied AI), VNIT Nagpur, 2025–26.
