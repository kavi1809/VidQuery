# ⚡ VidQuery — AI-Powered YouTube Video Q&A

> Ask questions from any YouTube video without watching it.
> Powered by **RAG + Hybrid Search + Timestamp Grounding**, delivered as a Chrome Extension.

---

## 📌 What it does

- **Open any YouTube video** → VidQuery auto-fetches the transcript
- **Ask natural language questions** → get grounded answers in seconds
- **Click timestamp chips** → jump directly to the relevant moment in the video
- **Summarize in one click** → get key points without watching

---

## 🏗️ Architecture

```
Chrome Extension (content.js)
        │
        │  HTTP POST /process, /ask, /summarize
        ▼
FastAPI Backend (main.py)
        │
        ├── transcript.py    → YouTube Transcript API → chunks
        ├── embedder.py      → Sentence-BERT → ChromaDB vector store
        ├── retriever.py     → BM25 + Vector → RRF Hybrid Search
        └── llm.py           → Claude API → grounded answer
```

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/VidQuery.git
cd VidQuery
```

### 2. Set up Python backend
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add your API key
```bash
# Inside backend/ folder
cp .env.example .env
# Open .env and paste your Anthropic API key
```
Get your key at: https://console.anthropic.com

### 4. Start the backend server
```bash
uvicorn main:app --reload --port 8000
```

You should see: `Uvicorn running on http://127.0.0.1:8000`

### 5. Load the Chrome Extension
1. Open Chrome → go to `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. The VidQuery icon appears in your Chrome toolbar

### 6. Use it!
1. Open any YouTube video (with captions)
2. The VidQuery sidebar auto-appears on the right
3. Wait ~5 seconds for indexing
4. Ask anything about the video

---

## 📁 Project Structure

```
VidQuery/
├── backend/
│   ├── main.py           ← FastAPI server (all endpoints)
│   ├── transcript.py     ← YouTube transcript fetcher & chunker
│   ├── embedder.py       ← Sentence-BERT embeddings + ChromaDB
│   ├── retriever.py      ← Hybrid search (BM25 + Vector + RRF)
│   ├── llm.py            ← Claude API answer generation
│   ├── requirements.txt
│   └── .env.example
│
├── extension/
│   ├── manifest.json     ← Chrome extension config (Manifest v3)
│   ├── popup.html        ← Extension popup UI
│   ├── popup.js          ← Popup logic
│   ├── src/
│   │   ├── content.js    ← YouTube sidebar UI + API calls
│   │   └── background.js ← Service worker
│   └── icons/
│
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Chrome Extension | JavaScript (Manifest v3) |
| Backend API | Python, FastAPI |
| Transcript | youtube-transcript-api |
| Embeddings | Sentence-BERT (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Re-ranking | Reciprocal Rank Fusion |
| LLM | Claude (claude-sonnet-4-6) |

---

## 🔗 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check if server is running |
| `/process` | POST | Fetch + embed video transcript |
| `/ask` | POST | Answer a question (hybrid search + LLM) |
| `/summarize` | POST | Summarize full video |

---

## 📄 License

MIT License — Final Year Project by Kaviya
