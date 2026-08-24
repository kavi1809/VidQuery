/**
* content.js
* Injected into every YouTube /watch page.
* Builds the VidQuery sidebar UI and communicates with the FastAPI backend.
*/
 
const API_BASE = "http://127.0.0.1:8000";
 
// ── State ──────────────────────────────────────────────────────────────────
let currentVideoId  = null;
let chatHistory     = [];       // [{role, content}] for multi-turn Q&A
let isProcessed     = false;
let sidebarVisible  = false;
 
// ── Helpers ────────────────────────────────────────────────────────────────
function getVideoId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("v");
}
 
function formatTimestamp(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
 
function jumpToTime(seconds) {
  const video = document.querySelector("video");
  if (video) {
    video.currentTime = seconds;
    video.play();
  }
}
 
// ── API calls ──────────────────────────────────────────────────────────────
async function processVideo(videoId) {
  const res = await fetch(`${API_BASE}/process`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ video_url: videoId })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to process video");
  }
  return res.json();
}
 
async function askQuestion(videoId, question) {
  const res = await fetch(`${API_BASE}/ask`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      video_id:    videoId,
      question:    question,
      chat_history: chatHistory
    })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to get answer");
  }
  return res.json();
}
 
async function summarizeVideo(videoId) {
  const res = await fetch(`${API_BASE}/summarize`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ video_id: videoId })
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to summarize");
  }
  return res.json();
}
 
// ── Sidebar HTML ───────────────────────────────────────────────────────────
function createSidebar() {
  const sidebar = document.createElement("div");
  sidebar.id = "vidquery-sidebar";
  sidebar.innerHTML = `
<div id="vq-header">
<div id="vq-logo">⚡ VidQuery</div>
<button id="vq-close" title="Close">✕</button>
</div>
 
    <div id="vq-status">
<div id="vq-status-text">Loading video info...</div>
<div id="vq-progress" style="display:none">
<div id="vq-progress-bar"></div>
</div>
</div>
 
    <div id="vq-actions">
<button id="vq-summarize-btn" disabled>📝 Summarize video</button>
</div>
 
    <div id="vq-chat">
<div id="vq-messages"></div>
<div id="vq-input-area">
<input
          id="vq-question-input"
          type="text"
          placeholder="Ask something about this video..."
          disabled
        />
<button id="vq-send-btn" disabled>Ask</button>
</div>
</div>
  `;
 
  injectStyles();
  return sidebar;
}
 
function injectStyles() {
  if (document.getElementById("vq-styles")) return;
 
  const style = document.createElement("style");
  style.id = "vq-styles";
  style.textContent = `
    #vidquery-sidebar {
      position: fixed;
      top: 60px;
      right: 0;
      width: 360px;
      height: calc(100vh - 60px);
      background: #0f172a;
      color: #e2e8f0;
      display: flex;
      flex-direction: column;
      z-index: 9999;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 14px;
      border-left: 1px solid #1e293b;
      box-shadow: -4px 0 20px rgba(0,0,0,0.4);
      transform: translateX(0);
      transition: transform 0.3s ease;
    }
    #vidquery-sidebar.hidden {
      transform: translateX(100%);
    }
 
    #vq-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      background: #0d1424;
      border-bottom: 1px solid #1e293b;
      flex-shrink: 0;
    }
    #vq-logo {
      font-size: 16px;
      font-weight: 600;
      color: #38bdf8;
      letter-spacing: 0.5px;
    }
    #vq-close {
      background: none;
      border: none;
      color: #64748b;
      font-size: 16px;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 4px;
    }
    #vq-close:hover { color: #e2e8f0; background: #1e293b; }
 
    #vq-status {
      padding: 12px 16px;
      background: #0d1424;
      border-bottom: 1px solid #1e293b;
      flex-shrink: 0;
    }
    #vq-status-text { color: #94a3b8; font-size: 13px; }
    #vq-progress {
      height: 3px;
      background: #1e293b;
      border-radius: 2px;
      margin-top: 8px;
      overflow: hidden;
    }
    #vq-progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      border-radius: 2px;
      animation: vq-progress-anim 1.5s ease-in-out infinite;
    }
    @keyframes vq-progress-anim {
      0%   { width: 0%; margin-left: 0; }
      50%  { width: 60%; margin-left: 20%; }
      100% { width: 0%; margin-left: 100%; }
    }
 
    #vq-actions {
      padding: 10px 16px;
      border-bottom: 1px solid #1e293b;
      flex-shrink: 0;
    }
    #vq-summarize-btn {
      width: 100%;
      padding: 8px;
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #334155;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      transition: background 0.15s;
    }
    #vq-summarize-btn:hover:not(:disabled) { background: #273549; }
    #vq-summarize-btn:disabled { opacity: 0.4; cursor: not-allowed; }
 
    #vq-chat {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #vq-messages {
      flex: 1;
      overflow-y: auto;
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    #vq-messages::-webkit-scrollbar { width: 4px; }
    #vq-messages::-webkit-scrollbar-track { background: transparent; }
    #vq-messages::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }
 
    .vq-msg {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .vq-msg-label {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .vq-msg.user .vq-msg-label { color: #38bdf8; }
    .vq-msg.assistant .vq-msg-label { color: #818cf8; }
    .vq-msg-bubble {
      padding: 10px 12px;
      border-radius: 10px;
      line-height: 1.55;
      font-size: 13px;
    }
    .vq-msg.user .vq-msg-bubble {
      background: #1e293b;
      color: #e2e8f0;
    }
    .vq-msg.assistant .vq-msg-bubble {
      background: #0d1f36;
      color: #e2e8f0;
      border: 1px solid #1e3a5f;
    }
    .vq-msg.error .vq-msg-bubble {
      background: #2d1516;
      color: #fca5a5;
      border: 1px solid #7f1d1d;
    }
 
    .vq-timestamps {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .vq-ts-chip {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 10px;
      background: #0f3460;
      color: #38bdf8;
      border: 1px solid #1e4a80;
      border-radius: 20px;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .vq-ts-chip:hover { background: #1a4a7a; }
    .vq-ts-chip::before { content: "▶"; font-size: 9px; }
 
    #vq-input-area {
      display: flex;
      gap: 8px;
      padding: 12px 16px;
      border-top: 1px solid #1e293b;
      flex-shrink: 0;
      background: #0d1424;
    }
    #vq-question-input {
      flex: 1;
      padding: 9px 12px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #e2e8f0;
      font-size: 13px;
      outline: none;
    }
    #vq-question-input:focus { border-color: #38bdf8; }
    #vq-question-input:disabled { opacity: 0.4; }
    #vq-question-input::placeholder { color: #475569; }
 
    #vq-send-btn {
      padding: 9px 16px;
      background: #38bdf8;
      color: #0f172a;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: background 0.15s;
      white-space: nowrap;
    }
    #vq-send-btn:hover:not(:disabled) { background: #7dd3fc; }
    #vq-send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
 
    /* Adjust YouTube layout when sidebar is open */
    #vidquery-sidebar ~ ytd-app,
    #secondary { margin-right: 360px; }
  `;
  document.head.appendChild(style);
}
 
// ── UI helpers ─────────────────────────────────────────────────────────────
function setStatus(text, showProgress = false) {
  document.getElementById("vq-status-text").textContent = text;
  document.getElementById("vq-progress").style.display = showProgress ? "block" : "none";
}
 
function enableInput() {
  document.getElementById("vq-question-input").disabled = false;
  document.getElementById("vq-send-btn").disabled = false;
  document.getElementById("vq-summarize-btn").disabled = false;
}
 
function addMessage(role, text, timestamps = []) {
  const messages = document.getElementById("vq-messages");
 
  const msgDiv = document.createElement("div");
  msgDiv.className = `vq-msg ${role}`;
 
  const label = document.createElement("div");
  label.className = "vq-msg-label";
  label.textContent = role === "user" ? "You" : role === "assistant" ? "VidQuery" : "Error";
 
  const bubble = document.createElement("div");
  bubble.className = "vq-msg-bubble";
  bubble.textContent = text;
 
  msgDiv.appendChild(label);
  msgDiv.appendChild(bubble);
 
  // Add timestamp chips if provided
  if (timestamps && timestamps.length > 0) {
    const tsRow = document.createElement("div");
    tsRow.className = "vq-timestamps";
    timestamps.forEach(ts => {
      const chip = document.createElement("span");
      chip.className = "vq-ts-chip";
      chip.textContent = ts.timestamp;
      chip.title = ts.text_preview;
      chip.onclick = () => jumpToTime(ts.start_seconds);
      tsRow.appendChild(chip);
    });
    msgDiv.appendChild(tsRow);
  }
 
  messages.appendChild(msgDiv);
  messages.scrollTop = messages.scrollHeight;
 
  return msgDiv;
}
 
function addLoadingMessage() {
  const messages = document.getElementById("vq-messages");
  const div = document.createElement("div");
  div.className = "vq-msg assistant";
  div.id = "vq-loading-msg";
  div.innerHTML = `
<div class="vq-msg-label">VidQuery</div>
<div class="vq-msg-bubble" style="color:#64748b">Thinking...</div>
  `;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}
 
function removeLoadingMessage() {
  const el = document.getElementById("vq-loading-msg");
  if (el) el.remove();
}
 
// ── Main init ──────────────────────────────────────────────────────────────
async function initVidQuery() {
  const videoId = getVideoId();
  if (!videoId || videoId === currentVideoId) return;
 
  currentVideoId = videoId;
  chatHistory    = [];
  isProcessed    = false;
 
  setStatus("Processing video transcript...", true);
 
  try {
    const result = await processVideo(videoId);
    isProcessed = true;
    setStatus(`Ready — ${result.chunk_count} segments indexed`);
    enableInput();
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}
 
// ── Event listeners ────────────────────────────────────────────────────────
function attachListeners() {
  // Send button
  document.getElementById("vq-send-btn").addEventListener("click", handleSend);
 
  // Enter key in input
  document.getElementById("vq-question-input").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) handleSend();
  });
 
  // Close button
  document.getElementById("vq-close").addEventListener("click", toggleSidebar);
 
  // Summarize button
  document.getElementById("vq-summarize-btn").addEventListener("click", handleSummarize);
}
 
async function handleSend() {
  const input = document.getElementById("vq-question-input");
  const question = input.value.trim();
  if (!question || !isProcessed) return;
 
  input.value = "";
  document.getElementById("vq-send-btn").disabled = true;
  input.disabled = true;
 
  addMessage("user", question);
  addLoadingMessage();
 
  try {
    const result = await askQuestion(currentVideoId, question);
    removeLoadingMessage();
    addMessage("assistant", result.answer, result.timestamps);
 
    // Update chat history for multi-turn context
    chatHistory.push({ role: "user",      content: question });
    chatHistory.push({ role: "assistant", content: result.answer });
 
    // Keep history short (last 6 turns = 3 exchanges)
    if (chatHistory.length > 6) chatHistory = chatHistory.slice(-6);
 
  } catch (err) {
    removeLoadingMessage();
    addMessage("error", `Error: ${err.message}`);
  } finally {
    document.getElementById("vq-send-btn").disabled = false;
    input.disabled = false;
    input.focus();
  }
}
 
async function handleSummarize() {
  if (!isProcessed) return;
 
  const btn = document.getElementById("vq-summarize-btn");
  btn.disabled = true;
  btn.textContent = "Summarizing...";
 
  addLoadingMessage();
 
  try {
    const result = await summarizeVideo(currentVideoId);
    removeLoadingMessage();
    addMessage("assistant", result.summary);
  } catch (err) {
    removeLoadingMessage();
    addMessage("error", `Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "📝 Summarize video";
  }
}
 
// ── Toggle sidebar visibility ──────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.getElementById("vidquery-sidebar");
  sidebarVisible = !sidebarVisible;
  sidebar.classList.toggle("hidden", !sidebarVisible);
}
 
// ── Inject sidebar and initialise ─────────────────────────────────────────
function mountSidebar() {
  // Don't inject twice
  if (document.getElementById("vidquery-sidebar")) return;
 
  const sidebar = createSidebar();
  document.body.appendChild(sidebar);
  attachListeners();
  sidebarVisible = true;
 
  initVidQuery();
}
 
// Run on page load
mountSidebar();
 
// Re-run when YouTube navigates to a new video (SPA navigation)
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    if (url.includes("/watch")) {
      setTimeout(() => initVidQuery(), 1500); // wait for page to settle
    }
  }
}).observe(document, { subtree: true, childList: true });