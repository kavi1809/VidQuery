/**
 * popup.js
 * Toolbar popup: reports what the backend has available, and toggles the panel.
 */

const API_BASE = "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);

function show(id, visible) {
  const node = $(id);
  if (node) node.hidden = !visible;
}

async function checkServer() {
  const dot = $("server-dot");
  const label = $("server-status");

  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(2500),
    });
    if (!res.ok) throw new Error(String(res.status));

    const health = await res.json();

    dot.classList.add("ok");
    label.textContent = health.gemini?.configured
      ? "Server running"
      : "Server running — no API key";

    $("v-model").textContent = health.gemini?.model || "not set";
    $("v-embed").textContent = health.embeddings || "unknown";

    const describe = (p) => (p?.available ? "ready" : "not installed");
    $("v-asr").textContent = describe(health.pipelines?.asr);
    $("v-visual").textContent = describe(health.pipelines?.visual);

    show("server-card", true);
    show("server-hint", false);
  } catch {
    dot.classList.add("bad");
    label.textContent = "Server offline";
    show("server-card", false);
    show("server-hint", true);
  }
}

async function checkTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const onWatchPage = Boolean(tab?.url?.includes("youtube.com/watch"));

  show("on-youtube", onWatchPage);
  show("off-youtube", !onWatchPage);

  if (onWatchPage) {
    $("toggle-btn").addEventListener("click", async () => {
      // Message the tab directly — the service worker is only a relay, and
      // going straight to the content script is one less thing to fail.
      try {
        await chrome.tabs.sendMessage(tab.id, { type: "TOGGLE_SIDEBAR" });
      } catch {
        // Content script not injected yet (e.g. the page predates an install).
      }
      window.close();
    });
  }
}

checkServer();
checkTab();
