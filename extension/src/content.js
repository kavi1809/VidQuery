/**
 * content.js
 * The VidQuery sidebar, injected into every YouTube watch page.
 *
 * Classic content script. src/api.js loads first (see the manifest) and
 * exposes window.__VidQuery in the isolated world.
 */

(() => {
  "use strict";

  if (window.__vidqueryLoaded) return;
  window.__vidqueryLoaded = true;

  // Set by src/api.js, which the manifest loads immediately before this file.
  const { api, ApiError } = window.__VidQuery || {};
  if (!api) {
    console.error("[VidQuery] src/api.js did not load — check the manifest order.");
    return;
  }

  const SIDEBAR_ID = "vidquery-sidebar";

  // ── State ───────────────────────────────────────────────────────────
  const state = {
    videoId: null,
    ready: false,
    collapsed: false,
    chatHistory: [],
    lastQuestion: null,
    controller: null,        // aborts in-flight work when the video changes
    recognition: null,
  };

  // ── DOM helpers ─────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function getVideoId() {
    return new URLSearchParams(location.search).get("v");
  }

  function seekTo(seconds) {
    const video = document.querySelector("video.html5-main-video, video");
    if (!video) return;
    video.currentTime = seconds;
    video.play?.().catch(() => {});
  }

  // ── Build ───────────────────────────────────────────────────────────
  function buildSidebar() {
    const root = el("div");
    root.id = SIDEBAR_ID;
    root.setAttribute("role", "complementary");
    root.setAttribute("aria-label", "VidQuery");

    root.innerHTML = `
      <div id="vq-header">
        <span id="vq-logo">VidQuery</span>
        <span id="vq-pipeline-badge"></span>
        <button id="vq-collapse" class="vq-icon" title="Hide panel" aria-label="Hide panel">&#10005;</button>
      </div>

      <div id="vq-status">
        <div id="vq-status-text">Checking this video&hellip;</div>
        <div id="vq-progress" hidden><div id="vq-progress-bar"></div></div>
      </div>

      <div id="vq-actions">
        <button id="vq-summarize" class="vq-action" disabled>Summarise video</button>
        <button id="vq-clear" class="vq-action" disabled>Clear chat</button>
      </div>

      <div id="vq-messages">
        <div id="vq-empty">
          Ask anything about this video. Answers come only from what was said
          or shown, and every one links back to the moment it came from.
          <ul>
            <li>What is the main argument?</li>
            <li>Explain the part about&hellip;</li>
            <li>What did they say at the start?</li>
          </ul>
        </div>
      </div>

      <div id="vq-input-area">
        <input id="vq-question" type="text" autocomplete="off"
               placeholder="Ask about this video&hellip;" disabled />
        <button id="vq-mic" title="Ask by voice" aria-label="Ask by voice" disabled>&#127908;</button>
        <button id="vq-send" disabled>Ask</button>
      </div>
    `;
    return root;
  }

  // ── Status + progress ───────────────────────────────────────────────
  function setStatus(text, { error = false, progress = null } = {}) {
    const label = $("vq-status-text");
    if (!label) return;
    label.textContent = text;
    label.classList.toggle("vq-error", error);

    const bar = $("vq-progress");
    const fill = $("vq-progress-bar");
    if (progress === null) {
      bar.hidden = true;
      fill.classList.remove("vq-indeterminate");
    } else {
      bar.hidden = false;
      if (progress < 0) {
        fill.classList.add("vq-indeterminate");
        fill.style.width = "";
      } else {
        fill.classList.remove("vq-indeterminate");
        fill.style.width = `${Math.round(progress * 100)}%`;
      }
    }
  }

  function setPipelineBadge(pipeline) {
    const badge = $("vq-pipeline-badge");
    if (!badge) return;
    const labels = { captions: "captions", asr: "whisper", visual: "visual ocr", cached: "cached" };
    const label = labels[pipeline];
    badge.textContent = label || "";
    badge.classList.toggle("vq-on", Boolean(label));
  }

  function setEnabled(enabled) {
    state.ready = enabled;
    ["vq-question", "vq-send", "vq-mic", "vq-summarize", "vq-clear"].forEach((id) => {
      const node = $(id);
      if (node) node.disabled = !enabled;
    });
  }

  // ── Messages ────────────────────────────────────────────────────────
  function clearEmptyState() {
    $("vq-empty")?.remove();
  }

  function addMessage(role, text, { chips = [], confidence = null } = {}) {
    clearEmptyState();
    const list = $("vq-messages");

    const wrap = el("div", `vq-msg ${role}`);
    const labels = { user: "You", assistant: "VidQuery", error: "Error" };
    wrap.appendChild(el("div", "vq-msg-label", labels[role] || role));
    wrap.appendChild(el("div", "vq-bubble", text));

    if (chips.length) {
      const row = el("div", "vq-chips");
      chips.forEach((chip) => {
        const button = el("button", "vq-chip", chip.timestamp);
        button.type = "button";
        button.title = chip.text_preview || "Jump here";
        button.addEventListener("click", () => {
          seekTo(chip.start_seconds);
          // The implicit reward signal behind the retrieval boost.
          api.feedback(state.videoId, chip.chunk_id, state.lastQuestion);
        });
        row.appendChild(button);
      });
      wrap.appendChild(row);
    }

    if (confidence) {
      const level = confidence.confidence || "unknown";
      const note = el("div", "vq-confidence");
      note.appendChild(el("span", `vq-dot ${level}`));

      const wording = {
        high: "Grounded in the transcript",
        medium: "Partly grounded — check the timestamps",
        low: "Low confidence — may not be supported by this video",
        unknown: "Not checked",
      };
      note.appendChild(el("span", null, wording[level] || wording.unknown));

      if (confidence.bertscore != null) {
        note.title = `BERTScore ${confidence.bertscore.toFixed(3)} (threshold ${confidence.threshold})`;
      } else if (confidence.ungrounded?.length) {
        note.title = `Not found in the excerpts: ${confidence.ungrounded.join(", ")}`;
      }
      wrap.appendChild(note);
    }

    list.appendChild(wrap);
    list.scrollTop = list.scrollHeight;
    return wrap;
  }

  function addThinking(label = "Thinking…") {
    clearEmptyState();
    const list = $("vq-messages");
    const wrap = el("div", "vq-msg assistant");
    wrap.id = "vq-thinking";
    wrap.appendChild(el("div", "vq-msg-label", "VidQuery"));
    wrap.appendChild(el("div", "vq-bubble vq-thinking", label));
    list.appendChild(wrap);
    list.scrollTop = list.scrollHeight;
  }

  const removeThinking = () => $("vq-thinking")?.remove();

  // ── Actions ─────────────────────────────────────────────────────────
  async function handleAsk() {
    const input = $("vq-question");
    const question = input.value.trim();
    if (!question || !state.ready) return;

    input.value = "";
    setEnabled(false);
    state.lastQuestion = question;

    addMessage("user", question);
    addThinking();

    try {
      const result = await api.ask(
        state.videoId,
        question,
        state.chatHistory,
        state.controller?.signal
      );
      removeThinking();
      addMessage("assistant", result.answer, {
        chips: result.timestamps,
        confidence: result.confidence,
      });

      state.chatHistory.push({ role: "user", content: question });
      state.chatHistory.push({ role: "assistant", content: result.answer });
      if (state.chatHistory.length > 6) {
        state.chatHistory = state.chatHistory.slice(-6);
      }
    } catch (err) {
      removeThinking();
      if (err.name !== "AbortError") addMessage("error", err.message);
    } finally {
      setEnabled(true);
      $("vq-question")?.focus();
    }
  }

  async function handleSummarize() {
    if (!state.ready) return;
    setEnabled(false);
    addThinking("Summarising…");
    try {
      const result = await api.summarize(state.videoId, state.controller?.signal);
      removeThinking();
      addMessage("assistant", result.summary);
    } catch (err) {
      removeThinking();
      if (err.name !== "AbortError") addMessage("error", err.message);
    } finally {
      setEnabled(true);
    }
  }

  function handleClear() {
    state.chatHistory = [];
    const list = $("vq-messages");
    list.replaceChildren();
    const empty = el("div", null, "Chat cleared. Ask something else about this video.");
    empty.id = "vq-empty";
    list.appendChild(empty);
  }

  // ── Voice ───────────────────────────────────────────────────────────
  function handleVoice() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const mic = $("vq-mic");

    if (!Recognition) {
      addMessage("error", "Voice input needs Chrome's Web Speech API, which this browser does not expose.");
      return;
    }

    if (state.recognition) {
      state.recognition.stop();
      return;
    }

    const recognition = new Recognition();
    // navigator.language is the browser's UI language, which is a reasonable
    // default but not necessarily the language the user intends to speak.
    recognition.lang = navigator.language || "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      state.recognition = recognition;
      mic.classList.add("vq-recording");
      mic.title = "Listening — click to stop";
    };

    recognition.onresult = (event) => {
      const said = event.results[0][0].transcript;
      $("vq-question").value = said;
      handleAsk();
    };

    recognition.onerror = (event) => {
      const reasons = {
        "not-allowed": "Microphone access was blocked. Allow it in the address bar to use voice.",
        "no-speech": "No speech detected — try again.",
        "audio-capture": "No microphone found.",
      };
      addMessage("error", reasons[event.error] || `Voice input failed (${event.error}).`);
    };

    recognition.onend = () => {
      state.recognition = null;
      mic.classList.remove("vq-recording");
      mic.title = "Ask by voice";
    };

    try {
      recognition.start();
    } catch {
      /* already started */
    }
  }

  // ── Collapse / reopen ───────────────────────────────────────────────
  // The original had no way back once the panel was closed: the popup sent a
  // TOGGLE_SIDEBAR message that nothing in the page listened for.
  function setCollapsed(collapsed) {
    state.collapsed = collapsed;
    const root = $(SIDEBAR_ID);
    root?.classList.toggle("vq-collapsed", collapsed);
    document.documentElement.classList.toggle("vq-sidebar-open", !collapsed);

    let tab = $("vq-reopen");
    if (collapsed && !tab) {
      tab = el("button", null, "VIDQUERY");
      tab.id = "vq-reopen";
      tab.type = "button";
      tab.title = "Show VidQuery";
      tab.addEventListener("click", () => setCollapsed(false));
      document.body.appendChild(tab);
    } else if (!collapsed && tab) {
      tab.remove();
    }

    // Let the player recompute its layout against the new width.
    window.dispatchEvent(new Event("resize"));
  }

  chrome.runtime.onMessage.addListener((message) => {
    if (message?.type === "TOGGLE_SIDEBAR") setCollapsed(!state.collapsed);
  });

  // ── Init per video ──────────────────────────────────────────────────
  async function initForVideo() {
    const videoId = getVideoId();
    if (!videoId || videoId === state.videoId) return;

    state.controller?.abort();
    state.controller = new AbortController();

    state.videoId = videoId;
    state.chatHistory = [];
    setEnabled(false);
    setPipelineBadge(null);
    handleClear();

    setStatus("Checking this video…", { progress: -1 });

    try {
      const result = await api.processVideo(
        videoId,
        ({ stage, progress, pipeline }) => {
          setStatus(stage, { progress: progress > 0 ? progress : -1 });
          if (pipeline) setPipelineBadge(pipeline);
        },
        { signal: state.controller.signal }
      );

      setPipelineBadge(result.pipeline);
      const count = result.chunk_count;
      setStatus(count ? `Ready — ${count} segments indexed` : "Ready");
      setEnabled(true);
      $("vq-question")?.focus();
    } catch (err) {
      if (err.name === "AbortError") return;
      setStatus(err.message, { error: true });
      if (err instanceof ApiError && err.status === 422) {
        addMessage("error", err.message);
      }
    }
  }

  // ── Mount ───────────────────────────────────────────────────────────
  function mount() {
    if (document.getElementById(SIDEBAR_ID)) return;
    if (!document.body) return;

    document.body.appendChild(buildSidebar());
    document.documentElement.classList.add("vq-sidebar-open");

    $("vq-send").addEventListener("click", handleAsk);
    $("vq-question").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleAsk();
      }
    });
    $("vq-mic").addEventListener("click", handleVoice);
    $("vq-summarize").addEventListener("click", handleSummarize);
    $("vq-clear").addEventListener("click", handleClear);
    $("vq-collapse").addEventListener("click", () => setCollapsed(true));

    initForVideo();
  }

  // YouTube is a single-page app: navigating between videos never reloads.
  // yt-navigate-finish is YouTube's own signal and fires reliably; the URL
  // poll is the belt-and-braces fallback for layout changes that skip it.
  window.addEventListener("yt-navigate-finish", () => {
    if (location.pathname === "/watch") {
      mount();
      initForVideo();
    }
  });

  let lastUrl = location.href;
  setInterval(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      if (location.pathname === "/watch") {
        mount();
        initForVideo();
      }
    }
  }, 1000);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount, { once: true });
  } else {
    mount();
  }
})();
