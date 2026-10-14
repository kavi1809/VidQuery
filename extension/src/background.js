/**
 * background.js
 * Service worker. Two jobs: relaying the popup's toggle, and proxying every
 * backend call.
 *
 * The proxy is the important one. A fetch issued from a content script is
 * made in the page's context, so it inherits everything the page is subject
 * to: CORS against https://www.youtube.com, Private/Local Network Access
 * checks for reaching loopback from a public site, and mixed-content rules
 * for http:// from an https:// page. Any one of those failing surfaces as an
 * opaque network error with no status and no detail.
 *
 * A fetch from the service worker is made in the EXTENSION's context, where
 * host_permissions grant cross-origin access outright and none of those three
 * apply. It is the same privilege the popup has always had — which is exactly
 * why the popup could reach /health while the sidebar could not.
 */

const ALLOWED_PREFIXES = [
  "http://127.0.0.1:8000/",
  "http://localhost:8000/",
];

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === "install") {
    console.log("VidQuery installed. Start the backend on port 8000.");
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // ── Backend proxy ──────────────────────────────────────────────────
  if (message?.type === "VQ_FETCH") {
    // Only ever proxy to the local backend. A content script is the least
    // trusted part of an extension — it shares a page with whatever the site
    // is running — so it does not get to name arbitrary URLs for the worker
    // to fetch with extension privileges.
    const url = String(message.url || "");
    if (!ALLOWED_PREFIXES.some((prefix) => url.startsWith(prefix))) {
      sendResponse({ ok: false, status: 0, error: "Blocked non-backend URL" });
      return true;
    }

    (async () => {
      try {
        const res = await fetch(url, {
          method: message.method || "GET",
          headers: message.body ? { "Content-Type": "application/json" } : undefined,
          body: message.body || undefined,
        });
        sendResponse({
          ok: res.ok,
          status: res.status,
          body: await res.text(),
        });
      } catch (err) {
        sendResponse({ ok: false, status: 0, error: String(err?.message || err) });
      }
    })();

    return true;   // response is async
  }

  // ── Toggle relay ───────────────────────────────────────────────────
  if (message?.type === "TOGGLE_SIDEBAR" && !sender.tab) {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (!tab?.id) {
        sendResponse({ ok: false });
        return;
      }
      chrome.tabs.sendMessage(tab.id, { type: "TOGGLE_SIDEBAR" })
        .then(() => sendResponse({ ok: true }))
        .catch(() => sendResponse({ ok: false }));
    });
    return true;
  }

  return false;
});
