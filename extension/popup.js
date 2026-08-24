/**
* popup.js
* Logic for the VidQuery extension popup.
*/
 
const API_BASE = "http://127.0.0.1:8000";
 
// Check if backend server is running
async function checkServer() {
  const dot    = document.getElementById("server-dot");
  const status = document.getElementById("server-status");
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      dot.classList.remove("offline");
      status.textContent = "Backend server running ✓";
    } else {
      throw new Error("not ok");
    }
  } catch {
    dot.classList.add("offline");
    status.textContent = "Server offline — run uvicorn";
  }
}
 
// Check if current tab is YouTube
async function checkTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const isYouTube = tab?.url?.includes("youtube.com/watch");
 
  document.getElementById("main-content").style.display = isYouTube ? "block" : "none";
  document.getElementById("not-youtube").style.display  = isYouTube ? "none"  : "block";
}
 
// Toggle sidebar via background message
document.getElementById("toggle-btn")?.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "TOGGLE_SIDEBAR" });
  window.close(); // close popup after clicking
});
 
// Init
checkServer();
checkTab();