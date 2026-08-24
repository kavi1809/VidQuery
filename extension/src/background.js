/**
 * background.js
 * Service worker for the VidQuery Chrome extension.
 * Handles messages between popup and content scripts.
 */

chrome.runtime.onInstalled.addListener(() => {
  console.log("VidQuery extension installed.");
});

// Listen for messages from popup or content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "TOGGLE_SIDEBAR") {
    // Tell the content script to show/hide the sidebar
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "TOGGLE_SIDEBAR" });
      }
    });
  }
  return true;
});
