/**
 * api.js
 * Every call to the VidQuery backend.
 *
 * A classic content script, not an ES module. Manifest V3 cannot declare a
 * content script as type="module", and a dynamic import() from the isolated
 * world is unreliable against a strict page CSP — which YouTube has. When
 * that import fails it throws before anything renders, so the panel never
 * appears and nothing obvious explains why. Loading this file first and
 * hanging the exports off one namespace avoids the whole problem.
 *
 * /process no longer blocks: it returns a job id, and processVideo() polls
 * /status until the job finishes. An ASR or visual run takes minutes, so a
 * request that waited for the result would hit the browser's fetch timeout
 * long before the work was done.
 */

(() => {
  "use strict";

  const API_BASE = "http://127.0.0.1:8000";
  const POLL_INTERVAL_MS = 900;
  const POLL_TIMEOUT_MS = 30 * 60 * 1000;   // visual pipelines are genuinely slow

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  /**
   * Calls go through the service worker, not fetch() from this page.
   *
   * A content-script fetch runs in the page's context and is therefore
   * subject to the page's CORS, to Private/Local Network Access rules for
   * reaching loopback from a public origin, and to mixed-content rules for
   * http:// from https://. Any of those failing gives an opaque network error
   * with no status to diagnose. The worker fetches with extension privileges,
   * where host_permissions already cover the backend and none of that applies.
   */
  async function request(path, { method = "GET", body, signal } = {}) {
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");

    let reply;
    try {
      reply = await chrome.runtime.sendMessage({
        type: "VQ_FETCH",
        url: `${API_BASE}${path}`,
        method,
        body: body ? JSON.stringify(body) : null,
      });
    } catch {
      // The worker was not reachable at all — usually the extension was
      // reloaded while this page stayed open.
      throw new ApiError(
        "VidQuery lost its connection to the extension. Refresh the page.",
        0
      );
    }

    // The request outlived what asked for it (the video changed).
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");

    if (!reply) {
      throw new ApiError("No response from the extension background worker.", 0);
    }

    if (reply.status === 0) {
      throw new ApiError(
        "Can't reach the VidQuery server. Start it with: uvicorn main:app --port 8000",
        0
      );
    }

    if (!reply.ok) {
      let detail = `Request failed (${reply.status})`;
      try {
        const data = JSON.parse(reply.body);
        if (data && data.detail) detail = data.detail;
      } catch {
        /* response had no JSON body */
      }
      throw new ApiError(detail, reply.status);
    }

    try {
      return JSON.parse(reply.body);
    } catch {
      throw new ApiError("Backend returned a malformed response.", reply.status);
    }
  }

  const api = {
    health: () => request("/health"),

    detect: (video) => request("/detect", { method: "POST", body: { video } }),

    /**
     * Submits the video and polls until it is indexed.
     * @param {function} onStage called with ({stage, progress, pipeline})
     */
    async processVideo(video, onStage, { signal, force = false } = {}) {
      const job = await request("/process", {
        method: "POST",
        body: { video, force },
        signal,
      });

      if (job.cached || job.state === "done") {
        onStage?.({ stage: job.stage, progress: 1, pipeline: job.pipeline });
        return job;
      }

      onStage?.({ stage: job.stage, progress: job.progress });

      const deadline = Date.now() + POLL_TIMEOUT_MS;
      let jobId = job.job_id;

      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        if (signal?.aborted) throw new DOMException("aborted", "AbortError");

        let status;
        try {
          status = await request(`/status/${jobId}`, { signal });
        } catch (err) {
          if (err.status !== 404) throw err;

          // The job vanished mid-poll. Jobs are held in memory, so any server
          // restart drops them — which `--reload` does on every file save —
          // and they are swept an hour after finishing. None of that means
          // the work failed: chunks and vectors are on disk. Ask again. If
          // the video is indexed the work completed, otherwise we get a new
          // job to follow.
          const retry = await request("/process", {
            method: "POST",
            body: { video, force: false },
            signal,
          });

          if (retry.cached || retry.state === "done") {
            onStage?.({ stage: retry.stage, progress: 1, pipeline: retry.pipeline });
            return retry;
          }

          jobId = retry.job_id;
          continue;
        }

        onStage?.({
          stage: status.stage,
          progress: status.progress,
          pipeline: status.pipeline,
        });

        if (status.state === "done") return status;
        if (status.state === "error") {
          throw new ApiError(status.error || "Processing failed", 500);
        }
      }

      throw new ApiError("Processing timed out", 504);
    },

    ask: (videoId, question, chatHistory, signal) =>
      request("/ask", {
        method: "POST",
        body: { video_id: videoId, question, chat_history: chatHistory },
        signal,
      }),

    summarize: (videoId, signal) =>
      request("/summarize", { method: "POST", body: { video_id: videoId }, signal }),

    /**
     * A chip click is the implicit reward signal behind the retrieval boost.
     * Fire-and-forget: feedback must never interrupt playback.
     */
    feedback: (videoId, chunkId, question) =>
      request("/feedback", {
        method: "POST",
        body: { video_id: videoId, chunk_id: chunkId, clicked: true, question },
      }).catch(() => {}),
  };

  // The isolated world's window, shared with content.js.
  window.__VidQuery = { api, ApiError, API_BASE };
})();
