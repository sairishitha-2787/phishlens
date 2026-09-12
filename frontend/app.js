/* PhishLens detection UI (P track) — static frontend, no build step.
 * API contract: POST {API_BASE}/api/predict with body {"text": combined}
 * Response: {id,label,confidence,probabilities,explanation:{top_features},model_version,truncated,created_at}
 * Security: ALL dynamic content rendered via textContent/createElement. Never innerHTML for API/email content.
 */
(function () {
  "use strict";

  // Single centralized API configuration — backend URL is public, not a secret.
  var PHISHLENS_API_BASE = "https://phishlens-api-tlx8.onrender.com";
  var PREDICT_URL = PHISHLENS_API_BASE + "/api/predict";
  var HEALTH_URL = PHISHLENS_API_BASE + "/api/health";
  var REQUEST_TIMEOUT_MS = 90000; // Render free tier cold start ~30-60s

  var form = document.getElementById("analyzeForm");
  var subjectEl = document.getElementById("subject");
  var bodyEl = document.getElementById("body");
  var analyzeBtn = document.getElementById("analyzeBtn");
  var clearBtn = document.getElementById("clearBtn");
  var statusEl = document.getElementById("status");
  var resultEl = document.getElementById("result");
  var verdictBadge = document.getElementById("verdictBadge");
  var rawLabel = document.getElementById("rawLabel");
  var confText = document.getElementById("confText");
  var confBar = document.getElementById("confBar");
  var probsEl = document.getElementById("probs");
  var featuresEl = document.getElementById("features");
  var modelVer = document.getElementById("modelVer");
  var truncWarn = document.getElementById("truncWarn");
  var charCount = document.getElementById("charCount");
  var warmupEl = document.getElementById("warmup");
  var verdictIcon = document.getElementById("verdictIcon");
  var technicalLabel = document.getElementById("technicalLabel");
  var apiStatus = document.getElementById("apiStatus");
  var apiStatusText = document.getElementById("apiStatusText");
  var againBtn = document.getElementById("againBtn");
  var toastEl = document.getElementById("toast");
  var resultId = document.getElementById("resultId");

  var isAnalyzing = false;
  var currentController = null;
  var cancellationRequested = false;
  var loadingTimer = null;
  var loadingMessages = ["Scanning content...", "Checking phishing signals...", "Generating verdict..."];

  function setText(el, s) { el.textContent = (s === undefined || s === null) ? "" : String(s); }

  function clearChildren(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  function showError(msg) {
    clearChildren(statusEl);
    var d = document.createElement("div");
    d.className = "error";
    setText(d, msg);
    statusEl.appendChild(d);
  }

  function showToast(msg) {
    setText(toastEl, msg);
    toastEl.classList.add("visible");
    window.setTimeout(function () { toastEl.classList.remove("visible"); }, 2800);
  }

  function showLoading(msg) {
    clearChildren(statusEl);
    var d = document.createElement("div");
    d.className = "loading";
    setText(d, msg);
    statusEl.appendChild(d);
  }

  function showHint(msg) {
    clearChildren(statusEl);
    var d = document.createElement("div");
    d.className = "ok-hint";
    setText(d, msg);
    statusEl.appendChild(d);
  }

  function clearStatus() { clearChildren(statusEl); }

  function setAnalyzing(on) {
    isAnalyzing = on;
    analyzeBtn.disabled = on;
    analyzeBtn.classList.toggle("is-loading", on);
    setText(analyzeBtn.querySelector(".button-label"), on ? "Analyzing email..." : "Analyze Email");
    if (on) {
      var messageIndex = 0;
      showLoading(loadingMessages[messageIndex]);
      loadingTimer = window.setInterval(function () {
        messageIndex = (messageIndex + 1) % loadingMessages.length;
        showLoading(loadingMessages[messageIndex]);
      }, 1700);
    } else if (loadingTimer) {
      window.clearInterval(loadingTimer);
      loadingTimer = null;
    }
  }

  function verdictFor(label) {
    if (label === "legitimate") return { text: "Legitimate Email", cls: "legit", icon: "✓" };
    if (label === "human_phishing") return { text: "Phishing Detected", cls: "phish", icon: "!" };
    if (label === "ai_phishing") return { text: "AI-Generated Phishing", cls: "ai-phish", icon: "!" };
    return { text: "Unknown Result", cls: "unknown", icon: "?" };
  }

  function friendlyHttpError(status, detail) {
    if (status === 400) return "That email looks empty to the server. Please paste some email text and try again.";
    if (status === 422) return "The server could not understand that request (validation error). Please shorten or simplify the input and try again.";
    if (status === 500 || status === 502 || status === 503) return "The detection server had a problem. Please wait a moment and try again.";
    if (detail) return "Request failed. Please try again.";
    return "Request failed (HTTP " + status + "). Please try again.";
  }

  function renderResult(data) {
    var label = data && data.label;
    if (typeof label !== "string" || !label) {
      showError("Unexpected API response: missing prediction label. Please try again.");
      resultEl.hidden = true;
      return;
    }
    var v = verdictFor(label);
    verdictBadge.className = v.cls;
    setText(verdictBadge, v.text);
    verdictIcon.className = "verdict-icon " + v.cls;
    setText(verdictIcon, v.icon);
    setText(rawLabel, label);
    setText(technicalLabel, label);

    var conf = Number(data.confidence);
    if (!isFinite(conf) || conf < 0 || conf > 1) conf = NaN;
    setText(confText, isFinite(conf) ? Math.round(conf * 100) + "%" : "n/a");
    confBar.style.width = "0%";
    window.requestAnimationFrame(function () {
      confBar.style.width = isFinite(conf) ? Math.round(conf * 100) + "%" : "0%";
    });

    clearChildren(probsEl);
    var probs = data.probabilities;
    if (probs && typeof probs === "object") {
      Object.keys(probs).sort().forEach(function (k) {
        var row = document.createElement("div");
        row.className = "probability-row";
        var heading = document.createElement("div");
        heading.className = "probability-heading";
        var name = document.createElement("span");
        var names = { ai_phishing: "AI Phishing", human_phishing: "Human Phishing", legitimate: "Legitimate" };
        setText(name, names[k] || k);
        var value = document.createElement("strong");
        var numericValue = Number(probs[k]);
        setText(value, isFinite(numericValue) ? Math.round(numericValue * 100) + "%" : "n/a");
        heading.appendChild(name); heading.appendChild(value);
        var track = document.createElement("div"); track.className = "probability-track";
        var fill = document.createElement("div"); fill.className = "probability-fill";
        fill.style.width = "0%"; track.appendChild(fill); row.appendChild(heading); row.appendChild(track); probsEl.appendChild(row);
        window.requestAnimationFrame(function () { fill.style.width = isFinite(numericValue) ? Math.round(numericValue * 100) + "%" : "0%"; });
      });
    } else {
      var emptyProbs = document.createElement("p");
      emptyProbs.className = "empty-state";
      setText(emptyProbs, "Not returned by server.");
      probsEl.appendChild(emptyProbs);
    }

    clearChildren(featuresEl);
    var feats = data.explanation && data.explanation.top_features;
    if (Array.isArray(feats) && feats.length) {
      feats.forEach(function (f) {
        var li = document.createElement("span");
        li.className = "feature-pill";
        setText(li, f);
        featuresEl.appendChild(li);
      });
    } else {
      var emptyFeatures = document.createElement("p");
      emptyFeatures.className = "empty-state";
      setText(emptyFeatures, "No top features returned.");
      featuresEl.appendChild(emptyFeatures);
    }

    setText(modelVer, data.model_version || "unknown");
    setText(truncWarn, data.truncated ? "Truncated to 3000 characters" : "Complete");
    truncWarn.className = data.truncated ? "mono warning-text" : "mono";
    setText(resultId, data.id ? "ID " + String(data.id).slice(0, 12) : "Live analysis");
    resultEl.hidden = false;
  }

  function combinedText() {
    var s = subjectEl.value.trim();
    var b = bodyEl.value.trim();
    if (s && b) return "Subject: " + s + "\n\n" + b;
    return s || b || "";
  }

  function updateCount() {
    setText(charCount, String(combinedText().length));
  }
  subjectEl.addEventListener("input", updateCount);
  bodyEl.addEventListener("input", updateCount);
  updateCount();

  clearBtn.addEventListener("click", function () {
    cancellationRequested = true;
    if (isAnalyzing && currentController) currentController.abort();
    form.reset();
    updateCount();
    clearStatus();
    resultEl.hidden = true;
    setAnalyzing(false);
    showToast("Analysis cleared");
  });

  againBtn.addEventListener("click", function () {
    resultEl.hidden = true;
    subjectEl.focus();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Non-blocking warm-up so first Analyze is less likely to hit a cold start.
  try {
    fetch(HEALTH_URL, { method: "GET" }).then(function (r) {
      if (r.ok) {
        apiStatus.classList.add("online");
        setText(apiStatusText, "API Online");
        warmupEl.className = "warmup ready";
        setText(warmupEl, "Detection server is awake and ready.");
      } else {
        apiStatus.classList.add("offline");
        setText(apiStatusText, "API Warming");
        setText(warmupEl, "Detection server is warming up — first analysis may take up to a minute.");
      }
    }).catch(function () {
      apiStatus.classList.add("offline");
      setText(apiStatusText, "API Warming");
      setText(warmupEl, "Detection server is warming up — first analysis may take up to a minute.");
    });
  } catch (e) {
    apiStatus.classList.add("offline");
    setText(apiStatusText, "API Warming");
    setText(warmupEl, "Detection server is warming up — first analysis may take up to a minute.");
  }

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    if (isAnalyzing) return; // prevent duplicate submissions

    var text = combinedText();
    if (!text) {
      resultEl.hidden = true;
      showError("Please paste an email subject or body before clicking Analyze.");
      (subjectEl.value.trim() ? bodyEl : subjectEl).focus();
      return;
    }

    setAnalyzing(true);
    cancellationRequested = false;
    resultEl.hidden = true;
    currentController = ("AbortController" in window) ? new AbortController() : null;
    var timeoutId = setTimeout(function () {
      if (currentController) currentController.abort();
    }, REQUEST_TIMEOUT_MS);

    var fetchOpts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    };
    if (currentController) fetchOpts.signal = currentController.signal;

    fetch(PREDICT_URL, fetchOpts).then(function (resp) {
      clearTimeout(timeoutId);
      if (!resp.ok) {
        var msg = friendlyHttpError(resp.status, null);
        return resp.text().then(function () { throw new Error(msg + " [HTTP " + resp.status + "]"); });
      }
      return resp.json().catch(function () {
        throw new Error("Unexpected API response: invalid JSON. Please try again.");
      });
    }).then(function (data) {
      clearStatus();
      renderResult(data);
    }).catch(function (err) {
      clearTimeout(timeoutId);
      if (cancellationRequested) return;
      resultEl.hidden = true;
      var m = err && err.message ? err.message : "";
      if (err && err.name === "AbortError") {
        showError("Request timed out or was cancelled. The server may be waking up — please wait ~30 seconds and try again.");
      } else if (m.indexOf("[HTTP") !== -1) {
        showError(m.replace(/ \[HTTP.*$/, ""));
      } else if (m === "Failed to fetch" || m.indexOf("NetworkError") !== -1 || m.indexOf("Load failed") !== -1) {
        showError("Unable to connect to the detection server. Please check your connection and try again.");
      } else {
        showError(m || "Unable to connect to the detection server. Please try again.");
      }
    }).then(function () {
      setAnalyzing(false);
      currentController = null;
    });
  });
})();
