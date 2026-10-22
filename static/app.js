/**
 * Frontend JavaScript Controller for AI Spam Message Detector.
 * Handles DOM interactions, input validation, live API calls, and metrics fetching.
 */

document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("messageInput");
  const charCounter = document.getElementById("charCounter");
  const sampleChips = document.querySelectorAll(".chip");

  // Character and word count updater
  const updateCounts = () => {
    const text = textarea.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    charCounter.textContent = `${chars} / 5000 chars (${words} words)`;
  };

  textarea.addEventListener("input", updateCounts);

  // Sample prompt chips quick-filler
  sampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const msg = chip.getAttribute("data-msg");
      if (msg) {
        textarea.value = msg;
        updateCounts();
        hideError();
        handlePredict();
      }
    });
  });

  // Load actual evaluation metrics on startup
  fetchMetrics();
});

function showError(message) {
  const alertBox = document.getElementById("errorAlert");
  alertBox.textContent = message;
  alertBox.style.display = "block";
}

function hideError() {
  const alertBox = document.getElementById("errorAlert");
  alertBox.style.display = "none";
  alertBox.textContent = "";
}

function handleClear() {
  const textarea = document.getElementById("messageInput");
  textarea.value = "";
  document.getElementById("charCounter").textContent = "0 / 5000 chars (0 words)";
  document.getElementById("resultCard").style.display = "none";
  hideError();
  textarea.focus();
}

async function handlePredict() {
  const textarea = document.getElementById("messageInput");
  const message = textarea.value.trim();
  const predictBtn = document.getElementById("predictBtn");
  const btnText = document.getElementById("btnText");
  const btnIcon = document.getElementById("btnIcon");
  const resultCard = document.getElementById("resultCard");

  hideError();

  // Client-side validation
  if (!message) {
    showError("Please enter or paste an SMS message to classify.");
    textarea.focus();
    return;
  }

  if (message.length > 5000) {
    showError("Message exceeds 5,000 characters limit. Please shorten your input.");
    return;
  }

  // Set loading state
  predictBtn.disabled = true;
  btnIcon.textContent = "⏳";
  btnText.textContent = "Analyzing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: message }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Server error while classifying message.");
    }

    renderResult(data);
  } catch (err) {
    showError(err.message || "Unable to reach prediction service. Ensure the server is running.");
  } finally {
    predictBtn.disabled = false;
    btnIcon.textContent = "🔍";
    btnText.textContent = "Classify Message";
  }
}

function renderResult(data) {
  const card = document.getElementById("resultCard");
  const badge = document.getElementById("resultBadge");
  const badgeIcon = document.getElementById("badgeIcon");
  const badgeLabel = document.getElementById("badgeLabel");
  const confidence = document.getElementById("confidenceScore");
  const spamProbPercent = document.getElementById("spamProbPercent");
  const progressBarFill = document.getElementById("progressBarFill");
  const explanation = document.getElementById("resultExplanation");
  const disclaimer = document.getElementById("resultDisclaimer");

  const isSpam = data.is_spam;
  const spamPercentage = (data.spam_probability * 100).toFixed(1);

  // Toggle styling classes
  card.className = `result-card ${isSpam ? "spam" : "ham"}`;
  badge.className = `result-badge ${isSpam ? "spam" : "ham"}`;
  progressBarFill.className = `progress-bar-fill ${isSpam ? "spam" : "ham"}`;

  // Content population
  badgeIcon.textContent = isSpam ? "🚨" : "✅";
  badgeLabel.textContent = isSpam ? "Spam Message" : "Not Spam (Ham)";
  confidence.textContent = `Confidence: ${data.confidence_score}%`;
  spamProbPercent.textContent = `${spamPercentage}%`;
  progressBarFill.style.width = `${spamPercentage}%`;

  explanation.textContent = data.explanation;
  disclaimer.textContent = data.disclaimer;

  card.style.display = "block";
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function fetchMetrics() {
  try {
    const response = await fetch("/metrics");
    if (!response.ok) return;

    const data = await response.json();
    if (!data.performance_metrics) return;

    const p = data.performance_metrics;
    document.getElementById("valAccuracy").textContent = `${(p.accuracy * 100).toFixed(1)}%`;
    document.getElementById("valPrecision").textContent = `${(p.precision * 100).toFixed(1)}%`;
    document.getElementById("valRecall").textContent = `${(p.recall * 100).toFixed(1)}%`;
    document.getElementById("valF1").textContent = `${(p.f1_score * 100).toFixed(1)}%`;

    if (data.confusion_matrix) {
      const cm = data.confusion_matrix;
      document.getElementById("cmTN").textContent = cm.true_negatives;
      document.getElementById("cmFP").textContent = cm.false_positives;
      document.getElementById("cmFN").textContent = cm.false_negatives;
      document.getElementById("cmTP").textContent = cm.true_positives;
    }
  } catch (e) {
    console.warn("Metrics could not be pre-loaded:", e);
  }
}
