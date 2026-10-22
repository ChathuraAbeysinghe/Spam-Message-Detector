/**
 * SpamDetector Dashboard Controller.
 * Powers dynamic predictions, analysis inspection, SVG donut charts,
 * recent history, views navigation, and light/dark theme switching.
 */

// Default initial recent messages matching reference UI
const INITIAL_RECENT_MESSAGES = [
  {
    text: "Hey, are we still meeting tomorrow?",
    is_spam: false,
    confidence: 96,
    time: "2 min ago",
  },
  {
    text: "You've won a free iPhone! Click here now!!",
    is_spam: true,
    confidence: 99,
    time: "6 min ago",
  },
  {
    text: "Your order #12345 has been shipped.",
    is_spam: false,
    confidence: 98,
    time: "12 min ago",
  },
  {
    text: "Claim $5000 now! Limited time offer!",
    is_spam: true,
    confidence: 99,
    time: "18 min ago",
  },
  {
    text: "Thanks for the update.",
    is_spam: false,
    confidence: 95,
    time: "25 min ago",
  },
];

let recentMessages = [];
let totalCheckedCount = 1248;

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initNavigation();
  initRecentMessages();
  initInputListeners();
  fetchModelMetrics();

  // Mobile menu drawer
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("sidebar");
  if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener("click", () => {
      sidebar.classList.toggle("mobile-open");
    });

    document.addEventListener("click", (e) => {
      if (
        sidebar.classList.contains("mobile-open") &&
        !sidebar.contains(e.target) &&
        !mobileMenuBtn.contains(e.target)
      ) {
        sidebar.classList.remove("mobile-open");
      }
    });
  }
});

/* -------------------------------------------------------------
   THEME TOGGLING (LIGHT / DARK)
------------------------------------------------------------- */
function initTheme() {
  const savedTheme = localStorage.getItem("spam_theme") || "light";
  document.body.setAttribute("data-theme", savedTheme);

  const toggleBtn = document.getElementById("themeToggleBtn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = document.body.getAttribute("data-theme") || "light";
      const next = current === "light" ? "dark" : "light";
      document.body.setAttribute("data-theme", next);
      localStorage.setItem("spam_theme", next);
    });
  }
}

/* -------------------------------------------------------------
   SIDEBAR NAVIGATION
------------------------------------------------------------- */
function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-item");
  const viewHome = document.getElementById("viewHome");
  const viewAbout = document.getElementById("viewAbout");

  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      navButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const view = btn.getAttribute("data-view");
      if (view === "about") {
        viewHome.style.display = "none";
        viewAbout.style.display = "block";
      } else if (view === "history") {
        viewHome.style.display = "block";
        viewAbout.style.display = "none";
        document.getElementById("recentList").scrollIntoView({ behavior: "smooth" });
      } else {
        // 'home' or 'analyze'
        viewHome.style.display = "block";
        viewAbout.style.display = "none";
        document.getElementById("messageInput").focus();
      }

      // Close mobile drawer if open
      const sidebar = document.getElementById("sidebar");
      if (sidebar) sidebar.classList.remove("mobile-open");
    });
  });
}

/* -------------------------------------------------------------
   INPUT LISTENERS & COUNTERS
------------------------------------------------------------- */
function initInputListeners() {
  const textarea = document.getElementById("messageInput");
  const counter = document.getElementById("charCounter");

  if (textarea && counter) {
    textarea.addEventListener("input", () => {
      counter.textContent = `${textarea.value.length}/500`;
      hideError();
    });
  }
}

function handleClear() {
  const textarea = document.getElementById("messageInput");
  textarea.value = "";
  document.getElementById("charCounter").textContent = "0/500";
  hideError();
  textarea.focus();
}

function showError(msg) {
  const banner = document.getElementById("errorBanner");
  banner.textContent = msg;
  banner.style.display = "block";
}

function hideError() {
  const banner = document.getElementById("errorBanner");
  banner.style.display = "none";
}

/* -------------------------------------------------------------
   CLASSIFICATION / INFERENCE LOGIC
------------------------------------------------------------- */
async function handlePredict() {
  const textarea = document.getElementById("messageInput");
  const message = textarea.value.trim();
  const predictBtn = document.getElementById("predictBtn");
  const predictText = document.getElementById("predictText");

  hideError();

  if (!message) {
    showError("Please enter a text message to classify.");
    textarea.focus();
    return;
  }

  // Set loading state
  predictBtn.disabled = true;
  predictText.textContent = "Analyzing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction request failed.");
    }

    renderPredictionResult(data);
    addRecentMessage(data);
    totalCheckedCount++;
    updateStatsDisplay();
  } catch (err) {
    showError(err.message || "Failed to contact prediction server.");
  } finally {
    predictBtn.disabled = false;
    predictText.textContent = "Check for Spam";
  }
}

function renderPredictionResult(data) {
  const panel = document.getElementById("resultPanel");
  const banner = document.getElementById("resultBanner");
  const iconCircle = document.getElementById("resultIconCircle");
  const heading = document.getElementById("resultHeading");
  const subtext = document.getElementById("resultSubtext");
  const pill = document.getElementById("confidencePill");

  const valKeywords = document.getElementById("valKeywords");
  const valLinks = document.getElementById("valLinks");
  const valSender = document.getElementById("valSender");
  const valPattern = document.getElementById("valPattern");

  const isSpam = data.is_spam;
  const confidence = Math.round(data.confidence_score);

  // Update Result Banner classes & styles
  if (isSpam) {
    banner.className = "result-banner spam";
    iconCircle.innerHTML = `
      <svg class="result-check-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
    `;
    heading.innerHTML = `This message is <span class="res-highlight">SPAM</span>`;
    subtext.textContent = data.explanation || "Warning: High-risk spam patterns detected.";
    pill.textContent = `Confidence: ${confidence}%`;
  } else {
    banner.className = "result-banner not-spam";
    iconCircle.innerHTML = `
      <svg class="result-check-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
    `;
    heading.innerHTML = `This message is <span class="res-highlight">NOT spam</span>`;
    subtext.textContent = data.explanation || "The content appears to be a legitimate message.";
    pill.textContent = `Confidence: ${confidence}%`;
  }

  // Update Analysis Details List
  const details = data.analysis_details || {};
  valKeywords.textContent = details.suspicious_keywords || "None found";
  valLinks.textContent = details.links_detected || "None found";
  valSender.textContent = details.sender_analysis || (isSpam ? "Suspicious tone" : "Looks legitimate");
  valPattern.textContent = details.message_pattern || (isSpam ? "Promotional scam pattern" : "Normal conversation pattern");

  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* -------------------------------------------------------------
   RECENT MESSAGES COMPONENT
------------------------------------------------------------- */
function initRecentMessages() {
  const saved = localStorage.getItem("spam_recents");
  if (saved) {
    try {
      recentMessages = JSON.parse(saved);
    } catch {
      recentMessages = [...INITIAL_RECENT_MESSAGES];
    }
  } else {
    recentMessages = [...INITIAL_RECENT_MESSAGES];
  }

  renderRecentList();
  updateStatsDisplay();
}

function renderRecentList() {
  const listContainer = document.getElementById("recentList");
  if (!listContainer) return;

  listContainer.innerHTML = "";

  recentMessages.slice(0, 6).forEach((item) => {
    const el = document.createElement("div");
    el.className = "recent-item";
    el.title = "Click to load and re-test this message";

    const badgeClass = item.is_spam ? "spam" : "ham";
    const badgeLabel = item.is_spam ? "Spam" : "Not Spam";

    el.innerHTML = `
      <div class="recent-item-left">
        <div class="recent-badge-row">
          <span class="recent-pill ${badgeClass}">
            <span class="legend-dot ${item.is_spam ? "dot-red" : "dot-green"}"></span>
            ${badgeLabel}
          </span>
        </div>
        <div class="recent-snippet">${escapeHtml(item.text)}</div>
      </div>
      <div class="recent-item-right">
        <span>${item.time}</span>
        <svg class="recent-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <polyline points="9 18 15 12 9 6"/>
        </svg>
      </div>
    `;

    el.addEventListener("click", () => {
      const textarea = document.getElementById("messageInput");
      textarea.value = item.text;
      document.getElementById("charCounter").textContent = `${item.text.length}/500`;
      handlePredict();
    });

    listContainer.appendChild(el);
  });
}

function addRecentMessage(data) {
  const newItem = {
    text: data.message,
    is_spam: data.is_spam,
    confidence: Math.round(data.confidence_score),
    time: "Just now",
  };

  recentMessages.unshift(newItem);
  if (recentMessages.length > 20) {
    recentMessages.pop();
  }

  localStorage.setItem("spam_recents", JSON.stringify(recentMessages));
  renderRecentList();
}

function handleClearHistory() {
  recentMessages = [];
  localStorage.removeItem("spam_recents");
  renderRecentList();
}

/* -------------------------------------------------------------
   QUICK STATS DONUT METER CALCULATION
------------------------------------------------------------- */
function updateStatsDisplay() {
  let spamCount = recentMessages.filter((m) => m.is_spam).length;
  let totalRecents = recentMessages.length;

  let hamPct = 87;
  let spamPct = 13;

  if (totalRecents > 0) {
    spamPct = Math.round((spamCount / totalRecents) * 100);
    hamPct = 100 - spamPct;
  }

  document.getElementById("centerStatPct").textContent = `${hamPct}%`;
  document.getElementById("centerStatLabel").textContent = "Not Spam";
  document.getElementById("statHamPct").textContent = `${hamPct}%`;
  document.getElementById("statSpamPct").textContent = `${spamPct}%`;
  document.getElementById("statTotalChecked").textContent = totalCheckedCount.toLocaleString();

  // SVG Circumference for r=38 is 238.76
  // Offset formula: 238.76 * (spamPct / 100)
  const ring = document.querySelector(".donut-ham-ring");
  if (ring) {
    const offset = 238.76 * (spamPct / 100);
    ring.style.strokeDashoffset = offset.toFixed(2);
  }
}

/* -------------------------------------------------------------
   FETCH MODEL METRICS FROM BACKEND
------------------------------------------------------------- */
async function fetchModelMetrics() {
  try {
    const res = await fetch("/metrics");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.performance_metrics) return;

    const p = data.performance_metrics;
    const cm = data.confusion_matrix || {};

    const elAcc = document.getElementById("aboutAccuracy");
    if (elAcc) elAcc.textContent = `${(p.accuracy * 100).toFixed(2)}%`;

    const elPrec = document.getElementById("aboutPrecision");
    if (elPrec) elPrec.textContent = `${(p.precision * 100).toFixed(2)}%`;

    const elRec = document.getElementById("aboutRecall");
    if (elRec) elRec.textContent = `${(p.recall * 100).toFixed(2)}%`;

    const elF1 = document.getElementById("aboutF1");
    if (elF1) elF1.textContent = `${(p.f1_score * 100).toFixed(2)}%`;

    if (cm.true_negatives !== undefined) {
      document.getElementById("aboutTN").textContent = cm.true_negatives;
      document.getElementById("aboutFP").textContent = cm.false_positives;
      document.getElementById("aboutFN").textContent = cm.false_negatives;
      document.getElementById("aboutTP").textContent = cm.true_positives;
    }
  } catch (e) {
    console.warn("Metrics fetch optional fallback:", e);
  }
}

function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
}
