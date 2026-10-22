/**
 * SpamDetector Dashboard Controller (v1.2.0)
 * Fully connected to SQLite backend APIs for real persistence,
 * batch processing, interactive CRUD history, live SVG donut charts,
 * theme switching, and mobile drawer navigation.
 */

// Sample test messages for quick testing
const SAMPLES = {
  urgent_prize: "URGENT! You have won a £1,000 Walmart gift card! Call 09061701461 to claim your reward immediately.",
  bank_alert: "URGENT ALERT: Your bank account #4928 has been suspended due to unauthorized login. Click http://verify-secure-bank.com to restore access.",
  legit_chat: "Hey! Are you still coming to the library for the study group this evening? Let me know.",
  otp_code: "Your security verification code is 849201. It is valid for 10 minutes. Please do not share this code with anyone.",
  delivery: "Your delivery package #GB92841 is scheduled to arrive tomorrow between 1:00 PM and 3:00 PM.",
};

// Global State
let currentView = "home";
let currentLastPredictionId = null;
let historyState = {
  page: 1,
  pageSize: 10,
  total: 0,
  search: "",
  filter: "all",
  sort: "newest",
  records: [],
};
let searchDebounceTimer = null;

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initNavigation();
  initInputListeners();
  initSampleChips();
  initUserPillDropdown();
  initBatchUpload();
  initMobileDrawer();
  
  // Load real data from SQLite backend
  refreshStatistics();
  loadRecentScans();
  loadHistoryTable();
  fetchModelMetrics();
  checkSystemHealth();
});

/* =============================================================
   1. THEME SWITCHER (LIGHT / DARK)
   ============================================================= */
function initTheme() {
  const savedTheme = localStorage.getItem("spam_detector_theme") || "light";
  document.body.setAttribute("data-theme", savedTheme);

  const toggleBtn = document.getElementById("themeToggleBtn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = document.body.getAttribute("data-theme") || "light";
      const next = current === "light" ? "dark" : "light";
      document.body.setAttribute("data-theme", next);
      localStorage.setItem("spam_detector_theme", next);
      showToast(`Switched to ${next} theme`, "info");
    });
  }
}

/* =============================================================
   2. NAVIGATION & VIEW SWITCHING
   ============================================================= */
function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-item");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.getAttribute("data-view");
      switchView(view);
    });
  });
}

function switchView(viewName) {
  currentView = viewName;

  // Update navigation active states
  document.querySelectorAll(".nav-item").forEach((btn) => {
    if (btn.getAttribute("data-view") === viewName) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Switch views
  const views = {
    home: document.getElementById("viewHome"),
    analyze: document.getElementById("viewAnalyze"),
    history: document.getElementById("viewHistory"),
    about: document.getElementById("viewAbout"),
  };

  Object.keys(views).forEach((k) => {
    if (views[k]) {
      views[k].style.display = k === viewName ? "block" : "none";
    }
  });

  // If switched to history, reload table
  if (viewName === "history") {
    loadHistoryTable();
  }

  // Close mobile drawer if open
  closeMobileDrawer();

  // Scroll to top
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* =============================================================
   3. MOBILE DRAWER & USER PILL DROPDOWN
   ============================================================= */
function initMobileDrawer() {
  const menuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebarBackdrop");

  if (menuBtn && sidebar && backdrop) {
    menuBtn.addEventListener("click", () => {
      sidebar.classList.toggle("mobile-open");
      backdrop.classList.toggle("active");
    });

    backdrop.addEventListener("click", closeMobileDrawer);
  }
}

function closeMobileDrawer() {
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebarBackdrop");
  if (sidebar) sidebar.classList.remove("mobile-open");
  if (backdrop) backdrop.classList.remove("active");
}

function initUserPillDropdown() {
  const userPillBtn = document.getElementById("userPillBtn");
  const userDropdown = document.getElementById("userDropdownMenu");

  if (userPillBtn && userDropdown) {
    userPillBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      userDropdown.classList.toggle("active");
      const isExpanded = userDropdown.classList.contains("active");
      userPillBtn.setAttribute("aria-expanded", isExpanded);
    });

    document.addEventListener("click", (e) => {
      if (!userDropdown.contains(e.target) && !userPillBtn.contains(e.target)) {
        userDropdown.classList.remove("active");
        userPillBtn.setAttribute("aria-expanded", "false");
      }
    });
  }
}

/* =============================================================
   4. INPUT LISTENERS, SAMPLES, & CHAR COUNTER
   ============================================================= */
function initInputListeners() {
  const textarea = document.getElementById("messageInput");
  const counter = document.getElementById("charCounter");

  if (textarea && counter) {
    textarea.addEventListener("input", () => {
      const len = textarea.value.length;
      const words = textarea.value.trim() ? textarea.value.trim().split(/\s+/).length : 0;
      counter.textContent = `${len}/5000 chars • ${words} words`;
      hideError();
    });
  }
}

function initSampleChips() {
  const chips = document.querySelectorAll(".sample-chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const sampleKey = chip.getAttribute("data-sample");
      const text = SAMPLES[sampleKey];
      if (text) {
        const textarea = document.getElementById("messageInput");
        textarea.value = text;
        const len = text.length;
        const words = text.trim().split(/\s+/).length;
        document.getElementById("charCounter").textContent = `${len}/5000 chars • ${words} words`;
        hideError();
        handlePredict();
      }
    });
  });
}

function handleClear() {
  const textarea = document.getElementById("messageInput");
  textarea.value = "";
  document.getElementById("charCounter").textContent = "0/5000 chars • 0 words";
  hideError();
  const panel = document.getElementById("resultPanel");
  if (panel) panel.style.display = "none";
  textarea.focus();
  showToast("Input cleared", "info");
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

/* =============================================================
   5. SINGLE PREDICTION LOGIC (HOME)
   ============================================================= */
async function handlePredict() {
  const textarea = document.getElementById("messageInput");
  const message = textarea.value.trim();
  const predictBtn = document.getElementById("predictBtn");
  const predictText = document.getElementById("predictText");

  hideError();

  if (!message) {
    showError("Please enter or paste an SMS message to classify.");
    textarea.focus();
    return;
  }

  // Loading state
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

    currentLastPredictionId = data.id;
    renderPredictionResult(data);
    refreshStatistics();
    loadRecentScans();
    showToast(`Classification complete: ${data.prediction.toUpperCase()}`, data.is_spam ? "error" : "success");
  } catch (err) {
    showError(err.message || "Failed to contact prediction server.");
    showToast(err.message || "Server error during classification", "error");
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
  const probBarFill = document.getElementById("probBarFill");
  const probBarText = document.getElementById("probBarText");
  const resultTimestamp = document.getElementById("resultTimestamp");

  const valKeywords = document.getElementById("valKeywords");
  const valLinks = document.getElementById("valLinks");
  const valSender = document.getElementById("valSender");
  const valPattern = document.getElementById("valPattern");

  const isSpam = data.is_spam;
  const confidence = Math.round(data.confidence_score);
  const spamProbPct = (data.spam_probability * 100).toFixed(1);

  // Update banner classes & styling
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

  // Update probability bar
  probBarText.textContent = `${spamProbPct}%`;
  probBarFill.style.width = `${spamProbPct}%`;

  // Update Analysis Details List
  const details = data.analysis_details || {};
  valKeywords.textContent = details.suspicious_keywords || "None found";
  valLinks.textContent = details.links_detected || "None found";
  valSender.textContent = details.sender_analysis || (isSpam ? "Suspicious tone" : "Looks legitimate");
  valPattern.textContent = details.message_pattern || (isSpam ? "Promotional scam pattern" : "Normal conversation pattern");
  resultTimestamp.textContent = "Just now";

  // Setup Feedback buttons
  const btnCorrect = document.getElementById("btnFeedbackCorrect");
  const btnWrong = document.getElementById("btnFeedbackWrong");

  if (btnCorrect && btnWrong) {
    btnCorrect.onclick = () => submitFeedback(data.id, "correct");
    btnWrong.onclick = () => submitFeedback(data.id, "misclassified");
  }

  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function submitFeedback(recordId, feedbackType) {
  if (!recordId) return;
  try {
    const res = await fetch(`/api/history/${recordId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feedback: feedbackType }),
    });
    if (res.ok) {
      showToast(feedbackType === "correct" ? "Thanks! Verified as accurate." : "Feedback recorded. Model logged for review.", "success");
    }
  } catch (e) {
    console.error("Feedback error:", e);
  }
}

/* =============================================================
   6. REAL-TIME SQLITE STATISTICS & DONUT METER
   ============================================================= */
async function refreshStatistics() {
  try {
    const res = await fetch("/api/history/stats");
    if (!res.ok) return;
    const stats = await res.json();

    const total = stats.total_checked || 0;
    const spamPct = stats.spam_percentage || 0;
    const hamPct = stats.ham_percentage || 0;

    // Update center label
    document.getElementById("centerStatPct").textContent = total > 0 ? `${hamPct}%` : "0%";
    document.getElementById("centerStatLabel").textContent = total > 0 ? "Clean" : "No Data";

    // Update legend
    document.getElementById("statHamPct").textContent = `${hamPct}%`;
    document.getElementById("statSpamPct").textContent = `${spamPct}%`;
    document.getElementById("statTotalChecked").textContent = total.toLocaleString();

    // Update sidebar nav badge
    const navBadge = document.getElementById("navHistoryCount");
    if (navBadge) navBadge.textContent = total.toLocaleString();

    // Update History View metrics
    const histTotal = document.getElementById("histTotalChecked");
    if (histTotal) {
      histTotal.textContent = total.toLocaleString();
      document.getElementById("histTotalSpam").textContent = (stats.spam_count || 0).toLocaleString();
      document.getElementById("histTotalHam").textContent = (stats.ham_count || 0).toLocaleString();
      document.getElementById("histSpamRatio").textContent = `${spamPct}%`;
    }

    // Update SVG Donut Circumference
    // Circle r=38, Circumference = 2 * PI * 38 ≈ 238.76
    const ring = document.querySelector(".donut-ham-ring");
    if (ring) {
      if (total === 0) {
        ring.style.strokeDashoffset = "238.76"; // Empty
      } else {
        const offset = 238.76 * (spamPct / 100);
        ring.style.strokeDashoffset = offset.toFixed(2);
      }
    }
  } catch (err) {
    console.warn("Failed to fetch SQLite stats:", err);
  }
}

/* =============================================================
   7. RECENT SCANS LIST (HOME)
   ============================================================= */
async function loadRecentScans() {
  const container = document.getElementById("recentList");
  if (!container) return;

  try {
    const res = await fetch("/api/history?limit=6&offset=0");
    if (!res.ok) return;
    const data = await res.json();
    const records = data.records || [];

    container.innerHTML = "";

    if (records.length === 0) {
      container.innerHTML = `
        <div style="text-align: center; padding: 1.5rem 1rem; color: var(--text-muted); font-size: 0.85rem;">
          No messages scanned yet.<br>
          <span style="font-size: 0.75rem; color: var(--text-light);">Try testing a sample above!</span>
        </div>
      `;
      return;
    }

    records.forEach((item) => {
      const el = document.createElement("div");
      el.className = "recent-item";
      el.title = "Click to load and re-classify this message";

      const badgeClass = item.is_spam ? "spam" : "ham";
      const badgeLabel = item.is_spam ? "Spam" : "Not Spam";

      el.innerHTML = `
        <div class="recent-item-left">
          <div class="recent-badge-row">
            <span class="recent-pill ${badgeClass}">
              <span class="legend-dot ${item.is_spam ? "dot-red" : "dot-green"}"></span>
              ${badgeLabel} (${Math.round(item.confidence_score)}%)
            </span>
          </div>
          <div class="recent-snippet">${escapeHtml(item.message)}</div>
        </div>
        <div class="recent-item-right">
          <span>${formatDateSnippet(item.created_at)}</span>
          <svg class="recent-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="9 18 15 12 9 6"/>
          </svg>
        </div>
      `;

      el.addEventListener("click", () => {
        const textarea = document.getElementById("messageInput");
        textarea.value = item.message;
        const len = item.message.length;
        const words = item.message.trim().split(/\s+/).length;
        document.getElementById("charCounter").textContent = `${len}/5000 chars • ${words} words`;
        handlePredict();
      });

      container.appendChild(el);
    });
  } catch (err) {
    console.error("Failed to load recent scans:", err);
  }
}

/* =============================================================
   8. HISTORY MANAGEMENT (CRUD, FILTERING, PAGINATION)
   ============================================================= */
function initHistoryControls() {
  const searchInput = document.getElementById("historySearchInput");
  const filterBtns = document.querySelectorAll(".filter-btn");
  const sortSelect = document.getElementById("historySortSelect");

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      clearTimeout(searchDebounceTimer);
      searchDebounceTimer = setTimeout(() => {
        historyState.search = e.target.value;
        historyState.page = 1;
        loadHistoryTable();
      }, 300);
    });
  }

  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      filterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      historyState.filter = btn.getAttribute("data-filter");
      historyState.page = 1;
      loadHistoryTable();
    });
  });

  if (sortSelect) {
    sortSelect.addEventListener("change", (e) => {
      historyState.sort = e.target.value;
      historyState.page = 1;
      loadHistoryTable();
    });
  }
}

async function loadHistoryTable() {
  const tbody = document.getElementById("historyTableBody");
  const emptyState = document.getElementById("historyEmptyState");
  const paginationBar = document.getElementById("historyPagination");

  if (!tbody) return;

  const offset = (historyState.page - 1) * historyState.pageSize;
  const url = `/api/history?limit=${historyState.pageSize}&offset=${offset}&search=${encodeURIComponent(historyState.search)}&filter_type=${historyState.filter}&sort=${historyState.sort}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to fetch history");
    const data = await res.json();

    historyState.total = data.total;
    historyState.records = data.records;

    tbody.innerHTML = "";

    if (data.total === 0) {
      tbody.parentElement.style.display = "none";
      emptyState.style.display = "block";
      paginationBar.style.display = "none";
      return;
    }

    tbody.parentElement.style.display = "table";
    emptyState.style.display = "none";
    paginationBar.style.display = "flex";

    data.records.forEach((row) => {
      const tr = document.createElement("tr");

      const badgeClass = row.is_spam ? "spam" : "ham";
      const badgeText = row.is_spam ? "Spam" : "Not Spam";
      const keywords = row.analysis_details ? row.analysis_details.suspicious_keywords : "None";

      tr.innerHTML = `
        <td style="color: var(--text-muted); font-weight: 600;">#${row.id}</td>
        <td>
          <span class="recent-pill ${badgeClass}">
            <span class="legend-dot ${row.is_spam ? "dot-red" : "dot-green"}"></span>
            ${badgeText}
          </span>
        </td>
        <td><strong>${Math.round(row.confidence_score)}%</strong></td>
        <td>
          <div class="table-snippet" title="${escapeHtml(row.message)}">${escapeHtml(row.message)}</div>
        </td>
        <td><span style="font-size: 0.78rem; color: var(--text-muted);">${escapeHtml(keywords)}</span></td>
        <td><span style="font-size: 0.78rem; color: var(--text-muted);">${formatDate(row.created_at)}</span></td>
        <td>
          <div class="table-actions">
            <button type="button" class="action-icon-btn" title="Re-test message" onclick="retestMessage('${escapeAttr(row.message)}');">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            </button>
            <button type="button" class="action-icon-btn delete-btn" title="Delete record" onclick="deleteHistoryItem(${row.id});">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });

    updatePaginationUI();
  } catch (err) {
    console.error("Error loading history table:", err);
    showToast("Error loading history data", "error");
  }
}

function updatePaginationUI() {
  const total = historyState.total;
  const start = total === 0 ? 0 : (historyState.page - 1) * historyState.pageSize + 1;
  const end = Math.min(historyState.page * historyState.pageSize, total);
  const totalPages = Math.ceil(total / historyState.pageSize) || 1;

  document.getElementById("paginationInfo").textContent = `Showing ${start}-${end} of ${total} records`;
  document.getElementById("pageNumberLabel").textContent = `Page ${historyState.page} of ${totalPages}`;

  document.getElementById("btnPrevPage").disabled = historyState.page <= 1;
  document.getElementById("btnNextPage").disabled = historyState.page >= totalPages;
}

function handlePrevPage() {
  if (historyState.page > 1) {
    historyState.page--;
    loadHistoryTable();
  }
}

function handleNextPage() {
  const totalPages = Math.ceil(historyState.total / historyState.pageSize);
  if (historyState.page < totalPages) {
    historyState.page++;
    loadHistoryTable();
  }
}

function retestMessage(msg) {
  switchView("home");
  const textarea = document.getElementById("messageInput");
  textarea.value = msg;
  const len = msg.length;
  const words = msg.trim().split(/\s+/).length;
  document.getElementById("charCounter").textContent = `${len}/5000 chars • ${words} words`;
  handlePredict();
}

async function deleteHistoryItem(id) {
  try {
    const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Delete failed");
    showToast(`Record #${id} deleted`, "info");
    loadHistoryTable();
    refreshStatistics();
    loadRecentScans();
  } catch (err) {
    showToast("Failed to delete record", "error");
  }
}

/* Modal: Clear History */
function openClearModal() {
  const modal = document.getElementById("clearHistoryModal");
  if (modal) modal.style.display = "flex";
}

function closeClearModal() {
  const modal = document.getElementById("clearHistoryModal");
  if (modal) modal.style.display = "none";
}

async function confirmClearHistory() {
  closeClearModal();
  try {
    const res = await fetch("/api/history", { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Clear history failed");

    showToast(`Cleared ${data.deleted_count} records from database`, "success");
    historyState.page = 1;
    loadHistoryTable();
    refreshStatistics();
    loadRecentScans();
  } catch (err) {
    showToast("Failed to clear history", "error");
  }
}

function exportHistoryCsv() {
  showToast("Downloading database CSV export...", "info");
  window.location.href = "/api/history/export";
}

/* =============================================================
   9. BATCH CSV / TXT FILE UPLOAD SCANNER
   ============================================================= */
function initBatchUpload() {
  const dropzone = document.getElementById("uploadDropzone");
  const fileInput = document.getElementById("batchFileInput");
  const browseBtn = document.getElementById("browseFileBtn");

  if (!dropzone || !fileInput) return;

  browseBtn.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (e) => {
    if (e.target !== browseBtn) fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleBatchFileUpload(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleBatchFileUpload(e.dataTransfer.files[0]);
    }
  });

  initHistoryControls();
}

async function handleBatchFileUpload(file) {
  if (!file.name.endsWith(".csv") && !file.name.endsWith(".txt")) {
    showToast("Please upload a .csv or .txt file", "error");
    return;
  }

  const progress = document.getElementById("batchProgress");
  const resultsPanel = document.getElementById("batchResultsPanel");
  const tableBody = document.getElementById("batchTableBody");

  progress.style.display = "block";
  resultsPanel.style.display = "none";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/batch-upload", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Batch classification failed.");
    }

    // Render batch summary cards
    document.getElementById("batchTotalNum").textContent = data.total_processed;
    document.getElementById("batchSpamNum").textContent = data.spam_count;
    document.getElementById("batchHamNum").textContent = data.ham_count;
    document.getElementById("batchSpamPct").textContent = `${data.spam_percentage}%`;

    // Render results table
    tableBody.innerHTML = "";
    data.results.forEach((r, idx) => {
      const tr = document.createElement("tr");
      const badgeClass = r.is_spam ? "spam" : "ham";
      const badgeText = r.is_spam ? "Spam" : "Not Spam";
      const keywords = r.analysis_details ? r.analysis_details.suspicious_keywords : "None";

      tr.innerHTML = `
        <td style="color: var(--text-muted);">${idx + 1}</td>
        <td>
          <span class="recent-pill ${badgeClass}">
            <span class="legend-dot ${r.is_spam ? "dot-red" : "dot-green"}"></span>
            ${badgeText}
          </span>
        </td>
        <td><strong>${Math.round(r.confidence_score)}%</strong></td>
        <td><div class="table-snippet" title="${escapeHtml(r.message)}">${escapeHtml(r.message)}</div></td>
        <td><span style="font-size: 0.78rem; color: var(--text-muted);">${escapeHtml(keywords)}</span></td>
      `;
      tableBody.appendChild(tr);
    });

    resultsPanel.style.display = "block";
    refreshStatistics();
    loadRecentScans();
    showToast(`Batch completed: ${data.total_processed} messages analyzed!`, "success");
  } catch (err) {
    showToast(err.message || "Failed to process batch file", "error");
  } finally {
    progress.style.display = "none";
  }
}

/* =============================================================
   10. FETCH MODEL METRICS & SYSTEM HEALTH
   ============================================================= */
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

async function checkSystemHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) return;
    const data = await res.json();
    const dot = document.getElementById("systemStatusDot");
    const text = document.getElementById("systemStatusText");

    if (data.status === "healthy") {
      dot.style.background = "var(--ham-green)";
      text.textContent = "Model & DB Active";
    } else {
      dot.style.background = "#f59e0b";
      text.textContent = "Degraded";
    }
  } catch {
    const dot = document.getElementById("systemStatusDot");
    const text = document.getElementById("systemStatusText");
    if (dot && text) {
      dot.style.background = "var(--spam-red)";
      text.textContent = "Offline";
    }
  }
}

/* =============================================================
   11. TOAST NOTIFICATION SYSTEM
   ============================================================= */
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const icon = type === "success" ? "✓" : type === "error" ? "✕" : "ℹ";

  toast.innerHTML = `
    <span>${icon}</span>
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

/* =============================================================
   12. UTILITY HELPERS
   ============================================================= */
function escapeHtml(text) {
  if (!text) return "";
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

function escapeAttr(text) {
  if (!text) return "";
  return String(text).replace(/'/g, "\\'").replace(/"/g, "&quot;");
}

function formatDate(isoStr) {
  if (!isoStr) return "N/A";
  try {
    const d = new Date(isoStr + "Z");
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return isoStr;
  }
}

function formatDateSnippet(isoStr) {
  if (!isoStr) return "Just now";
  try {
    const d = new Date(isoStr + "Z");
    const diff = (new Date() - d) / 1000;
    if (diff < 60) return "Just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return "Recently";
  }
}
