const state = {
  mode: "text",
  currentResult: null,
  history: [],
  metrics: null
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

function showPage(page) {
  $$(".page").forEach(p => p.classList.toggle("active", p.id === page));
  $$(".nav-item").forEach(n => n.classList.toggle("active", n.dataset.page === page));
  const titles = {
    dashboard: "Good afternoon 👋",
    analyze: "New Verification Scan",
    result: "Forensic & Combined Report",
    history: "Verification Audit History",
    reports: "Combined Forensic Reports",
    settings: "Workspace Settings"
  };
  const titleEl = $("#pageTitle");
  if (titleEl) titleEl.textContent = titles[page] || "VeritiScan AI";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function selectVerifyMode(mode) {
  state.mode = mode;
  $$(".mode-card").forEach(c => c.classList.toggle("selected", c.dataset.mode === mode));

  const configs = {
    text: {
      title: "Text Claim to Analyze",
      hint: "Paste a news claim, article body, or statement to evaluate with the trained TF-IDF + LogisticRegression model.",
      label: "Analyze Text Claim",
      type: "TEXT NLP"
    },
    url: {
      title: "News Article or Video URL Verification",
      hint: "Paste any news article link, YouTube video link, or direct media URL. VeritiScan fetches the live text and embedded media automatically.",
      label: "Fetch & Verify URL Now",
      type: "LIVE URL SCANNER"
    },
    image: {
      title: "Image Forensics",
      hint: "Upload an image for Vision Transformer (ViT), Error Level Analysis (ELA), DCT frequency, and Laplacian sharpness checks.",
      label: "Analyze Image",
      type: "IMAGE FORENSICS"
    },
    video: {
      title: "Video Frame Forensics",
      hint: "Upload a video file for 1-FPS frame ELA sampling and inter-frame temporal consistency analysis.",
      label: "Analyze Video",
      type: "VIDEO FORENSICS"
    },
    combined: {
      title: "Combined All-in-One Analysis (Text + Image + Video)",
      hint: "Provide Text, upload an Image, and upload a Video below. All three modalities are analyzed in parallel and fused into a unified Combined Forensic Report.",
      label: "Run Combined Multimodal Analysis & Generate Report",
      type: "ALL-IN-ONE COMBINED"
    }
  };

  const c = configs[mode] || configs.text;
  if ($("#inputTitle")) $("#inputTitle").textContent = c.title;
  if ($("#inputHint")) $("#inputHint").textContent = c.hint;
  if ($("#analyzeBtn")) $("#analyzeBtn").innerHTML = `${c.label} <span>✦</span>`;
  if ($("#inputType")) $("#inputType").textContent = c.type;

  const boxText = $("#boxText");
  const boxUrl = $("#boxUrl");
  const boxSingleFile = $("#boxSingleFile");
  const boxCombined = $("#boxCombined");

  if (boxText) boxText.classList.toggle("hidden", mode !== "text");
  if (boxUrl) boxUrl.classList.toggle("hidden", mode !== "url");
  if (boxSingleFile) boxSingleFile.classList.toggle("hidden", !(mode === "image" || mode === "video"));
  if (boxCombined) boxCombined.classList.toggle("hidden", mode !== "combined");

  if (mode === "image" || mode === "video") {
    const fi = $("#fileInput");
    if (fi) {
      fi.accept = mode === "image" ? "image/*" : "video/*";
      fi.value = "";
    }
    if ($("#dropTitle")) $("#dropTitle").textContent = `Drop your ${mode} here`;
    if ($("#fileMeta")) $("#fileMeta").textContent = mode === "image" ? "PNG, JPG, WEBP • Up to 20 MB" : "MP4, MOV, WEBM • Up to 50 MB";
    if ($("#preview")) {
      $("#preview").classList.add("hidden");
      $("#preview").innerHTML = "";
    }
  }
}

function openModeAndPage(mode) {
  selectVerifyMode(mode);
  showPage("analyze");
}

function fillSampleUrl(url, inputId) {
  const el = document.getElementById(inputId || "urlInput");
  if (el) {
    el.value = url;
    showToast("Sample URL loaded — click Verify to scan!");
  }
}

// Expose globally for inline onclick handlers
window.selectVerifyMode = selectVerifyMode;
window.openModeAndPage = openModeAndPage;
window.fillSampleUrl = fillSampleUrl;
window.showPage = showPage;

document.addEventListener("click", e => {
  const pageBtn = e.target.closest("[data-page]");
  if (pageBtn) {
    showPage(pageBtn.dataset.page);
    return;
  }
  const modeBtn = e.target.closest(".mode-card[data-mode]");
  if (modeBtn) {
    selectVerifyMode(modeBtn.dataset.mode);
  }
});

const logoutBtn = document.getElementById("logoutBtn");
if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("userIdentifier");
    window.location.href = "../login.html";
  });
}

const refreshBtn = document.getElementById("refreshBtn");
if (refreshBtn) {
  refreshBtn.addEventListener("click", () => {
    fetchMetricsAndHistory().then(() => showToast("Live metrics & history refreshed"));
  });
}

function updateCombinedPills() {
  const hasText = Boolean($("#combinedTextInput")?.value.trim());
  const hasImg = Boolean($("#combinedImageInput")?.files?.[0]);
  const hasVid = Boolean($("#combinedVideoInput")?.files?.[0]);

  const pText = $("#pillTextStatus");
  const pImg = $("#pillImageStatus");
  const pVid = $("#pillVideoStatus");

  if (pText) {
    pText.textContent = hasText ? "1. Text: Ready ✓" : "1. Text: Empty";
    pText.classList.toggle("ready", hasText);
  }
  if (pImg) {
    pImg.textContent = hasImg ? "2. Image: Ready ✓" : "2. Image: Empty";
    pImg.classList.toggle("ready", hasImg);
  }
  if (pVid) {
    pVid.textContent = hasVid ? "3. Video: Ready ✓" : "3. Video: Empty";
    pVid.classList.toggle("ready", hasVid);
  }
}

if ($("#textInput")) {
  $("#textInput").addEventListener("input", e => {
    if ($("#charCount")) $("#charCount").textContent = `${e.target.value.length} characters`;
  });
}

if ($("#combinedTextInput")) {
  $("#combinedTextInput").addEventListener("input", e => {
    if ($("#combinedCharCount")) $("#combinedCharCount").textContent = `${e.target.value.length} characters`;
    updateCombinedPills();
  });
}

// Single file input preview
if ($("#fileInput")) {
  $("#fileInput").addEventListener("change", e => {
    const file = e.target.files[0];
    if (!file) return;
    const preview = $("#preview");
    if (preview) {
      preview.classList.remove("hidden");
      const url = URL.createObjectURL(file);
      preview.innerHTML = state.mode === "image"
        ? `<img src="${url}" alt="Selected image preview">`
        : `<video src="${url}" controls></video>`;
    }
    if ($("#dropTitle")) $("#dropTitle").textContent = file.name;
    if ($("#fileMeta")) $("#fileMeta").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB • Ready for forensic scan`;
  });
}

// Combined Image input preview
if ($("#combinedImageInput")) {
  $("#combinedImageInput").addEventListener("change", e => {
    const file = e.target.files[0];
    const preview = $("#combinedImgPreview");
    if (!file) {
      if (preview) { preview.classList.add("hidden"); preview.innerHTML = ""; }
      if ($("#combinedImgTitle")) $("#combinedImgTitle").textContent = "Click to select Image";
      updateCombinedPills();
      return;
    }
    if (preview) {
      preview.classList.remove("hidden");
      const url = URL.createObjectURL(file);
      preview.innerHTML = `<img src="${url}" alt="Combined image preview" style="max-height:130px;margin-top:8px;">`;
    }
    if ($("#combinedImgTitle")) $("#combinedImgTitle").textContent = file.name;
    if ($("#combinedImgMeta")) $("#combinedImgMeta").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB • Image ready ✓`;
    updateCombinedPills();
  });
}

// Combined Video input preview
if ($("#combinedVideoInput")) {
  $("#combinedVideoInput").addEventListener("change", e => {
    const file = e.target.files[0];
    const preview = $("#combinedVidPreview");
    if (!file) {
      if (preview) { preview.classList.add("hidden"); preview.innerHTML = ""; }
      if ($("#combinedVidTitle")) $("#combinedVidTitle").textContent = "Click to select Video";
      updateCombinedPills();
      return;
    }
    if (preview) {
      preview.classList.remove("hidden");
      const url = URL.createObjectURL(file);
      preview.innerHTML = `<video src="${url}" controls style="max-height:130px;margin-top:8px;"></video>`;
    }
    if ($("#combinedVidTitle")) $("#combinedVidTitle").textContent = file.name;
    if ($("#combinedVidMeta")) $("#combinedVidMeta").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB • Video ready ✓`;
    updateCombinedPills();
  });
}

// Live API submission handler
if ($("#analyzeBtn")) {
  $("#analyzeBtn").addEventListener("click", async () => {
    const formData = new FormData();

    if (state.mode === "text") {
      const textVal = $("#textInput")?.value.trim() || "";
      if (!textVal) { showToast("Please enter text to analyze."); return; }
      formData.append("text", textVal);
    } else if (state.mode === "url") {
      const urlVal = $("#urlInput")?.value.trim() || "";
      if (!urlVal) { showToast("Please enter a news or video URL to verify."); return; }
      formData.append("url", urlVal);
    } else if (state.mode === "image") {
      const singleFile = $("#fileInput")?.files?.[0];
      if (!singleFile) { showToast("Please select an image file first."); return; }
      formData.append("image_file", singleFile);
    } else if (state.mode === "video") {
      const singleFile = $("#fileInput")?.files?.[0];
      if (!singleFile) { showToast("Please select a video file first."); return; }
      formData.append("video_file", singleFile);
    } else if (state.mode === "combined") {
      const combText = $("#combinedTextInput")?.value.trim() || $("#textInput")?.value.trim() || "";
      const combUrl = $("#combinedUrlInput")?.value.trim() || "";
      const combImg = $("#combinedImageInput")?.files?.[0];
      const combVid = $("#combinedVideoInput")?.files?.[0];

      if (!combText && !combUrl && !combImg && !combVid) {
        showToast("Please provide Text, Image, Video, or URL in the Combined workspace.");
        return;
      }
      if (combText) formData.append("text", combText);
      if (combUrl) formData.append("url", combUrl);
      if (combImg) formData.append("image_file", combImg);
      if (combVid) formData.append("video_file", combVid);
    }

    const btn = $("#analyzeBtn");
    const oldHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = "Running Multimodal Forensic Pipelines... <span>◌</span>";

    try {
      const response = await fetch("/api/v1/verify", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Verification failed");
      }

      state.currentResult = data;
      populateResult(data);
      await fetchMetricsAndHistory();
      showPage("result");
      showToast("Verification & Combined Report complete!");
    } catch (err) {
      showToast("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = oldHtml;
    }
  });
}

function formatTime(isoStr) {
  if (!isoStr) return "Just now";
  try {
    const d = new Date(isoStr);
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch (e) {
    return isoStr;
  }
}

function deriveModalityType(record) {
  const cr = record.combined_report;
  if (cr && Array.isArray(cr.modalities_analyzed) && cr.modalities_analyzed.length > 0) {
    if (cr.modalities_analyzed.length > 1) {
      return "Combined (" + cr.modalities_analyzed.join("+") + ")";
    }
    return cr.modalities_analyzed[0];
  }
  const summary = record.input_summary || {};
  if (summary.url) return "URL";
  if (summary.content_type && summary.content_type.startsWith("image/")) return "Image";
  if (summary.content_type && summary.content_type.startsWith("video/")) return "Video";
  return "Text";
}

function deriveRecordTitle(record) {
  const summary = record.input_summary || {};
  const bd = record.breakdown || {};
  if (summary.title) return summary.title;
  if (bd.url_title) return bd.url_title;
  if (bd.url_source) return bd.url_source;
  if (summary.filename) return summary.filename;
  return `Scan #${(record.record_id || "").slice(0, 8)}`;
}

function statusClassForVerdict(verdict) {
  if (!verdict) return "";
  if (verdict.includes("Manipulated")) return "bad";
  if (verdict.includes("Contradiction")) return "review";
  return "";
}

function iconForType(typeStr) {
  const t = (typeStr || "").toLowerCase();
  if (t.includes("combined")) return "⊕";
  if (t.includes("url")) return "🌐";
  if (t.includes("image")) return "◉";
  if (t.includes("video")) return "▶";
  return "Aa";
}

function renderRecent() {
  const container = $("#recentList");
  if (!container) return;
  if (!state.history.length) {
    container.innerHTML = `<div class="empty-state" style="padding:30px 15px;"><p>No scans recorded yet. Run your first analysis!</p></div>`;
    return;
  }
  container.innerHTML = state.history.slice(0, 6).map((item, idx) => {
    const typeLabel = deriveModalityType(item);
    const title = deriveRecordTitle(item);
    const sClass = statusClassForVerdict(item.verdict);
    return `
      <div class="analysis-row" data-history-idx="${idx}">
        <div class="type-icon type-text">${iconForType(typeLabel)}</div>
        <div>
          <div class="row-title">${escapeHtml(title)}</div>
          <div class="row-sub">${escapeHtml(typeLabel)} • ${formatTime(item.timestamp)}</div>
        </div>
        <div class="score">${Number(item.confidence_score || 0).toFixed(1)}%</div>
        <div class="status ${sClass}">${escapeHtml(item.verdict || "Completed")}</div>
      </div>`;
  }).join("");

  container.querySelectorAll("[data-history-idx]").forEach(row => {
    row.addEventListener("click", () => {
      const item = state.history[Number(row.dataset.historyIdx)];
      if (item) {
        state.currentResult = item;
        populateResult(item);
        showPage("result");
      }
    });
  });
}

function renderHistory() {
  const container = $("#historyList");
  if (!container) return;
  const q = ($("#historySearch")?.value || "").toLowerCase();
  const filter = $("#historyFilter")?.value || "all";

  const filtered = state.history.filter(item => {
    const typeLabel = deriveModalityType(item);
    const title = deriveRecordTitle(item).toLowerCase();
    const verdict = (item.verdict || "").toLowerCase();
    const matchesFilter =
      filter === "all" ||
      (filter === "Combined" && typeLabel.startsWith("Combined")) ||
      typeLabel.toLowerCase().includes(filter.toLowerCase());
    const matchesSearch = !q || title.includes(q) || verdict.includes(q) || typeLabel.toLowerCase().includes(q);
    return matchesFilter && matchesSearch;
  });

  if (!filtered.length) {
    container.innerHTML = `<div class="empty-state"><h3>No matching analyses found</h3><p>Try a different search query or run a new verification scan.</p></div>`;
    return;
  }

  container.innerHTML = `
    <div class="history-item header"><span>Submission</span><span>Modalities</span><span>Confidence</span><span>Verdict</span></div>
    ${filtered.map(item => {
      const idx = state.history.indexOf(item);
      const typeLabel = deriveModalityType(item);
      const title = deriveRecordTitle(item);
      const sClass = statusClassForVerdict(item.verdict);
      return `
        <div class="history-item" data-history-idx="${idx}">
          <div class="history-main">${escapeHtml(title)}<small>${formatTime(item.timestamp)} • ID: ${(item.record_id || "").slice(0, 8)}</small></div>
          <div>${escapeHtml(typeLabel)}</div>
          <div><strong>${Number(item.confidence_score || 0).toFixed(1)}%</strong></div>
          <div class="status ${sClass}">${escapeHtml(item.verdict)}</div>
        </div>`;
    }).join("")}
  `;

  container.querySelectorAll("[data-history-idx]").forEach(row => {
    row.addEventListener("click", () => {
      const item = state.history[Number(row.dataset.historyIdx)];
      if (item) {
        state.currentResult = item;
        populateResult(item);
        showPage("result");
      }
    });
  });
}

function renderReportsPage() {
  const container = $("#reportsListContainer");
  if (!container) return;
  if (!state.history.length) {
    container.innerHTML = `
      <div class="empty-state panel">
        <div class="empty-icon">▤</div>
        <h3>No forensic reports generated yet</h3>
        <p>Run a single-modality, URL, or Combined (Text + Image + Video) scan to generate downloadable forensic reports.</p>
        <button class="primary-btn" type="button" onclick="window.openModeAndPage('combined')">Run Combined Analysis</button>
      </div>`;
    return;
  }

  container.innerHTML = state.history.map((item, idx) => {
    const cr = item.combined_report;
    const typeLabel = deriveModalityType(item);
    const title = deriveRecordTitle(item);
    const sClass = statusClassForVerdict(item.verdict);
    const summaryText = (cr && cr.executive_summary) ? cr.executive_summary : (item.explanation || "");
    const agreement = (cr && cr.cross_modal_agreement) ? cr.cross_modal_agreement : "Verified";
    return `
      <div class="report-dossier-card">
        <div class="report-dossier-main">
          <div class="result-badges">
            <span class="pill">${escapeHtml(typeLabel)}</span>
            <span class="status ${sClass}">${escapeHtml(item.verdict)} (${Number(item.confidence_score || 0).toFixed(1)}%)</span>
            <span class="agreement-badge ${agreement.includes("Contradiction") ? "conflict" : ""}">${escapeHtml(agreement)}</span>
          </div>
          <h3>${escapeHtml(title)}</h3>
          <p>${escapeHtml(summaryText)}</p>
          <small style="color:var(--muted);font-size:10px;display:block;margin-top:6px;">${formatTime(item.timestamp)} • Record ID: ${item.record_id}</small>
        </div>
        <div class="report-dossier-actions">
          <button type="button" class="secondary-btn view-report-btn" data-idx="${idx}">View Full Report</button>
          <button type="button" class="primary-btn download-item-btn" data-idx="${idx}" style="padding:9px 14px;font-size:11px;">⬇ Download</button>
        </div>
      </div>`;
  }).join("");

  container.querySelectorAll(".view-report-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const item = state.history[Number(btn.dataset.idx)];
      if (item) {
        state.currentResult = item;
        populateResult(item);
        showPage("result");
      }
    });
  });

  container.querySelectorAll(".download-item-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const item = state.history[Number(btn.dataset.idx)];
      if (item) downloadCombinedReportHTML(item);
    });
  });
}

function updateDashboardStats() {
  const total = state.history.length;
  const urlCount = state.history.filter(r => {
    const cr = r.combined_report;
    return (r.input_summary && r.input_summary.url) || (cr && cr.modalities_analyzed && cr.modalities_analyzed.includes("URL"));
  }).length;
  const combinedCount = state.history.filter(r => {
    const cr = r.combined_report;
    return (cr && cr.is_multimodal) || (r.input_summary && r.input_summary.is_multimodal);
  }).length;

  if ($("#statTotal")) $("#statTotal").textContent = state.metrics ? Math.max(state.metrics.total_verifications, total) : total;
  if ($("#statUrl")) $("#statUrl").textContent = urlCount;
  if ($("#statCombined")) $("#statCombined").textContent = combinedCount;

  if (state.metrics) {
    const acc = (state.metrics.model_accuracy * 100).toFixed(1) + "%";
    const f1 = (state.metrics.f1_score * 100).toFixed(1) + "%";
    if ($("#heroAccuracy")) $("#heroAccuracy").textContent = acc;
    if ($("#avgConfidence")) $("#avgConfidence").textContent = acc;
    if ($("#statF1")) $("#statF1").textContent = f1;
    if ($("#avgDonut")) $("#avgDonut").style.background = `conic-gradient(#6371ff 0 ${(state.metrics.model_accuracy * 100).toFixed(1)}%, #edf0f6 ${(state.metrics.model_accuracy * 100).toFixed(1)}%)`;
  }
}

async function fetchMetricsAndHistory() {
  try {
    const [mRes, hRes] = await Promise.all([
      fetch("/api/v1/metrics"),
      fetch("/api/v1/history?limit=100")
    ]);
    if (mRes.ok) {
      state.metrics = await mRes.json();
    }
    if (hRes.ok) {
      const hData = await hRes.json();
      state.history = hData.records || [];
    }
    updateDashboardStats();
    renderRecent();
    renderHistory();
    renderReportsPage();
  } catch (err) {
    console.error("Failed to fetch metrics/history:", err);
  }
}

function populateResult(res) {
  const cr = res.combined_report || {};
  const bd = res.breakdown || {};
  const conf = Number(res.confidence_score || 0);
  const isManipulated = (res.verdict || "").includes("Manipulated");
  const isConflict = (res.verdict || "").includes("Contradiction");

  if ($("#resultLabel")) $("#resultLabel").textContent = res.verdict || "Analysis Complete";
  if ($("#resultDescription")) $("#resultDescription").textContent = res.explanation || "";
  if ($("#confidenceValue")) $("#confidenceValue").textContent = conf.toFixed(1) + "%";
  if ($("#detailConfidence")) $("#detailConfidence").textContent = conf.toFixed(1) + "%";

  const ringColor = isManipulated ? "#df4855" : isConflict ? "#ff9f43" : "#22b573";
  if ($("#confidenceRing")) $("#confidenceRing").style.background = `conic-gradient(${ringColor} 0 ${conf}%, #e9ebf3 ${conf}% 100%)`;

  const modalitiesList = cr.modalities_analyzed && cr.modalities_analyzed.length
    ? cr.modalities_analyzed.join(" + ")
    : deriveModalityType(res);

  if ($("#resultType")) $("#resultType").textContent = modalitiesList.toUpperCase();
  if ($("#detailType")) $("#detailType").textContent = modalitiesList;
  if ($("#resultTime")) $("#resultTime").textContent = formatTime(res.timestamp);
  if ($("#resultRecordId")) $("#resultRecordId").textContent = `ID: ${(res.record_id || "").slice(0, 12)}`;

  const agreement = cr.cross_modal_agreement || "Verified";
  const agBadge = $("#agreementBadge");
  if (agBadge) {
    agBadge.textContent = agreement;
    agBadge.classList.toggle("conflict", agreement.includes("Contradiction"));
  }

  const authScore = cr.overall_authenticity_score !== undefined ? cr.overall_authenticity_score : (isManipulated ? (100 - conf) : conf);
  const riskScore = cr.overall_manipulation_risk !== undefined ? cr.overall_manipulation_risk : (100 - authScore);
  if ($("#reportAuthScore")) $("#reportAuthScore").textContent = Number(authScore).toFixed(1) + "%";
  if ($("#reportRiskScore")) $("#reportRiskScore").textContent = Number(riskScore).toFixed(1) + "%";
  if ($("#reportExecSummary")) $("#reportExecSummary").textContent = cr.executive_summary || res.explanation || "No summary available.";

  const modCardsContainer = $("#modalityCardsGrid");
  const modReports = cr.modality_reports || [];
  if (modCardsContainer) {
    if (modReports.length > 0) {
      modCardsContainer.innerHTML = modReports.map(m => {
        const isBad = m.verdict.includes("Manipulated");
        const authPct = Number(m.authenticity_score || 0).toFixed(1);
        const riskPct = (Number(m.manipulation_probability || 0) * 100).toFixed(1);
        return `
          <div class="mod-report-card">
            <div class="mod-card-top">
              <span class="mod-badge">${iconForType(m.modality)} ${escapeHtml(m.modality)}</span>
              <span class="mod-verdict-tag ${isBad ? "bad" : ""}">${escapeHtml(m.verdict)}</span>
            </div>
            <div class="mod-bar-wrap">
              <div class="mod-bar-fill ${isBad ? "bad" : ""}" style="width:${isBad ? riskPct : authPct}%"></div>
            </div>
            <div class="mod-card-scores">
              <span>Authenticity: <strong>${authPct}%</strong></span>
              <span>Manipulation Risk: <strong>${riskPct}%</strong></span>
            </div>
            <p class="mod-card-summary">${escapeHtml(m.summary)}</p>
          </div>
        `;
      }).join("");
    } else {
      modCardsContainer.innerHTML = `<p style="font-size:12px;color:var(--muted);">Single-stream summary recorded.</p>`;
    }
  }

  const findingsList = $("#reportFindingsList");
  const findings = cr.key_findings || [];
  if (findingsList) {
    findingsList.innerHTML = findings.length
      ? findings.map(f => `<li>${escapeHtml(f)}</li>`).join("")
      : `<li>${escapeHtml(res.explanation || "Standard forensic checks completed.")}</li>`;
  }

  if ($("#reportRecommendation")) $("#reportRecommendation").textContent = cr.recommendation || "Review forensic metrics before publication.";

  if ($("#detailTextProb")) {
    $("#detailTextProb").textContent = bd.text_fake_probability !== null && bd.text_fake_probability !== undefined
      ? (bd.text_fake_probability * 100).toFixed(2) + "% fake risk"
      : "N/A";
  }
  if ($("#detailEla")) {
    $("#detailEla").textContent = bd.compression_discrepancy !== null && bd.compression_discrepancy !== undefined
      ? bd.compression_discrepancy.toFixed(4)
      : "N/A";
  }
  if ($("#detailEntropy")) {
    $("#detailEntropy").textContent = bd.visual_entropy !== null && bd.visual_entropy !== undefined
      ? bd.visual_entropy.toFixed(4)
      : "N/A";
  }
  if ($("#detailConflict")) {
    $("#detailConflict").textContent = bd.cross_modal_conflict_detected ? "YES (High Risk)" : "No Conflict";
  }

  const previewParts = [];
  if (bd.url_source) {
    previewParts.push(`🌐 Source URL: ${bd.url_source}`);
    if (bd.url_title) previewParts.push(`📰 Extracted Title: ${bd.url_title}`);
    if (bd.url_type) previewParts.push(`📂 URL Content Type: ${bd.url_type}`);
    if (bd.url_extracted_text_preview) {
      previewParts.push(`\n--- Extracted URL Content Preview ---\n${bd.url_extracted_text_preview}`);
    }
  }
  const userText = (state.mode === "combined" ? $("#combinedTextInput")?.value.trim() : $("#textInput")?.value.trim()) || "";
  if (userText && (state.mode === "text" || state.mode === "combined")) {
    previewParts.push(`\n--- Submitted Text Claim ---\n${userText}`);
  }
  if (bd.image_ela_score !== null && bd.image_ela_score !== undefined) {
    previewParts.push(`\n🖼 Image Forensics:\n  • ELA / ViT Manipulation Score: ${bd.image_ela_score.toFixed(4)}\n  • Laplacian Sharpness Anomaly: ${(bd.image_laplacian_score || 0).toFixed(4)}`);
  }
  if (bd.video_mean_ela !== null && bd.video_mean_ela !== undefined) {
    previewParts.push(`\n🎬 Video Forensics:\n  • Mean Frame ELA: ${bd.video_mean_ela.toFixed(4)}\n  • Temporal Consistency Variance: ${(bd.video_temporal_variance || 0).toFixed(4)}`);
  }
  if ($("#submissionPreview")) {
    $("#submissionPreview").textContent = previewParts.join("\n") || "Forensic telemetry recorded in audit log.";
  }
}

const downloadReportBtn = document.getElementById("downloadReportBtn");
if (downloadReportBtn) {
  downloadReportBtn.addEventListener("click", () => {
    if (state.currentResult) downloadCombinedReportHTML(state.currentResult);
  });
}

const printReportBtn = document.getElementById("printReportBtn");
if (printReportBtn) {
  printReportBtn.addEventListener("click", () => {
    window.print();
  });
}

function downloadCombinedReportHTML(res) {
  const cr = res.combined_report || {};
  const bd = res.breakdown || {};
  const modalities = (cr.modalities_analyzed || []).join(", ") || "Single Modality";
  const modRows = (cr.modality_reports || []).map(m => `
    <tr>
      <td style="padding:10px;border:1px solid #ddd;font-weight:bold;">${escapeHtml(m.modality)}</td>
      <td style="padding:10px;border:1px solid #ddd;">${escapeHtml(m.verdict)}</td>
      <td style="padding:10px;border:1px solid #ddd;">${Number(m.authenticity_score).toFixed(1)}%</td>
      <td style="padding:10px;border:1px solid #ddd;">${(Number(m.manipulation_probability) * 100).toFixed(1)}%</td>
      <td style="padding:10px;border:1px solid #ddd;">${escapeHtml(m.summary)}</td>
    </tr>
  `).join("");

  const findingsHtml = (cr.key_findings || []).map(f => `<li>${escapeHtml(f)}</li>`).join("");

  const htmlDoc = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>VeritiScan AI — Combined Forensic Report (${res.record_id || "report"})</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #17213a; max-width: 900px; margin: 40px auto; padding: 0 24px; line-height: 1.6; }
    .header { border-bottom: 2px solid #5b6cff; padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }
    .badge { display: inline-block; padding: 6px 12px; border-radius: 999px; background: #eef0ff; color: #5b6cff; font-weight: bold; font-size: 13px; }
    .box { background: #f7f9ff; border: 1px solid #dce1ef; border-radius: 12px; padding: 18px; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; margin: 16px 0 24px; font-size: 14px; }
    th { background: #eef0ff; text-align: left; padding: 10px; border: 1px solid #ddd; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1 style="margin:0;">VeritiScan AI — Multimodal Forensic Report</h1>
      <small>Record ID: ${escapeHtml(res.record_id || "")} • Timestamp: ${escapeHtml(res.timestamp || "")}</small>
    </div>
    <span class="badge">${escapeHtml(res.verdict)} (${Number(res.confidence_score).toFixed(1)}% Confidence)</span>
  </div>

  <div class="box">
    <h3 style="margin-top:0;">Executive Summary</h3>
    <p>${escapeHtml(cr.executive_summary || res.explanation || "")}</p>
    <p><strong>Modalities Analyzed:</strong> ${escapeHtml(modalities)} | <strong>Cross-Modal Agreement:</strong> ${escapeHtml(cr.cross_modal_agreement || "Verified")}</p>
    <p><strong>Overall Authenticity Score:</strong> ${Number(cr.overall_authenticity_score || 0).toFixed(1)}% | <strong>Overall Manipulation Risk:</strong> ${Number(cr.overall_manipulation_risk || 0).toFixed(1)}%</p>
  </div>

  <h3>Modality-by-Modality Forensic Breakdown</h3>
  <table>
    <thead>
      <tr><th>Modality</th><th>Verdict</th><th>Authenticity</th><th>Manipulation Risk</th><th>Forensic Summary</th></tr>
    </thead>
    <tbody>${modRows}</tbody>
  </table>

  <div class="box">
    <h3 style="margin-top:0;">Key Forensic Findings</h3>
    <ul>${findingsHtml}</ul>
    <h3>Editorial Recommendation</h3>
    <p><strong>${escapeHtml(cr.recommendation || "")}</strong></p>
  </div>

  ${bd.url_source ? `<div class="box"><h3>URL Source Metadata</h3><p><strong>URL:</strong> ${escapeHtml(bd.url_source)}<br><strong>Title:</strong> ${escapeHtml(bd.url_title || "")}<br><strong>Domain:</strong> ${escapeHtml(bd.url_domain || "")}</p></div>` : ""}
</body>
</html>`;

  const blob = new Blob([htmlDoc], { type: "text/html" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `VeritiScan_Combined_Report_${(res.record_id || "scan").slice(0, 8)}.html`;
  link.click();
  showToast("Combined Forensic Report downloaded!");
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

if ($("#historySearch")) $("#historySearch").addEventListener("input", renderHistory);
if ($("#historyFilter")) $("#historyFilter").addEventListener("change", renderHistory);

function showToast(message) {
  const t = $("#toast");
  if (!t) return;
  t.textContent = message;
  t.classList.add("show");
  clearTimeout(window.__toast);
  window.__toast = setTimeout(() => t.classList.remove("show"), 2600);
}

// Initialize mode and load live data from backend
selectVerifyMode("text");
fetchMetricsAndHistory();
