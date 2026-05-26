const API_BASE = window.localStorage.getItem("MASI_API_BASE") || window.location.origin;

const crosshairPlugin = {
  id: "crosshairGuide",
  afterDatasetsDraw(chart) {
    const active = chart.tooltip?.getActiveElements?.() || [];
    if (!active.length) return;

    const { ctx, chartArea, scales } = chart;
    const point = active[0].element;
    const x = point.x;
    const y = point.y;
    const yScale = scales?.y;
    if (!chartArea || !yScale) return;

    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(31, 41, 55, 0.35)";

    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(chartArea.left, y);
    ctx.lineTo(chartArea.right, y);
    ctx.stroke();
    ctx.restore();
  },
};

if (window.Chart) {
  Chart.register(crosshairPlugin);
}

const state = {
  selectedHorizon: 1,
  latestForecast: null,
  priceSeries: null,
  riskSeries: null,
  testPredictions: [],
  backtest: null,
  report: null,
  admin: {
    pipelines: [],
    runs: [],
    selectedRunId: null,
    pollTimer: null,
  },
  charts: {},
};

const el = {
  apiStatus: document.getElementById("apiStatus"),
  errorBox: document.getElementById("errorBox"),
  refreshButton: document.getElementById("refreshButton"),
  runMeta: document.getElementById("runMeta"),
  varValue: document.getElementById("varValue"),
  varTarget: document.getElementById("varTarget"),
  esValue: document.getElementById("esValue"),
  returnValue: document.getElementById("returnValue"),
  meanValue: document.getElementById("meanValue"),
  meanTarget: document.getElementById("meanTarget"),
  volatilityValue: document.getElementById("volatilityValue"),
  volatilityTarget: document.getElementById("volatilityTarget"),
  regimeValue: document.getElementById("regimeValue"),
  weightValue: document.getElementById("weightValue"),
  violationRate: document.getElementById("violationRate"),
  violationStatus: document.getElementById("violationStatus"),
  nViolations: document.getElementById("nViolations"),
  breachLabel: document.getElementById("breachLabel"),
  esResidualMean: document.getElementById("esResidualMean"),
  esResidualLabel: document.getElementById("esResidualLabel"),
  finalWealth: document.getElementById("finalWealth"),
  drawdownValue: document.getElementById("drawdownValue"),
  bhWealthLabel: document.getElementById("bhWealthLabel"),
  bhDrawdownLabel: document.getElementById("bhDrawdownLabel"),
  sharpeValue: document.getElementById("sharpeValue"),
  bhSharpeLabel: document.getElementById("bhSharpeLabel"),
  reportContent: document.getElementById("reportContent"),
  downloadReportButton: document.getElementById("downloadReportButton"),
  testPredictionRows: document.getElementById("testPredictionRows"),
  refreshAdminButton: document.getElementById("refreshAdminButton"),
  pipelineGrid: document.getElementById("pipelineGrid"),
  pipelineRunRows: document.getElementById("pipelineRunRows"),
  pipelineLog: document.getElementById("pipelineLog"),
  adminUploadForm: document.getElementById("adminUploadForm"),
  adminDataFile: document.getElementById("adminDataFile"),
  adminFileName: document.getElementById("adminFileName"),
  adminUploadStatus: document.getElementById("adminUploadStatus"),
  uploadDataButton: document.getElementById("uploadDataButton"),
  runCleaningButton: document.getElementById("runCleaningButton"),
  chatButton: document.getElementById("chatButton"),
  chatOverlay: document.getElementById("chatOverlay"),
  chatPanel: document.getElementById("chatPanel"),
  chatCloseButton: document.getElementById("chatCloseButton"),
  chatMessages: document.getElementById("chatMessages"),
  chatStarterPrompts: document.getElementById("chatStarterPrompts"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  chatSendButton: document.getElementById("chatSendButton"),
  chatVoiceButton: document.getElementById("chatVoiceButton"),
  chatVoiceMeter: document.getElementById("chatVoiceMeter"),
  chatVoiceStatus: document.getElementById("chatVoiceStatus"),
};

const chatState = {
  isOpen: false,
  isSending: false,
  isListening: false,
  history: [],
  recognition: null,
};

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${path} failed (${response.status}): ${detail}`);
  }
  return response.json();
}

async function apiPost(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const body = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = isJson ? body?.detail || JSON.stringify(body) : body;
    throw new Error(detail || `${path} failed (${response.status})`);
  }

  return body;
}

async function apiPostStream(path, payload, onEvent) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      Accept: "application/x-ndjson",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${path} failed (${response.status})`);
  }

  if (!response.body) {
    const result = await apiPost("/chat/ask", payload);
    onEvent({ type: "done", answer: result.answer || "" });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      onEvent(JSON.parse(trimmed));
    }
  }

  buffer += decoder.decode();
  const trailing = buffer.trim();
  if (trailing) onEvent(JSON.parse(trailing));
}

async function apiUpload(path, formData) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { Accept: "application/json" },
    body: formData,
  });
  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const body = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = isJson ? body?.detail || JSON.stringify(body) : body;
    throw new Error(detail || `${path} failed (${response.status})`);
  }

  return body;
}

function fmtPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString("fr-FR", { maximumFractionDigits: digits });
}

function fmtPrice(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString("fr-FR", { maximumFractionDigits: 2 });
}

function formatDate(value) {
  if (!value) return "--";
  return new Date(value).toLocaleDateString("fr-FR", { day: "2-digit", month: "short", year: "numeric" });
}

function showError(message) {
  el.errorBox.hidden = false;
  el.errorBox.textContent = message;
}

function clearError() {
  el.errorBox.hidden = true;
  el.errorBox.textContent = "";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_])_([^_\n]+)_/g, "$1<em>$2</em>");
}

function renderMath(container) {
  if (!container || typeof window.renderMathInElement !== "function") return;
  window.renderMathInElement(container, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
}

function normalizeLatexContent(content) {
  return String(content || "")
    .replace(/\\\\\[/g, "\\[")
    .replace(/\\\\\]/g, "\\]")
    .replace(/\\\\\(/g, "\\(")
    .replace(/\\\\\)/g, "\\)")
    .replace(/\r\n/g, "\n");
}

function renderTextBlock(block) {
  const lines = block.split("\n").map((line) => line.trimRight());
  const html = [];
  let paragraph = [];
  let listType = null;
  let listItems = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!listType) return;
    const tag = listType === "ordered" ? "ol" : "ul";
    html.push(`<${tag}>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${tag}>`);
    listType = null;
    listItems = [];
  };

  lines.forEach((line) => {
    const trimmed = line.trim();
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    const orderedMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    const unorderedMatch = trimmed.match(/^[-*]\s+(.+)$/);

    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = Math.min(headingMatch[1].length + 2, 5);
      html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      return;
    }

    if (orderedMatch || unorderedMatch) {
      flushParagraph();
      const nextType = orderedMatch ? "ordered" : "unordered";
      if (listType && listType !== nextType) flushList();
      listType = nextType;
      listItems.push(orderedMatch ? orderedMatch[1] : unorderedMatch[1]);
      return;
    }

    flushList();
    paragraph.push(trimmed);
  });

  flushParagraph();
  flushList();
  return html.join("");
}

function renderChatContent(node, content) {
  if (!node) return;

  const normalized = normalizeLatexContent(content);
  const displayMathPattern = /(\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)/g;
  const segments = normalized.split(displayMathPattern).filter((segment) => segment && segment.trim());

  const html = segments
    .map((segment) => {
      const trimmed = segment.trim();
      if (/^(\\\[[\s\S]*\\\]|\$\$[\s\S]*\$\$)$/.test(trimmed)) {
        return `<div class="chat-math-block">${escapeHtml(trimmed)}</div>`;
      }

      return trimmed
        .split(/\n\s*\n/)
        .map((block) => block.trim())
        .filter(Boolean)
        .map(renderTextBlock)
        .join("");
    })
    .join("");

  node.innerHTML = `<div class="chat-rich-text">${html}</div>`;
  renderMath(node);
}

function setLoading(isLoading) {
  el.refreshButton.disabled = isLoading;
  el.refreshButton.textContent = isLoading ? "Loading..." : "Refresh";
}

function setChatLoading(isLoading) {
  chatState.isSending = isLoading;
  if (el.chatSendButton) el.chatSendButton.disabled = isLoading;
  if (el.chatVoiceButton) el.chatVoiceButton.disabled = isLoading;
  if (el.chatInput) el.chatInput.disabled = isLoading;
  if (el.chatCloseButton) el.chatCloseButton.disabled = isLoading;
  if (el.chatSendButton) el.chatSendButton.textContent = isLoading ? "Envoi..." : "Envoyer";
}

function getSpeechRecognitionConstructor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function setVoiceStatus(message) {
  if (el.chatVoiceStatus) el.chatVoiceStatus.textContent = message;
}

function appendTranscript(transcript) {
  if (!el.chatInput || !transcript) return;
  const current = el.chatInput.value.trim();
  el.chatInput.value = current ? `${current} ${transcript}` : transcript;
  el.chatInput.focus();
}

async function startVoiceInput() {
  if (chatState.isSending || chatState.isListening) return;
  const Recognition = getSpeechRecognitionConstructor();
  if (!Recognition) {
    setVoiceStatus("Dictee vocale non supportee par ce navigateur.");
    if (el.chatVoiceMeter) el.chatVoiceMeter.hidden = false;
    return;
  }

  chatState.isListening = true;
  if (el.chatVoiceButton) el.chatVoiceButton.classList.add("is-listening");
  if (el.chatVoiceMeter) el.chatVoiceMeter.hidden = false;
  setVoiceStatus("Ecoute en cours...");

  const recognition = new Recognition();
  recognition.lang = "fr-FR";
  recognition.continuous = false;
  recognition.interimResults = true;
  chatState.recognition = recognition;

  let finalTranscript = "";
  recognition.onresult = (event) => {
    let interimTranscript = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const piece = event.results[index][0]?.transcript || "";
      if (event.results[index].isFinal) {
        finalTranscript += piece;
      } else {
        interimTranscript += piece;
      }
    }
    setVoiceStatus(interimTranscript ? `Transcription: ${interimTranscript.trim()}` : "Ecoute en cours...");
  };
  recognition.onerror = (event) => {
    setVoiceStatus(event.error === "not-allowed" ? "Micro refuse par le navigateur." : "Dictee interrompue.");
  };
  recognition.onend = () => {
    if (finalTranscript.trim()) appendTranscript(finalTranscript.trim());
    stopVoiceInput(false);
  };

  try {
    recognition.start();
  } catch (error) {
    setVoiceStatus("Impossible de demarrer la dictee.");
    stopVoiceInput(false);
  }
}

function stopVoiceInput(stopRecognition = true) {
  if (stopRecognition && chatState.recognition) {
    try {
      chatState.recognition.stop();
    } catch (error) {
      console.warn("Speech recognition stop failed", error);
    }
  }
  chatState.recognition = null;
  chatState.isListening = false;
  if (el.chatVoiceButton) el.chatVoiceButton.classList.remove("is-listening");
  if (el.chatVoiceMeter) el.chatVoiceMeter.hidden = true;
}

function toggleVoiceInput() {
  if (chatState.isListening) {
    stopVoiceInput(true);
    return;
  }
  startVoiceInput();
}

function appendChatMessage(role, content, options = {}) {
  if (!el.chatMessages) return null;
  const article = document.createElement("article");
  article.className = `chat-message chat-message-${role}`;
  if (options.pending) article.classList.add("is-pending");
  renderChatContent(article, content);
  el.chatMessages.appendChild(article);
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  return article;
}

function hideStarterPrompts() {
  if (!el.chatStarterPrompts) return;
  el.chatStarterPrompts.hidden = true;
}

function maybeShowStarterPrompts() {
  if (!el.chatStarterPrompts) return;
  el.chatStarterPrompts.hidden = chatState.history.length > 0;
}

function updateChatMessage(node, content) {
  if (!node) return;
  node.classList.remove("is-pending");
  renderChatContent(node, content);
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
}

function openChatPanel() {
  if (!el.chatPanel || !el.chatOverlay) return;
  chatState.isOpen = true;
  el.chatOverlay.hidden = false;
  el.chatPanel.classList.add("is-open");
  el.chatPanel.setAttribute("aria-hidden", "false");
  maybeShowStarterPrompts();
  window.requestAnimationFrame(() => el.chatInput?.focus());
}

function closeChatPanel() {
  if (!el.chatPanel || !el.chatOverlay || chatState.isSending) return;
  if (chatState.isListening) stopVoiceInput(true);
  chatState.isOpen = false;
  el.chatPanel.classList.remove("is-open");
  el.chatPanel.setAttribute("aria-hidden", "true");
  el.chatOverlay.hidden = true;
}

function buildCurrentDashboardState() {
  const meta = state.latestForecast || {};
  const forecastPoint = selectedForecastPoint() || meta;
  const riskPoint = state.riskSeries?.forecasts?.find((item) => Number(item.horizon) === 1) || {};
  const backtest = state.backtest || {};

  return {
    selected_horizon: state.selectedHorizon,
    run_date: meta.run_date || null,
    model_version: meta.model_version || null,
    data_version: meta.data_version || null,
    target_date: forecastPoint.target_date || null,
    return_forecast: forecastPoint.return_forecast ?? null,
    var_forecast: forecastPoint.var_forecast ?? null,
    es_forecast: forecastPoint.es_forecast ?? null,
    mean_forecast: riskPoint.mean_forecast ?? forecastPoint.mean_forecast ?? null,
    egarch_volatility: riskPoint.volatility_forecast ?? forecastPoint.volatility_forecast ?? null,
    regime: meta.regime || null,
    weight: meta.weight ?? null,
    displayed_var: el.varValue?.textContent || null,
    displayed_es: el.esValue?.textContent || null,
    displayed_return: el.returnValue?.textContent || null,
    displayed_mean: el.meanValue?.textContent || null,
    displayed_egarch_volatility: el.volatilityValue?.textContent || null,
    displayed_regime: el.regimeValue?.textContent || null,
    backtest_violation_rate: backtest.violation_rate ?? null,
    backtest_expected_violation_rate: backtest.expected_violation_rate ?? null,
    backtest_n_violations: backtest.n_violations ?? null,
    backtest_kupiec_p_value: backtest.statistical?.kupiec_pof_p_value ?? null,
    backtest_christoffersen_p_value: backtest.statistical?.christoffersen_cc_p_value ?? null,
    backtest_es_residual_mean: backtest.es_tail_residual_mean ?? null,
    final_wealth: backtest.final_wealth ?? null,
    buy_hold_final_wealth: backtest.buy_hold_final_wealth ?? null,
    max_drawdown: backtest.max_drawdown ?? backtest.current_drawdown ?? null,
    buy_hold_max_drawdown: backtest.buy_hold_max_drawdown ?? null,
    annualized_sharpe: backtest.annualized_sharpe ?? null,
    buy_hold_sharpe: backtest.buy_hold_sharpe ?? null,
  };
}

async function submitChatQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed || chatState.isSending) return;
  if (chatState.isListening) stopVoiceInput(true);

  hideStarterPrompts();
  appendChatMessage("user", trimmed);
  chatState.history.push({ role: "user", content: trimmed });
  const pending = appendChatMessage("assistant", "Je reflechis...", { pending: true });
  setChatLoading(true);
  clearError();

  try {
    let streamedAnswer = "";
    let finalAnswer = "";
    const payload = {
      question: trimmed,
      debug: false,
      conversation_history: chatState.history,
      current_dashboard_state: buildCurrentDashboardState(),
    };

    await apiPostStream("/chat/ask/stream", payload, (event) => {
      if (event.type === "delta") {
        streamedAnswer += event.delta || "";
        updateChatMessage(pending, streamedAnswer || "...");
        return;
      }
      if (event.type === "done") {
        finalAnswer = event.answer || streamedAnswer;
        return;
      }
      if (event.type === "error") {
        throw new Error(event.detail || "Erreur streaming chatbot.");
      }
    });

    const answer = finalAnswer || streamedAnswer || "Aucune reponse exploitable n'a ete produite.";
    updateChatMessage(pending, answer);
    chatState.history.push({ role: "assistant", content: answer });
  } catch (error) {
    console.error(error);
    const fallback = "Le chatbot n'a pas pu repondre pour le moment. Verifie l'API backend et la disponibilite du modele.";
    updateChatMessage(pending, fallback);
    chatState.history.push({ role: "assistant", content: fallback });
  } finally {
    setChatLoading(false);
    if (el.chatInput) {
      el.chatInput.value = "";
      el.chatInput.focus();
    }
  }
}

function renderStatus(ok) {
  el.apiStatus.textContent = ok ? "API connected" : "API unavailable";
  el.apiStatus.classList.toggle("is-ok", ok);
  el.apiStatus.classList.toggle("is-error", !ok);
}

function renderCards() {
  const meta = state.latestForecast;
  const point = selectedForecastPoint() || meta;
  const riskPoint = state.riskSeries?.forecasts?.find((item) => Number(item.horizon) === 1);
  if (!meta || !point) return;

  el.varValue.textContent = fmtPct(point.var_forecast);
  el.esValue.textContent = fmtPct(point.es_forecast);
  el.returnValue.textContent = fmtPct(point.return_forecast, 3);
  el.meanValue.textContent = fmtPct(riskPoint?.mean_forecast, 3);
  el.volatilityValue.textContent = fmtPct(riskPoint?.volatility_forecast, 3);
  el.regimeValue.textContent = (meta.regime || "--").replace("_", " ");
  el.weightValue.textContent =
    meta.weight === null || meta.weight === undefined || Number.isNaN(Number(meta.weight))
      ? "Backtest allocation only"
      : `Backtest weight ${fmtNumber(meta.weight, 3)}`;
  el.varTarget.textContent = `Target ${formatDate(point.target_date)}`;
  el.meanTarget.textContent = `Target ${formatDate(riskPoint?.target_date)}`;
  el.volatilityTarget.textContent = `Target ${formatDate(riskPoint?.target_date)}`;
  el.runMeta.textContent = `Run ${formatDate(meta.run_date)} | Model ${meta.model_version || "--"} | Data ${meta.data_version || "--"}`;
}

function renderBacktest() {
  const b = state.backtest;
  if (!b) return;

  const violationRate = Number(b.violation_rate);
  const expectedRate = Number(b.expected_violation_rate ?? 0.05);
  const violationGap = Number.isFinite(violationRate) ? violationRate - expectedRate : null;
  const finalWealth = Number(b.final_wealth);
  const buyHoldWealth = Number(b.buy_hold_final_wealth);
  const maxDrawdown = Number(b.max_drawdown ?? b.current_drawdown);
  const buyHoldDrawdown = Number(b.buy_hold_max_drawdown);
  const sharpe = Number(b.annualized_sharpe);
  const buyHoldSharpe = Number(b.buy_hold_sharpe);
  const esResidual = Number(b.es_tail_residual_mean);

  el.violationRate.textContent = fmtPct(b.violation_rate);
  el.nViolations.textContent = fmtNumber(b.n_violations, 0);
  if (el.breachLabel) el.breachLabel.textContent = `${fmtNumber(b.n_observations, 0)} observations test`;
  if (el.esResidualMean) el.esResidualMean.textContent = fmtPct(b.es_tail_residual_mean, 3);
  el.finalWealth.textContent = fmtNumber(b.final_wealth, 3);
  el.drawdownValue.textContent = fmtPct(maxDrawdown);
  if (el.sharpeValue) el.sharpeValue.textContent = fmtNumber(b.annualized_sharpe, 2);

  if (el.violationStatus) {
    const gapText = violationGap === null ? "Ecart indisponible" : `${violationGap >= 0 ? "+" : ""}${fmtPct(violationGap, 2)} vs cible`;
    el.violationStatus.textContent = `${gapText} | cible ${fmtPct(expectedRate, 0)}`;
  }
  if (el.esResidualLabel) {
    el.esResidualLabel.textContent = Number.isFinite(esResidual)
      ? `${esResidual <= 0 ? "ES conservateur" : "ES trop optimiste"} en moyenne`
      : "Realise - ES en queue";
  }
  if (el.bhWealthLabel) {
    const delta = finalWealth - buyHoldWealth;
    el.bhWealthLabel.textContent = `${formatSignedNumber(delta, 3)} vs B&H (${fmtNumber(buyHoldWealth, 3)})`;
  }
  if (el.bhDrawdownLabel) {
    const delta = Math.abs(maxDrawdown) - Math.abs(buyHoldDrawdown);
    el.bhDrawdownLabel.textContent = `${delta <= 0 ? "Moins severe" : "Plus severe"} de ${fmtPct(Math.abs(delta), 2)} vs B&H`;
  }
  if (el.bhSharpeLabel) {
    const delta = sharpe - buyHoldSharpe;
    el.bhSharpeLabel.textContent = `${formatSignedNumber(delta, 2)} vs B&H (${fmtNumber(buyHoldSharpe, 2)})`;
  }

}

function formatSignedNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "--";
  return `${value >= 0 ? "+" : ""}${fmtNumber(value, digits)}`;
}

function formatReport(content) {
  if (!content) return "No report available.";
  const lines = content
    .replace(/```text|```markdown|```/g, "")
    .split("\n")
    .filter((line) => !line.trim().startsWith("Plot:"))
    .filter((line) => !line.includes(":\\"))
    .filter((line) => !line.includes(":/"));

  return lines.join("\n").trim();
}

function renderReport() {
  el.reportContent.textContent = formatReport(state.report?.content);
}

function chartBaseOptions(yTitle) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: "index" },
    layout: {
      padding: { right: 18 },
    },
    plugins: {
      legend: {
        display: true,
        position: "bottom",
        labels: { usePointStyle: true, boxWidth: 8, font: { family: "Poppins", size: 11 } },
      },
      tooltip: {
        backgroundColor: "#111827",
        padding: 12,
        titleFont: { family: "Poppins", size: 12, weight: "700" },
        bodyFont: { family: "Poppins", size: 11, weight: "500" },
        callbacks: {
          label(context) {
            return `${context.dataset.label}: ${fmtPrice(context.parsed.y)}`;
          },
        },
      },
      zoom: {
        pan: {
          enabled: true,
          mode: "x",
          threshold: 6,
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true,
          },
          drag: {
            enabled: false,
          },
          mode: "x",
        },
      },
    },
    scales: {
      x: {
        offset: true,
        grid: { display: false },
        ticks: { color: "#667085", maxTicksLimit: 8, font: { family: "Poppins", size: 10 } },
      },
      y: {
        title: { display: true, text: yTitle, color: "#667085", font: { family: "Poppins", size: 11 } },
        grid: { color: "#e8edf3" },
        ticks: { color: "#667085", font: { family: "Poppins", size: 10 } },
      },
    },
    elements: {
      point: { radius: 0, hoverRadius: 4 },
      line: { borderWidth: 2, tension: 0.18 },
    },
  };
}

function zoomChart(chartKey, factor) {
  const chart = state.charts[chartKey];
  if (!chart?.zoom) return;
  chart.zoom(factor);
}

function resetChartZoom(chartKey) {
  const chart = state.charts[chartKey];
  if (!chart?.resetZoom) return;
  chart.resetZoom();
}

function selectedForecastPoint() {
  return state.priceSeries?.forecasts?.find((item) => Number(item.horizon) === Number(state.selectedHorizon));
}

function horizontalLine(length, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return [];
  return Array.from({ length }, () => Number(value));
}

function forecastThresholdLine(_historyLength, totalLength, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return [];
  return horizontalLine(totalLength, value);
}

function renderPriceChart() {
  const series = state.priceSeries;
  if (!series?.observed?.length) return;

  const forecast = selectedForecastPoint() || series.forecasts?.[0];
  const observed = series.observed;
  const labels = observed.map((point) => point.date);
  const observedPrices = observed.map((point) => point.close);
  const lastPrice = observedPrices[observedPrices.length - 1];
  const rightPaddingLabels = ["", " "];
  const extendedLabels = forecast ? [...labels, forecast.target_date, ...rightPaddingLabels] : labels;

  const forecastLine = forecast
    ? [...Array(labels.length - 1).fill(null), lastPrice, forecast.price_forecast_proxy, ...rightPaddingLabels.map(() => null)]
    : [];
  const varLine = forecast ? forecastThresholdLine(labels.length, extendedLabels.length, forecast.var_price_proxy) : [];
  const esLine = forecast ? forecastThresholdLine(labels.length, extendedLabels.length, forecast.es_price_proxy) : [];

  const data = {
    labels: extendedLabels,
    datasets: [
      {
        label: "Observed price",
        data: forecast ? [...observedPrices, null, ...rightPaddingLabels.map(() => null)] : observedPrices,
        borderColor: "#111827",
        backgroundColor: "#111827",
      },
      {
        label: `Forecast proxy ${state.selectedHorizon}D`,
        data: forecastLine,
        borderColor: "#0f9f6e",
        backgroundColor: "#0f9f6e",
        pointRadius: 4,
        borderDash: [5, 4],
      },
      {
        label: `VaR 5% threshold ${state.selectedHorizon}D`,
        data: varLine,
        borderColor: "#d89b00",
        backgroundColor: "#d89b00",
        pointRadius: 0,
        pointHoverRadius: 0,
        borderDash: [6, 4],
      },
      {
        label: `ES 5% threshold ${state.selectedHorizon}D`,
        data: esLine,
        borderColor: "#d92d20",
        backgroundColor: "#d92d20",
        pointRadius: 0,
        pointHoverRadius: 0,
        borderDash: [6, 4],
      },
    ],
  };

  if (state.charts.price) {
    state.charts.price.data = data;
    state.charts.price.update();
    return;
  }

  state.charts.price = new Chart(document.getElementById("priceChart"), {
    type: "line",
    data,
    options: chartBaseOptions("MASI price"),
  });
}

function renderRiskChart() {
  const series = state.riskSeries;
  if (!series?.observed?.length) return;

  const forecast = series.forecasts?.find((item) => Number(item.horizon) === 1) || series.forecasts?.[0];
  const observed = series.observed;
  const labels = observed.map((point) => point.date);
  const lastMean = Number(observed[observed.length - 1].mean) * 100;
  const lastVolatility = Number(observed[observed.length - 1].volatility) * 100;
  const extendedLabels = forecast ? [...labels, forecast.target_date] : labels;

  const meanLine = forecast
    ? [...Array(labels.length - 1).fill(null), lastMean, Number(forecast.mean_forecast) * 100]
    : [];
  const volatilityLine = forecast
    ? [...Array(labels.length - 1).fill(null), lastVolatility, Number(forecast.volatility_forecast) * 100]
    : [];

  const data = {
    labels: extendedLabels,
    datasets: [
      {
        label: "Observed EGARCH mean",
        data: [...observed.map((point) => Number(point.mean) * 100), null],
        borderColor: "#111827",
        backgroundColor: "#111827",
      },
      {
        label: "Observed EGARCH volatility",
        data: [...observed.map((point) => Number(point.volatility) * 100), null],
        borderColor: "#ED1C24",
        backgroundColor: "#ED1C24",
      },
      {
        label: "Forecast mean 1D",
        data: meanLine,
        borderColor: "#111827",
        backgroundColor: "#111827",
        pointRadius: 4,
        borderDash: [5, 4],
      },
      {
        label: "Forecast volatility 1D",
        data: volatilityLine,
        borderColor: "#ED1C24",
        backgroundColor: "#ED1C24",
        pointRadius: 4,
        borderDash: [5, 4],
      },
    ],
  };

  const options = chartBaseOptions("Forecast metric (%)");
  options.plugins.tooltip.callbacks.label = (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(3)}%`;

  if (state.charts.risk) {
    state.charts.risk.data = data;
    state.charts.risk.options = options;
    state.charts.risk.update();
    return;
  }

  state.charts.risk = new Chart(document.getElementById("riskChart"), {
    type: "line",
    data,
    options,
  });
}

function renderWealthChart() {
  const rows = state.testPredictions || [];
  if (!rows.length || !rows[0].strategy_wealth) return;

  const labels = rows.map((row) => formatDate(row.date));
  const data = {
    labels,
    datasets: [
      {
        label: "Risk-Managed Simulation",
        data: rows.map((row) => row.strategy_wealth),
        borderColor: "#0f9f6e",
        backgroundColor: "transparent",
        borderWidth: 2,
      },
      {
        label: "Buy & Hold",
        data: rows.map((row) => row.buy_hold_wealth),
        borderColor: "#111827",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        borderDash: [4, 4],
      },
    ],
  };

  const options = chartBaseOptions("Simulated wealth (initial = 1.0)");
  options.elements.point.radius = 0;

  if (state.charts.wealth) {
    state.charts.wealth.data = data;
    state.charts.wealth.update();
    return;
  }

  state.charts.wealth = new Chart(document.getElementById("wealthChart"), {
    type: "line",
    data,
    options,
  });
}

function renderTestChart() {
  const rows = state.testPredictions || [];
  if (!rows.length) return;

  const labels = rows.map((row) => formatDate(row.date));
  const data = {
    labels,
    datasets: [
      {
        label: "Realized return",
        data: rows.map((row) => Number(row.realized_return) * 100),
        borderColor: "#111827",
        backgroundColor: "#111827",
      },
      {
        label: "Return forecast",
        data: rows.map((row) => Number(row.return_pred) * 100),
        borderColor: "#0f9f6e",
        backgroundColor: "#0f9f6e",
        borderDash: [5, 4],
      },
      {
        label: "VaR 5%",
        data: rows.map((row) => Number(row.var_pred) * 100),
        borderColor: "#f2c94c",
        backgroundColor: "#f2c94c",
      },
      {
        label: "ES 5%",
        data: rows.map((row) => Number(row.es_pred) * 100),
        borderColor: "#d92d20",
        backgroundColor: "#d92d20",
      },
    ],
  };
  const options = chartBaseOptions("Return (%)");
  options.plugins.tooltip.callbacks.label = (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;

  if (state.charts.test) {
    state.charts.test.data = data;
    state.charts.test.options = options;
    state.charts.test.update();
    return;
  }

  state.charts.test = new Chart(document.getElementById("testChart"), {
    type: "line",
    data,
    options,
  });
}

function renderTestTable() {
  const rows = (state.testPredictions || []).slice(-12).reverse();
  el.testPredictionRows.innerHTML = "";

  if (!rows.length) {
    el.testPredictionRows.innerHTML = '<tr><td colspan="6">No test predictions available.</td></tr>';
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const breach = Number(row.violation) === 1;
    tr.innerHTML = `
      <td>${formatDate(row.date)}</td>
      <td>${fmtPct(row.realized_return, 3)}</td>
      <td>${fmtPct(row.return_pred, 3)}</td>
      <td>${fmtPct(row.var_pred, 3)}</td>
      <td>${fmtPct(row.es_pred, 3)}</td>
      <td><span class="breach-pill ${breach ? "is-breach" : ""}">${breach ? "Yes" : "No"}</span></td>
    `;
    el.testPredictionRows.appendChild(tr);
  });
}

function downloadReport() {
  const content = state.report?.content || "";
  if (!content) {
    showError("No report content available to download.");
    return;
  }
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = state.report?.report_name || "latest_forecast_report.md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function renderAdmin() {
  if (!el.pipelineGrid || !el.pipelineRunRows) return;

  el.pipelineGrid.innerHTML = "";
  const hasActiveRun = state.admin.runs.some((run) => ["queued", "running"].includes(run.status));

  if (!state.admin.pipelines.length) {
    el.pipelineGrid.innerHTML = '<p class="admin-empty">Aucun pipeline disponible.</p>';
  }

  const grouped = groupPipelinesByCategory(state.admin.pipelines);
  grouped.forEach(([category, pipelines]) => {
    const section = document.createElement("section");
    section.className = "pipeline-group";
    section.innerHTML = `
      <h3>${escapeHtml(category)}</h3>
      <div class="pipeline-list"></div>
    `;
    const list = section.querySelector(".pipeline-list");
    pipelines.forEach((pipeline) => {
      const card = document.createElement("article");
      card.className = "pipeline-card";
      card.innerHTML = `
        <div>
          <div class="pipeline-card-header">
            <h4>${escapeHtml(pipeline.name)}</h4>
            <span>${escapeHtml(pipeline.estimated_duration || "")}</span>
          </div>
          <p>${escapeHtml(pipeline.description)}</p>
        </div>
        <button class="pipeline-run-button" type="button" data-pipeline-id="${escapeHtml(pipeline.id)}" ${hasActiveRun ? "disabled" : ""}>
          Lancer
        </button>
      `;
      list.appendChild(card);
    });
    el.pipelineGrid.appendChild(section);
  });

  el.pipelineRunRows.innerHTML = "";
  if (!state.admin.runs.length) {
    el.pipelineRunRows.innerHTML = '<tr><td colspan="5">Aucun pipeline lance depuis cette session.</td></tr>';
  } else {
    state.admin.runs.forEach((run) => {
      const tr = document.createElement("tr");
      tr.className = state.admin.selectedRunId === run.id ? "is-selected-run" : "";
      tr.dataset.runId = run.id;
      tr.innerHTML = `
        <td>${escapeHtml(run.pipeline_name)}</td>
        <td><span class="run-status run-status-${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></td>
        <td>${formatDateTime(run.started_at)}</td>
        <td>${run.finished_at ? formatDateTime(run.finished_at) : "--"}</td>
        <td>${run.return_code ?? "--"}</td>
      `;
      el.pipelineRunRows.appendChild(tr);
    });
  }

  document.querySelectorAll("[data-pipeline-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      await startPipeline(button.dataset.pipelineId);
    });
  });

  document.querySelectorAll("[data-run-id]").forEach((row) => {
    row.addEventListener("click", async () => {
      state.admin.selectedRunId = row.dataset.runId;
      await loadRunDetail(row.dataset.runId);
      renderAdmin();
    });
  });

  const selected = state.admin.runs.find((run) => run.id === state.admin.selectedRunId) || state.admin.runs[0];
  if (selected && !state.admin.selectedRunId) {
    state.admin.selectedRunId = selected.id;
    el.pipelineLog.textContent = selected.log_tail || "Logs en attente...";
  } else if (!selected && el.pipelineLog) {
    el.pipelineLog.textContent = "Aucun run selectionne.";
  }
}

function groupPipelinesByCategory(pipelines) {
  const order = ["Donnees", "Production", "Modele", "Recherche"];
  const groups = new Map();
  pipelines.forEach((pipeline) => {
    const category = pipeline.category || "Autres";
    if (!groups.has(category)) groups.set(category, []);
    groups.get(category).push(pipeline);
  });
  return Array.from(groups.entries()).sort((a, b) => {
    const aIndex = order.indexOf(a[0]);
    const bIndex = order.indexOf(b[0]);
    return (aIndex === -1 ? 99 : aIndex) - (bIndex === -1 ? 99 : bIndex);
  });
}

function formatDateTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("fr-FR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function loadAdmin() {
  if (!el.pipelineGrid) return;
  try {
    const data = await apiGet("/admin/pipelines");
    state.admin.pipelines = data.pipelines || [];
    state.admin.runs = data.runs || [];
    renderAdmin();
    updateAdminPolling();
  } catch (error) {
    console.error(error);
    showError(`Admin API error: ${error.message}`);
  }
}

async function loadRunDetail(runId) {
  if (!runId || !el.pipelineLog) return;
  try {
    const detail = await apiGet(`/admin/pipelines/runs/${runId}`);
    el.pipelineLog.textContent = detail.log_tail || "Logs en attente...";
  } catch (error) {
    console.error(error);
    el.pipelineLog.textContent = `Impossible de charger les logs: ${error.message}`;
  }
}

async function startPipeline(pipelineId) {
  if (!pipelineId) return;
  clearError();
  try {
    const run = await apiPost(`/admin/pipelines/${pipelineId}/run`, {});
    state.admin.selectedRunId = run.id;
    await loadAdmin();
    await loadRunDetail(run.id);
  } catch (error) {
    console.error(error);
    showError(`Pipeline error: ${error.message}`);
  }
}

async function uploadAdminData() {
  const file = el.adminDataFile?.files?.[0];
  if (!file) {
    if (el.adminUploadStatus) el.adminUploadStatus.textContent = "Selectionne d'abord un fichier.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  if (el.uploadDataButton) el.uploadDataButton.disabled = true;
  if (el.adminUploadStatus) el.adminUploadStatus.textContent = "Upload en cours...";
  clearError();

  try {
    const result = await apiUpload("/admin/data/upload", formData);
    if (el.adminUploadStatus) {
      el.adminUploadStatus.textContent = `Fichier envoye: ${result.filename} (${fmtNumber(result.size_bytes, 0)} octets). Pret pour le cleaning.`;
    }
    await loadAdmin();
  } catch (error) {
    console.error(error);
    if (el.adminUploadStatus) el.adminUploadStatus.textContent = `Upload echoue: ${error.message}`;
    showError(`Upload error: ${error.message}`);
  } finally {
    if (el.uploadDataButton) el.uploadDataButton.disabled = false;
  }
}

function updateAdminPolling() {
  const hasActiveRun = state.admin.runs.some((run) => ["queued", "running"].includes(run.status));
  if (hasActiveRun && !state.admin.pollTimer) {
    state.admin.pollTimer = window.setInterval(async () => {
      await loadAdmin();
      if (state.admin.selectedRunId) await loadRunDetail(state.admin.selectedRunId);
    }, 2500);
  }
  if (!hasActiveRun && state.admin.pollTimer) {
    window.clearInterval(state.admin.pollTimer);
    state.admin.pollTimer = null;
  }
}

async function loadDashboard() {
  clearError();
  setLoading(true);

  try {
    await apiGet("/health");
    renderStatus(true);

    const [latest, priceSeries, riskSeries, backtest, report, testPredictions] = await Promise.all([
      apiGet("/forecast/latest?horizon=1"),
      apiGet("/forecast/price-series?history_limit=260"),
      apiGet("/forecast/risk-series?history_limit=180"),
      apiGet("/backtest/latest"),
      apiGet("/report/latest"),
      apiGet("/backtest/test-predictions?limit=250"),
    ]);

    state.latestForecast = latest;
    state.priceSeries = priceSeries;
    state.riskSeries = riskSeries;
    state.backtest = backtest;
    state.report = report;
    state.testPredictions = testPredictions.predictions || [];

    renderCards();
    renderBacktest();
    renderReport();
    renderPriceChart();
    renderRiskChart();
    renderWealthChart();
    renderTestChart();
    renderTestTable();
  } catch (error) {
    console.error(error);
    renderStatus(false);
    showError(`API error: ${error.message}. Start FastAPI on ${API_BASE}.`);
  } finally {
    setLoading(false);
  }
}

document.querySelectorAll("[data-horizon]").forEach((button) => {
  button.addEventListener("click", () => {
    state.selectedHorizon = Number(button.dataset.horizon);
    document.querySelectorAll("[data-horizon]").forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");
    renderCards();
    renderPriceChart();
    renderRiskChart();
  });
});

document.querySelectorAll("[data-page-link]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    const pageId = link.dataset.pageLink;
    document.querySelectorAll("[data-page-link]").forEach((item) => item.classList.remove("is-active"));
    document.querySelectorAll(".page-section").forEach((section) => section.classList.remove("is-active"));
    link.classList.add("is-active");
    document.getElementById(pageId)?.classList.add("is-active");

    if (pageId === "backtestPage" && state.charts.test) {
      setTimeout(() => state.charts.test.resize(), 0);
    }
    if (pageId === "forecastPage" && state.charts.price) {
      setTimeout(() => state.charts.price.resize(), 0);
    }
    if (pageId === "forecastPage" && state.charts.risk) {
      setTimeout(() => state.charts.risk.resize(), 0);
    }
    if (pageId === "adminPage") {
      loadAdmin();
    }
  });
});

el.refreshButton.addEventListener("click", loadDashboard);
el.downloadReportButton.addEventListener("click", downloadReport);
if (el.refreshAdminButton) el.refreshAdminButton.addEventListener("click", loadAdmin);
if (el.adminDataFile) {
  el.adminDataFile.addEventListener("change", () => {
    const file = el.adminDataFile.files?.[0];
    if (el.adminFileName) el.adminFileName.textContent = file ? file.name : "Aucun fichier selectionne";
  });
}
if (el.adminUploadForm) {
  el.adminUploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await uploadAdminData();
  });
}
if (el.runCleaningButton) {
  el.runCleaningButton.addEventListener("click", async () => {
    await startPipeline("data_cleaning");
  });
}
if (el.chatButton) el.chatButton.addEventListener("click", openChatPanel);
if (el.chatOverlay) el.chatOverlay.addEventListener("click", closeChatPanel);
if (el.chatCloseButton) el.chatCloseButton.addEventListener("click", closeChatPanel);
if (el.chatCloseButton) el.chatCloseButton.innerHTML = "&times;";
if (el.chatVoiceButton) el.chatVoiceButton.addEventListener("click", toggleVoiceInput);
if (el.chatMessages) {
  const introParagraph = el.chatMessages.querySelector(".chat-message-intro .chat-message-shell p:last-child");
  if (introParagraph) {
    introParagraph.textContent =
      "Pose une question sur le forecast, la VaR, l'ES, le regime ou le backtest.";
  }
}
if (el.chatForm) {
  el.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitChatQuestion(el.chatInput?.value || "");
  });
}
if (el.chatInput) {
  el.chatInput.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      await submitChatQuestion(el.chatInput.value || "");
    }
    if (event.key === "Escape") closeChatPanel();
  });
}
document.querySelectorAll("[data-chat-prompt]").forEach((button) => {
  button.addEventListener("click", async () => {
    const prompt = button.dataset.chatPrompt || "";
    if (el.chatInput) el.chatInput.value = prompt;
    await submitChatQuestion(prompt);
  });
});
document.querySelectorAll("[data-chart-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const chartId = button.dataset.chartId;
    const action = button.dataset.chartAction;
    if (action === "zoom-in") zoomChart(chartId, 1.15);
    if (action === "zoom-out") zoomChart(chartId, 0.85);
    if (action === "reset") resetChartZoom(chartId);
  });
});
loadDashboard();
