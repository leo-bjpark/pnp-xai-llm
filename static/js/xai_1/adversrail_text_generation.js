(function () {
  "use strict";

  const rowBody = document.getElementById("adversarial-rows-body");
  const template = document.getElementById("adversarial-row-template");
  const hiddenInput = document.querySelector("[data-task-input='adversarial_rows']");
  const addRowBtn = document.getElementById("adversarial-add-row");
  const savedRows = Array.isArray(window.PNP_ADVERSARIAL_PRESET_ROWS)
    ? window.PNP_ADVERSARIAL_PRESET_ROWS
    : [];
  const savedResults = Array.isArray(window.PNP_ADVERSARIAL_PAST_RESULTS)
    ? window.PNP_ADVERSARIAL_PAST_RESULTS
    : [];
  const escapeFn = typeof escapeHtml === "function" ? escapeHtml : (s) => String(s ?? "");
  const DEFAULT_RESULT_TEXT = "Waiting for RUN";
  const DEFAULT_ROWS = [
    {
      input_string: "오늘 메뉴는 뭐로할까?",
      target_text: "자장면",
      seed_text: "차분하게 메뉴를 추천해줘",
    },
    {
      input_string: "오늘 메뉴는 뭐로할까?",
      target_text: "짬뽕",
      seed_text: "바삭한 식감의 메뉴로 추천해줘",
    },
  ];
  let rowCounter = 0;
  const COPY_RESET_MS = 1200;

  function buildPrefixedInput(prefix, input) {
    const parts = [];
    const safePrefix = String(prefix || "").trim();
    const safeInput = String(input || "").trim();
    if (safePrefix) parts.push(safePrefix);
    if (safeInput) parts.push(safeInput);
    return parts.join(" ");
  }

  function setButtonFlash(button, text) {
    if (!button) return;
    const original = button.textContent;
    button.textContent = text;
    button.setAttribute("disabled", "true");
    window.setTimeout(() => {
      button.textContent = original;
      button.removeAttribute("disabled");
    }, COPY_RESET_MS);
  }

  async function writeClipboard(text) {
    if (!text) return false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch (err) {
      // Fall back below.
    }
    const temp = document.createElement("textarea");
    temp.value = text;
    temp.setAttribute("readonly", "true");
    temp.style.position = "absolute";
    temp.style.left = "-9999px";
    document.body.appendChild(temp);
    temp.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (err) {
      ok = false;
    }
    document.body.removeChild(temp);
    return ok;
  }

  function setLastChatRowInput(value) {
    if (!chatRowsContainer) return false;
    const rows = Array.from(chatRowsContainer.querySelectorAll(".adversarial-test-row"));
    const lastRow = rows[rows.length - 1];
    if (!lastRow) return false;
    const textarea = lastRow.querySelector(".adversarial-test-row-input");
    if (!textarea) return false;
    textarea.value = value;
    textarea.focus();
    return true;
  }

  function createRow(data = {}) {
    if (!template || !rowBody) return null;
    const clone = template.content.firstElementChild.cloneNode(true);
    const rowId = data.row_id || `row-${++rowCounter}`;
    clone.dataset.rowId = rowId;

    const inputEl = clone.querySelector("[data-adversarial-input]");
    const targetEl = clone.querySelector("[data-adversarial-target]");
    const seedEl = clone.querySelector("[data-adversarial-seed]");
    const resultEl = clone.querySelector("[data-adversarial-result]");

    if (inputEl) {
      inputEl.value = data.input_string || data.input || "";
      inputEl.addEventListener("input", updateRowsInput);
    }
    if (targetEl) {
      targetEl.value = data.target_text || data.target || "";
      targetEl.addEventListener("input", updateRowsInput);
    }
    if (seedEl) {
      seedEl.value = data.seed_text || data.seed || "";
      seedEl.addEventListener("input", updateRowsInput);
    }

    const removeBtn = clone.querySelector("[data-adversarial-remove]");
    if (removeBtn) {
      removeBtn.addEventListener("click", (event) => {
        event.preventDefault();
        clone.remove();
        updateRowsInput();
      });
    }

    const copyBtn = clone.querySelector("[data-adversarial-copy]");
    if (copyBtn) {
      copyBtn.addEventListener("click", async (event) => {
        event.preventDefault();
        const prefixText = clone.dataset.prefixText || "";
        const inputValue = clone.querySelector("[data-adversarial-input]")?.value || "";
        if (!prefixText) {
          setButtonFlash(copyBtn, "Run first");
          return;
        }
        const combined = buildPrefixedInput(prefixText, inputValue);
        if (!combined) {
          setButtonFlash(copyBtn, "Empty");
          return;
        }
        const ok = await writeClipboard(combined);
        if (ok) {
          ensureChatRows();
          setLastChatRowInput(combined);
          setButtonFlash(copyBtn, "Copied");
        } else {
          setButtonFlash(copyBtn, "Failed");
        }
      });
    }

    if (resultEl) {
      resultEl.textContent = DEFAULT_RESULT_TEXT;
      resultEl.classList.add("adversarial-row-result-empty");
    }

    rowBody.appendChild(clone);
    updateRowsInput();
    return clone;
  }

  if (rowBody) {
    rowBody.addEventListener("click", (event) => {
      const removeBtn = event.target.closest("[data-adversarial-remove]");
      if (!removeBtn) return;
      event.preventDefault();
      const row = removeBtn.closest(".adversarial-row");
      if (!row) return;
      row.remove();
      updateRowsInput();
    });
  }

  function updateRowsInput() {
    if (!hiddenInput || !rowBody) return;
    const rows = Array.from(rowBody.querySelectorAll(".adversarial-row")).map((rowEl) => {
      const rowId = rowEl.dataset.rowId || "";
      const inputValue = rowEl.querySelector("[data-adversarial-input]")?.value || "";
      const targetValue = rowEl.querySelector("[data-adversarial-target]")?.value || "";
      const seedSentence = rowEl.querySelector("[data-adversarial-seed]")?.value || "";
      return {
        row_id: rowId,
        input_string: inputValue,
        target_text: targetValue,
        seed_text: seedSentence,
      };
    });
    hiddenInput.value = JSON.stringify(rows);
  }

  function setRowResult(rowId, result) {
    if (!rowBody || !rowId) return;
    const rowEl = rowBody.querySelector(`.adversarial-row[data-row-id="${rowId}"]`);
    if (!rowEl) return;
    const resultEl = rowEl.querySelector("[data-adversarial-result]");
    if (!resultEl) return;

    if (result?.error) {
      resultEl.innerHTML = `<span class="adversarial-result-error">${escapeFn(result.error)}</span>`;
      resultEl.classList.remove("adversarial-row-result-empty");
      return;
    }

    const pieces = [];
    const prefixText = result.prefix_text || (Array.isArray(result.prefix_tokens) && result.prefix_tokens.join(" ")) || "";
    if (prefixText) pieces.push(`<span><strong>Prefix</strong> ${escapeFn(prefixText)}</span>`);
    if (typeof result.loss === "number") {
      pieces.push(`<span><strong>Loss</strong> ${escapeFn(result.loss.toFixed(3))}</span>`);
    }
    resultEl.innerHTML = pieces.length ? pieces.join("") : escapeFn("No prefix found");
    resultEl.classList.remove("adversarial-row-result-empty");
    rowEl.dataset.prefixText = prefixText;
  }

  function refreshResults(results) {
    if (!Array.isArray(results)) return;
    results.forEach((result) => {
      setRowResult(result.row_id, result);
    });
  }

  function ensureRows() {
    const initialRows = savedRows.length ? savedRows : DEFAULT_ROWS;
    if (savedRows.length) {
      savedRows.forEach((row) => createRow(row));
    } else {
      DEFAULT_ROWS.forEach((row) => createRow(row));
    }
    if (!rowBody.querySelector(".adversarial-row")) {
      createRow();
    }
    updateRowsInput();
  }

  function renderSummary(results) {
    if (!Array.isArray(results) || !results.length) {
      return "<p>No adversarial rows available yet.</p>";
    }
    const lines = results.map((result) => {
      const target = result.target_text || result.target || "—";
      const prefix = result.prefix_text || (Array.isArray(result.prefix_tokens) && result.prefix_tokens.join(" ")) || "—";
      const extras = [];
      if (typeof result.loss === "number") extras.push(`loss ${result.loss.toFixed(3)}`);
      if (result.error) extras.push(`error ${escapeFn(result.error)}`);
      return `<li><span class="adversarial-summary-key">${escapeFn(target)}</span>→ ${escapeFn(prefix)}${extras.length ? ` · ${extras.join(" · ")}` : ""}</li>`;
    });
    return `<ul class="adversarial-summary-list">${lines.join("")}</ul>`;
  }

  window.PNP_renderAdversarialResultHTML = function (res, esc) {
    const results = Array.isArray(res?.adversarial_results) ? res.adversarial_results : [];
    const summary = renderSummary(results);
    refreshResults(results);
    return `
      <div class="results-completion-wrap">
        <h3>Adversarial search</h3>
        ${summary}
        <details class="results-completion-meta">
          <summary>Parameters &amp; full result</summary>
          <pre class="results-json">${esc(JSON.stringify(res || {}, null, 2))}</pre>
        </details>
      </div>
    `;
  };

  window.PNP_applyAdversarialResults = function (res) {
    const results = Array.isArray(res?.adversarial_results) ? res.adversarial_results : [];
    refreshResults(results);
  };

  const testPanel = document.getElementById("adversarial-test-panel");
  const testButton = document.getElementById("adversarial-test-run");
  const testStatus = document.getElementById("adversarial-test-status");
  const chatRowsContainer = document.getElementById("adversarial-test-rows");
  const chatRowTemplate = document.getElementById("adversarial-test-row-template");
  const addChatButton = document.getElementById("adversarial-add-chat");
  const maxTokensInput = document.getElementById("adversarial-max-tokens");
  const topPInput = document.getElementById("adversarial-top-p");
  const temperatureInput = document.getElementById("adversarial-temperature");
  const historyContainer = document.getElementById("adversarial-test-history");
  const sidebarModelSelect = document.getElementById("sidebar-model");
  const sidebarTreatmentSelect = document.getElementById("sidebar-treatment");
  let testRunning = false;
  const API = window.PNP_API;
  let chatRowCounter = 0;

  function normalizeNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function chatRowCount() {
    if (!chatRowsContainer) return 0;
    return chatRowsContainer.querySelectorAll(".adversarial-test-row").length;
  }

  function refreshRowTitles() {
    if (!chatRowsContainer) return;
    chatRowsContainer.querySelectorAll(".adversarial-test-row").forEach((row, idx) => {
      const titleEl = row.querySelector(".adversarial-test-row-title");
      if (titleEl) titleEl.textContent = `Turn ${idx + 1}`;
    });
  }

  function createChatRow(value = "") {
    if (!chatRowsContainer || !chatRowTemplate) return;
    const clone = chatRowTemplate.content.firstElementChild.cloneNode(true);
    const textarea = clone.querySelector(".adversarial-test-row-input");
    if (textarea) {
      textarea.value = value;
    }
    const outputEl = clone.querySelector("[data-adversarial-output]");
    if (outputEl) {
      outputEl.textContent = "대기 중";
    }
    const removeBtn = clone.querySelector(".adversarial-test-row-remove");
    if (removeBtn) {
      removeBtn.addEventListener("click", (event) => {
        event.preventDefault();
        clone.remove();
        refreshRowTitles();
      });
    }
    chatRowsContainer.appendChild(clone);
    refreshRowTitles();
  }

  if (chatRowsContainer) {
    chatRowsContainer.addEventListener("click", (event) => {
      const removeBtn = event.target.closest(".adversarial-test-row-remove");
      if (!removeBtn) return;
      event.preventDefault();
      const row = removeBtn.closest(".adversarial-test-row");
      if (!row) return;
      row.remove();
      refreshRowTitles();
    });
  }

  function ensureChatRows() {
    if (!chatRowsContainer) return;
    if (!chatRowCount()) {
      createChatRow();
    }
  }

  let historyCounter = 0;

  function getChatMessages() {
    if (!chatRowsContainer) return [];
    return Array.from(chatRowsContainer.querySelectorAll(".adversarial-test-row-input"))
      .map((ta) => (ta.value || "").trim())
      .filter(Boolean);
  }

  function setLastRowOutput(text) {
    if (!chatRowsContainer) return;
    const rows = Array.from(chatRowsContainer.querySelectorAll(".adversarial-test-row"));
    const lastRow = rows[rows.length - 1];
    if (!lastRow) return;
    const outputEl = lastRow.querySelector("[data-adversarial-output]");
    if (outputEl) outputEl.textContent = text;
  }

  function appendHistoryEntry(prompt, generated) {
    if (!historyContainer) return;
    historyCounter += 1;
    const entry = document.createElement("div");
    entry.className = "adversarial-test-history-entry";
    const pair = document.createElement("div");
    pair.className = "adversarial-test-history-pair";

    const promptCell = document.createElement("div");
    promptCell.className = "adversarial-test-history-cell";
    const promptLabel = document.createElement("span");
    promptLabel.className = "adversarial-test-history-label";
    promptLabel.textContent = `Turn ${historyCounter}`;
    const promptEl = document.createElement("p");
    promptEl.className = "adversarial-test-history-prompt";
    promptEl.textContent = prompt;
    promptCell.appendChild(promptLabel);
    promptCell.appendChild(promptEl);

    const responseCell = document.createElement("div");
    responseCell.className = "adversarial-test-history-cell";
    const responseLabel = document.createElement("span");
    responseLabel.className = "adversarial-test-history-label";
    responseLabel.textContent = `Turn ${historyCounter} output`;
    const responseEl = document.createElement("p");
    responseEl.className = "adversarial-test-history-response";
    responseEl.textContent = generated;
    responseCell.appendChild(responseLabel);
    responseCell.appendChild(responseEl);

    pair.appendChild(promptCell);
    pair.appendChild(responseCell);

    entry.appendChild(pair);
    if (historyContainer.firstChild && historyContainer.firstChild.tagName === "EM") {
      historyContainer.innerHTML = "";
    }
    historyContainer.prepend(entry);
  }

  let loadingEntry = null;

  function showLoadingIndicator() {
    if (!historyContainer) return;
    removeLoadingIndicator();
    loadingEntry = document.createElement("div");
    loadingEntry.className = "adversarial-test-history-entry adversarial-test-history-loading";
    loadingEntry.textContent = "생성 중...";
    historyContainer.prepend(loadingEntry);
  }

  function removeLoadingIndicator() {
    if (loadingEntry && loadingEntry.parentNode) {
      loadingEntry.parentNode.removeChild(loadingEntry);
    }
    loadingEntry = null;
  }

  async function runTestGeneration() {
    if (testRunning) return;
    if (!testStatus) return;
    const messages = getChatMessages();
    if (!messages.length) {
      testStatus.textContent = "대화 메시지를 추가하세요.";
      return;
    }
    const model = sidebarModelSelect ? sidebarModelSelect.value : "";
    const treatment = sidebarTreatmentSelect ? (sidebarTreatmentSelect.value || "").trim() : "";
    if (!API) {
      testStatus.textContent = "API 준비되지 않음";
      return;
    }
    try {
      testRunning = true;
      testButton?.setAttribute("disabled", "true");
      const lockTargets = Array.from(
        chatRowsContainer?.querySelectorAll(".adversarial-test-row") || []
      );
      lockTargets.forEach((row) => {
        row.dataset.lockPending = "true";
        row.querySelector(".adversarial-test-row-input")?.setAttribute("disabled", "true");
      });
      testStatus.textContent = "Generating…";
      showLoadingIndicator();
      await API.loadModel({ model, treatment });
      const payload = {
        model,
        treatment,
        input_setting: {
          messages: messages.map((text) => ({ role: "user", content: text })),
          temperature: normalizeNumber(temperatureInput?.value, 0.7),
          max_new_tokens: normalizeNumber(maxTokensInput?.value, 256),
          top_p: normalizeNumber(topPInput?.value, 1.0),
        },
      };
      const res = await API.run(payload);
      if (!res.ok || res.error) {
        throw new Error(res.error || "생성 실패");
      }
      const generated = res.generated_text ?? "응답이 없습니다.";
      setLastRowOutput(generated);
      testStatus.textContent = "완료";
      lockTargets.forEach((row) => {
        if (row.dataset.lockPending === "true") {
          row.dataset.locked = "true";
          delete row.dataset.lockPending;
        }
      });
    } catch (err) {
      const message = err?.message || "오류";
      testStatus.textContent = message;
      setLastRowOutput(message);
      chatRowsContainer?.querySelectorAll(".adversarial-test-row").forEach((row) => {
        if (row.dataset.lockPending === "true") {
          delete row.dataset.lockPending;
          row.querySelector(".adversarial-test-row-input")?.removeAttribute("disabled");
        }
      });
    } finally {
      testRunning = false;
      testButton?.removeAttribute("disabled");
      removeLoadingIndicator();
    }
  }

  if (addChatButton) {
    addChatButton.addEventListener("click", (event) => {
      event.preventDefault();
      createChatRow();
    });
  }

  if (testButton) {
    testButton.addEventListener("click", (event) => {
      event.preventDefault();
      runTestGeneration();
    });
  }

  if (addRowBtn) {
    addRowBtn.addEventListener("click", (event) => {
      event.preventDefault();
      createRow();
    });
  }

  ensureRows();
  refreshResults(savedResults);
  ensureChatRows();
  ensureHistoryEffect();
})();
