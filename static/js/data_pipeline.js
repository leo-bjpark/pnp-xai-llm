/**
 * Data Pipeline: Raw | Processed. Python process(example) for map. 5 examples + "5 more" button.
 */

(function () {
  const pipelineId = document.body.dataset.pipelineId;
  if (!pipelineId) return;

  const statusEl = document.getElementById("pipeline-status");
  const pathInput = document.getElementById("hf-dataset-path");
  const configInput = document.getElementById("hf-config-name");
  const splitInput = document.getElementById("hf-split");
  const randomNInput = document.getElementById("hf-random-n");
  const seedInput = document.getElementById("hf-seed");
  const loadBtn = document.getElementById("btn-load-dataset");
  const loadMessage = document.getElementById("load-dataset-message");
  const processingCodeTextarea = document.getElementById("processing-code");
  const applyBtn = document.getElementById("btn-apply-processing");
  const processingMessage = document.getElementById("processing-message");
  const placeholderLeft = document.getElementById("data-preview-placeholder-left");
  const rawContent = document.getElementById("data-raw-content");
  const placeholderRight = document.getElementById("data-preview-placeholder");
  const processedContent = document.getElementById("data-processed-content");
  const rawDataNameCell = document.getElementById("raw-data-name-cell");
  const rawInfoCell = document.getElementById("raw-info-cell");
  const rawExampleCell = document.getElementById("raw-example-cell");
  const btnRawShowMore = document.getElementById("btn-raw-show-more");
  const processedInfoCell = document.getElementById("processed-info-cell");
  const processedExampleCell = document.getElementById("processed-example-cell");
  const btnProcessedShowMore = document.getElementById("btn-processed-show-more");
  const btnSaveDataVar = document.getElementById("btn-save-data-var");
  const dataVarAdditionalInput = document.getElementById("data-var-additional-naming");

  const EXAMPLES_STEP = 5;
  const INDENT_SPACES = 4;
  let rawVisibleCount = EXAMPLES_STEP;
  let processedVisibleCount = EXAMPLES_STEP;
  let currentDatasetInfo = null;
  let currentProcessedInfo = null;
  let currentRawDataName = "";
  let currentProcessingCode = "";

  function setStatus(status) {
    if (statusEl) statusEl.textContent = status;
  }

  function showMessage(el, text, type) {
    if (!el) return;
    el.textContent = text || "";
    el.className = "data-pipeline-message" + (type ? " " + type : "");
  }

  function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function cellContent(row, col) {
    const v = row[col];
    const str = v === null || v === undefined ? "" : (typeof v === "object" ? JSON.stringify(v) : String(v));
    const clip = str.length > 200 ? str.slice(0, 200) + "…" : str;
    return { str, clip };
  }

  function renderExampleRows(rows, columns, maxRows) {
    if (!rows?.length || !columns?.length) return "";
    const toShow = rows.slice(0, maxRows);
    let html = "<table class=\"data-sample-table\"><thead><tr>";
    columns.forEach((c) => { html += "<th>" + escapeHtml(c) + "</th>"; });
    html += "</tr></thead><tbody>";
    toShow.forEach((row) => {
      html += "<tr>";
      columns.forEach((c) => {
        const { str, clip } = cellContent(row, c);
        html += "<td title=\"" + escapeHtml(str) + "\">" + escapeHtml(clip) + "</td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    return html;
  }

  function appendRowsToTbody(tbody, rows, columns, fromIdx, toIdx) {
    if (!tbody || !rows?.length || !columns?.length) return;
    const end = Math.min(toIdx, rows.length);
    for (let i = fromIdx; i < end; i++) {
      const row = rows[i];
      const tr = document.createElement("tr");
      columns.forEach((c) => {
        const { str, clip } = cellContent(row, c);
        const td = document.createElement("td");
        td.title = str;
        td.textContent = clip;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
  }

  function updateRawExampleTable(appendOnly) {
    if (!rawExampleCell || !currentDatasetInfo?.sample_rows) return;
    const firstSplit = currentDatasetInfo.splits?.[0] || Object.keys(currentDatasetInfo.sample_rows)[0];
    const rows = firstSplit ? (currentDatasetInfo.sample_rows[firstSplit] || []) : [];
    const cols = currentDatasetInfo.columns || (rows[0] ? Object.keys(rows[0]) : []);
    const total = rows.length;
    if (total === 0) {
      rawExampleCell.innerHTML = "<p class=\"data-pipeline-placeholder\">No sample.</p>";
      if (btnRawShowMore) btnRawShowMore.style.display = "none";
      return;
    }
    if (!appendOnly) {
      const maxShow = Math.min(EXAMPLES_STEP, total);
      rawExampleCell.innerHTML = renderExampleRows(rows, cols, maxShow);
      rawVisibleCount = maxShow;
    } else {
      const tbody = rawExampleCell.querySelector("table tbody");
      if (tbody) {
        const fromIdx = rawVisibleCount;
        rawVisibleCount = Math.min(rawVisibleCount + EXAMPLES_STEP, total);
        appendRowsToTbody(tbody, rows, cols, fromIdx, rawVisibleCount);
      }
    }
    if (btnRawShowMore) {
      btnRawShowMore.style.display = rawVisibleCount < total ? "block" : "none";
    }
  }

  function updateProcessedExampleTable(appendOnly) {
    if (!processedExampleCell || !currentProcessedInfo?.sample_rows) return;
    const firstSplit = currentProcessedInfo.splits?.[0] || Object.keys(currentProcessedInfo.sample_rows)[0];
    const rows = firstSplit ? (currentProcessedInfo.sample_rows[firstSplit] || []) : [];
    const cols = currentProcessedInfo.columns || (rows[0] ? Object.keys(rows[0]) : []);
    const total = rows.length;
    if (total === 0) {
      processedExampleCell.innerHTML = "<p class=\"data-pipeline-placeholder\">No sample.</p>";
      if (btnProcessedShowMore) btnProcessedShowMore.style.display = "none";
      return;
    }
    if (!appendOnly) {
      const maxShow = Math.min(EXAMPLES_STEP, total);
      processedExampleCell.innerHTML = renderExampleRows(rows, cols, maxShow);
      processedVisibleCount = maxShow;
    } else {
      const tbody = processedExampleCell.querySelector("table tbody");
      if (tbody) {
        const fromIdx = processedVisibleCount;
        processedVisibleCount = Math.min(processedVisibleCount + EXAMPLES_STEP, total);
        appendRowsToTbody(tbody, rows, cols, fromIdx, processedVisibleCount);
      }
    }
    if (btnProcessedShowMore) {
      btnProcessedShowMore.style.display = processedVisibleCount < total ? "block" : "none";
    }
  }

  function renderBatch(datasetInfo, rawDataName, processedDatasetInfo, processingCode) {
    const hasData = datasetInfo && (datasetInfo.splits?.length || datasetInfo.columns?.length);
    const hasProcessed = processedDatasetInfo && (processedDatasetInfo.splits?.length || processedDatasetInfo.columns?.length);

    currentDatasetInfo = datasetInfo || null;
    currentProcessedInfo = processedDatasetInfo || null;
    currentRawDataName = rawDataName || "";
    currentProcessingCode = processingCode || "";
    rawVisibleCount = EXAMPLES_STEP;
    processedVisibleCount = EXAMPLES_STEP;

    if (placeholderLeft) placeholderLeft.style.display = hasData ? "none" : "block";
    if (rawContent) rawContent.style.display = hasData ? "block" : "none";
    if (placeholderRight) placeholderRight.style.display = hasProcessed ? "none" : "block";
    if (processedContent) processedContent.style.display = hasProcessed ? "block" : "none";

    if (!hasData) return;

    if (rawDataNameCell) rawDataNameCell.textContent = rawDataName || "—";

    if (rawInfoCell && datasetInfo) {
      let html = "<div class=\"data-meta-block\"><strong>Columns</strong><ul class=\"data-meta-list\">";
      (datasetInfo.columns || []).forEach((col) => {
        html += "<li><code>" + escapeHtml(col) + "</code></li>";
      });
      html += "</ul></div><div class=\"data-meta-block\"><strong>Row count</strong><ul class=\"data-meta-list\">";
      Object.entries(datasetInfo.num_rows || {}).forEach(([split, n]) => {
        html += "<li><span class=\"data-meta-split\">" + escapeHtml(split) + "</span>: " + escapeHtml(String(n)) + "</li>";
      });
      html += "</ul></div>";
      rawInfoCell.innerHTML = html;
    }

    updateRawExampleTable();

    if (processedInfoCell) {
      if (hasProcessed && currentProcessingCode) {
        processedInfoCell.innerHTML = "<pre class=\"data-batch-config\">" + escapeHtml(currentProcessingCode) + "</pre>";
      } else if (hasProcessed) {
        processedInfoCell.innerHTML = "<p class=\"data-pipeline-placeholder\">—</p>";
      } else {
        processedInfoCell.innerHTML = "<p class=\"data-pipeline-placeholder\">Apply processing to see info.</p>";
      }
    }

    if (processedExampleCell && !hasProcessed) {
      processedExampleCell.innerHTML = "<p class=\"data-pipeline-placeholder\">Apply processing to see first example.</p>";
    } else {
      updateProcessedExampleTable();
    }
  }

  if (btnRawShowMore) {
    btnRawShowMore.addEventListener("click", () => {
      updateRawExampleTable(true);
    });
  }

  if (btnProcessedShowMore) {
    btnProcessedShowMore.addEventListener("click", () => {
      updateProcessedExampleTable(true);
    });
  }

  // ----- Code editor behavior for Processing function (Python) -----
  if (processingCodeTextarea) {
    processingCodeTextarea.addEventListener("keydown", (e) => {
      const ta = processingCodeTextarea;
      if (e.key === "Tab") {
        e.preventDefault();
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        const indent = " ".repeat(INDENT_SPACES);
        const before = ta.value.slice(0, start);
        const after = ta.value.slice(end);
        ta.value = before + indent + after;
        ta.selectionStart = ta.selectionEnd = start + INDENT_SPACES;
        return;
      }
      if (e.key === "Enter") {
        const start = ta.selectionStart;
        const textBefore = ta.value.slice(0, start);
        const lineStart = textBefore.lastIndexOf("\n") + 1;
        const currentLine = textBefore.slice(lineStart);
        const match = currentLine.match(/^(\s*)/);
        const indent = match ? match[1] : "";
        const afterCursor = ta.value.slice(ta.selectionEnd);
        const nextChar = afterCursor.charAt(0);
        let extra = "";
        if (nextChar === "}" || nextChar === ")" || nextChar === "]") {
          extra = "";
        } else if (currentLine.trim().endsWith(":") || currentLine.trim().endsWith("(")) {
          extra = " ".repeat(INDENT_SPACES);
        }
        e.preventDefault();
        const inserted = "\n" + indent + extra;
        const newStart = start + inserted.length;
        ta.value = textBefore + inserted + afterCursor;
        ta.selectionStart = ta.selectionEnd = newStart;
        return;
      }
    });
    processingCodeTextarea.setAttribute("spellcheck", "false");
  }

  if (loadBtn && pathInput) {
    loadBtn.addEventListener("click", async () => {
      const path = pathInput.value.trim();
      if (!path) {
        showMessage(loadMessage, "Enter a dataset path.", "error");
        return;
      }
      const config_name = (configInput && configInput.value.trim()) || undefined;
      const split = (splitInput && splitInput.value.trim()) || undefined;
      const randomN = randomNInput && randomNInput.value.trim() !== "" ? parseInt(randomNInput.value, 10) : undefined;
      const seed = seedInput && seedInput.value.trim() !== "" ? parseInt(seedInput.value, 10) : undefined;

      showMessage(loadMessage, "Loading…");
      try {
        const res = await fetch(`/api/dataset-pipelines/${pipelineId}/load`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            path,
            hf_dataset_path: path,
            config: config_name,
            config_name: config_name,
            split: split || (splitInput && splitInput.value ? splitInput.value : null),
            random_n: !isNaN(randomN) ? randomN : undefined,
            seed: !isNaN(seed) ? seed : undefined,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          showMessage(loadMessage, data.error || "Load failed", "error");
          return;
        }
        showMessage(loadMessage, "Loaded: " + (data.dataset_info?.splits?.join(", ") || path), "success");
        setStatus("loaded");
        const code = (processingCodeTextarea && processingCodeTextarea.value) || "";
        renderBatch(data.dataset_info, path, window.PNP_PROCESSED_DATASET_INFO || null, code);
      } catch (err) {
        showMessage(loadMessage, err.message || "Load failed", "error");
      }
    });
  }

  if (applyBtn && processingCodeTextarea) {
    applyBtn.addEventListener("click", async () => {
      const code = (processingCodeTextarea.value || "").trim();
      if (!code) {
        showMessage(processingMessage, "Enter Python code defining process(example).", "error");
        return;
      }
      showMessage(processingMessage, "Applying…");
      try {
        const res = await fetch(`/api/dataset-pipelines/${pipelineId}/process`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ processing_code: code }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          showMessage(processingMessage, data.error || "Process failed", "error");
          return;
        }
        showMessage(processingMessage, "Processing applied.", "success");
        setStatus("processed");
        currentProcessedInfo = data.processed_dataset_info || null;
        currentProcessingCode = code;
        window.PNP_PROCESSED_DATASET_INFO = currentProcessedInfo;
        processedVisibleCount = EXAMPLES_STEP;
        if (processedContent) processedContent.style.display = "block";
        if (placeholderRight) placeholderRight.style.display = "none";
        if (processedInfoCell) processedInfoCell.innerHTML = "<pre class=\"data-batch-config\">" + escapeHtml(code) + "</pre>";
        updateProcessedExampleTable();
      } catch (err) {
        showMessage(processingMessage, err.message || "Failed", "error");
      }
    });
  }

  const SAVE_BTN_LABEL = "Save To Variable";
  if (btnSaveDataVar) {
    btnSaveDataVar.addEventListener("click", async () => {
      const btn = btnSaveDataVar;
      const originalText = btn.textContent;
      try {
        btn.disabled = true;
        btn.textContent = "Save To Variable...";
        const additionalNaming = (dataVarAdditionalInput && dataVarAdditionalInput.value) ? dataVarAdditionalInput.value.trim() : "";
        const res = await fetch("/api/data-vars/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pipeline_id: pipelineId, additional_naming: additionalNaming || undefined }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || "Save failed");
        if (typeof window.refreshDataVarsList === "function") window.refreshDataVarsList();
        btn.textContent = "Variable Saved ✅";
        setTimeout(() => {
          btn.textContent = SAVE_BTN_LABEL;
          btn.disabled = false;
        }, 2000);
      } catch (err) {
        btn.textContent = originalText;
        btn.disabled = false;
        alert("Save failed: " + err.message);
      }
    });
  }

  if (window.PNP_DATASET_INFO && (window.PNP_DATASET_INFO.splits?.length || window.PNP_DATASET_INFO.columns?.length)) {
    renderBatch(
      window.PNP_DATASET_INFO,
      window.PNP_RAW_DATA_NAME || "",
      window.PNP_PROCESSED_DATASET_INFO || null,
      window.PNP_PROCESSING_CODE || (processingCodeTextarea && processingCodeTextarea.value) || ""
    );
  }
})();
