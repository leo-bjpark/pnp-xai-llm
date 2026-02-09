/**
 * PnP-XAI-LLM - Main app logic
 * - Session management (Loaded Model + Treatment)
 * - Create XAI = new session
 * - Run with mismatch -> confirm modal
 * - Task persistence
 */

(function () {
  "use strict";

  const API = {
    tasks: () => fetch("/api/tasks").then((r) => r.json()).then((d) => d.tasks || d),
    tasksWithMeta: () => fetch("/api/tasks").then((r) => r.json()),
    task: (id) => fetch(`/api/tasks/${id}`).then((r) => r.json()),
    createTask: (data) =>
      fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json()),
    updateTask: (taskId, data) =>
      fetch(`/api/tasks/${taskId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json()),
    session: () => fetch("/api/session").then((r) => r.json()),
    setSession: (data) =>
      fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json()),
    loadModel: (data) =>
      fetch("/api/load_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json()),
    run: (data) =>
      fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }).then((r) => r.json().then((body) => ({ ok: r.ok, ...body }))),
    modelStatus: (model) =>
      fetch(`/api/model_status?model=${encodeURIComponent(model)}`).then((r) => r.json()),
    modelLayerNames: (model) =>
      fetch(`/api/model_layer_names?model=${encodeURIComponent(model)}`).then((r) => r.json()),
    cudaEnvGet: () => fetch("/api/cuda_env").then((r) => r.json()),
    cudaEnvSet: (value) =>
      fetch("/api/cuda_env", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
      }).then((r) => r.json()),
  };

  // State
  let session = { loaded_model: null, treatment: null };
  let loadInProgress = false;
  let pendingRunAfterConfirm = null;
  let currentTaskLevel = "0.1"; // Selected XAI level when creating task
  let runAbortController = null; // abort the current /api/run request when user clicks Stop
  let taskLinkNavigateTimeout = null; // delay single-click navigate so double-click can cancel it for rename
  let modelSpecTooltipEl = null;
  let modelSpecShowTimeout = null;
  let modelSpecHideTimeout = null;
  let modelSpecCache = {}; // modelKey -> status
  let cudaPanelHideTimeout = null;

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const el = {
    btnCreateTask: $("#btn-create-task"),
    createTaskWrap: document.querySelector(".create-task-wrap"),
    createTaskDropdown: $("#create-task-dropdown"),
    loadedModelDisplay: $("#loaded-model-display"),
    loadedModelText: $("#loaded-model-text"),
    taskPanelList: $("#task-panel-list"),
    inputSettingPanel: $("#input-setting-panel"),
    inputSettingTrigger: $("#input-setting-trigger"),
    sidebarModel: $("#sidebar-model"),
    sidebarTreatment: $("#sidebar-treatment"),
    btnCreateTreatment: $("#btn-create-treatment"),
    createTreatmentDropdown: $("#create-treatment-dropdown"),
    treatmentStatusIndicator: $("#treatment-status-indicator"),
    treatmentPanel: $("#treatment-panel"),
    treatmentPanelClose: $("#treatment-panel-close"),
    treatmentResidualVar: $("#treatment-residual-var"),
    treatmentLayerGrid: $("#treatment-layer-grid"),
    treatmentAlpha: $("#treatment-alpha"),
    treatmentDelta: $("#treatment-delta"),
    treatmentNormalize: $("#treatment-normalize"),
    btnTreatmentApply: $("#btn-treatment-apply"),
    btnTreatmentClear: $("#btn-treatment-clear"),
    btnLoadModel: $("#btn-load-model"),
    btnRun: $("#btn-run"),
    resultsPlaceholder: $("#results-placeholder"),
    resultsContent: $("#results-content"),
    modalConfirm: $("#modal-confirm-load"),
    modalMessage: $("#modal-message"),
    modalCancel: $("#modal-cancel"),
    modalConfirmBtn: $("#modal-confirm"),
    btnCudaSetting: $("#btn-cuda-setting"),
    cudaSettingPanel: $("#cuda-setting-panel"),
    cudaVisibleDevicesInput: $("#cuda-visible-devices-input"),
    btnCudaExport: $("#btn-cuda-export"),
    cudaSettingEchoValue: $("#cuda-setting-echo-value"),
  };

  // Debug: Treatment / Steering DOM 상태 확인
  console.log("[treatment:init]", {
    sidebarTreatment: el.sidebarTreatment,
    btnCreateTreatment: el.btnCreateTreatment,
    createTreatmentDropdown: el.createTreatmentDropdown,
    treatmentPanel: el.treatmentPanel,
  });

  // ----- Sync UI with session -----
  function updateSessionUI() {
    const wantModel = session.loaded_model || "";
    const wantHasModel = !!session.loaded_model;
    const curText = (el.loadedModelText && el.loadedModelText.textContent) || "";
    const curHasModel = el.loadedModelDisplay && el.loadedModelDisplay.classList.contains("has-model");
    const displayValue = wantModel || "—";
    if (curText !== displayValue || curHasModel !== wantHasModel) {
      if (el.loadedModelDisplay) {
        if (wantHasModel) el.loadedModelDisplay.classList.add("has-model");
        else el.loadedModelDisplay.classList.remove("has-model");
      }
      if (el.loadedModelText) el.loadedModelText.textContent = displayValue;
    }
    updateLoadButtonState();
    updateInputSettingTriggerText();
  }

  function xaiLevelToTaskTypeLabel(level) {
    const map = { "0.1.1": "Completion", "0.1.2": "Conversation", "1.0.1": "Response Attribution" };
    return (map[level] != null ? map[level] : (level || "—"));
  }

  // Input Setting trigger: Task type (Completion|Conversation) | Model | Treatment | User-set name
  function updateInputSettingTriggerText() {
    const trigger = el.inputSettingTrigger;
    if (!trigger) return;
    const taskType = trigger.dataset.taskType || "—";
    const userName = trigger.dataset.taskName != null && trigger.dataset.taskName !== "" ? trigger.dataset.taskName : "—";
    const onTaskPage = !!window.PNP_CURRENT_TASK_ID;
    const hasTaskModel = onTaskPage && trigger.dataset.taskModel != null && trigger.dataset.taskModel !== "";
    const model = hasTaskModel
      ? trigger.dataset.taskModel
      : (onTaskPage ? "None" : ((el.sidebarModel?.value || "").trim() || "—"));
    const treatment = (onTaskPage && trigger.dataset.taskTreatment != null)
      ? (trigger.dataset.taskTreatment || "None")
      : (onTaskPage ? "None" : ((el.sidebarTreatment?.value || "").trim() || "None"));
    trigger.textContent = `${taskType}  |  ${model}  |  ${treatment}  |  ${userName}`;
  }

  function updateTreatmentStatusUI() {
    const indicator = el.treatmentStatusIndicator;
    if (!indicator) return;
    const cfg = parseCurrentTreatmentJSON();
    if (cfg) {
      indicator.textContent = "Simple Steering: Active";
      indicator.classList.add("active");
    } else {
      indicator.textContent = "Simple Steering: Inactive";
      indicator.classList.remove("active");
    }
  }

  // Load button: disabled while loading, or when selected model === loaded model
  function updateLoadButtonState() {
    if (!el.btnLoadModel) return;
    const selected = el.sidebarModel ? el.sidebarModel.value : "";
    if (loadInProgress) {
      el.btnLoadModel.disabled = true;
      el.btnLoadModel.textContent = "Loading…";
    } else if (selected && selected === session.loaded_model) {
      el.btnLoadModel.disabled = true;
      el.btnLoadModel.textContent = "Load";
    } else {
      el.btnLoadModel.disabled = false;
      el.btnLoadModel.textContent = "Load";
    }
  }

  // ----- Load Model (left sidebar) -----
  el.sidebarModel?.addEventListener("change", () => {
    updateLoadButtonState();
    updateInputSettingTriggerText();
  });
  el.sidebarTreatment?.addEventListener("input", () => {
    updateInputSettingTriggerText();
    updateTreatmentStatusUI();
  });
  el.sidebarTreatment?.addEventListener("change", () => {
    updateInputSettingTriggerText();
    updateTreatmentStatusUI();
  });

  // ----- Treatments: Simple Steering (Residual) -----
  let treatmentSelectedKeys = new Set();
  let treatmentDragState = null; // { startRow, startCol, isSelecting }

  async function refreshResidualVarOptions() {
    if (!el.treatmentResidualVar) return;
    try {
      const res = await fetch("/api/data-vars");
      const data = await res.json().catch(() => ({}));
      const vars = Array.isArray(data.variables) ? data.variables : [];
      const residuals = vars.filter((v) => v.type === "residual");
      const select = el.treatmentResidualVar;
      const current = select.value;
      select.innerHTML = '<option value="">— Select saved residual —</option>';
      residuals.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.name;
        opt.textContent = v.name;
        select.appendChild(opt);
      });
      if (current && residuals.some((v) => v.name === current)) {
        select.value = current;
      }
    } catch {
      // ignore
    }
  }

  function buildTreatmentGrid(directions) {
    const grid = el.treatmentLayerGrid;
    if (!grid) return;
    grid.innerHTML = "";
    treatmentSelectedKeys = new Set(treatmentSelectedKeys); // keep existing

    const keys = Object.keys(directions || {});
    const pattern = /^(.*)\.(\d+)\.(attn_out|attn_block_out|mlp_out|mlp_block_out)$/;
    const byLayer = new Map();
    keys.forEach((k) => {
      const m = pattern.exec(k);
      if (!m) return;
      const layerIdx = parseInt(m[2], 10);
      const kind = m[3];
      if (!byLayer.has(layerIdx)) byLayer.set(layerIdx, {});
      byLayer.get(layerIdx)[kind] = k;
    });
    const sortedLayers = Array.from(byLayer.keys()).sort((a, b) => a - b);

    const kinds = ["attn_out", "attn_block_out", "mlp_out", "mlp_block_out"];
    const kindLabels = {
      attn_out: "ATTN",
      attn_block_out: "ATTN_block",
      mlp_out: "MLP",
      mlp_block_out: "MLP_block",
    };

    // Header row
    grid.appendChild(createGridHeaderCell(""));
    kinds.forEach((k) => {
      const cell = createGridHeaderCell(kindLabels[k] || k);
      grid.appendChild(cell);
    });

    sortedLayers.forEach((layerIdx) => {
      const rowData = byLayer.get(layerIdx) || {};
      // Layer label cell
      const labelCell = document.createElement("div");
      labelCell.className = "treatment-layer-cell layer-label";
      labelCell.textContent = `L${layerIdx}`;
      labelCell.dataset.row = String(layerIdx);
      labelCell.dataset.col = "-1";
      grid.appendChild(labelCell);

      kinds.forEach((kind, colIdx) => {
        const key = rowData[kind];
        const cell = document.createElement("div");
        cell.className = "treatment-layer-cell";
        cell.dataset.row = String(layerIdx);
        cell.dataset.col = String(colIdx);
        if (key) {
          cell.dataset.key = key;
          cell.textContent = "●";
          if (treatmentSelectedKeys.has(key)) {
            cell.classList.add("selected");
          }
        } else {
          cell.textContent = "";
          cell.classList.add("disabled");
        }
        grid.appendChild(cell);
      });
    });
  }

  function createGridHeaderCell(text) {
    const cell = document.createElement("div");
    cell.className = "treatment-layer-grid-header-cell";
    cell.textContent = text;
    return cell;
  }

  async function refreshLayerKeysOptions() {
    if (!el.treatmentResidualVar || !el.treatmentLayerGrid) return;
    const name = el.treatmentResidualVar.value;
    el.treatmentLayerGrid.innerHTML = "";
    if (!name) {
      treatmentSelectedKeys.clear();
      return;
    }
    try {
      const res = await fetch("/api/residual-vars/" + encodeURIComponent(name));
      const rv = await res.json();
      if (rv.error) return;
      const dirs = rv.directions || {};
      buildTreatmentGrid(dirs);
    } catch {
      // ignore
    }
  }

  function parseCurrentTreatmentJSON() {
    if (!el.sidebarTreatment) return null;
    const raw = (el.sidebarTreatment.value || "").trim();
    if (!raw) return null;
    try {
      const obj = JSON.parse(raw);
      if (obj && typeof obj === "object" && obj.type === "simple_steering") return obj;
    } catch {
      // not JSON or not our type
    }
    return null;
  }

  function syncTreatmentUIFromField() {
    const cfg = parseCurrentTreatmentJSON();
    treatmentSelectedKeys.clear();
    if (!cfg) {
      if (el.treatmentResidualVar) el.treatmentResidualVar.value = "";
      if (el.treatmentAlpha) el.treatmentAlpha.value = "1.0";
      if (el.treatmentDelta) el.treatmentDelta.value = "0.0";
      if (el.treatmentNormalize) el.treatmentNormalize.checked = true;
      if (el.treatmentLayerGrid) el.treatmentLayerGrid.innerHTML = "";
      return;
    }
    if (el.treatmentResidualVar && cfg.residual_var) {
      el.treatmentResidualVar.value = String(cfg.residual_var);
    }
    if (el.treatmentAlpha && cfg.alpha != null) {
      el.treatmentAlpha.value = String(cfg.alpha);
    }
    if (el.treatmentDelta && cfg.delta != null) {
      el.treatmentDelta.value = String(cfg.delta);
    }
    if (el.treatmentNormalize) {
      el.treatmentNormalize.checked = cfg.normalize !== false;
    }
    if (Array.isArray(cfg.layer_keys)) {
      cfg.layer_keys.forEach((k) => treatmentSelectedKeys.add(String(k)));
    }
    // rebuild grid for current residual var & selection
    refreshLayerKeysOptions();
  }

  el.treatmentResidualVar?.addEventListener("change", async () => {
    await refreshLayerKeysOptions();
  });

  // Grid interactions: click / drag to select rectangle
  function handleTreatmentGridMouseDown(e) {
    const cell = e.target.closest(".treatment-layer-cell");
    if (!cell || !cell.dataset || cell.classList.contains("layer-label") || !el.treatmentLayerGrid) return;
    const row = parseInt(cell.dataset.row || "-1", 10);
    const col = parseInt(cell.dataset.col || "-1", 10);
    if (row < 0 || col < 0) return;
    const key = cell.dataset.key;
    const isAlreadySelected = key && treatmentSelectedKeys.has(key);
    treatmentDragState = { startRow: row, startCol: col, isSelecting: !isAlreadySelected };
    e.preventDefault();
  }

  function handleTreatmentGridMouseMove(e) {
    if (!treatmentDragState || !el.treatmentLayerGrid) return;
    const cell = e.target.closest(".treatment-layer-cell");
    if (!cell || !cell.dataset) return;
    const row = parseInt(cell.dataset.row || "-1", 10);
    const col = parseInt(cell.dataset.col || "-1", 10);
    if (row < 0 || col < 0) return;
    const { startRow, startCol } = treatmentDragState;
    const minRow = Math.min(startRow, row);
    const maxRow = Math.max(startRow, row);
    const minCol = Math.min(startCol, col);
    const maxCol = Math.max(startCol, col);
    el.treatmentLayerGrid.querySelectorAll(".treatment-layer-cell").forEach((c) => {
      const r = parseInt(c.dataset.row || "-1", 10);
      const kCol = parseInt(c.dataset.col || "-1", 10);
      if (r >= minRow && r <= maxRow && kCol >= minCol && kCol <= maxCol && !c.classList.contains("layer-label")) {
        c.classList.add("dragging-preview");
      } else {
        c.classList.remove("dragging-preview");
      }
    });
  }

  function handleTreatmentGridMouseUp(e) {
    if (!el.treatmentLayerGrid) {
      treatmentDragState = null;
      return;
    }

    // 대상 셀 모으기: 드래그가 있었다면 preview 셀들, 아니면 클릭한 한 셀
    const previewCells = el.treatmentLayerGrid.querySelectorAll(".treatment-layer-cell.dragging-preview");
    let targetCells = [];
    if (previewCells.length > 0) {
      targetCells = Array.from(previewCells);
    } else if (e) {
      const single = e.target.closest(".treatment-layer-cell");
      if (single) targetCells = [single];
    }

    // preview 클래스는 항상 제거
    previewCells.forEach((c) => c.classList.remove("dragging-preview"));

    if (!treatmentDragState || targetCells.length === 0) {
      treatmentDragState = null;
      return;
    }

    const { isSelecting } = treatmentDragState;
    targetCells.forEach((c) => {
      const key = c.dataset.key;
      const row = parseInt(c.dataset.row || "-1", 10);
      const col = parseInt(c.dataset.col || "-1", 10);
      if (!key || row < 0 || col < 0 || c.classList.contains("layer-label")) return;
      if (isSelecting) {
        treatmentSelectedKeys.add(key);
        c.classList.add("selected");
      } else if (c.classList.contains("selected")) {
        treatmentSelectedKeys.delete(key);
        c.classList.remove("selected");
      }
    });
    treatmentDragState = null;
  }

  if (el.treatmentLayerGrid) {
    el.treatmentLayerGrid.addEventListener("mousedown", handleTreatmentGridMouseDown);
    el.treatmentLayerGrid.addEventListener("mousemove", handleTreatmentGridMouseMove);
    window.addEventListener("mouseup", handleTreatmentGridMouseUp);
  }

  el.btnTreatmentApply?.addEventListener("click", async () => {
    if (!el.sidebarTreatment) return;
    const residualVar = el.treatmentResidualVar?.value || "";
    if (!residualVar) {
      alert("Residual variable을 먼저 선택하세요.");
      return;
    }
    const alphaStr = el.treatmentAlpha?.value ?? "1.0";
    let alpha = 1.0;
    try {
      alpha = parseFloat(alphaStr);
    } catch {
      alpha = 1.0;
    }
    const deltaStr = el.treatmentDelta?.value ?? "0.0";
    let delta = 0.0;
    try {
      delta = Math.max(0, Math.min(1, parseFloat(deltaStr)));
    } catch {
      delta = 0.0;
    }
    const normalize = el.treatmentNormalize ? !!el.treatmentNormalize.checked : true;
    const layerKeys = Array.from(treatmentSelectedKeys);
    if (!layerKeys.length) {
      alert("최소 한 개의 layer key를 선택하세요.");
      return;
    }
    const cfg = {
      type: "simple_steering",
      residual_var: residualVar,
      alpha,
      delta,
      normalize,
      layer_keys: layerKeys,
    };
    el.sidebarTreatment.value = JSON.stringify(cfg);
    updateInputSettingTriggerText();
    updateTreatmentStatusUI();
    // 세션에도 즉시 반영하여 새로고침 후에도 유지되도록 한다.
    try {
      session = {
        loaded_model: session.loaded_model || null,
        treatment: el.sidebarTreatment.value,
      };
      await API.setSession({
        loaded_model: session.loaded_model,
        treatment: session.treatment,
      });
    } catch {
      // 세션 저장 실패는 조용히 무시 (UI는 그대로 유지)
    }
    // 사용자가 업데이트 완료를 직관적으로 알 수 있도록 버튼 상태를 잠시 변경
    if (el.btnTreatmentApply) {
      const prevText = el.btnTreatmentApply.textContent;
      el.btnTreatmentApply.disabled = true;
      el.btnTreatmentApply.textContent = "Updated";
      setTimeout(() => {
        el.btnTreatmentApply.textContent = prevText || "Apply Steering";
        el.btnTreatmentApply.disabled = false;
      }, 900);
    }
  });

  el.btnTreatmentClear?.addEventListener("click", () => {
    if (el.sidebarTreatment) {
      el.sidebarTreatment.value = "";
    }
    treatmentSelectedKeys.clear();
    if (el.treatmentResidualVar) el.treatmentResidualVar.value = "";
    if (el.treatmentLayerGrid) el.treatmentLayerGrid.innerHTML = "";
    if (el.treatmentAlpha) el.treatmentAlpha.value = "1.0";
    if (el.treatmentDelta) el.treatmentDelta.value = "0.0";
    if (el.treatmentNormalize) el.treatmentNormalize.checked = true;
    updateInputSettingTriggerText();
    updateTreatmentStatusUI();
  });

  // Open Treatment panel (Simple Steering) attached to sidebar
  async function openSimpleSteeringPanel() {
    console.log("[treatment] openSimpleSteeringPanel called", el.treatmentPanel);
    if (!el.treatmentPanel) return;
    await refreshResidualVarOptions();
    syncTreatmentUIFromField();
    el.treatmentPanel.classList.add("visible");
  }

  // 버튼 클릭 시에도 바로 패널 열기
  el.btnCreateTreatment?.addEventListener("click", (e) => {
    console.log("[treatment:event] click on btnCreateTreatment", e.target);
    e.preventDefault();
    openSimpleSteeringPanel();
  });

  el.createTreatmentDropdown?.addEventListener("mousedown", (e) => {
    console.log("[treatment:event] mousedown on dropdown", e.target);
    const opt = e.target.closest(".create-treatment-option");
    if (!opt) {
      console.log("[treatment:event] no option found");
      return;
    }
    const type = opt.dataset.treatmentType || "simple_steering";
    console.log("[treatment:event] option type =", type);
    if (type === "simple_steering") {
      e.preventDefault();
      openSimpleSteeringPanel();
    }
  });

  el.treatmentPanelClose?.addEventListener("click", () => {
    if (el.treatmentPanel) el.treatmentPanel.classList.remove("visible");
  });

  el.btnLoadModel.addEventListener("click", async () => {
    const model = el.sidebarModel.value;
    const treatment = el.sidebarTreatment.value.trim() || "";
    loadInProgress = true;
    updateLoadButtonState();
    try {
      const res = await API.loadModel({ model, treatment });
      if (res.error) throw new Error(res.error);
      session = { loaded_model: model, treatment };
      updateSessionUI();
      if (typeof window.refreshMemorySummary === "function") window.refreshMemorySummary();
    } catch (err) {
      alert("Model load failed: " + err.message);
      updateLoadButtonState();
    } finally {
      loadInProgress = false;
      updateLoadButtonState();
    }
  });

  // ----- Setting: CUDA_VISIBLE_DEVICES export & ECHO 결과 -----
  async function refreshCudaEcho() {
    if (!el.cudaSettingEchoValue) return;
    try {
      const res = await API.cudaEnvGet();
      const echo = res.echo != null ? String(res.echo) : (res.CUDA_VISIBLE_DEVICES != null ? String(res.CUDA_VISIBLE_DEVICES) : "—");
      el.cudaSettingEchoValue.textContent = echo === "" ? "(empty)" : echo;
      if (el.cudaVisibleDevicesInput) el.cudaVisibleDevicesInput.value = echo === "(empty)" ? "" : echo;
    } catch (_) {
      el.cudaSettingEchoValue.textContent = "—";
    }
  }

  function showCudaPanel() {
    if (cudaPanelHideTimeout) clearTimeout(cudaPanelHideTimeout);
    cudaPanelHideTimeout = null;
    const panel = el.cudaSettingPanel;
    if (panel) {
      panel.classList.add("visible");
      panel.setAttribute("aria-hidden", "false");
      refreshCudaEcho();
    }
  }

  function scheduleHideCudaPanel() {
    if (cudaPanelHideTimeout) clearTimeout(cudaPanelHideTimeout);
    cudaPanelHideTimeout = setTimeout(() => {
      cudaPanelHideTimeout = null;
      if (el.cudaSettingPanel) {
        el.cudaSettingPanel.classList.remove("visible");
        el.cudaSettingPanel.setAttribute("aria-hidden", "true");
      }
    }, 180);
  }

  el.btnCudaSetting?.addEventListener("click", (e) => {
    e.stopPropagation();
  });

  el.btnCudaSetting?.addEventListener("mouseenter", showCudaPanel);
  el.btnCudaSetting?.addEventListener("mouseleave", scheduleHideCudaPanel);
  el.cudaSettingPanel?.addEventListener("mouseenter", () => {
    if (cudaPanelHideTimeout) clearTimeout(cudaPanelHideTimeout);
    cudaPanelHideTimeout = null;
  });
  el.cudaSettingPanel?.addEventListener("mouseleave", scheduleHideCudaPanel);

  el.btnCudaExport?.addEventListener("click", async () => {
    const input = el.cudaVisibleDevicesInput;
    if (!input || !el.cudaSettingEchoValue) return;
    const value = input.value.trim();
    try {
      const res = await API.cudaEnvSet(value);
      const echo = res.echo != null ? String(res.echo) : (res.CUDA_VISIBLE_DEVICES != null ? String(res.CUDA_VISIBLE_DEVICES) : "");
      el.cudaSettingEchoValue.textContent = echo === "" ? "(empty)" : echo;
    } catch (err) {
      el.cudaSettingEchoValue.textContent = "Error: " + (err.message || "failed");
    }
  });

  // ----- Loaded Model: hover to show Model Spec -----
  function getModelSpecTooltipEl() {
    if (modelSpecTooltipEl) return modelSpecTooltipEl;
    const el_ = document.createElement("div");
    el_.id = "model-spec-tooltip";
    el_.className = "model-spec-tooltip";
    el_.setAttribute("role", "tooltip");
    el_.setAttribute("aria-hidden", "true");
    document.body.appendChild(el_);
    modelSpecTooltipEl = el_;
    el_.addEventListener("mouseenter", () => {
      if (modelSpecHideTimeout) clearTimeout(modelSpecHideTimeout);
      modelSpecHideTimeout = null;
    });
    el_.addEventListener("mouseleave", () => {
      scheduleHideModelSpec();
    });
    return el_;
  }

  window.PNP_LAYER_CONFIG = window.PNP_LAYER_CONFIG || {};

  function scheduleShowModelSpec() {
    if (modelSpecShowTimeout) clearTimeout(modelSpecShowTimeout);
    modelSpecShowTimeout = null;
    const modelKey = session.loaded_model;
    if (!modelKey || !el.loadedModelDisplay) return;
    modelSpecShowTimeout = setTimeout(async () => {
      modelSpecShowTimeout = null;
      const tip = getModelSpecTooltipEl();
      tip.classList.add("loading");
      tip.setAttribute("aria-hidden", "false");
      const displayRect = el.loadedModelDisplay.getBoundingClientRect();
      tip.style.left = `${displayRect.left}px`;
      tip.style.top = `${displayRect.bottom + 6}px`;
      try {
        const [statusRes, layerRes] = await Promise.all([
          API.modelStatus(modelKey),
          API.modelLayerNames(modelKey),
        ]);
        const status = statusRes.error ? null : (modelSpecCache[modelKey] = statusRes);
        const layerStruct = layerRes.error ? {} : (layerRes.layer_structure || {});
        const memStr = (status?.device_status || [])
          .map((d) => `${d.device}: ${d.memory_gb != null ? d.memory_gb + " GB" : "—"}`)
          .join("\n");
        const configStr = status?.config && Object.keys(status.config).length
          ? JSON.stringify(status.config, null, 2)
          : "—";
        const modulesStr = (status?.modules || "").trim() ? status.modules : "—";

        const def = (v) => (v != null && v !== "" ? String(v) : "");
        const layerBaseVal = def(layerStruct.layers_base) || (layerRes.layers_base || "");
        const attnVal = def(layerStruct.attn_name);
        const mlpVal = def(layerStruct.mlp_name);
        const oProjVal = def(layerStruct.o_proj_name);
        const downProjVal = def(layerStruct.down_proj_name);

        window.PNP_LAYER_CONFIG[modelKey] = {
          layer_base: layerBaseVal || "model.model.layers",
          attn_name: attnVal || "self_attn",
          mlp_name: mlpVal || "mlp",
          o_proj_name: oProjVal || "o_proj",
          down_proj_name: downProjVal || "down_proj",
        };

        tip.classList.remove("loading");
        tip.innerHTML = `
          <div class="model-spec-tooltip-title">Model Spec</div>
          <dl class="model-spec-dl">
            <dt>Name</dt><dd>${escapeHtml(String(status?.name || status?.model_key || modelKey || "—"))}</dd>
            <dt>Layers</dt><dd>${status?.num_layers != null ? escapeHtml(String(status.num_layers)) : "—"}</dd>
            <dt>Heads</dt><dd>${status?.num_heads != null ? escapeHtml(String(status.num_heads)) : "—"}</dd>
            <dt>Memory</dt><dd><pre class="model-spec-memory">${escapeHtml(memStr)}</pre></dd>
          </dl>
          <div class="model-spec-layer-config">
            <div class="model-spec-tooltip-title model-spec-layer-config-title">Layer Config</div>
            <div class="model-spec-layer-row"><label>1) Layers</label><input type="text" class="model-spec-layer-input" data-key="layer_base" value="${escapeHtml(layerBaseVal)}" placeholder="please find name" /></div>
            <div class="model-spec-layer-row"><label>2) attn</label><input type="text" class="model-spec-layer-input" data-key="attn_name" value="${escapeHtml(attnVal)}" placeholder="please find name" /></div>
            <div class="model-spec-layer-row"><label>3) mlp</label><input type="text" class="model-spec-layer-input" data-key="mlp_name" value="${escapeHtml(mlpVal)}" placeholder="please find name" /></div>
            <div class="model-spec-layer-row"><label>4) o_proj</label><input type="text" class="model-spec-layer-input" data-key="o_proj_name" value="${escapeHtml(oProjVal)}" placeholder="please find name" /></div>
            <div class="model-spec-layer-row"><label>5) down_proj</label><input type="text" class="model-spec-layer-input" data-key="down_proj_name" value="${escapeHtml(downProjVal)}" placeholder="please find name" /></div>
          </div>
          <details class="model-spec-config">
            <summary>Model structure (config)</summary>
            <pre class="model-spec-config-json">${escapeHtml(configStr)}</pre>
          </details>
          <details class="model-spec-modules">
            <summary>Module structure (modules)</summary>
            <pre class="model-spec-modules-pre">${escapeHtml(modulesStr)}</pre>
          </details>
        `;

        tip.querySelectorAll(".model-spec-layer-input").forEach((input) => {
          input.addEventListener("input", () => {
            const key = input.dataset.key;
            const val = (input.value || "").trim();
            if (!window.PNP_LAYER_CONFIG[modelKey]) window.PNP_LAYER_CONFIG[modelKey] = {};
            window.PNP_LAYER_CONFIG[modelKey][key] = val || (key === "layer_base" ? "model.model.layers" : key === "attn_name" ? "self_attn" : key === "mlp_name" ? "mlp" : key === "o_proj_name" ? "o_proj" : "down_proj");
          });
        });
      } catch (err) {
        tip.classList.remove("loading");
        tip.innerHTML = `<div class="model-spec-tooltip-error">${escapeHtml(err.message || "Failed to load spec")}</div>`;
      }
      tip.classList.add("visible");
    }, 350);
  }

  function scheduleHideModelSpec() {
    if (modelSpecHideTimeout) clearTimeout(modelSpecHideTimeout);
    modelSpecHideTimeout = setTimeout(() => {
      modelSpecHideTimeout = null;
      if (modelSpecTooltipEl) {
        modelSpecTooltipEl.classList.remove("visible", "loading");
        modelSpecTooltipEl.setAttribute("aria-hidden", "true");
      }
    }, 150);
  }

  function cancelShowModelSpec() {
    if (modelSpecShowTimeout) {
      clearTimeout(modelSpecShowTimeout);
      modelSpecShowTimeout = null;
    }
  }

  if (el.loadedModelDisplay) {
    el.loadedModelDisplay.addEventListener("mouseenter", () => {
      if (!el.loadedModelDisplay.classList.contains("has-model")) return;
      cancelShowModelSpec();
      if (modelSpecHideTimeout) clearTimeout(modelSpecHideTimeout);
      modelSpecHideTimeout = null;
      scheduleShowModelSpec();
    });
    el.loadedModelDisplay.addEventListener("mouseleave", () => {
      cancelShowModelSpec();
      scheduleHideModelSpec();
    });
  }

  // ----- Sidebar section toggle (Loaded Model, Treatments, Created Task Panels) -----
  document.querySelector(".sidebar")?.addEventListener("click", (e) => {
    const title = e.target.closest(".sidebar-section-title");
    if (!title) return;
    // Don't toggle when clicking inner task-panel-group-title
    if (e.target.closest(".task-panel-group-title")) return;
    const section = title.closest(".sidebar-section");
    if (!section || !section.querySelector(".sidebar-section-body")) return;
    e.preventDefault();
    e.stopPropagation();
    section.classList.toggle("collapsed");
  });

  // ----- Input Setting panel toggle (task-specific, right side) -----
  if (el.inputSettingTrigger && el.inputSettingPanel) {
    el.inputSettingTrigger.addEventListener("click", () => {
      el.inputSettingPanel.classList.toggle("visible");
      el.inputSettingTrigger.classList.toggle("has-setting", el.inputSettingPanel.classList.contains("visible"));
    });
  }

  // Data variables (Working Memory) panel is now managed by static/js/working_memory.js

  // ----- Create XAI: show full-screen name input, then add task -----
  const modalCreateName = document.getElementById("modal-create-task-name");
  const createNameInput = document.getElementById("create-task-name-input");
  const createTaskCancel = document.getElementById("create-task-cancel");
  const createTaskConfirm = document.getElementById("create-task-confirm");

  let pendingCreate = null; // { level, name }

  function showCreateTaskModal(level, defaultName) {
    if (!level) return;
    pendingCreate = { level, defaultName };
    if (createNameInput) createNameInput.value = defaultName || "";
    if (createNameInput) createNameInput.placeholder = "Enter task name";
    modalCreateName?.classList.add("visible");
    createNameInput?.focus();
  }

  function hideCreateTaskModal() {
    modalCreateName.classList.remove("visible");
    pendingCreate = null;
  }

  async function submitCreateTask() {
    if (!pendingCreate) return;
    const title = createNameInput.value.trim() || "Task";
    const { level } = pendingCreate;
    hideCreateTaskModal();
    try {
      const res = await API.createTask({
        xai_level: level,
        title,
        model: "",
        treatment: "",
        result: {},
      });
      if (res.error) throw new Error(res.error);
      const data = await API.tasksWithMeta();
      renderTaskList(data.tasks || data, data.xai_level_names || {});
    } catch (err) {
      alert("Failed to add task: " + err.message);
    }
  }

  // Create XAI: click main button to toggle dropdown (hover also works)
  el.btnCreateTask?.addEventListener("click", (e) => {
    e.stopPropagation();
    el.createTaskWrap?.classList.toggle("dropdown-open");
  });

  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (e.target.closest(".create-task-wrap")) return;
    el.createTaskWrap?.classList.remove("dropdown-open");
  });

  $$(".create-task-option").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      el.createTaskWrap?.classList.remove("dropdown-open");
      const level = btn.dataset.level;
      const name = btn.dataset.name || "";
      if (level) showCreateTaskModal(level, name);
    });
  });

  createTaskCancel?.addEventListener("click", hideCreateTaskModal);
  createTaskConfirm?.addEventListener("click", submitCreateTask);
  createNameInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") submitCreateTask();
    if (e.key === "Escape") hideCreateTaskModal();
  });
  modalCreateName?.addEventListener("click", (e) => {
    if (e.target === modalCreateName) hideCreateTaskModal();
  });

  // Run button: show Play+RUN or Square+Stop
  function setRunButtonState(isRunning) {
    if (!el.btnRun) return;
    if (isRunning) {
      el.btnRun.classList.add("is-running");
      el.btnRun.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg> Stop';
    } else {
      el.btnRun.classList.remove("is-running");
      el.btnRun.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> RUN';
    }
  }

  // ----- Run with session check -----
  async function doRun(forceLoadModel = false) {
    const model = el.sidebarModel.value;
    const treatment = el.sidebarTreatment.value.trim() || "";
    const inputSetting = { ...gatherTaskInput(), model, treatment };
    if (window.PNP_CURRENT_TASK_LEVEL === "2.0.1") {
      let layerCfg = window.PNP_LAYER_CONFIG && window.PNP_LAYER_CONFIG[model];
      if (!layerCfg && model && inputSetting.variable_name) {
        try {
          const res = await API.modelLayerNames(model);
          if (!res.error && res.layer_structure) {
            const s = res.layer_structure;
            layerCfg = {
              layer_base: s.layers_base || res.layers_base || "model.model.layers",
              attn_name: s.attn_name || "self_attn",
              mlp_name: s.mlp_name || "mlp",
              o_proj_name: s.o_proj_name || "o_proj",
              down_proj_name: s.down_proj_name || "down_proj",
            };
            window.PNP_LAYER_CONFIG = window.PNP_LAYER_CONFIG || {};
            window.PNP_LAYER_CONFIG[model] = layerCfg;
          }
        } catch (_) {}
      }
      if (layerCfg) Object.assign(inputSetting, layerCfg);
    }

    if (window.PNP_CURRENT_TASK_LEVEL === "0.1.2") {
      const contentEl = document.getElementById("conversation-user-input");
      const content = (contentEl && contentEl.value ? contentEl.value : "").trim();
      if (!content) {
        alert("메시지를 입력하세요.");
        return;
      }
    }

    if (forceLoadModel) {
      try {
        const res = await API.loadModel({ model, treatment });
        if (res.error) throw new Error(res.error);
        session = { loaded_model: model, treatment };
        updateSessionUI();
      } catch (err) {
        alert("Model load failed: " + err.message);
        return;
      }
    }

    runAbortController = new AbortController();
    const generationStatus = document.getElementById("generation-status");
    setRunButtonState(true);
    if (generationStatus) {
      generationStatus.textContent = "Working on it...";
      generationStatus.classList.add("visible");
    }
    try {
      if (window.PNP_CURRENT_TASK_LEVEL === "0.1.2") {
        const contentEl = document.getElementById("conversation-user-input");
        const userContent = (contentEl && contentEl.value ? contentEl.value : "").trim();
        if (userContent && window.PNP_appendConversationMessage && window.PNP_appendConversationMessageGenerating) {
          window.PNP_appendConversationMessage("user", userContent);
          window.PNP_appendConversationMessageGenerating();
          if (window.PNP_clearConversationUserInput) window.PNP_clearConversationUserInput();
        }
      }
      let res;
      const runBody = { model, treatment, input_setting: inputSetting };
      const isResidualStream = window.PNP_CURRENT_TASK_LEVEL === "2.0.1" && (inputSetting.variable_name || "").trim();

      if (isResidualStream) {
        try {
          const r = await fetch("/api/run/residual-concept-stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(runBody),
            signal: runAbortController.signal,
          });
          if (!r.ok || !r.body) {
            const errText = await r.text().catch(() => "");
            res = { ok: false, error: errText || "Stream failed (status " + r.status + ")" };
          } else {
            const reader = r.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              buffer += decoder.decode(value, { stream: true });
              const events = buffer.split("\n\n");
              buffer = events.pop() || "";
              for (const ev of events) {
                const dataLine = ev.split("\n").find((l) => l.startsWith("data: "));
                if (dataLine) {
                  try {
                    const msg = JSON.parse(dataLine.slice(6));
                    if (msg.type === "progress" && generationStatus) {
                      generationStatus.textContent = msg.message || "Forward batch " + msg.batch + "/" + msg.total;
                    } else if (msg.type === "done") {
                      res = { ok: true, ...(msg.result || {}) };
                    } else if (msg.type === "error") {
                      res = { ok: false, error: msg.error };
                    }
                  } catch (e) { /* ignore parse */ }
                }
              }
            }
            if (buffer) {
              const dataLine = buffer.split("\n").find((l) => l.startsWith("data: "));
              if (dataLine) {
                try {
                  const msg = JSON.parse(dataLine.slice(6));
                  if (msg.type === "done") res = { ok: true, ...(msg.result || {}) };
                  else if (msg.type === "error") res = { ok: false, error: msg.error };
                } catch (e) { /* ignore */ }
              }
            }
            if (!res) res = { ok: false, error: "No result from stream" };
          }
        } catch (streamErr) {
          if (streamErr.name === "AbortError") throw streamErr;
          res = { ok: false, error: streamErr.message || "Stream error" };
        }
        if (res && res.error === "No result from stream") {
          const fallback = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(runBody),
            signal: runAbortController.signal,
          });
          const text = await fallback.text();
          try {
            res = { ok: fallback.ok, ...JSON.parse(text) };
          } catch (_) {
            res = res || { ok: false, error: "No result from stream" };
          }
        }
      } else {
        const r = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(runBody),
          signal: runAbortController.signal,
        });
        const text = await r.text();
        try {
          res = { ok: r.ok, ...JSON.parse(text) };
        } catch (parseErr) {
          console.error("Run response was not JSON:", text.slice(0, 200));
          if (window.PNP_finishGeneratingMessage) window.PNP_finishGeneratingMessage("Invalid response from server.", true);
          alert("Server returned an invalid response (status " + r.status + "). Check the console for details.");
          return;
        }
      }

      if (!res.ok && res.error === "session_mismatch") {
        // 세션 불일치는 팝업 없이 자동으로 세션을 동기화한 뒤 한 번 더 시도한다.
        if (!forceLoadModel) {
          await doRun(true);
          return;
        }
      }

      if (res.error) {
        if (window.PNP_finishGeneratingMessage) window.PNP_finishGeneratingMessage(res.error, true);
        alert(res.error);
        return;
      }

      const taskId = window.PNP_CURRENT_TASK_ID;
      const isConversationResult = res && "conversation_list" in res;
      const isCompletionResult = res && "generated_text" in res;
      const isAttributionResult = res && Array.isArray(res.input_tokens) && Array.isArray(res.token_scores);
      const isResidualConceptResult = res && res.directions && typeof res.directions === "object";

      if (isConversationResult && window.PNP_finishGeneratingMessage) {
        window.PNP_finishGeneratingMessage(res.generated_text != null ? res.generated_text : "");
        if (window.PNP_appendAssistantMessage) window.PNP_appendAssistantMessage(res.generated_text || "");
        if (window.PNP_updateConversationCacheCount && res.cache_message_count != null) {
          window.PNP_updateConversationCacheCount(res.cache_message_count);
        }
        const wrap = document.getElementById("conversation-wrap");
        if (wrap) {
          const prev = parseInt(wrap.dataset.cumulativeGenerated || "0", 10);
          const cum = prev + (res.generated_tokens || 0);
          wrap.dataset.cumulativeGenerated = String(cum);
          if (window.PNP_updateConversationTokenCount) {
            window.PNP_updateConversationTokenCount(res.input_tokens, res.generated_tokens, cum);
          }
        }
        if (window.PNP_clearConversationUserInput) window.PNP_clearConversationUserInput();
        const resultToShow = { ...res };
        if (wrap && wrap.dataset.cumulativeGenerated) resultToShow.cumulative_generated_tokens = parseInt(wrap.dataset.cumulativeGenerated, 10);
        if (inputSetting.system_instruction !== undefined) resultToShow.system_instruction = inputSetting.system_instruction;
        if (taskId) {
          await API.updateTask(taskId, { result: resultToShow, model, treatment });
          if (el.inputSettingTrigger) {
            el.inputSettingTrigger.dataset.taskModel = model;
            el.inputSettingTrigger.dataset.taskTreatment = treatment;
          }
          updateInputSettingTriggerText();
          const taskTitle = (el.inputSettingTrigger && el.inputSettingTrigger.dataset.taskName) || "";
          fetch("/api/memory/session/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task_id: taskId, model, treatment, name: taskTitle }),
          }).catch(() => {});
          const data = await API.tasksWithMeta();
          renderTaskList(data.tasks || data, data.xai_level_names || {});
          if (typeof window.refreshResultList === "function") window.refreshResultList();
          if (typeof window.refreshMemorySummary === "function") window.refreshMemorySummary();
        }
        const jsonEl = document.getElementById("conversation-result-json");
        if (jsonEl) jsonEl.textContent = JSON.stringify(resultToShow, null, 2);
        return;
      }

      function renderResultContent() {
        if (isAttributionResult && window.PNP_renderAttributionResultHTML) {
          return window.PNP_renderAttributionResultHTML(res, escapeHtml);
        }
        if (isResidualConceptResult) {
          window.PNP_LAST_RESIDUAL_RESULT = res;
          const numKeys = res.num_keys ?? Object.keys(res.directions || {}).length;
          const modelDim = res.model_dim ?? (Object.values(res.directions || {})[0]?.length ?? 0);
          const keys = Object.keys(res.directions || {});
          const keysStr = keys.length > 4
            ? keys.slice(0, 2).join(", ") + ", … , " + keys.slice(-1).join(", ") + ` (${numKeys} keys)`
            : keys.join(", ");
          return `
            <div class="results-completion-wrap">
              <h3>Residual Direction Vectors</h3>
              <p><strong>Keys:</strong> ${escapeHtml(String(numKeys))} · <strong>Dimension:</strong> ${escapeHtml(String(modelDim))}</p>
              <p>Format: <code>{ module: dimension }</code></p>
              <p>Modules: ${escapeHtml(keysStr)}</p>
              <p>Positive: ${res.n_positive ?? "—"} · Negative: ${res.n_negative ?? "—"} · Batches: ${res.num_batches ?? "—"}</p>
              <div class="residual-save-row">
                <button type="button" class="btn-save-residual-var">Save To Variable</button>
                <input type="text" class="residual-var-additional-input" placeholder="Additional naming (optional)" value="" />
              </div>
              <details class="results-completion-meta">
                <summary>Parameters &amp; full result</summary>
                <pre class="results-json">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
              </details>
            </div>
          `;
        }
        if (isCompletionResult) {
          return `
            <div class="results-completion-wrap">
              <h3>Generated</h3>
              <pre class="results-completion-text">${escapeHtml(String(res.generated_text ?? ""))}</pre>
              <details class="results-completion-meta">
                <summary>Parameters &amp; full result</summary>
                <pre class="results-json">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
              </details>
            </div>
          `;
        }
        return `
          <div style="padding: 16px;">
            <h3>Run Result</h3>
            <pre style="background: var(--panel); padding: 12px; border-radius: 6px; overflow: auto; font-size: 12px;">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
          </div>
        `;
      }

      if (el.resultsPlaceholder) el.resultsPlaceholder.classList.add("hidden");
      if (el.resultsContent) {
        el.resultsContent.classList.add("visible");
        el.resultsContent.innerHTML = renderResultContent();
        if (isAttributionResult && window.PNP_initAttributionGradientControls) window.PNP_initAttributionGradientControls(el.resultsContent);
      }

      if (taskId) {
        const resultToSave = { ...res };
        if (isResidualConceptResult && inputSetting) {
          resultToSave.variable_name = inputSetting.variable_name;
          resultToSave.text_key = inputSetting.text_key;
          resultToSave.label_key = inputSetting.label_key;
          resultToSave.positive_label = inputSetting.positive_label;
          resultToSave.negative_label = inputSetting.negative_label;
          resultToSave.layer_base = inputSetting.layer_base;
          resultToSave.attn_name = inputSetting.attn_name;
          resultToSave.mlp_name = inputSetting.mlp_name;
          resultToSave.o_proj_name = inputSetting.o_proj_name;
          resultToSave.down_proj_name = inputSetting.down_proj_name;
          resultToSave.token_location_mode = document.getElementById("input-token-location-mode")?.value || "full";
          resultToSave.token_ids = document.getElementById("input-token-ids")?.value || "";
          resultToSave.batch_size = inputSetting.batch_size;
        }
        await API.updateTask(taskId, { result: resultToSave, model, treatment });
        if (el.inputSettingTrigger) {
          el.inputSettingTrigger.dataset.taskModel = model;
          el.inputSettingTrigger.dataset.taskTreatment = treatment;
        }
        updateInputSettingTriggerText();
        const resContent = document.getElementById("results-content");
        const resPlaceholder = document.getElementById("results-placeholder");
        if (resContent) resContent.innerHTML = renderResultContent();
        resPlaceholder?.classList.add("hidden");
        resContent?.classList.add("visible");
        if (isAttributionResult && resContent && window.PNP_initAttributionGradientControls) {
          window.PNP_initAttributionGradientControls(resContent);
        }
      } else {
        const title = prompt("Enter task title (will be saved):", "Task " + new Date().toLocaleString());
        if (title) {
          await API.createTask({
            xai_level: currentTaskLevel,
            title,
            model,
            treatment,
            result: res,
          });
          const data = await API.tasksWithMeta();
          renderTaskList(data.tasks || data, data.xai_level_names || {});
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        if (window.PNP_finishGeneratingMessage) window.PNP_finishGeneratingMessage("Cancelled.", true);
        return;
      }
      if (window.PNP_finishGeneratingMessage) window.PNP_finishGeneratingMessage("Error: " + (err.message || String(err)), true);
      alert(err.message || String(err));
    } finally {
      runAbortController = null;
      setRunButtonState(false);
      if (document.getElementById("generation-status")) {
        const gs = document.getElementById("generation-status");
        gs.textContent = "";
        gs.classList.remove("visible");
      }
    }
  }

  // ----- Gather task-specific input from right Input Setting panel -----
  function gatherTaskInput() {
    const body = document.getElementById("input-setting-body");
    const inputs = body ? body.querySelectorAll("[data-task-input]") : [];
    const obj = {};
    inputs.forEach((el) => {
      const key = el.dataset.taskInput || el.name || el.id;
      if (key) obj[key] = el.value !== undefined ? el.value : el.textContent;
    });
    if (window.PNP_CURRENT_TASK_LEVEL === "0.1.2") {
      const contentEl = document.getElementById("conversation-user-input");
      const systemEl = document.getElementById("input-system-instruction");
      const content = (contentEl && contentEl.value ? contentEl.value : "").trim();
      if (systemEl) obj.system_instruction = (systemEl.value || "").trim();
      // JS에서 관리하는 messages 전달
      if (window.PNP_getConversationMessagesForRun) {
        obj.messages = window.PNP_getConversationMessagesForRun(content);
      } else {
        obj.content = content;
      }
    }
    if (window.PNP_CURRENT_TASK_LEVEL === "2.0.1") {
      const modeEl = document.getElementById("input-token-location-mode");
      const idsEl = document.getElementById("input-token-ids");
      const mode = (modeEl && modeEl.value) || "full";
      obj.token_location = mode === "ids" && idsEl && idsEl.value.trim()
        ? idsEl.value.trim().replace(/\s+/g, ",").split(",").map((x) => parseInt(x, 10)).filter((n) => !isNaN(n))
        : "full";
    }
    return obj;
  }

  if (el.btnRun) {
    el.btnRun.addEventListener("click", () => {
      if (el.btnRun.classList.contains("is-running")) {
        if (!confirm("Generation을 중단하시겠습니까?")) return;
        if (runAbortController) runAbortController.abort();
        return;
      }
      doRun(false);
    });
  }

  const conversationInput = document.getElementById("conversation-user-input");
  if (conversationInput) {
    conversationInput.addEventListener("keydown", function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (el.btnRun && !el.btnRun.classList.contains("is-running")) {
          doRun(false);
        }
      }
    });
  }

  if (el.modalCancel) {
    el.modalCancel.addEventListener("click", () => {
      el.modalConfirm?.classList.remove("visible");
      pendingRunAfterConfirm = null;
    });
  }

  // ----- Export / Import -----
  const btnExport = document.getElementById("btn-export");
  const btnImport = document.getElementById("btn-import");
  const importFileInput = document.getElementById("import-file-input");

  if (btnExport) {
    btnExport.addEventListener("click", () => {
      window.location.href = "/api/memory/export";
    });
  }

  if (btnImport && importFileInput) {
    btnImport.addEventListener("click", () => importFileInput.click());
    importFileInput.addEventListener("change", async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        const res = await fetch("/api/memory/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
        const json = await res.json();
        if (json.error) throw new Error(json.error);
        location.reload();
      } catch (err) {
        alert("Import failed: " + err.message);
      }
      importFileInput.value = "";
    });
  }

  // ----- Data Management: delete button and right-click context menu -----
  const datasetPipelineList = document.getElementById("dataset-pipeline-list");
  async function deleteDatasetPipeline(pipelineId, item) {
    if (!confirm("이 pipeline을 삭제하시겠습니까?")) return;
    try {
      const res = await fetch("/api/dataset-pipelines/" + encodeURIComponent(pipelineId), { method: "DELETE" });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || "Delete failed");
      if (item) item.remove();
      if (datasetPipelineList && datasetPipelineList.children.length === 0 && !datasetPipelineList.querySelector(".dataset-pipeline-empty")) {
        const li = document.createElement("li");
        li.className = "dataset-pipeline-empty";
        li.textContent = "No pipelines yet.";
        datasetPipelineList.appendChild(li);
      }
      if (document.body.dataset.pipelineId === pipelineId) {
        window.location.href = "/";
      }
    } catch (err) {
      alert("Delete failed: " + err.message);
    }
  }
  if (datasetPipelineList) {
    datasetPipelineList.addEventListener("click", (e) => {
      const btn = e.target.closest(".dataset-pipeline-delete");
      if (btn) {
        e.preventDefault();
        e.stopPropagation();
        const pipelineId = btn.dataset.pipelineId;
        const item = btn.closest(".dataset-pipeline-item");
        if (pipelineId) deleteDatasetPipeline(pipelineId, item);
      }
    });
    datasetPipelineList.addEventListener("contextmenu", (e) => {
      const item = e.target.closest(".dataset-pipeline-item");
      if (!item) return;
      e.preventDefault();
      const pipelineId = item.dataset.pipelineId;
      if (!pipelineId) return;
      const menu = document.createElement("div");
      menu.className = "context-menu";
      menu.innerHTML = '<button type="button" class="context-menu-item context-menu-delete">Delete</button>';
      menu.style.position = "fixed";
      menu.style.left = e.clientX + "px";
      menu.style.top = e.clientY + "px";
      menu.style.zIndex = "9999";
      document.body.appendChild(menu);
      const hide = () => {
        menu.remove();
        document.removeEventListener("click", hide);
      };
      document.addEventListener("click", hide);
      menu.addEventListener("click", (ev) => ev.stopPropagation());
      menu.querySelector(".context-menu-delete").addEventListener("click", async (ev) => {
        ev.stopPropagation();
        hide();
        deleteDatasetPipeline(pipelineId, item);
      });
    });
  }

  // ----- Data Management: create new Dataset Pipeline -----
  const btnCreatePipeline = document.getElementById("btn-create-pipeline");
  if (btnCreatePipeline) {
    btnCreatePipeline.addEventListener("click", async () => {
      const name = (prompt("Pipeline name:") || "").trim() || "Unnamed";
      try {
        const res = await fetch("/api/dataset-pipelines", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(json.error || "Create failed");
        const id = json.id || json.pipeline?.id;
        if (id) window.location.href = "/data/" + id;
        else location.reload();
      } catch (err) {
        alert("Create pipeline failed: " + err.message);
      }
    });
  }

  if (el.modalConfirmBtn) {
    el.modalConfirmBtn.addEventListener("click", async () => {
      el.modalConfirm?.classList.remove("visible");
      if (pendingRunAfterConfirm) {
        const { model, treatment } = pendingRunAfterConfirm;
        pendingRunAfterConfirm = null;
        await doRun(true);
      }
    });
  }

  // ----- Render task list -----
  function renderTaskList(tasks, xaiLevelNames = {}) {
    if (!tasks || typeof tasks !== "object") return;
    const entries = Object.entries(tasks).filter(([, items]) => Array.isArray(items) && items.length > 0);

    // Skip re-render if current DOM already matches new data (avoids flicker when refetch runs)
    if (el.taskPanelList) {
      if (entries.length === 0) {
        if (el.taskPanelList.querySelector(".task-panel-empty")) return;
      } else {
        const newItems = entries.flatMap(([, items]) => items.map((t) => ({ id: t.id, title: (t.title || "").trim() })));
        const existingItems = Array.from(el.taskPanelList.querySelectorAll(".task-panel-item")).map((li) => ({
          id: li.dataset.taskId || "",
          title: (li.querySelector(".task-panel-item-link")?.textContent || "").trim(),
        }));
        if (newItems.length === existingItems.length && newItems.every((n, i) => existingItems[i] && n.id === existingItems[i].id && n.title === existingItems[i].title)) {
          return;
        }
      }
    }

    const fragment = document.createDocumentFragment();
    for (const [level, items] of entries) {
      const levelKey = level.replace("xai_level_", "").replace(/_/g, ".");
      const levelLabel = levelKey + (xaiLevelNames[levelKey] ? " — " + xaiLevelNames[levelKey] : "");
      const group = document.createElement("li");
      group.className = "task-panel-group";
      group.innerHTML = `
        <div class="task-panel-group-title">
          <span class="task-panel-chevron" aria-hidden="true">▼</span>
          <span class="task-panel-group-label">${escapeHtml(levelLabel)}</span>
        </div>
        <ul class="task-panel-sublist">
          ${items
                .map(
                  (t) => `
            <li class="task-panel-item" data-task-id="${t.id}" data-level="${level}">
              <a href="/task/${t.id}" class="task-panel-item-link">${escapeHtml(t.title)}</a>
              <button type="button" class="task-panel-delete" data-task-id="${t.id}" title="Delete">×</button>
            </li>
          `
                )
                .join("")}
        </ul>
      `;
      fragment.appendChild(group);
    }

    el.taskPanelList.innerHTML = "";
    if (entries.length === 0) {
      el.taskPanelList.innerHTML = '<li class="task-panel-empty">No tasks created yet.</li>';
    } else {
      el.taskPanelList.appendChild(fragment);
    }
  }

  // ----- Load task into right panel (click panel item) -----
  function loadTaskIntoView(task) {
    if (!task) return;
    window.PNP_CURRENT_TASK_ID = task.id;
    currentTaskLevel = task.xai_level || "0.1";

    // Load default (current session), overlay with task data if present
    if (el.sidebarModel && task.model) el.sidebarModel.value = task.model;
    if (el.sidebarTreatment && task.treatment !== undefined) el.sidebarTreatment.value = String(task.treatment);
    session = {
      loaded_model: task.model || session.loaded_model,
      treatment: task.treatment !== undefined && task.treatment !== null ? String(task.treatment) : session.treatment,
    };
    updateSessionUI();

    if (el.inputSettingTrigger) {
      el.inputSettingTrigger.dataset.taskType = xaiLevelToTaskTypeLabel(task.xai_level);
      el.inputSettingTrigger.dataset.taskName = (task.name != null && task.name !== undefined ? String(task.name) : (task.title || "")).trim();
      el.inputSettingTrigger.dataset.taskModel = task.model != null && task.model !== undefined ? String(task.model) : "";
      el.inputSettingTrigger.dataset.taskTreatment = task.treatment !== undefined && task.treatment !== null ? String(task.treatment) : "";
    }
    updateInputSettingTriggerText();

    if (task.xai_level === "0.1.1" || task.xai_level === "0.1.2" || task.xai_level === "1.0.1") {
      el.inputSettingPanel?.classList.add("visible");
      el.inputSettingTrigger?.classList.add("has-setting");
    }

    // Results: overlay task.result if present, else show placeholder
    if (task.result && Object.keys(task.result).length > 0) {
      el.resultsPlaceholder?.classList.add("hidden");
      el.resultsContent?.classList.add("visible");
      const res = task.result;
      const isAttribution = res && Array.isArray(res.input_tokens) && Array.isArray(res.token_scores);
      const isCompletion = res && "generated_text" in res;
      if (isAttribution && window.PNP_renderAttributionResultHTML) {
        el.resultsContent.innerHTML = window.PNP_renderAttributionResultHTML(res, escapeHtml);
        if (window.PNP_initAttributionGradientControls) window.PNP_initAttributionGradientControls(el.resultsContent);
      } else if (isCompletion) {
        el.resultsContent.innerHTML = `
          <div class="results-completion-wrap">
            <h3>Generated</h3>
            <pre class="results-completion-text">${escapeHtml(String(res.generated_text ?? ""))}</pre>
            <details class="results-completion-meta">
              <summary>Parameters &amp; full result</summary>
              <pre class="results-json">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
            </details>
          </div>
        `;
      } else {
        el.resultsContent.innerHTML = `
          <div style="padding: 16px;">
            <h3 style="margin-top:0;">${escapeHtml(task.title || "")}</h3>
            <p><strong>XAI Level:</strong> ${escapeHtml(task.xai_level || "—")}</p>
            <p><strong>Model:</strong> ${escapeHtml(task.model || "—")}</p>
            <p><strong>Treatment:</strong> ${escapeHtml(task.treatment || "—")}</p>
            <p><strong>Created:</strong> ${escapeHtml(task.created_at || "—")}</p>
            <pre style="background: var(--panel); padding: 12px; border-radius: 6px; overflow: auto; font-size: 12px;">${escapeHtml(JSON.stringify(task.result || {}, null, 2))}</pre>
          </div>
        `;
      }
    } else {
      el.resultsPlaceholder?.classList.remove("hidden");
      el.resultsContent?.classList.remove("visible");
      if (el.resultsContent) el.resultsContent.innerHTML = "";
      if (el.resultsPlaceholder) el.resultsPlaceholder.innerHTML = `
        <span class="brand-title">PnP-XAI-LLM</span>
        <ul class="feature-list"><li></li><li></li><li></li></ul>
      `;
    }

    // Update active state
    el.taskPanelList?.querySelectorAll(".task-panel-item").forEach((li) => {
      li.classList.toggle("active", String(li.dataset.taskId) === String(task.id));
    });
  }

  // Double-click task title → inline rename (cancel delayed single-click navigate)
  el.taskPanelList?.addEventListener("dblclick", (e) => {
    const link = e.target.closest(".task-panel-item-link");
    if (!link) return;
    e.preventDefault();
    if (taskLinkNavigateTimeout) {
      clearTimeout(taskLinkNavigateTimeout);
      taskLinkNavigateTimeout = null;
    }
    const item = link.closest(".task-panel-item");
    const taskId = item?.dataset.taskId;
    if (!taskId) return;

    const currentTitle = link.textContent.trim();
    const input = document.createElement("input");
    input.type = "text";
    input.className = "task-panel-item-edit";
    input.value = currentTitle;
    input.setAttribute("data-task-id", taskId);

    const finish = (save) => {
      const newTitle = input.value.trim();
      input.removeEventListener("blur", onBlur);
      input.removeEventListener("keydown", onKey);
      link.textContent = save && newTitle ? newTitle : currentTitle;
      link.style.display = "";
      input.replaceWith(link);
      if (save && newTitle && newTitle !== currentTitle) {
        API.updateTask(taskId, { title: newTitle }).then((res) => {
          if (res.error) link.textContent = currentTitle;
        });
        if (el.inputSettingTrigger && window.PNP_CURRENT_TASK_ID === taskId) {
          el.inputSettingTrigger.dataset.taskName = newTitle;
          updateInputSettingTriggerText();
        }
      }
    };

    const onBlur = () => finish(true);
    const onKey = (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        input.blur();
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        finish(false);
      }
    };

    link.style.display = "none";
    link.after(input);
    input.focus();
    input.select();
    input.addEventListener("blur", onBlur);
    input.addEventListener("keydown", onKey);
  });

  // Toggle group expand/collapse (VSCode-style)
  el.taskPanelList?.addEventListener("click", async (e) => {
    const header = e.target.closest(".task-panel-group-title");
    if (header) {
      e.preventDefault();
      const group = header.closest(".task-panel-group");
      if (group) group.classList.toggle("collapsed");
      return;
    }

    const link = e.target.closest(".task-panel-item-link");
    if (link) {
      e.preventDefault();
      const href = link.getAttribute("href");
      if (href && href.startsWith("/task/")) {
        if (taskLinkNavigateTimeout) clearTimeout(taskLinkNavigateTimeout);
        taskLinkNavigateTimeout = setTimeout(() => {
          taskLinkNavigateTimeout = null;
          window.location.href = href;
        }, 250);
      }
      return;
    }

    const btn = e.target.closest(".task-panel-delete");
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const taskId = btn.dataset.taskId;
    if (!taskId) return;
    try {
      const sessRes = await fetch("/api/memory/session/list");
      const sessData = await sessRes.json().catch(() => ({}));
      const sessCaches = Array.isArray(sessData.caches) ? sessData.caches : [];
      for (const sc of sessCaches) {
        if (sc.task === taskId && sc.key) {
          await fetch("/api/memory/session/unregister/" + encodeURIComponent(sc.key), { method: "DELETE" });
        }
      }
      const res = await fetch(`/api/tasks/${taskId}`, { method: "DELETE" });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const meta = await API.tasksWithMeta();
      renderTaskList(meta.tasks || meta, meta.xai_level_names || {});
      if (typeof window.refreshSessionList === "function") window.refreshSessionList();
      if (typeof window.refreshResultList === "function") window.refreshResultList();
      if (window.PNP_CURRENT_TASK_ID === taskId) {
        window.location.href = "/";
      }
    } catch (err) {
      alert("Failed to delete: " + err.message);
    }
  });

  function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  // ----- Sidebar: Variable Cache list -----
  async function refreshSidebarVariableList() {
    const listEl = document.getElementById("sidebar-variable-list");
    const emptyEl = document.getElementById("sidebar-variable-empty");
    if (!listEl) return;
    try {
      const res = await fetch("/api/data-vars");
      const data = await res.json().catch(() => ({}));
      const variables = Array.isArray(data.variables) ? data.variables : [];
      listEl.querySelectorAll(".sidebar-variable-item").forEach((el) => el.remove());
      if (emptyEl) emptyEl.style.display = variables.length > 0 ? "none" : "list-item";
      variables.forEach((v) => {
        const li = document.createElement("li");
        li.className = "sidebar-variable-item";
        const name = (v.name || "").trim() || "—";
        li.innerHTML = '<span class="sidebar-variable-name" title="' + escapeHtml(name) + '">' + escapeHtml(name) + "</span>";
        listEl.appendChild(li);
      });
    } catch (_) {
      if (emptyEl) emptyEl.style.display = "list-item";
    }
  }
  window.refreshSidebarVariableList = refreshSidebarVariableList;

  // ----- Init -----
  async function init() {
    try {
      const params = new URLSearchParams(location.search);
      const createLevel = params.get("create");
      if (createLevel) {
        currentTaskLevel = createLevel;
        session = { loaded_model: null, treatment: null };
        history.replaceState({}, "", "/");
      } else {
        session = await API.session();
      }
      if (session.loaded_model && el.sidebarModel) {
        el.sidebarModel.value = session.loaded_model;
      }
      if (session.treatment && el.sidebarTreatment) {
        el.sidebarTreatment.value = session.treatment;
      }
      updateSessionUI();
      const data = await API.tasksWithMeta();
      renderTaskList(data.tasks || data, data.xai_level_names || {});
      refreshSidebarVariableList();
      await refreshResidualVarOptions();
      syncTreatmentUIFromField();
      updateTreatmentStatusUI();
      if (el.treatmentPanel) {
        el.treatmentPanel.classList.remove("visible");
      }
      if (createLevel) {
        const option = document.querySelector(`.create-task-option[data-level="${createLevel}"]`);
        showCreateTaskModal(createLevel, "");
      }
      // 0.1.1 Completion / 0.1.2 Conversation / 1.0.1 Response Attribution / 2.0.1 Residual Concept: open Input Setting panel
      if ((window.PNP_CURRENT_TASK_LEVEL === "0.1.1" || window.PNP_CURRENT_TASK_LEVEL === "0.1.2" || window.PNP_CURRENT_TASK_LEVEL === "1.0.1" || window.PNP_CURRENT_TASK_LEVEL === "2.0.1") && el.inputSettingPanel) {
        el.inputSettingPanel.classList.add("visible");
        el.inputSettingTrigger?.classList.add("has-setting");
      }
      if (el.resultsContent && el.resultsContent.querySelector(".results-attribution-wrap")) {
        if (window.PNP_initAttributionGradientControls) window.PNP_initAttributionGradientControls(el.resultsContent);
      }
    } catch (e) {
      console.error("Init error:", e);
    }
  }

  init();
})();
