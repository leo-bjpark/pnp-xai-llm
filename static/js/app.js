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
  el.sidebarTreatment?.addEventListener("input", () => updateInputSettingTriggerText());
  el.sidebarTreatment?.addEventListener("change", () => updateInputSettingTriggerText());

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
        const status = modelSpecCache[modelKey] || await API.modelStatus(modelKey);
        if (status.error) throw new Error(status.error);
        modelSpecCache[modelKey] = status;
        const memStr = (status.device_status || [])
          .map((d) => `${d.device}: ${d.memory_gb != null ? d.memory_gb + " GB" : "—"}`)
          .join("\n");
        const configStr = status.config && Object.keys(status.config).length
          ? JSON.stringify(status.config, null, 2)
          : "—";
        const modulesStr = status.modules && status.modules.trim() ? status.modules : "—";
        tip.classList.remove("loading");
        tip.innerHTML = `
          <div class="model-spec-tooltip-title">Model Spec</div>
          <dl class="model-spec-dl">
            <dt>Name</dt><dd>${escapeHtml(String(status.name || status.model_key || "—"))}</dd>
            <dt>Layers</dt><dd>${status.num_layers != null ? escapeHtml(String(status.num_layers)) : "—"}</dd>
            <dt>Heads</dt><dd>${status.num_heads != null ? escapeHtml(String(status.num_heads)) : "—"}</dd>
            <dt>Memory</dt><dd><pre class="model-spec-memory">${escapeHtml(memStr)}</pre></dd>
          </dl>
          <details class="model-spec-config">
            <summary>Model structure (config)</summary>
            <pre class="model-spec-config-json">${escapeHtml(configStr)}</pre>
          </details>
          <details class="model-spec-modules">
            <summary>Module structure (modules)</summary>
            <pre class="model-spec-modules-pre">${escapeHtml(modulesStr)}</pre>
          </details>
        `;
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

  // ----- Right panel: open in separate window (independent of main page) -----
  let panelWindow = null;
  const btnRightPanel = document.getElementById("btn-right-panel");
  const PANEL_NAME = "pnpRightPanel";
  const PANEL_FEATURES = "width=360,height=800,scrollbars=yes,resizable=yes";

  function updatePanelButtonActive() {
    if (!btnRightPanel) return;
    btnRightPanel.classList.toggle("active", panelWindow && !panelWindow.closed);
  }

  function toggleRightPanel(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (!btnRightPanel) return;
    if (panelWindow && !panelWindow.closed) {
      panelWindow.close();
      panelWindow = null;
    } else {
      panelWindow = window.open("/panel", PANEL_NAME, PANEL_FEATURES);
    }
    updatePanelButtonActive();
  }

  if (btnRightPanel) {
    btnRightPanel.addEventListener("click", toggleRightPanel, true);
  }
  setInterval(function () {
    if (panelWindow && panelWindow.closed) {
      panelWindow = null;
      updatePanelButtonActive();
    }
  }, 500);

  // ----- Data variables (global saved state): table Name | Memory (GPU/RAM) | Delete -----
  const btnDataVars = document.getElementById("btn-data-vars");
  const dataVarsDropdown = document.getElementById("data-vars-dropdown");
  const dataVarsTbody = document.getElementById("data-vars-tbody");
  const dataVarsEmptyRow = document.getElementById("data-vars-empty-row");

  function formatMemory(gpuGb, ramGb) {
    const g = gpuGb != null ? "GPU: " + gpuGb + " GB" : "";
    const r = ramGb != null ? "RAM: " + ramGb + " GB" : "";
    return [g, r].filter(Boolean).join(" · ") || "—";
  }

  async function refreshDataVarsList() {
    if (!dataVarsTbody || !dataVarsEmptyRow) return;
    try {
      const res = await fetch("/api/data-vars");
      const data = await res.json().catch(() => ({}));
      const loadedModel = data.loaded_model || null;
      const variables = data.variables || [];
      const hasAny = loadedModel || variables.length > 0;

      dataVarsEmptyRow.style.display = hasAny ? "none" : "table-row";
      dataVarsTbody.querySelectorAll(".data-vars-data-row").forEach((el) => el.remove());

      if (loadedModel) {
        const tr = document.createElement("tr");
        tr.className = "data-vars-data-row data-vars-model-row";
        const memStr = formatMemory(loadedModel.memory_gpu_gb, loadedModel.memory_ram_gb);
        tr.innerHTML =
          "<td class=\"data-vars-td-name\" title=\"Loaded model\">" + escapeHtml(loadedModel.name || "") + "</td>" +
          "<td class=\"data-vars-td-memory\">" + escapeHtml(memStr) + "</td>" +
          "<td class=\"data-vars-td-action\"><button type=\"button\" class=\"data-vars-btn-delete\" data-kind=\"model\" title=\"Unload model\">×</button></td>";
        const btn = tr.querySelector(".data-vars-btn-delete");
        btn.addEventListener("click", async () => {
          try {
            const r = await fetch("/api/empty_cache", { method: "POST" });
            if (r.ok) {
              if (typeof session !== "undefined" && typeof updateSessionUI === "function") {
                session = { loaded_model: null, treatment: null };
                updateSessionUI();
              }
              refreshDataVarsList();
            }
          } catch (e) {}
        });
        dataVarsTbody.appendChild(tr);
      }

      variables.forEach((v) => {
        const tr = document.createElement("tr");
        tr.className = "data-vars-data-row data-vars-var-row";
        const memStr = v.memory_ram_mb != null ? "RAM: ~" + v.memory_ram_mb + " MB" : "—";
        const varName = v.name || "";
        tr.innerHTML =
          "<td class=\"data-vars-td-name\" title=\"" + escapeHtml(varName) + "\">" + escapeHtml(varName) + "</td>" +
          "<td class=\"data-vars-td-memory\">" + escapeHtml(memStr) + "</td>" +
          "<td class=\"data-vars-td-action\"><button type=\"button\" class=\"data-vars-btn-delete\" data-kind=\"var\" title=\"Remove variable\">×</button></td>";
        const btn = tr.querySelector(".data-vars-btn-delete");
        btn.addEventListener("click", async () => {
          try {
            const r = await fetch("/api/data-vars/" + encodeURIComponent(varName), { method: "DELETE" });
            if (r.ok) refreshDataVarsList();
          } catch (e) {}
        });
        dataVarsTbody.appendChild(tr);
      });
    } catch (err) {
      dataVarsEmptyRow.style.display = "table-row";
      const cell = document.getElementById("data-vars-empty");
      if (cell) cell.textContent = "Failed to load.";
    }
  }
  window.refreshDataVarsList = refreshDataVarsList;

  if (btnDataVars && dataVarsDropdown) {
    btnDataVars.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const isOpen = dataVarsDropdown.classList.contains("visible");
      if (isOpen) {
        dataVarsDropdown.classList.remove("visible");
        dataVarsDropdown.setAttribute("aria-hidden", "true");
      } else {
        refreshDataVarsList();
        dataVarsDropdown.classList.add("visible");
        dataVarsDropdown.setAttribute("aria-hidden", "false");
      }
    });
    document.addEventListener("click", () => {
      if (dataVarsDropdown.classList.contains("visible")) {
        dataVarsDropdown.classList.remove("visible");
        dataVarsDropdown.setAttribute("aria-hidden", "true");
      }
    });
    dataVarsDropdown.addEventListener("click", (e) => e.stopPropagation());
  }

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

    if (window.PNP_CURRENT_TASK_LEVEL === "0.1.2") {
      if (!(inputSetting.content || "").trim()) {
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
        const userContent = (inputSetting.content || "").trim();
        if (userContent && window.PNP_appendConversationMessage && window.PNP_appendConversationMessageGenerating) {
          window.PNP_appendConversationMessage("user", userContent);
          window.PNP_appendConversationMessageGenerating();
          if (window.PNP_clearConversationUserInput) window.PNP_clearConversationUserInput();
        }
      }
      const r = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, treatment, input_setting: inputSetting }),
        signal: runAbortController.signal,
      });
      const text = await r.text();
      let res;
      try {
        res = { ok: r.ok, ...JSON.parse(text) };
      } catch (parseErr) {
        console.error("Run response was not JSON:", text.slice(0, 200));
        if (window.PNP_finishGeneratingMessage) window.PNP_finishGeneratingMessage("Invalid response from server.", true);
        alert("Server returned an invalid response (status " + r.status + "). Check the console for details.");
        return;
      }

      if (!res.ok && res.error === "session_mismatch") {
        if (window.PNP_finishGeneratingMessage) window.PNP_finishGeneratingMessage("Session mismatch. Please load the model.", true);
        pendingRunAfterConfirm = { model, treatment };
        el.modalMessage.textContent =
          "Loaded Model + Treatment does not match the current session. Load the model with this setting?";
        el.modalConfirm.classList.add("visible");
        return;
      }

      if (res.error) {
        if (window.PNP_finishGeneratingMessage) window.PNP_finishGeneratingMessage(res.error, true);
        alert(res.error);
        return;
      }

      const taskId = window.PNP_CURRENT_TASK_ID;
      const isConversationResult = res && "conversation_id" in res;
      const isCompletionResult = res && "generated_text" in res;
      const isAttributionResult = res && Array.isArray(res.input_tokens) && Array.isArray(res.token_scores);

      if (isConversationResult && window.PNP_finishGeneratingMessage) {
        window.PNP_finishGeneratingMessage(res.generated_text != null ? res.generated_text : "");
        if (window.PNP_setConversationId) window.PNP_setConversationId(res.conversation_id || "");
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
        }
        const jsonEl = document.getElementById("conversation-result-json");
        if (jsonEl) jsonEl.textContent = JSON.stringify(resultToShow, null, 2);
        return;
      }

      function renderResultContent() {
        if (isAttributionResult && window.PNP_renderAttributionResultHTML) {
          return window.PNP_renderAttributionResultHTML(res, escapeHtml);
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
        await API.updateTask(taskId, { result: res, model, treatment });
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
      const cidEl = document.getElementById("conversation-id");
      const contentEl = document.getElementById("conversation-user-input");
      const systemEl = document.getElementById("input-system-instruction");
      if (cidEl) obj.conversation_id = cidEl.value || "";
      if (contentEl) obj.content = (contentEl.value || "").trim();
      if (systemEl) obj.system_instruction = (systemEl.value || "").trim();
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

  // ----- Empty Cache (reset model / CUDA / session) -----
  const btnEmptyCache = document.getElementById("btn-empty-cache");
  if (btnEmptyCache) {
    btnEmptyCache.addEventListener("click", async () => {
      const warning =
        "Warning: This will clear the loaded model, session state, conversation cache, and CUDA cache. " +
        "You will need to load a model again to run. Continue?";
      if (!confirm(warning)) return;
      try {
        const res = await fetch("/api/empty_cache", { method: "POST" });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(json.error || "Empty cache failed");
        session = { loaded_model: null, treatment: null };
        updateSessionUI();
      } catch (err) {
        alert("Empty cache failed: " + err.message);
      }
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
      const res = await fetch(`/api/tasks/${taskId}`, { method: "DELETE" });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const meta = await API.tasksWithMeta();
      renderTaskList(meta.tasks || meta, meta.xai_level_names || {});
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
      if (createLevel) {
        const option = document.querySelector(`.create-task-option[data-level="${createLevel}"]`);
        showCreateTaskModal(createLevel, "");
      }
      // 0.1.1 Completion / 0.1.2 Conversation / 1.0.1 Response Attribution: open Input Setting panel
      if ((window.PNP_CURRENT_TASK_LEVEL === "0.1.1" || window.PNP_CURRENT_TASK_LEVEL === "0.1.2" || window.PNP_CURRENT_TASK_LEVEL === "1.0.1") && el.inputSettingPanel) {
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
