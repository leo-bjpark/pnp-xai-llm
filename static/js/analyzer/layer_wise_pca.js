/**
 * Layer Residual PCA (frontend).
 *
 * - Lets user pick a residual variable.
 * - Calls /api/analyzer/layer-residual-pca/<var_id> to get singular values.
 * - Renders summary, scree-like sparkline, and table of singular values
 *   sorted in descending order.
 */

(function () {
  "use strict";

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function clear(el) {
    if (el) el.innerHTML = "";
  }

  function loadResidualOptions() {
    const sel = $("#residual-pca-var-select");
    if (!sel) return;
    fetch("/api/data-vars")
      .then((r) => r.json())
      .then((data) => {
        const vars = (data.variables || []).filter((v) => v.type === "residual");
        sel.innerHTML = '<option value="">— Select residual variable —</option>';
        vars.forEach((v) => {
          const opt = document.createElement("option");
          opt.value = v.id || "";
          opt.textContent = (v.name || v.id || "") + (v.task_name ? " (" + v.task_name + ")" : "");
          sel.appendChild(opt);
        });
      })
      .catch(() => {});
  }

  function renderSparkline(svg, values) {
    if (!svg) return;
    svg.innerHTML = "";
    const w = 400;
    const h = 80;
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    if (!values.length) return;
    const maxV = values[0] || 1e-12;
    const n = values.length;
    const padX = 4;
    const padY = 4;
    const usableW = w - padX * 2;
    const usableH = h - padY * 2;
    const stepX = n > 1 ? usableW / (n - 1) : 0;
    let d = "";
    values.forEach((v, idx) => {
      const x = padX + stepX * idx;
      const y = padY + usableH * (1 - v / maxV);
      d += (idx === 0 ? "M" : "L") + x + " " + y + " ";
    });
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d.trim());
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "var(--accent)");
    path.setAttribute("stroke-width", "1.5");
    svg.appendChild(path);
  }

  function renderList(container, singularValues, explained) {
    if (!container) return;
    clear(container);
    if (!singularValues.length) {
      container.textContent = "No singular values.";
      return;
    }
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");
    ["Component", "Singular value", "Explained variance"].forEach((label) => {
      const th = document.createElement("th");
      th.textContent = label;
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    singularValues.forEach((v, idx) => {
      const tr = document.createElement("tr");
      const cIdx = document.createElement("td");
      cIdx.textContent = "PC" + (idx + 1);
      const cVal = document.createElement("td");
      cVal.textContent = v.toFixed(5);
      const cVar = document.createElement("td");
      const frac = explained && explained[idx] != null ? explained[idx] : 0;
      cVar.textContent = (frac * 100).toFixed(2) + " %";
      tr.appendChild(cIdx);
      tr.appendChild(cVal);
      tr.appendChild(cVar);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
  }

  function runAnalysis() {
    const sel = $("#residual-pca-var-select");
    const btn = $("#residual-pca-btn-analyze");
    const status = $("#residual-pca-status");
    const list = $("#residual-pca-list");
    const sparkline = $("#residual-pca-sparkline");
    const summary = $("#residual-pca-summary");
    const varId = (sel && sel.value) || "";
    if (!varId) {
      if (status) status.textContent = "Select a residual variable first.";
      return;
    }
    if (btn) btn.disabled = true;
    if (status) status.textContent = "Running PCA…";
    clear(list);
    if (sparkline) sparkline.innerHTML = "";
    if (summary) summary.textContent = "";

    const ensureFn = window.PNP_ensureVariableLoaded || window.ensureVariableLoaded;
    const proceed = () => {
      fetch("/api/analyzer/layer-residual-pca/" + encodeURIComponent(varId))
        .then((r) => r.json().then((body) => ({ ok: r.ok, body })))
        .then(({ ok, body }) => {
          if (!ok || body.error) throw new Error(body.error || "Analyzer error");
          const sv = body.singular_values || [];
          const ev = body.explained_variance || [];
          if (status) status.textContent = "Done. Showing " + sv.length + " components.";
          if (summary) {
            summary.innerHTML =
              "<strong>Variable:</strong> " +
              (body.variable_id || varId) +
              (body.task_name ? " · <strong>Task:</strong> " + body.task_name : "") +
              (body.model ? " · <strong>Model:</strong> " + body.model : "") +
              "<br><strong>Directions:</strong> " +
              (body.num_directions || "?") +
              " · <strong>Dimension:</strong> " +
              (body.model_dim || "?");
          }
          renderSparkline(sparkline, sv);
          renderList(list, sv, ev);
        })
        .catch((err) => {
          if (status) status.textContent = "Error: " + (err.message || "Unknown");
        })
        .finally(() => {
          if (btn) btn.disabled = false;
        });
    };

    if (typeof ensureFn === "function") {
      ensureFn(varId).then((ok) => {
        if (!ok) {
          if (status) status.textContent = "Variable must be loaded to run PCA.";
          if (btn) btn.disabled = false;
          return;
        }
        proceed();
      });
    } else {
      proceed();
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    loadResidualOptions();
    const btn = $("#residual-pca-btn-analyze");
    if (btn) btn.addEventListener("click", runAnalysis);
    if (typeof window.refreshSidebarVariableList === "function") {
      const orig = window.refreshSidebarVariableList;
      window.refreshSidebarVariableList = function () {
        orig();
        loadResidualOptions();
      };
    }
  });
})();

