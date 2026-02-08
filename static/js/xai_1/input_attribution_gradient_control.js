/**
 * Input attribution gradient controls (min_clip, max_clip, colors) + KDE visualization.
 * Exposes window.PNP_initAttributionGradientControls(root) for app.js.
 */
(function () {
  "use strict";

  function drawAttributionKDE(wrap) {
    const chartEl = wrap.querySelector(".attribution-kde-chart");
    const pathEl = chartEl?.querySelector(".kde-path");
    const firstTokensWrap = wrap.querySelector(".attribution-tokens-wrap");
    const tokens = firstTokensWrap ? firstTokensWrap.querySelectorAll(".attribution-token") : [];
    if (!chartEl || !pathEl) return;
    const scores = Array.from(tokens).map((t) => parseFloat(t.dataset.score) || 0).filter((s) => Number.isFinite(s));
    const n = scores.length;
    const gridRes = 120;
    const padding = { top: 8, right: 8, bottom: 20, left: 28 };
    const w = 220;
    const h = 100;
    const baseY = 72;
    const innerW = w - padding.left - padding.right;
    const innerH = baseY - padding.top;
    if (n === 0) {
      pathEl.setAttribute("d", `M ${padding.left},${baseY} L ${w - padding.right},${baseY} Z`);
      return;
    }
    const mean = scores.reduce((a, b) => a + b, 0) / n;
    const variance = scores.reduce((a, s) => a + (s - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance) || 0.1;
    const bandwidth = Math.max(0.02, 1.06 * std * Math.pow(n, -0.2));
    function gaussian(u) {
      return Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
    }
    const densities = [];
    for (let i = 0; i <= gridRes; i++) {
      const x = i / gridRes;
      let sum = 0;
      for (let j = 0; j < n; j++) sum += gaussian((x - scores[j]) / bandwidth);
      densities.push((1 / (n * bandwidth)) * sum);
    }
    const maxD = Math.max(...densities, 1e-9);
    const pts = densities.map((d, i) => {
      const x = padding.left + (i / gridRes) * innerW;
      const y = baseY - (d / maxD) * innerH;
      return [x, y];
    });
    const d = `M ${padding.left},${baseY} L ${pts.map(([x, y]) => `${x},${y}`).join(" L ")} L ${w - padding.right},${baseY} Z`;
    pathEl.setAttribute("d", d);
  }

  function hexToRgb(hex) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [232, 232, 232];
  }

  function interpolateRgb(rgb1, rgb2, t) {
    t = Math.max(0, Math.min(1, t));
    return [
      Math.round(rgb1[0] + (rgb2[0] - rgb1[0]) * t),
      Math.round(rgb1[1] + (rgb2[1] - rgb1[1]) * t),
      Math.round(rgb1[2] + (rgb2[2] - rgb1[2]) * t),
    ];
  }

  function applyAttributionColors(wrap) {
    const track = wrap.querySelector(".attribution-range-track");
    const minIn = wrap.querySelector(".attribution-range-min");
    const maxIn = wrap.querySelector(".attribution-range-max");
    const leftColor = wrap.querySelector(".attribution-color-left");
    const rightColor = wrap.querySelector(".attribution-color-right");
    const valuesEl = wrap.querySelector(".attribution-range-values");
    const tokens = wrap.querySelectorAll(".attribution-token");
    if (!minIn || !maxIn || !leftColor || !rightColor) return;
    let minClip = parseFloat(minIn.value) || 0;
    let maxClip = parseFloat(maxIn.value) || 1;
    if (minClip > maxClip) maxClip = minClip;
    if (maxClip < minClip) minClip = maxClip;
    minIn.value = String(minClip);
    maxIn.value = String(maxClip);
    const leftHex = leftColor.value || "#e8e8e8";
    const rightHex = rightColor.value || "#3b82f6";
    const leftRgb = hexToRgb(leftHex);
    const rightRgb = hexToRgb(rightHex);
    if (track) {
      track.style.background = `linear-gradient(to right, ${leftHex}, ${rightHex})`;
    }
    if (valuesEl) {
      valuesEl.textContent = `${minClip.toFixed(2)} — ${maxClip.toFixed(2)}`;
    }
    const span = maxClip - minClip;
    tokens.forEach((tok) => {
      const score = parseFloat(tok.dataset.score) || 0;
      const t = span > 0 ? Math.max(0, Math.min(1, (score - minClip) / span)) : 0.5;
      const rgb = interpolateRgb(leftRgb, rightRgb, t);
      tok.style.background = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    });
  }

  function initAttributionGradientControls(root) {
    const wrap = (root || document).querySelector(".results-attribution-wrap");
    if (!wrap) return;
    if (window.PNP_tokenDisplayText) {
      wrap.querySelectorAll(".attribution-token").forEach((el) => {
        el.textContent = window.PNP_tokenDisplayText(el.textContent);
      });
    }
    const sliderWrap = wrap.querySelector(".attribution-range-slider-wrap");
    const minIn = wrap.querySelector(".attribution-range-min");
    const maxIn = wrap.querySelector(".attribution-range-max");
    const leftColor = wrap.querySelector(".attribution-color-left");
    const rightColor = wrap.querySelector(".attribution-color-right");
    if (!sliderWrap || !minIn || !maxIn) return;
    minIn.style.pointerEvents = "none";
    maxIn.style.pointerEvents = "none";
    let overlay = wrap.querySelector(".attribution-range-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.className = "attribution-range-overlay";
      overlay.setAttribute("aria-hidden", "true");
      sliderWrap.appendChild(overlay);
    }
    function setActive(input, active) {
      minIn.classList.toggle("attribution-thumb-active", input === minIn && active);
      maxIn.classList.toggle("attribution-thumb-active", input === maxIn && active);
    }
    function fractionFromEvent(e) {
      const rect = sliderWrap.getBoundingClientRect();
      const x = "touches" in e ? e.touches[0].clientX : e.clientX;
      return Math.max(0, Math.min(1, (x - rect.left) / rect.width));
    }
    let dragging = null;
    function startDrag(f) {
      const minVal = parseFloat(minIn.value) || 0;
      const maxVal = parseFloat(maxIn.value) || 1;
      const mid = (minVal + maxVal) / 2;
      dragging = (f < mid || Math.abs(f - minVal) < Math.abs(f - maxVal)) ? "min" : "max";
      setActive(dragging === "min" ? minIn : maxIn, true);
      updateValue(f);
    }
    function updateValue(f) {
      const minVal = parseFloat(minIn.value) || 0;
      const maxVal = parseFloat(maxIn.value) || 1;
      if (dragging === "min") {
        const v = Math.min(f, maxVal);
        minIn.value = String(v);
      } else {
        const v = Math.max(f, minVal);
        maxIn.value = String(v);
      }
      applyAttributionColors(wrap);
    }
    function endDrag() {
      if (dragging) setActive(dragging === "min" ? minIn : maxIn, false);
      dragging = null;
      document.removeEventListener("mouseup", onDocMouseUp);
      document.removeEventListener("mousemove", onDocMouseMove);
    }
    function onDocMouseUp() {
      endDrag();
    }
    function onDocMouseMove(e) {
      if (!dragging) return;
      const rect = sliderWrap.getBoundingClientRect();
      const f = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      updateValue(f);
    }
    overlay.addEventListener("mousedown", (e) => {
      e.preventDefault();
      startDrag(fractionFromEvent(e));
      document.addEventListener("mouseup", onDocMouseUp);
      document.addEventListener("mousemove", onDocMouseMove);
    });
    overlay.addEventListener("mousemove", (e) => { if (dragging) updateValue(fractionFromEvent(e)); });
    overlay.addEventListener("touchstart", (e) => { startDrag(fractionFromEvent(e)); }, { passive: true });
    overlay.addEventListener("touchmove", (e) => { if (dragging) { e.preventDefault(); updateValue(fractionFromEvent(e)); } }, { passive: false });
    overlay.addEventListener("touchend", (e) => { endDrag(); });
    overlay.addEventListener("touchcancel", endDrag);
    minIn.addEventListener("input", () => { if (parseFloat(minIn.value) > parseFloat(maxIn.value)) maxIn.value = minIn.value; applyAttributionColors(wrap); });
    maxIn.addEventListener("input", () => { if (parseFloat(maxIn.value) < parseFloat(minIn.value)) minIn.value = maxIn.value; applyAttributionColors(wrap); });
    if (leftColor) leftColor.addEventListener("input", () => applyAttributionColors(wrap));
    if (rightColor) rightColor.addEventListener("input", () => applyAttributionColors(wrap));
    applyAttributionColors(wrap);
    drawAttributionKDE(wrap);
  }

  window.PNP_initAttributionGradientControls = initAttributionGradientControls;
})();
