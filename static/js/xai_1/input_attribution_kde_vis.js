/**
 * Renders the attribution result HTML (controls + KDE chart + token blocks + generated).
 * Exposes window.PNP_renderAttributionResultHTML(res, escapeHtml).
 */
(function () {
  "use strict";

  function defaultEscapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  /** 토큰 표시용 후처리: Ġ(U+0120, 스페이스 심볼) → 공백으로 보여줌 */
  function tokenDisplayText(raw) {
    return String(raw).replace(/\u0120/g, " ");
  }

  function renderAttributionResultHTML(res, escapeHtml) {
    escapeHtml = escapeHtml || defaultEscapeHtml;
    const tokens = res.input_tokens || [];
    const scores = res.token_scores || [];
    const scoresDropSpecial = res.token_scores_drop_special || scores;
    const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    const tokenSpans = tokens
      .map((t, i) => {
        const score = scores[i] != null ? Number(scores[i]) : 0;
        const display = tokenDisplayText(t);
        return `<span class="attribution-token" data-score="${score}" title="score: ${score}">${esc(display)}</span>`;
      })
      .join("");
    const tokenSpansDropSpecial = tokens
      .map((t, i) => {
        const score = scoresDropSpecial[i] != null ? Number(scoresDropSpecial[i]) : 0;
        const display = tokenDisplayText(t);
        return `<span class="attribution-token" data-score="${score}" title="score: ${score}">${esc(display)}</span>`;
      })
      .join("");
    return `
    <div class="results-completion-wrap results-attribution-wrap">
      <h3>Input attribution</h3>
      <div class="attribution-range-row">
        <div class="attribution-range-controls">
          <label class="attribution-range-label">min_clip</label>
          <input type="color" class="attribution-color-left" value="#e8e8e8" title="Low (min) color" aria-label="Low color">
          <div class="attribution-range-slider-wrap">
            <div class="attribution-range-track" id="attribution-range-track"></div>
            <input type="range" class="attribution-range-min" min="0" max="1" step="0.01" value="0" aria-label="min_clip">
            <input type="range" class="attribution-range-max" min="0" max="1" step="0.01" value="1" aria-label="max_clip">
          </div>
          <input type="color" class="attribution-color-right" value="#3b82f6" title="High (max) color" aria-label="High color">
          <label class="attribution-range-label">max_clip</label>
          <span class="attribution-range-values" id="attribution-range-values">0.00 — 1.00</span>
        </div>
        <div class="attribution-kde-chart" id="attribution-kde-chart" aria-label="Score distribution (KDE)">
          <svg viewBox="0 0 220 100" preserveAspectRatio="xMidYMid meet">
            <g class="kde-axis">
              <line x1="28" y1="8" x2="28" y2="72" class="kde-yline"/>
              <line x1="28" y1="72" x2="212" y2="72" class="kde-xline"/>
              <text x="28" y="80" text-anchor="middle">0</text>
              <text x="212" y="80" text-anchor="middle">1</text>
              <text x="120" y="94" text-anchor="middle" class="kde-axis-label">Value</text>
              <text x="14" y="40" text-anchor="middle" class="kde-axis-label" transform="rotate(-90,14,40)">Density</text>
            </g>
            <path class="kde-path" d="M28,72 L212,72 L212,72 L28,72 Z"/>
          </svg>
        </div>
      </div>
      <div class="attribution-tokens-wrap" id="attribution-tokens-wrap">${tokenSpans}</div>
      <div class="attribution-tokens-drop-special-wrap">
        <div class="attribution-tokens-drop-special-label">Special tokens dropped</div>
        <div class="attribution-tokens-wrap" id="attribution-tokens-wrap-drop-special">${tokenSpansDropSpecial}</div>
      </div>
      <h3>Generated</h3>
      <pre class="results-completion-text">${escapeHtml(String(res.generated_text ?? ""))}</pre>
      <details class="results-completion-meta">
        <summary>Parameters &amp; full result</summary>
        <pre class="results-json">${escapeHtml(JSON.stringify(res, null, 2))}</pre>
      </details>
    </div>
  `;
  }

  window.PNP_renderAttributionResultHTML = renderAttributionResultHTML;
  window.PNP_tokenDisplayText = tokenDisplayText;
})();
