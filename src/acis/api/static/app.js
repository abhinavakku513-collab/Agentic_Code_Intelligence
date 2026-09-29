/* ACIS demo UI — vanilla, offline, no dependencies (D15).
 *
 * Everything shown comes from the API's response: similarities and ranks are the engine's own signals, timings
 * are the engine's stage timings, and every code block is the text the engine re-read from the content store by
 * hash (INV-1). Nothing here ranks, re-scores or fills in a value the engine did not return.
 */
"use strict";
const $ = (id) => document.getElementById(id);

/* -- small helpers ----------------------------------------------------------------------------------------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const KEYWORDS = new Set(("False None True and as assert async await break class continue def del elif else except " +
  "finally for from global if import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
function highlight(source) {
  const out = [];
  const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+\.?\d*\b)|([A-Za-z_][A-Za-z0-9_]*)/g;
  let last = 0, m, prevDef = false;
  while ((m = re.exec(source)) !== null) {
    out.push(escapeHtml(source.slice(last, m.index)));
    const [text, comment, str, num, word] = m;
    if (comment) out.push(`<span class="tok-com">${escapeHtml(text)}</span>`);
    else if (str) out.push(`<span class="tok-str">${escapeHtml(text)}</span>`);
    else if (num) out.push(`<span class="tok-num">${escapeHtml(text)}</span>`);
    else if (word && KEYWORDS.has(word)) out.push(`<span class="tok-kw">${escapeHtml(text)}</span>`);
    else if (word && prevDef) out.push(`<span class="tok-def">${escapeHtml(text)}</span>`);
    else out.push(escapeHtml(text));
    prevDef = word === "def" || word === "class";
    last = m.index + text.length;
  }
  out.push(escapeHtml(source.slice(last)));
  return out.join("");
}
async function api(path, body) {
  const response = await fetch(path, body
    ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
    : {});
  const payload = await response.json().catch(() => ({ message: response.statusText }));
  if (!response.ok) {
    const detail = payload.message || (Array.isArray(payload.detail) ? payload.detail.map((d) => d.msg).join("; ") : payload.detail);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return payload;
}
function segValue(id) { return document.querySelector(`#${id} button.on`).dataset.value; }
function bindSeg(id, onChange) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    if (onChange) onChange(b.dataset.value);
  });
}
function codeBlock(source, lines = 14) {
  const all = source.split("\n");
  const id = `c${Math.random().toString(36).slice(2, 9)}`;
  const more = all.length > lines
    ? `<button class="more" data-target="${id}">show all ${all.length} lines</button>` : "";
  return `<pre class="code" id="${id}">${highlight(source)}</pre>${more}`;
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("button.more"); if (!b) return;
  const pre = $(b.dataset.target); pre.classList.toggle("expanded");
  b.textContent = pre.classList.contains("expanded") ? "collapse" : b.textContent.replace("collapse", "show all");
});
function showError(target, err) {
  $(target).innerHTML = `<div class="error">${escapeHtml(err.message || err)}</div>`;
}

/* -- latency (this browser session) ------------------------------------------------------------------------ */
const latencies = [];
function recordLatency(ms, serverMs) {
  latencies.push(ms);
  const sorted = [...latencies].sort((a, b) => a - b);
  const pct = (p) => sorted[Math.min(sorted.length - 1, Math.round((p / 100) * (sorted.length - 1)))];
  $("latency").innerHTML = `last query <b>${ms.toFixed(0)} ms</b>` +
    (serverMs != null ? ` (engine ${serverMs.toFixed(0)} ms)` : "") +
    ` · session p50 <b>${pct(50).toFixed(0)} ms</b> · p95 <b>${pct(95).toFixed(0)} ms</b> · n=${sorted.length}`;
}

/* -- P0 / P1 hit rendering --------------------------------------------------------------------------------- */
const STAGE_COLORS = { route_encode: "#8a63d2", encode: "#b27ee0", dense: "#2f5bd3", bm25: "#e0a340",
  features: "#3aa876", ranker: "#d9534f" };
function renderStages(t) {
  const stages = Object.entries(t).filter(([k]) => k.startsWith("stage.")).map(([k, v]) => [k.slice(6), v]);
  const total = stages.reduce((a, [, v]) => a + v, 0) || 1;
  const bars = stages.map(([k, v]) => `<span style="width:${(100 * v) / total}%;background:${STAGE_COLORS[k] || "#999"}" title="${k} ${v.toFixed(1)} ms"></span>`).join("");
  const legend = stages.map(([k, v]) => `<span><i style="background:${STAGE_COLORS[k] || "#999"}"></i>${k} ${v.toFixed(1)}</span>`).join("");
  return `<span class="stages">${bars}</span> <span class="legend">${legend}</span>`;
}
function renderHits(target, hits, explanation) {
  $(target).innerHTML = hits.map((h) => {
    const s = h.signals || {};
    const sim = s.similarity;
    const chips = [];
    if (s.dense_rank != null) chips.push(`<span class="badge accent" title="rank by embedding similarity over the whole snapshot">dense #${s.dense_rank}</span>`);
    if (s.bm25_rank != null) chips.push(`<span class="badge" title="BM25 score ${s.bm25.toFixed(2)}">BM25 #${s.bm25_rank}</span>`);
    else if (explanation && explanation.channel !== "dense") chips.push(`<span class="badge" title="not in BM25's top 100">BM25 —</span>`);
    if (h.unit.version && h.unit.version !== "-" && h.unit.version !== "v0") chips.push(`<span class="badge">${escapeHtml(h.unit.version)}</span>`);
    chips.push(`<span class="badge" title="content hash (INV-1)">${escapeHtml(h.unit.body_hash.slice(0, 10))}</span>`);
    const simHtml = sim != null
      ? `<div class="sim-value">${sim.toFixed(3)}</div><div class="sim-label">cosine similarity</div><div class="bar"><span style="width:${Math.max(0, Math.min(1, sim)) * 100}%"></span></div>`
      : `<div class="sim-label">no dense signal</div>`;
    return `<article class="hit">
      <div class="hit-head">
        <div class="rank">${h.rank}</div>
        <div><div class="title">${escapeHtml(h.unit.key || h.unit.unit_id)}</div><div class="chips">${chips.join("")}</div></div>
        <div class="sim">${simHtml}</div>
      </div>
      ${codeBlock(h.source)}
    </article>`;
  }).join("");
}
function renderSummary(target, r, wallMs) {
  const e = r.explanation || {};
  $(target).hidden = false;
  $(target).innerHTML = [
    `<span>ordered by <b>${escapeHtml(e.ordered_by || "-")}</b></span>`,
    `<span>route <span class="badge">${escapeHtml(r.route)}</span></span>`,
    `<span>snapshot <span class="badge">${escapeHtml(r.snapshot.id)}</span>${r.snapshot.version && r.snapshot.version !== "v0" ? ` <span class="badge accent">${escapeHtml(r.snapshot.version)}</span>` : ""}</span>`,
    r.no_strong_match ? `<span class="badge warn">no strong match</span>` : `<span class="badge good">confidence ${escapeHtml(r.confidence)}</span>`,
    `<span><b>${r.timings_ms.total.toFixed(0)} ms</b> engine · ${wallMs.toFixed(0)} ms wall</span>`,
    renderStages(r.timings_ms),
    ...(r.degradations || []).map((d) => `<span class="badge warn">${escapeHtml(d)}</span>`),
  ].join("");
}

/* -- P0: search the APPS corpus ---------------------------------------------------------------------------- */
async function searchP0(e) {
  e?.preventDefault();
  const query = $("q-search").value.trim(); if (!query) return;
  $("go-search").disabled = true; $("empty-search").hidden = true;
  const t0 = performance.now();
  try {
    const r = await api("/v1/search", { query, repo_id: "-", top_k: Number($("topk-search").value) || 10, mode: segValue("channel"), explain: true });
    const wall = performance.now() - t0;
    renderHits("results-search", r.results, r.explanation);
    renderSummary("summary-search", r, wall);
    recordLatency(wall, r.timings_ms.total);
  } catch (err) { showError("results-search", err); }
  finally { $("go-search").disabled = false; }
}

/* -- P1: pinned versions ----------------------------------------------------------------------------------- */
async function loadVersions() {
  const repo = $("repo-versions").value; if (!repo) return;
  const { versions } = await api(`/v1/repos/${encodeURIComponent(repo)}/versions`);
  $("version-versions").innerHTML = `<option value="latest">latest (active)</option>` +
    versions.map((v) => `<option value="${escapeHtml(v.label)}">${escapeHtml(v.label)}${v.active ? " · active" : ""}</option>`).join("");
  $("vtable").innerHTML = versions.map((v) => `<tr class="${v.active ? "active" : ""}"><td>${escapeHtml(v.label)}</td><td>${v.n_units}</td><td class="snap">${escapeHtml(String(v.snapshot_id).slice(0, 12))}</td><td>${v.active ? '<span class="badge good">active</span>' : ""}</td></tr>`).join("");
}
async function searchP1(e) {
  e?.preventDefault();
  const query = $("q-versions").value.trim(); if (!query) return;
  const version = $("selector-versions").value.trim() || $("version-versions").value;
  $("empty-versions").hidden = true;
  const t0 = performance.now();
  try {
    const r = await api("/v1/search", { query, repo_id: $("repo-versions").value, version, top_k: 10, mode: "auto", explain: true });
    const wall = performance.now() - t0;
    renderHits("results-versions", r.results, r.explanation);
    renderSummary("summary-versions", r, wall);
    recordLatency(wall, r.timings_ms.total);
  } catch (err) { showError("results-versions", err); }
}
async function commit() {
  const repo = $("repo-versions").value; if (!repo) return;
  const box = $("commit-result"); box.hidden = false; box.className = "commit-result"; box.textContent = "building…";
  $("commit").disabled = true;
  try {
    const r = await api(`/v1/repos/${encodeURIComponent(repo)}/commit`, { edits: Number($("commit-edits").value), seed: Date.now() % 100000 });
    box.innerHTML = `<b>${escapeHtml(r.version)}</b> searchable in <b>${r.seconds_searchable.toFixed(2)} s</b> · ` +
      `<b>${r.units_new}</b> unit(s) embedded, ${r.units_reused} reused (of ${r.units_total})<br>` +
      `<span class="muted">changed: ${r.changed.map(escapeHtml).join(", ") || "none"}</span>`;
    await loadVersions();
  } catch (err) { box.className = "commit-result error"; box.textContent = err.message; }
  finally { $("commit").disabled = false; }
}
async function rollback() {
  const repo = $("repo-versions").value; if (!repo) return;
  try { await api(`/v1/repos/${encodeURIComponent(repo)}/rollback`, {}); await loadVersions(); }
  catch (err) { const box = $("commit-result"); box.hidden = false; box.className = "commit-result error"; box.textContent = err.message; }
}

/* -- Bonus: across every version --------------------------------------------------------------------------- */
function renderGroups(groups, labels) {
  $("results-evolution").innerHTML = groups.map((g, i) => {
    const byVersion = Object.fromEntries(g.timeline.map((s) => [s.version, s]));
    const steps = labels.map((v) => {
      const s = byVersion[v];
      const cls = !s ? "absent" : `${s.changed ? "changed" : ""} ${v === g.best.version ? "best" : ""}`;
      const title = !s ? `${v} · not present` : `${v} · ${s.relation}${s.inferred ? " (inferred)" : ""}`;
      return `<span class="step"><span class="dot ${cls}" title="${escapeHtml(title)}"></span><span class="vlabel">${escapeHtml(v)}</span></span>`;
    }).join('<span class="rail"></span>');
    const members = g.members.map((m) => `<code>${escapeHtml(m.version)}</code>`).join(", ");
    return `<article class="hit">
      <div class="hit-head">
        <div class="rank">${i + 1}</div>
        <div><div class="title">${escapeHtml(g.best.key)}</div>
          <div class="chips"><span class="badge accent">best revision ${escapeHtml(g.best.version)}</span>
          <span class="badge">${g.n_revisions} revision(s) · ${escapeHtml(g.span[0])} → ${escapeHtml(g.span[1])}</span>
          ${g.inferred ? '<span class="badge warn">contains an inferred link</span>' : ""}
          <span class="badge" title="lineage id">${escapeHtml(g.lineage_id)}</span></div></div>
        <div class="sim"><div class="sim-label">history</div></div>
      </div>
      <div class="timeline">${steps}</div>
      <div class="members">matched in: ${members}</div>
      ${g.best.source ? codeBlock(g.best.source) : ""}
    </article>`;
  }).join("");
}
function renderFlat(hits) {
  const seen = new Map();
  $("results-evolution").innerHTML = hits.map((h) => {
    const first = seen.get(h.unit.key);
    if (first == null) seen.set(h.unit.key, h.rank);
    const dup = first != null ? `<span class="badge warn">same unit as #${first}</span>` : "";
    return `<article class="hit">
      <div class="hit-head"><div class="rank">${h.rank}</div>
        <div><div class="title">${escapeHtml(h.unit.key)}</div>
          <div class="chips"><span class="badge accent">${escapeHtml(h.unit.version)}</span>${dup}
          <span class="badge">${escapeHtml(h.unit.body_hash.slice(0, 10))}</span></div></div>
        <div class="sim"></div></div>
      ${codeBlock(h.source, 8)}
    </article>`;
  }).join("");
}
async function searchBonus(e) {
  e?.preventDefault();
  const query = $("q-evolution").value.trim(); if (!query) return;
  const repo = $("repo-evolution").value; if (!repo) return;
  const flat = segValue("evo-mode") === "flat";
  $("empty-evolution").hidden = true;
  const t0 = performance.now();
  try {
    const [r, vs] = await Promise.all([
      api("/v1/evolve", { query, repo_id: repo, top_k: 10, flat }),
      api(`/v1/repos/${encodeURIComponent(repo)}/versions`),
    ]);
    const wall = performance.now() - t0;
    if (flat) renderFlat(r.flat_results); else renderGroups(r.groups, vs.versions.map((v) => v.label));
    const facts = Object.fromEntries((r.degradations || []).map((d) => d.split("=")));
    $("summary-evolution").hidden = false;
    $("summary-evolution").innerHTML = [
      `<span><b>${flat ? "flat: every matching revision" : "grouped: one answer per lineage"}</b></span>`,
      facts.lineages ? `<span>lineages <b>${facts.lineages}</b></span>` : "",
      facts.versions ? `<span>versions searched <b>${facts.versions}</b></span>` : "",
      facts.flat_duplicate_rate ? `<span title="share of flat hits that repeat a lineage already listed">flat duplicate rate <b>${(100 * Number(facts.flat_duplicate_rate)).toFixed(0)}%</b></span>` : "",
      `<span><b>${wall.toFixed(0)} ms</b> wall</span>`,
    ].join("");
    recordLatency(wall, null);
  } catch (err) { showError("results-evolution", err); }
}

/* -- start-up ---------------------------------------------------------------------------------------------- */
async function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${t.dataset.view}`));
  }));
  bindSeg("channel"); bindSeg("evo-mode");
  $("form-search").addEventListener("submit", searchP0);
  $("form-versions").addEventListener("submit", searchP1);
  $("form-evolution").addEventListener("submit", searchBonus);
  for (const id of ["q-search", "q-versions", "q-evolution"]) {
    $(id).addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $(id).form.requestSubmit(); }
    });
  }
  $("commit").addEventListener("click", commit);
  $("rollback").addEventListener("click", rollback);
  $("repo-versions").addEventListener("change", loadVersions);
  try {
    const [health, ready] = await Promise.all([api("/healthz"), api("/readyz")]);
    $("pill-encoder").textContent = health.encoder;
    $("pill-encoder").classList.toggle("ok", !!health.submission_capable);
    $("pill-profile").textContent = `${health.numeric_profile} · ${health.threads} threads`;
    $("pill-corpus").textContent = ready.p0_corpus ? `APPS corpus ready${ready.p0_units ? ` · ${ready.p0_units.toLocaleString()} units` : ""}` : "APPS corpus not loaded";
    $("pill-corpus").classList.toggle("ok", !!ready.p0_corpus);
    const repos = health.repos || [];
    const options = repos.map((r) => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`).join("");
    $("repo-versions").innerHTML = options; $("repo-evolution").innerHTML = options;
    const preferred = repos.includes("apps-history") ? "apps-history" : repos[0];
    if (preferred) { $("repo-versions").value = preferred; $("repo-evolution").value = preferred; await loadVersions(); }
  } catch (err) { $("pill-encoder").textContent = `API unavailable: ${err.message}`; }
}
init();
