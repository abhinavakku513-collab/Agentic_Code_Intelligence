/* ACIS UI — vanilla, offline, no dependencies (D15).
 *
 * Everything shown comes from the API. Similarities and ranks are the engine's own signals for this query, timings
 * are the engine's stage timings, every code block is the text the engine re-read from the content store by hash
 * (INV-1), and the P0 evaluation panel is HTML the server rendered from its ledger row, inserted unchanged. Nothing
 * here ranks, re-scores, estimates or fills in a value the backend did not return.
 */
"use strict";
const $ = (id) => document.getElementById(id);

/* -- helpers ----------------------------------------------------------------------------------------------- */
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
  const text = await response.text();
  let payload; try { payload = JSON.parse(text); } catch { payload = { message: text || response.statusText }; }
  if (!response.ok) {
    const detail = payload.message || (Array.isArray(payload.detail) ? payload.detail.map((d) => d.msg).join("; ") : payload.detail);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return payload;
}
async function apiText(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.text();
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
  const count = source.split("\n").length;
  const id = `c${Math.random().toString(36).slice(2, 9)}`;
  const more = count > lines ? `<button class="more" data-target="${id}" type="button">show all ${count} lines</button>` : "";
  return `<pre class="code" id="${id}">${highlight(source)}</pre>${more}`;
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("button.more"); if (!b) return;
  const pre = $(b.dataset.target); pre.classList.toggle("expanded");
  b.textContent = pre.classList.contains("expanded") ? "collapse" : `show all ${pre.textContent.split("\n").length} lines`;
});
function showError(target, err) { $(target).innerHTML = `<div class="error">${escapeHtml(err.message || err)}</div>`; }
function notice(target, kind, html) {
  const el = $(target);
  if (!html) { el.hidden = true; el.innerHTML = ""; return; }
  el.hidden = false; el.className = `notice ${kind}`; el.innerHTML = html;
}
const fmtInt = (n) => Number(n).toLocaleString();

/* -- custom dropdown ----------------------------------------------------------------------------------------
 * A button and a listbox, in place of the native <select>: styled like everything else, keyboard-operable
 * (↑ ↓ Home End Enter Esc, type-ahead by first letter), and its value lives in `data-value`. */
const dropdowns = {};
function dropdown(id, { options = [], value, onChange } = {}) {
  const root = $(id);
  const state = dropdowns[id] || { root, options: [], value: null, onChange: null, active: 0 };
  dropdowns[id] = state;
  if (!root.querySelector(".dd-button")) {
    root.innerHTML = `<button type="button" class="dd-button" aria-haspopup="listbox" aria-expanded="false"><span class="dd-label"></span><span class="dd-chev"></span></button><ul class="dd-list" role="listbox" tabindex="-1" hidden></ul>`;
    const button = root.querySelector(".dd-button"), list = root.querySelector(".dd-list");
    button.addEventListener("click", () => (root.classList.contains("open") ? close(id) : open(id)));
    button.addEventListener("keydown", (e) => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) { e.preventDefault(); open(id); }
    });
    list.addEventListener("keydown", (e) => keyNav(id, e));
    list.addEventListener("click", (e) => {
      const li = e.target.closest("li"); if (!li) return;
      choose(id, li.dataset.value); close(id); button.focus();
    });
  }
  state.options = options.map((o) => (typeof o === "object" ? o : { value: String(o), label: String(o) }));
  if (onChange) state.onChange = onChange;
  const wanted = value ?? root.dataset.value ?? state.value;
  const found = state.options.find((o) => o.value === String(wanted)) || state.options[0];
  setValue(id, found ? found.value : "", false);
  root.querySelector(".dd-button").disabled = state.options.length === 0;
  return state;
}
function setValue(id, value, fire = true) {
  const s = dropdowns[id]; const option = s.options.find((o) => o.value === value);
  s.value = option ? option.value : ""; s.root.dataset.value = s.value;
  s.root.querySelector(".dd-label").textContent = option ? option.label : "none";
  if (fire && s.onChange) s.onChange(s.value);
}
function choose(id, value) { if (dropdowns[id].value !== value) setValue(id, value); }
function ddValue(id) { return dropdowns[id] ? dropdowns[id].value : $(id).dataset.value; }
function renderList(id) {
  const s = dropdowns[id]; const list = s.root.querySelector(".dd-list");
  list.innerHTML = s.options.map((o, i) => `<li role="option" data-value="${escapeHtml(o.value)}" aria-selected="${o.value === s.value}" class="dd-option${i === s.active ? " active" : ""}"><span>${escapeHtml(o.label)}</span>${o.hint ? `<span class="hint">${escapeHtml(o.hint)}</span>` : ""}</li>`).join("");
}
function open(id) {
  Object.keys(dropdowns).forEach((other) => other !== id && close(other));
  const s = dropdowns[id]; if (!s.options.length) return;
  s.active = Math.max(0, s.options.findIndex((o) => o.value === s.value));
  renderList(id); s.root.classList.add("open");
  const list = s.root.querySelector(".dd-list"); list.hidden = false; list.focus();
  s.root.querySelector(".dd-button").setAttribute("aria-expanded", "true");
}
function close(id) {
  const s = dropdowns[id]; if (!s) return;
  s.root.classList.remove("open"); s.root.querySelector(".dd-list").hidden = true;
  s.root.querySelector(".dd-button").setAttribute("aria-expanded", "false");
}
function keyNav(id, e) {
  const s = dropdowns[id]; const n = s.options.length;
  if (e.key === "Escape") { close(id); s.root.querySelector(".dd-button").focus(); return; }
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(id, s.options[s.active].value); close(id); s.root.querySelector(".dd-button").focus(); return; }
  if (e.key === "ArrowDown") s.active = (s.active + 1) % n;
  else if (e.key === "ArrowUp") s.active = (s.active - 1 + n) % n;
  else if (e.key === "Home") s.active = 0;
  else if (e.key === "End") s.active = n - 1;
  else if (e.key.length === 1) {
    const i = s.options.findIndex((o, j) => j > s.active && o.label.toLowerCase().startsWith(e.key.toLowerCase()));
    s.active = i >= 0 ? i : Math.max(0, s.options.findIndex((o) => o.label.toLowerCase().startsWith(e.key.toLowerCase())));
  } else return;
  e.preventDefault(); renderList(id);
}
document.addEventListener("click", (e) => { if (!e.target.closest(".dd")) Object.keys(dropdowns).forEach(close); });

/* -- theme ------------------------------------------------------------------------------------------------- */
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("acis-theme", theme); } catch (e) { /* storage may be unavailable */ }
}
$("theme").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

/* -- latency (this browser session) ------------------------------------------------------------------------ */
const latencies = [];
function recordLatency(ms, serverMs) {
  latencies.push(ms);
  const sorted = [...latencies].sort((a, b) => a - b);
  const pct = (p) => sorted[Math.min(sorted.length - 1, Math.round((p / 100) * (sorted.length - 1)))];
  $("latency").innerHTML = `last query <b>${ms.toFixed(0)} ms</b>` +
    (serverMs != null ? ` (engine ${serverMs.toFixed(0)} ms)` : "") +
    ` · this session p50 <b>${pct(50).toFixed(0)} ms</b> · p95 <b>${pct(95).toFixed(0)} ms</b> · n=${sorted.length}`;
}

/* -- corpus pill: always the corpus the current tab searches ------------------------------------------------ */
const corpus = { search: null, versions: null, evolution: null, evaluation: "recorded evaluation · dev split" };
function currentTab() { return document.querySelector(".tab.active").dataset.view; }
function showCorpus() {
  const text = corpus[currentTab()];
  $("pill-corpus").textContent = text || "no corpus loaded";
  $("pill-corpus").classList.toggle("ok", !!text);
}
function setCorpus(tab, text) { corpus[tab] = text; showCorpus(); }

/* -- P0 / P1 hit rendering --------------------------------------------------------------------------------- */
const STAGES = { route: ["route", "--stage-route"], route_encode: ["encode", "--stage-encode"], encode: ["encode", "--stage-encode"],
  encode_wait: ["queued for the model", "--stage-wait"], dense: ["dense", "--stage-dense"], bm25: ["BM25", "--stage-bm25"],
  features: ["features", "--stage-features"], ranker: ["learned ranker", "--stage-ranker"], fusion: ["weighted fusion", "--stage-fusion"] };
function renderTrace(t) {
  const merged = {};
  for (const [k, v] of Object.entries(t)) {
    if (!k.startsWith("stage.")) continue;
    const [label, color] = STAGES[k.slice(6)] || [k.slice(6), "--muted"];
    merged[label] = merged[label] || { ms: 0, color }; merged[label].ms += v;
  }
  const steps = Object.entries(merged).map(([label, s]) =>
    `<span class="step"><i style="background:var(${s.color})"></i>${escapeHtml(label)} ${s.ms.toFixed(1)} ms</span>`);
  return `<span class="trace">${steps.join('<span class="arrow">→</span>')}</span>`;
}
function renderHits(target, hits, explanation) {
  $(target).innerHTML = hits.map((h) => {
    const s = h.signals || {};
    const chips = [];
    if (s.dense_rank != null) chips.push(`<span class="badge accent" title="rank of this unit by embedding similarity over the whole snapshot">dense #${s.dense_rank}</span>`);
    if (s.bm25_rank != null) chips.push(`<span class="badge" title="BM25 score ${s.bm25.toFixed(2)}">BM25 #${s.bm25_rank}</span>`);
    else if (explanation && explanation.channel !== "dense") chips.push(`<span class="badge" title="not in BM25's top 100 for this query">BM25 —</span>`);
    if (h.unit.version && h.unit.version !== "v0") chips.push(`<span class="badge">${escapeHtml(h.unit.version)}</span>`);
    chips.push(`<span class="badge" title="content hash: the code below was re-read from the store by this hash (INV-1)">${escapeHtml(h.unit.body_hash.slice(0, 10))}</span>`);
    const sim = s.similarity;
    const simHtml = sim != null
      ? `<div class="sim-value">${sim.toFixed(3)}</div><div class="sim-label" title="cosine between this query's embedding and this unit's. A per-query signal, not an accuracy measure.">cosine similarity</div><div class="bar"><span style="width:${Math.max(0, Math.min(1, sim)) * 100}%"></span></div>`
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
function confidenceBadge(r) {
  const e = r.explanation || {};
  const basis = e.confidence_basis;
  const title = basis ? escapeHtml(basis) : "";
  const facts = e.confidence || {};
  if (facts.identifier_matches) return `<span class="badge good" title="${title}">exact identifier match · ${facts.identifier_matches} unit(s)</span>`;
  if (facts.calibrated === false) return `<span class="badge" title="${title}">confidence not calibrated</span>`;
  if (r.no_strong_match) return `<span class="badge warn" title="${title}">weak match: treat these results as suggestions</span>`;
  const kind = r.confidence === "high" ? "good" : r.confidence === "medium" ? "accent" : "warn";
  const p = facts.p != null ? ` · est. P(top result relevant) ${Number(facts.p).toFixed(2)}` : "";
  return `<span class="badge ${kind}" title="${title}">confidence ${escapeHtml(r.confidence)}${escapeHtml(p)}</span>`;
}
function renderSummary(target, r, wallMs) {
  const e = r.explanation || {};
  const rd = e.route_decision || {};
  $(target).hidden = false;
  $(target).innerHTML = [
    `<span>ordered by <b>${escapeHtml(e.ordered_by || "-")}</b></span>`,
    `<span title="${escapeHtml(rd.reason || "")}">route <span class="badge">${escapeHtml(r.route)}</span></span>`,
    confidenceBadge(r),
    `<span><b>${r.timings_ms.total.toFixed(0)} ms</b> engine · ${wallMs.toFixed(0)} ms wall</span>`,
    renderTrace(r.timings_ms),
    ...(r.degradations || []).map((d) => `<span class="badge bad" title="a fallback this request took">${escapeHtml(d)}</span>`),
  ].join("");
}
function renderContext(target, r, { expectRepo, versionsById }) {
  const snap = r.snapshot;
  const repoLabel = snap.repo_id === "-" ? "APPS P0 corpus" : `repository <b>${escapeHtml(snap.repo_id)}</b>`;
  const owned = expectRepo === snap.repo_id && (!versionsById || versionsById[snap.id] !== undefined);
  $(target).hidden = false;
  $(target).innerHTML = `<span>answered from ${repoLabel}</span>` +
    (snap.version && snap.version !== "v0" ? `<span>version <b>${escapeHtml(snap.version)}</b></span>` : "") +
    `<span>snapshot <code>${escapeHtml(snap.id)}</code></span><span><b>${fmtInt(snap.n_units)}</b> units</span>` +
    (owned ? `<span class="badge good" title="the snapshot that answered belongs to the corpus selected on this tab">matches the selection</span>`
           : `<span class="badge bad">does not match the selection</span>`);
  return owned;
}

/* -- P0: search the APPS corpus ---------------------------------------------------------------------------- */
async function searchP0(e) {
  e?.preventDefault();
  const query = $("q-search").value.trim(); if (!query) return;
  $("go-search").disabled = true; $("empty-search").hidden = true;
  const t0 = performance.now();
  try {
    const r = await api("/v1/search", { query, repo_id: "-", top_k: Number(ddValue("topk-search")) || 10, mode: segValue("channel"), explain: true });
    const wall = performance.now() - t0;
    renderContext("context-search", r, { expectRepo: "-" });
    notice("notice-search", "warn", r.no_strong_match
      ? "<b>No strong match.</b> The top results are only loosely related to the query by every signal the engine has. They are shown ranked, but treat them as suggestions."
      : "");
    renderHits("results-search", r.results, r.explanation);
    renderSummary("summary-search", r, wall);
    setCorpus("search", `APPS P0 corpus · ${fmtInt(r.snapshot.n_units)} units`);
    recordLatency(wall, r.timings_ms.total);
  } catch (err) { showError("results-search", err); $("summary-search").hidden = true; $("context-search").hidden = true; }
  finally { $("go-search").disabled = false; }
}

/* -- P1: pinned versions ----------------------------------------------------------------------------------- */
let versionsState = { repo: null, byId: {}, list: [], loading: null };
async function loadVersions() {
  const repo = ddValue("repo-versions"); if (!repo) return;
  $("go-versions").disabled = true;                       // no search while the version list belongs to another repo
  $("selector-versions").value = "";                     // a selector typed for another repository is not carried over
  const loading = api(`/v1/repos/${encodeURIComponent(repo)}/versions`);
  versionsState.loading = loading;
  try {
    const { versions } = await loading;
    if (versionsState.loading !== loading) return;       // a newer selection superseded this one
    versionsState = { repo, byId: Object.fromEntries(versions.map((v) => [v.snapshot_id, v.label])), list: versions, loading: null };
    dropdown("version-versions", {
      options: [{ value: "latest", label: "latest (active)" }, ...versions.map((v) => ({ value: v.label, label: v.label, hint: `${fmtInt(v.n_units)} units${v.active ? " · active" : ""}` }))],
      value: "latest",
    });
    $("vtable-repo").textContent = repo;
    $("vtable").innerHTML = versions.map((v) => `<tr class="${v.active ? "active" : ""}"><td>${escapeHtml(v.label)}</td><td>${fmtInt(v.n_units)}</td><td class="snap">${escapeHtml(String(v.snapshot_id).slice(0, 12))}</td><td>${v.active ? '<span class="badge good">active</span>' : ""}</td></tr>`).join("");
    const active = versions.find((v) => v.active);
    setCorpus("versions", `${repo} · ${active ? active.label : "?"} active · ${fmtInt(active ? active.n_units : 0)} units`);
  } catch (err) { showError("results-versions", err); }
  finally { if (versionsState.repo === repo) $("go-versions").disabled = false; }
}
async function searchP1(e) {
  e?.preventDefault();
  const query = $("q-versions").value.trim(); if (!query) return;
  const repo = ddValue("repo-versions");
  if (!repo || versionsState.repo !== repo) return;       // the version list is still loading for this repository
  const version = $("selector-versions").value.trim() || ddValue("version-versions");
  $("empty-versions").hidden = true; $("go-versions").disabled = true;
  const t0 = performance.now();
  try {
    const r = await api("/v1/search", { query, repo_id: repo, version, top_k: 10, mode: "auto", explain: true });
    const wall = performance.now() - t0;
    const owned = renderContext("context-versions", r, { expectRepo: repo, versionsById: versionsState.byId });
    if (!owned) {
      notice("notice-versions", "bad", `<b>Refusing to show these results:</b> the snapshot that answered (<code>${escapeHtml(r.snapshot.id)}</code>, repository <code>${escapeHtml(r.snapshot.repo_id)}</code>) is not a version of <code>${escapeHtml(repo)}</code>.`);
      $("results-versions").innerHTML = ""; $("summary-versions").hidden = true; return;
    }
    notice("notice-versions", "warn", r.no_strong_match ? "<b>No strong match</b> in this version. Results are shown ranked; treat them as suggestions." : "");
    renderHits("results-versions", r.results, r.explanation);
    renderSummary("summary-versions", r, wall);
    setCorpus("versions", `${repo} · ${r.snapshot.version} · ${fmtInt(r.snapshot.n_units)} units`);
    recordLatency(wall, r.timings_ms.total);
  } catch (err) { showError("results-versions", err); $("summary-versions").hidden = true; }
  finally { $("go-versions").disabled = false; }
}
async function commit() {
  const repo = ddValue("repo-versions"); if (!repo) return;
  const box = $("commit-result"); box.hidden = false; box.className = "commit-result"; box.textContent = "applying synthetic edits and building…";
  $("commit").disabled = true;
  try {
    const r = await api(`/v1/repos/${encodeURIComponent(repo)}/commit`, { edits: Number($("commit-edits").value), seed: Date.now() % 100000 });
    box.innerHTML = `<b>${escapeHtml(r.version)}</b> (synthetic) searchable in <b>${r.seconds_searchable.toFixed(2)} s</b> · ` +
      `<b>${r.units_new}</b> unit(s) embedded, ${fmtInt(r.units_reused)} reused of ${fmtInt(r.units_total)}<br>` +
      `<span class="muted">edited: ${r.changed.map(escapeHtml).join(", ") || "none"}</span>`;
    await loadVersions();
  } catch (err) { box.className = "commit-result error"; box.textContent = err.message; }
  finally { $("commit").disabled = false; }
}
async function rollback() {
  const repo = ddValue("repo-versions"); if (!repo) return;
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
      const title = !s ? `${v} · not present` : `${v} · ${s.relation}${s.inferred ? " (inferred)" : ""}${s.same_content_as ? ` · same content as ${s.same_content_as} (revert)` : ""}`;
      return `<span class="tl-step"><span class="dot ${cls}" title="${escapeHtml(title)}"></span><span class="vlabel">${escapeHtml(v)}</span></span>`;
    }).join('<span class="rail"></span>');
    const members = g.members.map((m) => `<code>${escapeHtml(m.version)}</code>`).join(" ");
    return `<article class="hit">
      <div class="hit-head">
        <div class="rank">${i + 1}</div>
        <div><div class="title">${escapeHtml(g.best.key)}</div>
          <div class="chips"><span class="badge accent">best revision ${escapeHtml(g.best.version)}</span>
          <span class="badge">${g.n_revisions} revision(s) · ${escapeHtml(g.span[0])} → ${escapeHtml(g.span[1])}</span>
          ${g.inferred ? '<span class="badge warn" title="at least one link in this lineage was inferred rather than exact">contains an inferred link</span>' : ""}
          ${g.timeline.filter((s) => s.same_content_as).map((s) => `<span class="badge" title="the content at ${escapeHtml(s.version)} is byte-identical to ${escapeHtml(s.same_content_as)}">${escapeHtml(s.version)} reverts to ${escapeHtml(s.same_content_as)}</span>`).join("")}
          <span class="badge" title="lineage id from the lineage index">${escapeHtml(g.lineage_id)}</span></div></div>
        <div class="sim"><div class="sim-label">history</div></div>
      </div>
      <div class="timeline">${steps}</div>
      <div class="members">matched in ${members}</div>
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
  const repo = ddValue("repo-evolution"); if (!repo) return;
  const flat = segValue("evo-mode") === "flat";
  $("empty-evolution").hidden = true; $("go-evolution").disabled = true;
  const t0 = performance.now();
  try {
    const [r, vs] = await Promise.all([
      api("/v1/evolve", { query, repo_id: repo, top_k: 10, flat }),
      api(`/v1/repos/${encodeURIComponent(repo)}/versions`),
    ]);
    const wall = performance.now() - t0;
    const labels = vs.versions.map((v) => v.label);
    if (flat) renderFlat(r.flat_results); else renderGroups(r.groups, labels);
    const facts = Object.fromEntries((r.degradations || []).map((d) => d.split("=")));
    $("context-evolution").hidden = false;
    $("context-evolution").innerHTML = `<span>answered from repository <b>${escapeHtml(repo)}</b></span><span><b>${labels.length}</b> versions searched</span>` +
      (facts.lineages ? `<span><b>${fmtInt(facts.lineages)}</b> lineages in the index</span>` : "");
    $("summary-evolution").hidden = false;
    $("summary-evolution").innerHTML = [
      `<span><b>${flat ? "flat: every matching revision" : "grouped by the lineage index: one answer per lineage"}</b></span>`,
      facts.flat_duplicate_rate ? `<span title="share of flat hits that repeat a lineage already listed">flat duplicate rate <b>${(100 * Number(facts.flat_duplicate_rate)).toFixed(0)}%</b></span>` : "",
      `<span><b>${wall.toFixed(0)} ms</b> wall</span>`,
    ].join("");
    setCorpus("evolution", `${repo} · all ${labels.length} versions`);
    recordLatency(wall, null);
  } catch (err) { showError("results-evolution", err); }
  finally { $("go-evolution").disabled = false; }
}

/* -- P0 evaluation (recorded) ------------------------------------------------------------------------------ */
const evalState = { offset: 0, limit: 25, total: 0 };
async function loadEvaluation() {
  $("load-eval").disabled = true;
  try {
    // Server-rendered from the ledger row; inserted exactly as received — no number is computed on this page.
    const html = await apiText("/v1/benchmarks/p0/panel");
    $("eval-panel").innerHTML = html;
    evalState.offset = 0;
    await loadEvalRows();
  } catch (err) { $("eval-panel").innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`; }
  finally { $("load-eval").disabled = false; }
}
async function loadEvalRows() {
  try {
    const page = await api(`/v1/benchmarks/p0/queries?only=${encodeURIComponent(ddValue("eval-filter"))}&offset=${evalState.offset}&limit=${evalState.limit}`);
    evalState.total = page.total;
    $("eval-count").textContent = page.total ? `${fmtInt(evalState.offset + 1)}–${fmtInt(Math.min(page.total, evalState.offset + evalState.limit))} of ${fmtInt(page.total)}` : "none";
    $("eval-prev").disabled = evalState.offset === 0;
    $("eval-next").disabled = evalState.offset + evalState.limit >= page.total;
    const rank = (v) => (v == null ? '<span class="badge warn">not in top 100</span>' : v <= 10 ? `<b>${v}</b>` : String(v));
    $("eval-rows").innerHTML = page.records.map((r) => `<tr data-q="${escapeHtml(r.query_id)}">
      <td class="id">${escapeHtml(r.query_id)}</td><td>${escapeHtml(r.route.route)}</td><td>${escapeHtml(r.ordered_by)}</td>
      <td>${rank(r.rank_full)}</td><td>${rank(r.rank_dense)}</td><td class="head">${escapeHtml(r.query_head)}</td></tr>`).join("");
  } catch (err) { $("eval-rows").innerHTML = `<tr><td colspan="6" class="muted">${escapeHtml(err.message)}</td></tr>`; $("eval-count").textContent = ""; }
}
async function showQuery(qid) {
  const box = $("eval-detail"); box.hidden = false; box.innerHTML = '<div class="muted">loading…</div>';
  try {
    const r = await api(`/v1/benchmarks/p0/queries/${encodeURIComponent(qid)}`);
    const gold = new Set(r.gold);
    const top = r.top10_full.map((d, i) => `<li class="${gold.has(d) ? "gold" : ""}"><span>${i + 1}.</span><span>${escapeHtml(d)}</span>${gold.has(d) ? "<span>relevant</span>" : ""}</li>`).join("");
    const goldCode = (r.gold_code || []).map((g) => g.available ? codeBlock(g.source, 16) : `<div class="muted">${escapeHtml(g.doc_id)} (P0 corpus not loaded)</div>`).join("");
    box.innerHTML = `<h3>Query <code>${escapeHtml(r.query_id)}</code> · fold ${r.fold}</h3>
      <p class="muted">route <b>${escapeHtml(r.route.route)}</b> (${escapeHtml(r.route.reason)}), ordered by <b>${escapeHtml(r.ordered_by)}</b>.
      Relevant document at rank <b>${r.rank_full ?? "beyond 100"}</b> in the served pipeline, <b>${r.rank_dense ?? "beyond 100"}</b> dense only.</p>
      <div class="detail-grid">
        <div><h3>Statement (first 600 characters of ${fmtInt(r.query_chars)})</h3><div class="statement">${escapeHtml(r.query_excerpt)}</div>
          <h3 style="margin-top:14px">Served top 10</h3><ol class="toplist">${top}</ol></div>
        <div><h3>Relevant document: ${r.gold.map((g) => `<code>${escapeHtml(g)}</code>`).join(" ")}</h3>${goldCode}</div>
      </div>`;
  } catch (err) { box.innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`; }
}

/* -- start-up ---------------------------------------------------------------------------------------------- */
async function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${t.dataset.view}`));
    showCorpus();
    if (t.dataset.view === "evaluation" && !$("eval-panel").dataset.loaded) { $("eval-panel").dataset.loaded = "1"; loadEvaluation(); }
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
  $("load-eval").addEventListener("click", loadEvaluation);
  $("eval-prev").addEventListener("click", () => { evalState.offset = Math.max(0, evalState.offset - evalState.limit); loadEvalRows(); });
  $("eval-next").addEventListener("click", () => { evalState.offset += evalState.limit; loadEvalRows(); });
  $("eval-rows").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-q]"); if (tr) showQuery(tr.dataset.q); });
  dropdown("topk-search", { options: ["10", "25", "50"], value: "10" });
  dropdown("eval-filter", {
    options: [{ value: "all", label: "all queries" }, { value: "missed", label: "relevant doc not in top 10" },
      { value: "improved", label: "ranked higher than dense" }, { value: "worsened", label: "ranked lower than dense" },
      { value: "generic", label: "routed generic" }],
    value: "all", onChange: () => { evalState.offset = 0; loadEvalRows(); },
  });
  dropdown("repo-versions", { options: [], onChange: loadVersions });
  dropdown("repo-evolution", { options: [], onChange: (repo) => setCorpus("evolution", `${repo} · all versions`) });
  dropdown("version-versions", { options: [] });
  try {
    const [health, ready] = await Promise.all([api("/healthz"), api("/readyz")]);
    $("pill-encoder").textContent = health.encoder;
    $("pill-encoder").classList.toggle("ok", !!health.submission_capable);
    $("pill-encoder").classList.toggle("warn", !health.submission_capable);
    $("pill-profile").textContent = `${health.numeric_profile} · ${health.threads} threads`;
    setCorpus("search", ready.p0_corpus ? `APPS P0 corpus · ${fmtInt(ready.p0_units)} units` : null);
    const repos = health.repos || [];
    const preferred = repos.includes("apps-history") ? "apps-history" : repos[0];
    dropdown("repo-evolution", { options: repos, value: preferred });
    if (preferred) setCorpus("evolution", `${preferred} · all versions`);
    dropdown("repo-versions", { options: repos, value: preferred });
    if (preferred) await loadVersions();
  } catch (err) { $("pill-encoder").textContent = `API unavailable: ${err.message}`; $("pill-encoder").classList.add("warn"); }
}
init();
