/* ACIS demo UI. Vanilla, offline, no dependencies — the machine running this has no network (D15).
 *
 * Everything shown comes from the API's own response: the timings are the engine's timings, the badges are its
 * degradation counters, and the code in each result is the text the engine re-read from the content store by
 * hash. Nothing here re-ranks, re-scores or embellishes what the engine returned.
 */
const $ = (id) => document.getElementById(id);
const els = {
  form: $("form"), q: $("q"), go: $("go"), results: $("results"), empty: $("empty"),
  statusbar: $("statusbar"), timings: $("timings"), badges: $("badges"),
  repo: $("repo"), version: $("version"), topk: $("topk"), evolve: $("evolve"), evolveWrap: $("evolve-wrap"),
};

/* -- a five-token Python highlighter. Deliberately small: correctness here is cosmetic. ------------------- */
const KEYWORDS = new Set(("False None True and as assert async await break class continue def del elif else " +
  "except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield"
).split(" "));

function highlight(source) {
  const out = [];
  const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b\d+\.?\d*\b)|([A-Za-z_][A-Za-z0-9_]*)/g;
  let last = 0, m;
  while ((m = re.exec(source)) !== null) {
    out.push(escapeHtml(source.slice(last, m.index)));
    const [text, comment, str, num, word] = m;
    if (comment) out.push(`<span class="tok-com">${escapeHtml(text)}</span>`);
    else if (str) out.push(`<span class="tok-str">${escapeHtml(text)}</span>`);
    else if (num) out.push(`<span class="tok-num">${escapeHtml(text)}</span>`);
    else if (word && KEYWORDS.has(word)) out.push(`<span class="tok-kw">${escapeHtml(text)}</span>`);
    else if (word && (source[m.index - 1] === " ") && /\bdef |class /.test(source.slice(Math.max(0, m.index - 6), m.index))) {
      out.push(`<span class="tok-def">${escapeHtml(text)}</span>`);
    } else out.push(escapeHtml(text));
    last = m.index + text.length;
  }
  out.push(escapeHtml(source.slice(last)));
  return out.join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* -- API ------------------------------------------------------------------------------------------------- */
async function api(path, body) {
  const response = await fetch(path, body
    ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
    : {});
  const payload = await response.json().catch(() => ({ message: response.statusText }));
  if (!response.ok) throw new Error(payload.message || `HTTP ${response.status}`);
  return payload;
}

/* -- rendering ------------------------------------------------------------------------------------------- */
function firstLine(text) {
  return (text.split("\n").find((l) => l.trim()) || "").slice(0, 120);
}

function renderTimings(timings) {
  const entries = Object.entries(timings).filter(([k]) => k !== "total").sort();
  const total = timings.total || entries.reduce((a, [, v]) => a + v, 0) || 1;
  els.timings.innerHTML =
    `<strong>${(timings.total ?? total).toFixed(1)} ms</strong>` +
    entries.map(([name, ms]) =>
      `<span class="bar" title="${name}">${name} <i class="${name}" style="width:${Math.max(3, (ms / total) * 90)}px"></i> ${ms.toFixed(1)}ms</span>`
    ).join("");
}

function renderBadges(response, extra = []) {
  const badges = [];
  if (response.route) badges.push(["route " + response.route, ""]);
  if (response.confidence) badges.push(["confidence " + response.confidence, response.confidence === "high" ? "good" : ""]);
  if (response.no_strong_match) badges.push(["no strong match", "warn"]);
  if (response.snapshot) badges.push([`snapshot ${response.snapshot.id.slice(0, 12)}`, ""]);
  (response.degradations || []).forEach((d) => badges.push([d, "warn"]));
  extra.forEach((b) => badges.push(b));
  els.badges.innerHTML = badges.map(([text, cls]) => `<span class="badge ${cls}">${escapeHtml(text)}</span>`).join("");
}

function renderHits(hits) {
  els.results.innerHTML = hits.map((hit, i) => `
    <details class="hit" ${i === 0 ? "open" : ""}>
      <summary>
        <span class="rank">${hit.rank}</span>
        <span class="key">${escapeHtml(hit.unit.key || hit.unit.unit_id)}</span>
        <span class="peek">${escapeHtml(firstLine(hit.source))}</span>
        <span class="score">${hit.score.toFixed(4)}</span>
        <span class="hash">${escapeHtml(hit.unit.body_hash.slice(0, 10))}</span>
      </summary>
      <pre><code>${highlight(hit.source)}</code></pre>
    </details>`).join("");
}

function renderGroups(groups) {
  els.results.innerHTML = groups.map((g, i) => {
    const steps = g.timeline.map((step) => `
      <span class="step">
        <span class="dot ${step.changed ? "changed" : ""} ${step.version === g.best.version ? "best" : ""}"
              title="${escapeHtml(step.version)} · ${escapeHtml(step.relation)}${step.inferred ? " (inferred)" : ""}"></span>
        <span class="vlabel">${escapeHtml(step.version)}</span>
      </span>`).join('<span class="rail"></span>');
    return `
      <details class="hit group" ${i === 0 ? "open" : ""}>
        <summary>
          <span class="rank">${i + 1}</span>
          <span class="key">${escapeHtml(g.best.key)}</span>
          <span class="peek">${g.n_revisions} revisions · ${escapeHtml(g.span[0])} → ${escapeHtml(g.span[1])}</span>
          <span class="score">${g.score.toFixed(4)}</span>
        </summary>
        <div class="meta">
          best revision <strong>${escapeHtml(g.best.version)}</strong>
          ${g.inferred ? '<span class="badge warn">contains an inferred link</span>' : ""}
          <span class="hash">${escapeHtml(g.lineage_id)}</span>
        </div>
        <div class="timeline">${steps}</div>
        ${g.best.source ? `<pre><code>${highlight(g.best.source)}</code></pre>` : ""}
      </details>`;
  }).join("");
}

/* -- actions --------------------------------------------------------------------------------------------- */
async function search(event) {
  event?.preventDefault();
  const query = els.q.value.trim();
  if (!query) return;
  els.go.disabled = true;
  els.empty.hidden = true;
  const started = performance.now();
  try {
    const repo = els.repo.value;
    if (repo !== "-" && els.evolve.checked) {
      const response = await api("/v1/evolve", {
        query, repo_id: repo, top_k: Number(els.topk.value), flat: false, prefer: "best",
      });
      renderGroups(response.groups);
      renderTimings({ total: performance.now() - started });
      renderBadges({ degradations: response.degradations }, [["grouped by lineage", "good"]]);
    } else {
      const response = await api("/v1/search", {
        query, repo_id: repo, version: els.version.value, top_k: Number(els.topk.value),
        mode: document.querySelector('input[name="mode"]:checked').value, explain: true,
      });
      renderHits(response.results);
      renderTimings(response.timings_ms);
      renderBadges(response);
    }
    els.statusbar.hidden = false;
  } catch (error) {
    els.results.innerHTML = "";
    els.statusbar.hidden = false;
    els.timings.innerHTML = "";
    els.badges.innerHTML = `<span class="badge warn">${escapeHtml(error.message)}</span>`;
  } finally {
    els.go.disabled = false;
  }
}

async function loadVersions() {
  const repo = els.repo.value;
  els.evolveWrap.hidden = repo === "-";
  els.version.innerHTML = '<option value="latest">latest</option>';
  if (repo === "-") return;
  try {
    const { versions } = await api(`/v1/repos/${encodeURIComponent(repo)}/versions`);
    versions.forEach((v) => {
      const option = document.createElement("option");
      option.value = v.label;
      option.textContent = `${v.label}${v.active ? " (active)" : ""} · ${v.n_units} units`;
      els.version.appendChild(option);
    });
  } catch { /* a repository with no versions yet is not an error worth shouting about */ }
}

async function boot() {
  try {
    const health = await api("/healthz");
    $("v-encoder").textContent = health.encoder;
    $("v-profile").textContent = health.numeric_profile;
    $("v-threads").textContent = health.threads;
    const status = $("v-status");
    status.textContent = health.submission_capable ? health.status : `${health.status} · stand-in encoder`;
    status.className = health.submission_capable ? "good" : "warn";
    (health.repos || []).forEach((repo) => {
      const option = document.createElement("option");
      option.value = option.textContent = repo;
      els.repo.appendChild(option);
    });
  } catch (error) {
    $("v-status").textContent = "unreachable";
    $("v-status").className = "warn";
  }
  els.repo.addEventListener("change", loadVersions);
  els.form.addEventListener("submit", search);
  els.q.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) search(e); });
}

boot();
