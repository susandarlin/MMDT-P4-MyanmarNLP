/* MNLP — single-page UI (no build step, no dependencies). */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const view = $("#view");
  const LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"];
  const LABEL_CLASS = { SUPPORTS: "SUPPORTS", REFUTES: "REFUTES", "NOT ENOUGH INFO": "NEI" };
  const MYANMAR = /[က-႟ꩠ-ꩿ]/;
  let files = [];

  // ---------- utils ----------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const badge = (label) => LABEL_CLASS[label]
    ? `<span class="badge ${LABEL_CLASS[label]}">${esc(label)}</span>` : esc(label);
  const fmtBytes = (n) => n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
  const isNum = (v) => v !== "" && !isNaN(v);
  // Data menu order, and a short display name for <id>_predictions.csv
  const FILE_ORDER = ["corpus.csv", "claims.csv", "train_claims.csv", "test_claims.csv"];
  const isPredictions = (n) => /_predictions.csv$/i.test(n);
  const fileRank = (n) => isPredictions(n) ? FILE_ORDER.length : FILE_ORDER.includes(n) ? FILE_ORDER.indexOf(n) : FILE_ORDER.length + 1;
  const displayName = (n) => isPredictions(n) ? "predictions.csv" : n;

  async function api(url, opts) {
    const res = await fetch(url, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || data.description || res.statusText);
    return data;
  }

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove("show"), 2400);
  }

  function setHeader(crumb, title) {
    $("#crumb").textContent = crumb;
    $("#page-title").textContent = title;
    document.title = `${title} · MNLP`;
  }

  function loading() {
    view.innerHTML = `<div class="card placeholder"><div><div class="spinner" style="margin:0 auto 10px"></div>Loading…</div></div>`;
  }

  function errorCard(msg) {
    view.innerHTML = `<div class="card empty">⚠️ ${esc(msg)}</div>`;
  }

  // ---------- theme ----------
  const root = document.documentElement;
  const savedTheme = (() => { try { return localStorage.getItem("mnlp-theme"); } catch { return null; } })();
  root.dataset.theme = savedTheme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  $("#theme-toggle").onclick = () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("mnlp-theme", root.dataset.theme); } catch { /* ignore */ }
  };

  // ---------- mobile menu ----------
  const dd = $("#data-dd");
  const setDropdown = (open) => { dd.classList.toggle("open", open); $("#data-btn").setAttribute("aria-expanded", open); };
  const closeMenu = () => {
    $("#menu").classList.remove("open");
    $("#menu-btn").setAttribute("aria-expanded", "false");
    setDropdown(false);
  };
  $("#menu-btn").onclick = () => {
    const open = $("#menu").classList.toggle("open");
    $("#menu-btn").setAttribute("aria-expanded", open);
  };
  $("#data-btn").onclick = (e) => { e.stopPropagation(); setDropdown(!dd.classList.contains("open")); };
  document.addEventListener("click", (e) => { if (!dd.contains(e.target)) setDropdown(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setDropdown(false); });

  // ---------- modal ----------
  const modal = $("#modal");
  const closeModal = () => (modal.hidden = true);
  $("#modal-close").onclick = closeModal;
  modal.onclick = (e) => { if (e.target === modal) closeModal(); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

  function showRow(title, columns, row) {
    $("#modal-title").textContent = title;
    $("#modal-body").innerHTML = columns.map((c, i) => `
      <div class="kv"><div class="k">${esc(c)}</div>
      <div class="v ${MYANMAR.test(row[i]) ? "mm" : ""}">${LABEL_CLASS[row[i]] ? badge(row[i]) : esc(row[i]) || "<span class='meta'>—</span>"}</div></div>`).join("");
    modal.hidden = false;
  }

  // ---------- "Data Files" dropdown in the top menu ----------
  const FILE_ICON = `<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zm-1 7V3.5L18.5 9zM8 13h8v2H8zm0 4h8v2H8z"/></svg>`;
  async function loadFiles() {
    try {
      files = (await api("/api/files")).sort((a, b) => fileRank(a.name) - fileRank(b.name));
      // <id>_predictions.csv is also shown as "predictions.csv", so hide a plain duplicate
      if (files.some((f) => isPredictions(f.name))) files = files.filter((f) => f.name.toLowerCase() !== "predictions.csv");
    } catch (e) {
      $("#file-nav").innerHTML = `<div class="dd-empty">Could not load files</div>`;
      return;
    }
    $("#file-nav").innerHTML = files.length
      ? files.map((f) => `
        <a href="#/data/${encodeURIComponent(f.name)}" data-route="data/${esc(f.name)}" class="dd-item" title="${esc(displayName(f.name))}">
          ${FILE_ICON}<span>${esc(displayName(f.name))}</span>
          <span class="count">${f.rows ?? "?"} rows</span>
        </a>`).join("")
      : `<div class="dd-empty">No CSV files in data/</div>`;
    markActive();
  }

  function markActive() {
    const route = decodeURIComponent(location.hash.replace(/^#\//, "")) || "overview";
    document.querySelectorAll(".menu-link[data-route]").forEach((a) => {
      const r = a.dataset.route;
      a.classList.toggle("active", r === "data" ? route.startsWith("data/") : r === route);
    });
    document.querySelectorAll(".dd-item, .tab").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
  }

  // ---------- views ----------
  async function renderOverview() {
    setHeader("", "Home");
    loading();
    let s;
    try { s = await api("/api/stats"); } catch (e) { return errorCard(e.message); }

    const statCard = (label, value, hint, icon) => `
      <div class="card stat">
        <div class="glyph">${icon}</div>
        <div class="label">${label}</div>
        <div class="value">${value}</div>
        <div class="hint">${hint}</div>
      </div>`;
    const bars = (counts, labelKeys) => {
      const keys = labelKeys || Object.keys(counts);
      const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
      if (!keys.length) return `<div class="meta">No data</div>`;
      return keys.map((k) => `
        <div class="bar-row">
          <div class="name ${MYANMAR.test(k) ? "mm" : ""}" title="${esc(k)}">${LABEL_CLASS[k] ? badge(k) : esc(k)}</div>
          <div class="bar-track"><div class="bar-fill ${LABEL_CLASS[k] || ""}" style="width:0" data-w="${((counts[k] || 0) / total) * 100}"></div></div>
          <div class="num">${counts[k] || 0}</div>
        </div>`).join("");
    };
    const p = s.predictions;
    const ICON = {
      doc: `<svg viewBox="0 0 24 24"><path d="M4 6H2v14a2 2 0 0 0 2 2h14v-2H4zm16-4H8a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2zm-1 9H9V9h10zm-4 4H9v-2h6zm4-8H9V5h10z"/></svg>`,
      claim: `<svg viewBox="0 0 24 24"><path d="M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z"/></svg>`,
      split: `<svg viewBox="0 0 24 24"><path d="M14 4l2.29 2.29-2.88 2.88 1.42 1.42 2.88-2.88L20 10V4zm-4 0H4v6l2.290-2.29 4.71 4.7V20h2v-8.41l-5.29-5.3z"/></svg>`,
      acc: `<svg viewBox="0 0 24 24"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>`,
    };

    view.innerHTML = `
      <div class="card hero">
        <div>
          <h2>Check if a Myanmar claim is true</h2>
          <p>Type a claim and the system finds matching documents in the corpus, then decides whether they support it, refute it, or don't say enough.</p>
        </div>
        <a class="btn primary" href="#/verify">Verify a claim →</a>
      </div>
      <div class="grid stats">
        ${statCard("Corpus documents", s.corpus_docs, "corpus.csv", ICON.doc)}
        ${statCard("Annotated claims", s.claims, "claims.csv", ICON.claim)}
        ${statCard("Train / Test", `${s.train} <span class="meta" style="font-size:18px">/</span> ${s.test}`, "train_claims · test_claims", ICON.split)}
        ${statCard("Test accuracy", p && p.accuracy != null ? `${(p.accuracy * 100).toFixed(1)}%` : "—",
          p && p.accuracy != null ? `${p.evaluated} claims vs claims.csv gold` : "no predictions yet", ICON.acc)}
      </div>
      <div class="grid two">
        <div class="card">
          <h3>Claim label distribution</h3>
          <p class="sub">All annotated claims in claims.csv</p>
          ${bars(s.claim_labels, LABELS.filter((l) => l in s.claim_labels))}
        </div>
        <div class="card">
          <h3>Predictions</h3>
          <p class="sub">${p ? `${esc(displayName(p.file))} · ${p.rows} rows · mean confidence ${p.mean_confidence ?? "—"}` : "No predictions file yet"}</p>
          ${p ? bars(p.labels, LABELS.filter((l) => l in p.labels)) : `<p class="sub">Run <code>python app.py</code> locally to generate predictions.</p>`}
        </div>
      </div>`;
    requestAnimationFrame(() => document.querySelectorAll(".bar-fill[data-w]").forEach((b) => (b.style.width = b.dataset.w + "%")));
  }

  // ----- CSV table -----
  async function renderTable(name) {
    setHeader("Data Files", displayName(name));
    loading();
    let data;
    try { data = await api(`/api/files/${encodeURIComponent(name)}`); } catch (e) { return errorCard(e.message); }
    const info = files.find((f) => f.name === name);
    const { columns, rows } = data;
    const labelCol = columns.findIndex((c) => /label/i.test(c) &&
      rows.some((r) => LABEL_CLASS[r[columns.indexOf(c)]]));
    const confCol = columns.findIndex((c) => /confidence|score/i.test(c));
    const numericCols = new Set(columns.map((_, i) => i).filter((i) => rows.length && rows.every((r) => r[i] === "" || isNum(r[i]))));

    const state = { q: "", sort: null, dir: 1, page: 0, size: 25, label: null };

    view.innerHTML = `
      <div class="tabs">${files.map((f) => `
        <a class="tab" href="#/data/${encodeURIComponent(f.name)}" data-route="data/${esc(f.name)}">${esc(displayName(f.name))}<span class="count">${f.rows ?? "?"}</span></a>`).join("")}
      </div>
      <div class="toolbar">
        <div class="search">
          <svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
          <input class="input" id="q" placeholder="Search all columns…" autocomplete="off">
        </div>
        ${labelCol >= 0 ? `<div class="chips" id="chips">
          <button class="chip on" data-l="">All</button>
          ${LABELS.map((l) => `<button class="chip" data-l="${l}">${l}</button>`).join("")}
        </div>` : ""}
        <div class="spacer"></div>
        <span class="meta">${columns.length} columns${info ? ` · ${fmtBytes(info.size)} · updated ${info.modified.replace("T", " ")}` : ""}</span>
        <a class="btn" href="/api/files/${encodeURIComponent(name)}/raw">
          <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7zM5 18v2h14v-2z"/></svg>Download
        </a>
      </div>
      <div class="card table-card">
        <div class="table-wrap"><table>
          <thead><tr><th class="rownum">#</th>${columns.map((c, i) => `<th data-i="${i}">${esc(c)}<span class="arrow">↕</span></th>`).join("")}</tr></thead>
          <tbody id="tbody"></tbody>
        </table></div>
        <div class="pager">
          <span class="meta" id="pinfo"></span>
          <div class="btns">
            <select class="input" id="psize">${[10, 25, 50, 100].map((n) => `<option ${n === 25 ? "selected" : ""}>${n}</option>`).join("")}</select>
            <button class="btn" id="prev">‹ Prev</button>
            <button class="btn" id="next">Next ›</button>
          </div>
        </div>
      </div>`;

    const highlight = (text) => {
      const safe = esc(text);
      if (!state.q) return safe;
      const re = new RegExp(esc(state.q).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
      return safe.replace(re, (m) => `<mark>${m}</mark>`);
    };

    function cell(v, i) {
      if (i === labelCol && LABEL_CLASS[v]) return badge(v);
      if (i === confCol && isNum(v)) {
        const p = Math.max(0, Math.min(1, +v)) * 100;
        return `<div class="conf"><div class="bar-track"><div class="bar-fill" style="width:${p}%"></div></div><span class="num">${(+v).toFixed(2)}</span></div>`;
      }
      if (/^(DOC|CLM)[_\w]*$/.test(v)) return `<span class="pill">${highlight(v)}</span>`;
      return `<div class="cell">${highlight(v)}</div>`;
    }

    function filtered() {
      let out = rows.map((r, idx) => ({ r, idx }));
      if (state.label) out = out.filter(({ r }) => r[labelCol] === state.label);
      if (state.q) {
        const q = state.q.toLowerCase();
        out = out.filter(({ r }) => r.some((v) => v.toLowerCase().includes(q)));
      }
      if (state.sort != null) {
        const i = state.sort, numeric = numericCols.has(i);
        out.sort((a, b) => state.dir * (numeric
          ? (+a.r[i] || 0) - (+b.r[i] || 0)
          : a.r[i].localeCompare(b.r[i], undefined, { numeric: true })));
      }
      return out;
    }

    function draw() {
      const all = filtered();
      const pages = Math.max(1, Math.ceil(all.length / state.size));
      state.page = Math.min(state.page, pages - 1);
      const slice = all.slice(state.page * state.size, (state.page + 1) * state.size);
      $("#tbody").innerHTML = slice.length
        ? slice.map(({ r, idx }) => `<tr data-idx="${idx}"><td class="rownum">${idx + 1}</td>${r.map((v, i) =>
            `<td class="${numericCols.has(i) && i !== confCol ? "num" : ""}">${cell(v, i)}</td>`).join("")}</tr>`).join("")
        : `<tr><td colspan="${columns.length + 1}" class="empty">No matching rows</td></tr>`;
      const from = all.length ? state.page * state.size + 1 : 0;
      $("#pinfo").textContent = `Showing ${from}–${Math.min(all.length, (state.page + 1) * state.size)} of ${all.length} rows` +
        (all.length !== rows.length ? ` (filtered from ${rows.length})` : "");
      $("#prev").disabled = state.page === 0;
      $("#next").disabled = state.page >= pages - 1;
      document.querySelectorAll("th[data-i]").forEach((th) => {
        const on = +th.dataset.i === state.sort;
        th.classList.toggle("sorted", on);
        th.querySelector(".arrow").textContent = on ? (state.dir > 0 ? "↑" : "↓") : "↕";
      });
    }

    $("#q").oninput = (e) => { state.q = e.target.value.trim(); state.page = 0; draw(); };
    $("#psize").onchange = (e) => { state.size = +e.target.value; state.page = 0; draw(); };
    $("#prev").onclick = () => { state.page--; draw(); };
    $("#next").onclick = () => { state.page++; draw(); };
    document.querySelectorAll("th[data-i]").forEach((th) => (th.onclick = () => {
      const i = +th.dataset.i;
      if (state.sort === i) state.dir === 1 ? (state.dir = -1) : (state.sort = null);
      else { state.sort = i; state.dir = 1; }
      draw();
    }));
    const chips = $("#chips");
    if (chips) chips.onclick = (e) => {
      const b = e.target.closest(".chip");
      if (!b) return;
      chips.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c === b));
      state.label = b.dataset.l || null; state.page = 0; draw();
    };
    $("#tbody").onclick = (e) => {
      const tr = e.target.closest("tr[data-idx]");
      if (tr) showRow(`${displayName(name)} · row ${+tr.dataset.idx + 1}`, columns, rows[+tr.dataset.idx]);
    };
    draw();
    markActive();
  }

  // ----- verify -----
  async function renderVerify() {
    setHeader("", "Verify a Claim");
    view.innerHTML = `
      <div class="verify-grid">
        <div class="card">
          <h3>Enter a Myanmar claim</h3>
          <p class="sub">The claim is tokenized into syllables, matched against the corpus with TF-IDF cosine similarity, and checked against the best evidence.</p>
          <textarea class="input" id="claim" placeholder="ဥပမာ - သုတ္တန်ပိဋကတွင် နိကာယ်ငါးရပ် ပါဝင်သည်။"></textarea>
          <div style="display:flex;gap:10px;margin-top:12px;align-items:center">
            <button class="btn primary" id="go">
              <svg viewBox="0 0 24 24"><path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg>Verify claim
            </button>
            <button class="btn" id="clear">Clear</button>
            <span class="meta" style="margin-left:auto">Ctrl + Enter</span>
          </div>
          <div class="section-title">Try an example</div>
          <div class="examples" id="examples"><span class="meta">Loading…</span></div>
        </div>
        <div class="card" id="result">
          <div class="placeholder"><div>
            <div class="ico"><svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg></div>
            <b>No claim verified yet</b><br>Results, retrieved evidence and the grounded explanation will appear here.
          </div></div>
        </div>
      </div>`;

    const ta = $("#claim");
    $("#clear").onclick = () => { ta.value = ""; ta.focus(); };
    ta.onkeydown = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); };
    $("#go").onclick = run;

    // examples: two per label from claims.csv
    api("/api/files/claims.csv").then(({ columns, rows }) => {
      const t = columns.indexOf("claim_text"), l = columns.indexOf("label");
      if (t < 0) throw new Error();
      const picks = [];
      LABELS.forEach((lab) => rows.filter((r) => r[l] === lab).slice(0, 2).forEach((r) => picks.push(r)));
      $("#examples").innerHTML = picks.map((r) =>
        `<button class="example" data-t="${esc(r[t])}"><span>${esc(r[t])}</span>${l >= 0 ? badge(r[l]) : ""}</button>`).join("");
      $("#examples").onclick = (e) => {
        const b = e.target.closest(".example");
        if (b) { ta.value = b.dataset.t; run(); }
      };
    }).catch(() => ($("#examples").innerHTML = `<span class="meta">claims.csv not available</span>`));

    async function run() {
      const claim = ta.value.trim();
      if (!claim) return toast("Please enter a claim first");
      const btn = $("#go");
      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span>Verifying…`;
      try {
        const r = await api("/api/verify", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim }),
        });
        showResult(r);
      } catch (e) {
        $("#result").innerHTML = `<div class="empty">⚠️ ${esc(e.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg>Verify claim`;
      }
    }

    function showResult(r) {
      const cls = LABEL_CLASS[r.predicted_label] || "";
      const pct = Math.round(r.confidence * 100);
      const ringColor = { SUPPORTS: "var(--sup)", REFUTES: "var(--ref)", NEI: "var(--nei)" }[cls];
      const maxScore = Math.max(...r.top_k.map((d) => d.score), 0.0001);
      $("#result").innerHTML = `
        <div class="verdict">
          <div class="ring" style="--p:${pct};--ring-c:${ringColor}"><span>${pct}%</span></div>
          <div>
            <div class="meta">Predicted stance</div>
            <div class="big ${cls}">${esc(r.predicted_label)}</div>
            <div class="meta">rule fired: <span class="pill">${esc(r.reason)}</span> · best cosine ${r.similarity.toFixed(3)} · τ = ${r.tau}</div>
          </div>
        </div>
        <div class="section-title">Grounded explanation</div>
        <div class="mm">${esc(r.explanation)}</div>
        ${r.evidence_sentences?.length ? `
          <div class="section-title">Evidence sentences · ${esc(r.retrieved_evidence_id)}</div>
          ${r.evidence_sentences.map((s) => `<div class="evidence mm">${esc(s)}</div>`).join("")}` : ""}
        <div class="section-title">Top-${r.top_k.length} retrieved documents</div>
        ${r.top_k.map((d, i) => `
          <div class="doc-hit">
            <span class="pill">${esc(d.doc_id)}</span>
            <div style="min-width:0">
              <div class="title" title="${esc(d.title)}">${esc(d.title)}</div>
              <div class="bar-track" style="height:6px;margin-top:4px"><div class="bar-fill" style="width:${(d.score / maxScore) * 100}%;${i ? "opacity:.55" : ""}"></div></div>
            </div>
            <span class="score">${d.score.toFixed(3)}</span>
          </div>`).join("")}`;
    }
  }

  // ----- pipeline -----
  function renderPipeline() {
    setHeader("", "Run Pipeline");
    const steps = [
      ["Load", "corpus + claims (UTF-8)"], ["Preprocess", "NFC normalize, sylbreak, stopwords"],
      ["TF-IDF", "syllable uni + bigrams"], ["Naive Bayes", "baseline on claim text"],
      ["Retrieve", "cosine Top-3 + τ"], ["Verify", "stance rules + RAG explanation"], ["Export", "<id>_predictions.csv"],
    ];
    view.innerHTML = `
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
          <div>
            <h3>studentverifier()</h3>
            <p class="sub" style="margin:0">Runs the full pipeline from <code>app.py</code> on <code>data/</code> and regenerates the predictions CSV.</p>
          </div>
          <button class="btn primary" id="run"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>Run pipeline</button>
        </div>
        <div class="steps">${steps.map(([b, t], i) => `<div class="step"><b>${i + 1} · ${b}</b>${esc(t)}</div>`).join("")}</div>
        <div id="out"></div>
      </div>`;

    $("#run").onclick = async () => {
      const btn = $("#run");
      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span>Running…`;
      $("#out").innerHTML = `<pre class="log">Running studentverifier()…</pre>`;
      try {
        const r = await api("/api/run", { method: "POST" });
        const log = esc(r.log).replace(/^(STEP \d+:.*)$/gm, `<span class="h">$1</span>`);
        $("#out").innerHTML = `
          <div class="toolbar"><span class="badge SUPPORTS">Completed</span>
            <span class="meta">Wrote <b>${esc(displayName(r.file))}</b></span><div class="spacer"></div>
            <a class="btn" href="#/data/${encodeURIComponent(r.file)}">View predictions →</a></div>
          <pre class="log">${log}</pre>`;
        toast("Predictions regenerated");
        loadFiles();
      } catch (e) {
        $("#out").innerHTML = `<div class="toolbar"><span class="badge REFUTES">Failed</span><span class="meta">${esc(e.message)}</span></div>`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>Run again`;
      }
    };
  }

  // ---------- router ----------
  function route() {
    closeMenu();
    closeModal();
    const path = decodeURIComponent(location.hash.replace(/^#\//, ""));
    markActive();
    if (path === "data" || path === "data/") return files.length ? (location.hash = `#/data/${encodeURIComponent(files[0].name)}`) : renderOverview();
    if (path.startsWith("data/")) return renderTable(path.slice(5));
    if (path === "verify") return renderVerify();
    return renderOverview();
  }

  window.addEventListener("hashchange", route);
  loadFiles().then(route);
})();
