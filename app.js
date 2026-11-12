// Dashboard logic: drawing, calling /api/predict, rendering both models' output.
"use strict";

const $ = (id) => document.getElementById(id);
// STATIC = GitHub Pages build: models run in the browser (inference.js) instead
// of on the Python server. Everything else is the same code.
const STATIC = !!window.HWR_STATIC;
const REPORTS_BASE = STATIC ? "reports/" : "/reports/";
const guides = $("guides"), ink = $("ink"), overlay = $("overlay");
const gctx = guides.getContext("2d"), ictx = ink.getContext("2d"), octx = overlay.getContext("2d");

const state = {
  geom: null,          // canvas size, brush and guide-line fractions (from the server = training config)
  strokes: [],         // finished strokes, each an array of [x, y]
  current: null,       // stroke being drawn
  timer: null,         // auto-recognize debounce
  busy: false,
  pending: false,
  last: null,          // last prediction
};

// ------------------------------------------------------------------ setup
async function init() {
  let info;
  if (STATIC) {
    $("latency").textContent = "Loading models…";
    info = await fetch("config.json").then((r) => r.json());
    try {
      await HWR.load(info);
      $("latency").textContent = "Models loaded — everything runs in your browser.";
    } catch (err) {
      $("latency").textContent = "Could not load the models: " + err.message;
    }
    staticMode(info);
  } else {
    info = await fetch("/api/info").then((r) => r.json());
  }
  state.geom = info.geometry;
  for (const c of [guides, ink, overlay]) {
    c.width = state.geom.canvas_w;
    c.height = state.geom.canvas_h;
  }
  $("canvasWrap").style.aspectRatio = `${state.geom.canvas_w} / ${state.geom.canvas_h}`;
  drawGuides();
  renderInfo(info);
  bindEvents();
}

function drawGuides() {
  const { canvas_w: w, canvas_h: h, asc_line, xh_line, base_line, desc_line } = state.geom;
  const css = getComputedStyle(document.documentElement);
  gctx.clearRect(0, 0, w, h);
  const line = (frac, color, dash, width) => {
    gctx.beginPath();
    gctx.setLineDash(dash);
    gctx.strokeStyle = color;
    gctx.lineWidth = width;
    gctx.moveTo(0, frac * h);
    gctx.lineTo(w, frac * h);
    gctx.stroke();
  };
  line(asc_line, css.getPropertyValue("--guide"), [], 2);
  line(xh_line, css.getPropertyValue("--guide"), [14, 10], 2);
  line(base_line, css.getPropertyValue("--guide-strong"), [], 3);
  line(desc_line, css.getPropertyValue("--guide"), [], 2);
  gctx.setLineDash([]);
  gctx.fillStyle = css.getPropertyValue("--guide-strong");
  gctx.font = "16px system-ui";
  gctx.fillText("A b 7", 10, asc_line * h - 6);
  gctx.fillText("a c e", 10, xh_line * h - 6);
  gctx.fillText("baseline", 10, base_line * h - 6);
  gctx.fillText("g p y", 10, desc_line * h - 6);
}

// ------------------------------------------------------------------ drawing
function toCanvas(e) {
  const r = ink.getBoundingClientRect();
  return [(e.clientX - r.left) * (ink.width / r.width), (e.clientY - r.top) * (ink.height / r.height)];
}

function drawStroke(pts) {
  ictx.strokeStyle = ictx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink");
  ictx.lineWidth = state.geom.brush_px;
  ictx.lineCap = ictx.lineJoin = "round";
  if (pts.length === 1) {                       // a dot (e.g. on an 'i')
    ictx.beginPath();
    ictx.arc(pts[0][0], pts[0][1], state.geom.brush_px / 2, 0, 2 * Math.PI);
    ictx.fill();
    return;
  }
  ictx.beginPath();
  ictx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length - 1; i++) {    // smooth curve through midpoints
    const mx = (pts[i][0] + pts[i + 1][0]) / 2, my = (pts[i][1] + pts[i + 1][1]) / 2;
    ictx.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
  }
  const last = pts[pts.length - 1];
  ictx.lineTo(last[0], last[1]);
  ictx.stroke();
}

function redraw() {
  ictx.clearRect(0, 0, ink.width, ink.height);
  for (const s of state.strokes) drawStroke(s);
  if (state.current) drawStroke(state.current);
}

function bindEvents() {
  ink.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    ink.setPointerCapture(e.pointerId);
    state.current = [toCanvas(e)];
    clearTimeout(state.timer);
    octx.clearRect(0, 0, overlay.width, overlay.height);
    redraw();
  });
  ink.addEventListener("pointermove", (e) => {
    if (!state.current) return;
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of evs) state.current.push(toCanvas(ev));
    redraw();
  });
  const end = () => {
    if (!state.current) return;
    state.strokes.push(state.current);
    state.current = null;
    redraw();
    scheduleAuto();
  };
  ink.addEventListener("pointerup", end);
  ink.addEventListener("pointercancel", end);

  $("btnRecognize").onclick = recognize;
  $("btnUndo").onclick = undo;
  $("btnClear").onclick = clearAll;
  $("chkBoxes").onchange = () => drawBoxes(state.last);
  if ($("btnReload")) $("btnReload").onclick = reloadModels;      // absent in the static build
  if ($("sampleForm")) $("sampleForm").onsubmit = saveSample;
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;    // typing a label
    if (e.key === "Enter") { e.preventDefault(); recognize(); }
    else if (e.key === "Escape") clearAll();
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawGuides(); redraw(); });
}

function scheduleAuto() {
  clearTimeout(state.timer);
  if ($("chkAuto").checked) state.timer = setTimeout(recognize, 600);
}

function undo() {
  state.strokes.pop();
  redraw();
  if (state.strokes.length) scheduleAuto(); else resetResults();
}

function clearAll() {
  state.strokes = [];
  state.current = null;
  redraw();
  resetResults();
}

// ------------------------------------------------------------------ prediction
async function recognize() {
  if (!state.strokes.length) { resetResults(); return; }
  if (state.busy) { state.pending = true; return; }
  state.busy = true;
  try {
    if (STATIC) {
      render(await HWR.recognize(ink));
    } else {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: ink.toDataURL("image/png") }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      render(await res.json());
    }
  } catch (err) {
    $("latency").textContent = "Error: " + err.message;
  } finally {
    state.busy = false;
    if (state.pending) { state.pending = false; recognize(); }
  }
}

function confClass(p) { return p >= 0.9 ? "c-good" : p >= 0.6 ? "c-mid" : "c-bad"; }
function pct(p) { return (100 * p).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1) + "%"; }
function esc(s) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function render(r) {
  state.last = r;
  if (r.empty) { resetResults(); return; }
  const t = r.timing_ms || {};
  $("latency").textContent = `CRNN ${t.crnn ?? "–"} ms · Seg+CNN ${t.cnn ?? "–"} ms`;

  // ---- Model B: CRNN
  const c = r.crnn;
  if (c) {
    const el = $("crnnText");
    el.classList.toggle("placeholder", !c.text);
    if (!c.text) el.textContent = "(nothing recognized)";
    else if (c.char_conf) el.innerHTML = [...c.text].map((ch, i) =>
      `<span class="${confClass(c.char_conf[i])}" title="${pct(c.char_conf[i])}">${esc(ch)}</span>`).join("");
    else el.textContent = c.text;
    $("crnnConf").style.width = pct(c.confidence);
    $("crnnConfLabel").textContent = `P(text | image) = ${pct(c.confidence)}` +
      (c.greedy_text !== c.text ? ` · greedy decoding gave "${c.greedy_text}"` : "");
    $("beams").innerHTML = c.beams.map((b) => `<li><div class="row"><code>${esc(b.text) || "∅"}</code>
      <div class="bar"><div style="width:${pct(b.prob)}"></div></div><span>${pct(b.prob)}</span></div></li>`).join("");
    if ($("sampleText")) $("sampleText").value = c.text;
  } else {
    $("crnnText").textContent = "CRNN not trained yet";
  }

  // ---- Model A: segmentation + CharCNN
  const a = r.cnn;
  if (a) {
    $("cnnText").classList.toggle("placeholder", !a.text);
    $("cnnText").textContent = a.text || "(nothing found)";
    $("cnnInfo").textContent = `${a.chars.length} segment(s) found · product of top-1 probabilities ${pct(a.confidence)}`;
    $("cnnChars").innerHTML = a.chars.map((ch, i) => `<div class="charcard">
        <img src="${ch.image}" alt="segment ${i + 1}" title="28×28 input of segment ${i + 1}">
        ${ch.top3.map(([l, p]) => `<div class="t"><b>${esc(l)}</b>
          <div class="bar"><div style="width:${pct(p)}"></div></div><span>${pct(p)}</span></div>`).join("")}
      </div>`).join("");
    if ($("sampleText") && !r.crnn) $("sampleText").value = a.text;
  } else {
    $("cnnText").textContent = "CharCNN not trained yet";
    $("cnnChars").innerHTML = "";
  }
  drawBoxes(r);

  // ---- model input + CTC strip
  const [w, h] = r.model_input_size;
  const zoom = Math.max(2, Math.min(4, Math.floor(900 / w)));   // short words get a bigger zoom
  $("modelInputWrap").innerHTML =
    `<img src="${r.model_input}" style="width:${zoom * w}px;height:${zoom * h}px" alt="model input">`;
  if (c) {
    const cell = (zoom * w) / c.num_frames;
    $("ctcStrip").innerHTML = [...c.frames].map((f, i) => {
      const p = c.frame_conf[i];
      if (f === "·") return `<span class="blank" style="width:${cell}px" title="t=${i}: blank ${pct(p)}">·</span>`;
      return `<span style="width:${cell}px;background:hsl(${120 * p},55%,40%)" title="t=${i}: '${esc(f)}' ${pct(p)}">${esc(f)}</span>`;
    }).join("");
  } else $("ctcStrip").innerHTML = "";
}

function drawBoxes(r) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!r || !r.cnn || !$("chkBoxes").checked) return;
  const color = getComputedStyle(document.documentElement).getPropertyValue("--a");
  octx.strokeStyle = octx.fillStyle = color;
  octx.lineWidth = 2;
  octx.font = "bold 22px ui-monospace, monospace";
  for (const ch of r.cnn.chars) {
    const [x0, y0, x1, y1] = ch.box;
    octx.strokeRect(x0 - 3, y0 - 3, x1 - x0 + 6, y1 - y0 + 6);
    octx.fillText(ch.top3[0][0], x0, Math.max(20, y0 - 8));
  }
}

function resetResults() {
  state.last = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  for (const id of ["crnnText", "cnnText"]) { $(id).textContent = "—"; $(id).classList.add("placeholder"); }
  for (const id of ["beams", "cnnChars", "modelInputWrap", "ctcStrip"]) $(id).innerHTML = "";
  for (const id of ["crnnConfLabel", "cnnInfo", "latency"]) $(id).textContent = "";
  $("crnnConf").style.width = "0";
}

// ------------------------------------------------------------------ samples / models / reports
async function saveSample(e) {
  e.preventDefault();
  const text = $("sampleText").value.trim();
  if (!state.strokes.length) { $("sampleMsg").textContent = "Draw something first."; return; }
  if (!text) { $("sampleMsg").textContent = "Type the correct text first."; return; }
  const res = await fetch("/api/samples", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: ink.toDataURL("image/png"), text }),
  });
  const body = await res.json();
  $("sampleMsg").textContent = res.ok ? `Saved "${body.text}" — ${body.total} sample(s) collected.` : `Error: ${body.detail}`;
  if (res.ok) clearAll();
}

async function reloadModels() {
  const models = await fetch("/api/reload", { method: "POST" }).then((r) => r.json());
  renderInfo({ models });
}

function renderInfo(info) {
  const m = info.models;
  const chip = (name, meta, metric) => `<span class="chip ${meta.loaded ? "ok" : "missing"}">${name}` +
    (meta.loaded ? ` · ${(meta.params / 1e6).toFixed(2)}M params${metric ? " · " + metric : ""}` : " · not trained") + "</span>";
  const cv = m.crnn.test_metrics || m.crnn.val_metrics;
  const av = m.char_cnn.test_metrics || m.char_cnn.val_metrics;
  $("modelStatus").innerHTML =
    chip("CRNN", m.crnn, cv && `CER ${(100 * cv.cer).toFixed(1)}%`) +
    chip("CharCNN", m.char_cnn, av && `char acc ${(100 * av.acc).toFixed(1)}%`);
  $("footer").textContent = "Models: " + [m.crnn, m.char_cnn].map((x) => x.path + (x.finetuned ? " (fine-tuned)" : "")).join(" · ");
  if (info.reports) {
    $("reports").innerHTML = info.reports.map((n) =>
      `<figure><img src="${REPORTS_BASE}${n}" alt="${n}" loading="lazy"><figcaption>${n}</figcaption></figure>`).join("");
  }
  if (info.comparison_md !== undefined) $("comparison").innerHTML = info.comparison_md ? mdToHtml(info.comparison_md) : "";
  if (STATIC) $("footer").textContent = "Models run in your browser with ONNX Runtime Web — nothing you draw is uploaded.";
}

// GitHub Pages build: no server, so saving samples / fine-tuning needs the local version.
function staticMode(cfg) {
  const repo = cfg.repo;
  $("sub-links").innerHTML = `Runs entirely in your browser · <a href="${repo}">source code</a> · ` +
    `<a href="${repo}/blob/main/docs/THEORY.md">study guide</a>`;
  $("sampleSection").innerHTML = `<h2>Fine-tune on your own handwriting</h2>
    <p class="muted">Collecting samples and fine-tuning needs the local Python version:
    clone <a href="${repo}">${repo.replace("https://", "")}</a>, run <code>python -m app.server</code>,
    save ~30 corrected words, then <code>python -m hwr.train_crnn --finetune</code>.</p>`;
}

// minimal Markdown renderer (headings, tables, paragraphs) for reports/comparison.md
function mdToHtml(md) {
  const out = [];
  const lines = md.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      i--;
      const cells = (r) => r.split("|").slice(1, -1).map((s) => esc(s.trim()));
      const body = rows.filter((r) => !/^\|\s*-/.test(r));
      out.push("<table><tr>" + cells(body[0]).map((c) => `<th>${c}</th>`).join("") + "</tr>" +
        body.slice(1).map((r) => "<tr>" + cells(r).map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") + "</table>");
    } else if (l.startsWith("#")) out.push(`<h3>${esc(l.replace(/^#+\s*/, ""))}</h3>`);
    else if (l.trim()) out.push(`<p>${esc(l).replace(/`([^`]+)`/g, "<code>$1</code>")}</p>`);
  }
  return out.join("");
}

init();
