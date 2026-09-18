"""Human editability study v2 (rebuttal Q4/S3) — GROUPING vs PATH-ONLY.

Both conditions edit the SAME generated SVG (svg_ours_all); the manipulated
variable is the INTERACTION capability:
  - group : semantic-group panel available — clicking a model group
            (<g data-label>) selects all its paths at once (+ double-click).
  - path  : path-level selection ONLY (click / Ctrl+click); no group panel,
            no group double-click. Simulates editing without semantic
            structure.

Targets come from edit_gt_ours (authored on the same generated SVGs, so the
canvas starts pixel-identical to the target's "before"). Time is measured
per trial; "포기" (give up) is allowed and recorded. Success is rated by the
authors afterwards from the saved results.

Trials: task block (move → remove) × stems × 2 conditions; condition order
per (participant, task, stem) counterbalanced by hash.

Results: SSemBench/curated/human_test_results_v2/<participant>/
             <task>__<stem>__<cond>.svg / .png / .json

Run:
    .venv/bin/python human_test_v2.py     # http://localhost:8098
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

_PKG_ROOT = Path(__file__).resolve().parent
CURATED = _PKG_ROOT / "SSemBench" / "curated"
OURS = CURATED / "svg_ours_all"           # stimulus for BOTH conditions
EDIT_GT = CURATED / "edit_gt_ours"        # targets authored on ours outputs
RESULTS = CURATED / "human_test_results_v2"
CACHE = RESULTS / "_cache"
TASKS = ("move", "remove")
CONDS = ("group", "path")                 # ui capability, not svg source
PORT = 8098

SHAPE_TAGS = ("path", "rect", "circle", "ellipse", "polygon", "polyline", "line")
SHAPE_PATTERN = re.compile(
    r"<(" + "|".join(SHAPE_TAGS) + r")\b([^>]*?)(/>|>([\s\S]*?)</\1>)")
_NONRENDER_RE = re.compile(
    r"<(defs|clipPath|mask|symbol|pattern|marker)\b[^>]*>[\s\S]*?</\1>",
    re.IGNORECASE)
_OPEN_TAG_RE = re.compile(r"^(<\w+)([^>]*?)(/?>)")


def _iter_shapes(svg_text: str):
    skip = [(m.start(), m.end()) for m in _NONRENDER_RE.finditer(svg_text)]
    for m in SHAPE_PATTERN.finditer(svg_text):
        if any(s <= m.start() < e for s, e in skip):
            continue
        yield m


def _strip_attrs(raw: str, names: tuple[str, ...]) -> str:
    m = _OPEN_TAG_RE.match(raw)
    if not m:
        return raw
    name, attrs, closer = m.group(1), m.group(2), m.group(3)
    for a in names:
        attrs = re.sub(rf'\s*{a}\s*=\s*"[^"]*"', "", attrs)
    return name + attrs + closer + raw[m.end():]


def _viewbox(svg_text: str) -> tuple[float, float, float, float] | None:
    m = re.search(r'viewBox\s*=\s*"([^"]+)"', svg_text)
    if not m:
        return None
    p = [float(v) for v in re.split(r"[ ,]+", m.group(1).strip()) if v]
    return tuple(p) if len(p) == 4 else None


def normalize_stimulus(svg_text: str) -> str:
    """Ensure the root has a viewBox (vtracer files only carry width/height)."""
    if _viewbox(svg_text):
        return svg_text
    m = re.search(r"<svg\b[^>]*>", svg_text)
    if not m:
        return svg_text
    w = re.search(r'width\s*=\s*"([\d.]+)', m.group(0))
    h = re.search(r'height\s*=\s*"([\d.]+)', m.group(0))
    if not (w and h):
        return svg_text
    new_open = m.group(0)[:-1] + f' viewBox="0 0 {w.group(1)} {h.group(1)}">'
    return svg_text.replace(m.group(0), new_open, 1)


def stems_for(task: str) -> list[str]:
    return sorted(p.name.split("__")[0] for p in EDIT_GT.glob(f"*__{task}.json"))


def cond_order(participant: str, task: str, stem: str) -> list[str]:
    h = hashlib.md5(f"{participant}|{task}|{stem}".encode()).hexdigest()
    return ["group", "path"] if int(h, 16) % 2 == 0 else ["path", "group"]


def build_trials(participant: str) -> list[dict]:
    out = []
    for task in TASKS:
        for stem in stems_for(task):
            for cond in cond_order(participant, task, stem):
                out.append({"task": task, "stem": stem, "cond": cond})
    return out


def result_base(participant: str, t: dict) -> Path:
    return RESULTS / participant / f"{t['task']}__{t['stem']}__{t['cond']}"


app = FastAPI()


@app.get("/api/manifest")
def api_manifest(participant: str) -> JSONResponse:
    participant = participant.strip()
    if not participant:
        raise HTTPException(400, "participant required")
    trials = build_trials(participant)
    for t in trials:
        t["done"] = result_base(participant, t).with_suffix(".json").exists()
    return JSONResponse({"participant": participant, "trials": trials,
                         "n_total": len(trials),
                         "n_done": sum(t["done"] for t in trials)})


@app.get("/api/trial")
def api_trial(participant: str, idx: int) -> JSONResponse:
    trials = build_trials(participant.strip())
    if idx < 0 or idx >= len(trials):
        raise HTTPException(404, "trial index out of range")
    t = trials[idx]
    stim_p = OURS / f"{t['stem']}.svg"
    if not stim_p.exists():
        raise HTTPException(500, f"stimulus missing: {stim_p}")
    stim = normalize_stimulus(stim_p.read_text())
    return JSONResponse({
        "idx": idx, "n_total": len(trials),
        "task": t["task"], "stem": t["stem"], "cond": t["cond"],
        "stimulus_svg": stim,
        "gt_before": f"/gtpng/before/{t['stem']}.png",
        "gt_after": f"/gtpng/after/{t['stem']}__{t['task']}.png",
    })


@app.get("/gtpng/before/{stem}.png")
def gt_before(stem: str) -> Response:
    # "before" = the generated svg itself, so target-before and the canvas
    # start pixel-identical
    src = OURS / f"{stem}.svg"
    if not src.exists():
        raise HTTPException(404, "unknown stem")
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{stem}.png"
    if not cached.exists():
        import cairosvg
        cached.write_bytes(cairosvg.svg2png(
            bytestring=src.read_bytes(), output_width=512, output_height=512,
            background_color="#ffffff"))
    return FileResponse(cached, media_type="image/png")


@app.get("/gtpng/after/{name}.png")
def gt_after(name: str) -> FileResponse:
    p = EDIT_GT / f"{name}.png"
    if not p.exists():
        raise HTTPException(404, "unknown gt edit")
    return FileResponse(p, media_type="image/png")


class SaveBody(BaseModel):
    participant: str
    idx: int
    svg: str
    meta: dict


@app.post("/api/save")
def api_save(body: SaveBody) -> JSONResponse:
    participant = body.participant.strip()
    trials = build_trials(participant)
    if body.idx < 0 or body.idx >= len(trials):
        raise HTTPException(404, "trial index out of range")
    t = trials[body.idx]
    try:
        root = ET.fromstring(body.svg)
    except ET.ParseError as e:
        raise HTTPException(422, f"result svg not well-formed: {e}")
    if not root.tag.endswith("svg"):
        raise HTTPException(422, "root element is not <svg>")
    import cairosvg
    try:
        png = cairosvg.svg2png(bytestring=body.svg.encode(),
                               output_width=512, output_height=512,
                               background_color="#ffffff")
    except Exception as e:
        raise HTTPException(422, f"result svg failed to render: {e}")
    base = result_base(participant, t)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".svg").write_text(body.svg, encoding="utf-8")
    base.with_suffix(".png").write_bytes(png)
    meta = dict(body.meta)
    meta.update({"participant": participant, **t})
    base.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return JSONResponse({"ok": True})


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>SVG Editing Study</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #1f2937; }
  header { position: sticky; top: 0; z-index: 10; background: #1f2937; color: #fff; padding: 8px 14px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 14px; font-weight: 600; }
  header input { font-size: 13px; padding: 5px 9px; border-radius: 4px; border: 1px solid #4b5563; }
  header button { font-size: 13px; padding: 6px 12px; border-radius: 4px; border: 1px solid #4b5563; background: #374151; color: #fff; cursor: pointer; }
  header button:hover { background: #4b5563; }
  .taskbadge { font-size: 13px; font-weight: 700; padding: 3px 12px; border-radius: 12px; }
  .taskbadge.move { background: #2563eb; }
  .taskbadge.remove { background: #dc2626; }
  .progress, #timer { font-size: 13px; font-family: ui-monospace, monospace; }
  #timer { font-weight: 700; color: #fbbf24; min-width: 64px; }
  .grow { flex: 1; }
  main { display: grid; grid-template-columns: 320px 1fr 250px; gap: 10px; padding: 10px; }
  #treeBox { max-height: 74vh; overflow-y: auto; }
  .tnode { display: flex; align-items: center; gap: 6px; padding: 5px 6px; border-radius: 4px; font-size: 12px; cursor: pointer; }
  .tnode:hover { background: #eef2ff; }
  .tnode .cnt { font-size: 10px; color: #6b7280; font-family: ui-monospace, monospace; }
  .tnode .chip { font-size: 9px; padding: 1px 5px; border-radius: 8px; background: #ddd6fe; color: #5b21b6; }
  .tree-empty { color: #9ca3af; font-size: 12px; line-height: 1.6; }
  .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; }
  .panel h3 { margin: 0 0 6px; font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }
  .gtimg { width: 100%; border: 1px solid #e5e7eb; border-radius: 4px; background: #fff; }
  .arrow { text-align: center; font-size: 20px; color: #6b7280; margin: 2px 0; }
  .notice { font-size: 12px; color: #78350f; background: #fef3c7; border: 1px solid #fbbf24; border-radius: 4px; padding: 6px 8px; margin-top: 8px; line-height: 1.5; }
  .svg-box { background: repeating-conic-gradient(#f3f4f6 0% 25%, #fff 0% 50%) 50% / 22px 22px; border: 1px solid #eee; border-radius: 4px; display: flex; align-items: center; justify-content: center; min-height: 300px; }
  .svg-box svg { overflow: hidden; outline: 2px dashed rgba(244, 63, 94, .5); background: #fff; }
  #canvas svg :is(path,rect,circle,ellipse,polygon,polyline,line) { cursor: pointer; }
  .controls { margin-top: 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .controls button { padding: 9px 18px; font-size: 14px; border-radius: 5px; border: 1px solid #d1d5db; background: #fff; cursor: pointer; }
  .controls button:hover { background: #f3f4f6; }
  #doneBtn { background: #16a34a; color: #fff; border-color: #16a34a; font-weight: 700; }
  #doneBtn:hover { background: #15803d; }
  #giveupBtn { color: #b91c1c; }
  #delBtn { background: #ef4444; color: #fff; border-color: #ef4444; }
  .hint { font-size: 11px; color: #6b7280; margin-top: 6px; line-height: 1.7; }
  .kbd { background: #e5e7eb; border: 1px solid #d1d5db; border-radius: 3px; padding: 0 5px; font-size: 11px; font-family: ui-monospace, monospace; }
  #selInfo { font-size: 12px; font-family: ui-monospace, monospace; color: #374151; }
  #login { max-width: 460px; margin: 80px auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 28px; }
  #login h2 { margin: 0 0 12px; font-size: 18px; }
  #login p { font-size: 13px; color: #4b5563; line-height: 1.7; }
  #login input { width: 100%; font-size: 15px; padding: 9px 12px; border: 1px solid #d1d5db; border-radius: 5px; margin: 10px 0; }
  #login button { width: 100%; padding: 10px; font-size: 15px; background: #2563eb; color: #fff; border: none; border-radius: 5px; cursor: pointer; }
  #finish { max-width: 460px; margin: 100px auto; text-align: center; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 40px; display: none; }
  #toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #1f2937; color: #fff; padding: 10px 16px; border-radius: 6px; opacity: 0; transition: opacity .2s; pointer-events: none; z-index: 50; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<div id="login">
  <h2>SVG 편집 실험</h2>
  <p>왼쪽에 제시되는 <b>변경 전 → 변경 후</b> 그림을 보고, 편집 캔버스의 SVG에
  <b>같은 변경</b>을 적용해 주세요.<br><br>
  문제에 따라 오른쪽에 <b>구성 요소 목록</b>이 제공되기도 하고, 제공되지 않아
  조각을 직접 클릭해야 하기도 합니다.<br>
  너무 어렵거나 오래 걸리면 <b>포기</b> 버튼을 눌러 다음 문제로 넘어가도 됩니다.</p>
  <input id="pname" placeholder="이름 (참가자 ID)" autocomplete="off">
  <button id="startBtn">시작</button>
</div>
<div id="finish"><h2>실험 완료 🎉</h2><p>수고하셨습니다. 모든 결과가 저장되었습니다.</p></div>
<div id="study" style="display:none">
<header>
  <h1>SVG 편집 실험</h1>
  <span class="taskbadge" id="taskBadge"></span>
  <span class="progress" id="progress"></span>
  <span class="grow"></span>
  <span id="timer">0.0s</span>
  <button id="undo">↩ 되돌리기</button>
  <button id="reset">처음부터</button>
</header>
<main>
  <div class="panel">
    <h3>목표: 이 변경을 적용하세요</h3>
    <img class="gtimg" id="gtBefore" alt="before">
    <div class="arrow">⬇ 이렇게 바꾸세요 ⬇</div>
    <img class="gtimg" id="gtAfter" alt="after">
    <div class="notice" id="taskNotice"></div>
  </div>
  <div class="panel">
    <h3>편집 캔버스</h3>
    <div class="svg-box" id="canvas"></div>
    <div class="controls">
      <button id="delBtn" style="display:none">🗑 선택 삭제 (Del)</button>
      <span id="selInfo"></span>
      <span class="grow"></span>
      <button id="giveupBtn">이 문제 포기</button>
      <button id="doneBtn">✓ 완료 — 다음 문제</button>
    </div>
    <div class="hint" id="hintBar"></div>
  </div>
  <div class="panel">
    <h3>구성 요소 목록</h3>
    <div id="treeBox"></div>
  </div>
</main>
</div>
<div id="toast"></div>
<script>
const $ = id => document.getElementById(id);
const SHAPE_SEL = "path, rect, circle, ellipse, polygon, polyline, line";
const NONRENDER = "defs, clipPath, mask, pattern, marker, symbol, filter";
let participant = "";
let manifest = null;
let trial = null;        // current trial payload
let shapes = [];
let selected = new Set();
let actions = [];
let undoStack = [];
let drag = null;
let t0 = 0, firstMs = null, timerIv = null;

function toast(m) {
  const t = $("toast"); t.textContent = m; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

// ---------------------------------------------------------------- flow
$("startBtn").addEventListener("click", start);
$("pname").addEventListener("keydown", e => { if (e.key === "Enter") start(); });

async function start() {
  participant = $("pname").value.trim();
  if (!participant) { toast("이름을 입력하세요"); return; }
  manifest = await (await fetch(`/api/manifest?participant=${encodeURIComponent(participant)}`)).json();
  $("login").style.display = "none";
  $("study").style.display = "block";
  nextTrial();
}

function nextTrialIdx() {
  return manifest.trials.findIndex(t => !t.done);
}

async function nextTrial() {
  const idx = nextTrialIdx();
  if (idx < 0) {
    $("study").style.display = "none";
    $("finish").style.display = "block";
    return;
  }
  trial = await (await fetch(`/api/trial?participant=${encodeURIComponent(participant)}&idx=${idx}`)).json();
  $("taskBadge").textContent = trial.task.toUpperCase();
  $("taskBadge").className = "taskbadge " + trial.task;
  const doneCnt = manifest.trials.filter(t => t.done).length;
  $("progress").textContent = `${doneCnt + 1} / ${trial.n_total}`;
  $("gtBefore").src = trial.gt_before + "?v=" + idx;
  $("gtAfter").src = trial.gt_after + "?v=" + idx;
  $("taskNotice").textContent = trial.task === "move"
    ? "대상을 선택한 뒤 마우스로 끌어 아래 그림과 같은 위치로 옮기세요."
    : "대상을 선택하고 삭제하세요 (버튼 또는 Del 키).";
  $("delBtn").style.display = trial.task === "remove" ? "inline-block" : "none";
  const base = '클릭 = 조각 선택 · <span class="kbd">Ctrl</span>+클릭 = 누적 선택 · ' +
               '<span class="kbd">Esc</span> = 해제 · <span class="kbd">Ctrl+Z</span> = 되돌리기';
  $("hintBar").innerHTML = trial.cond === "group"
    ? base + ' · <b>더블클릭 또는 오른쪽 목록 클릭 = 묶음 전체 선택</b>'
    : base;
  resetCanvas();
  // timer
  firstMs = null;
  t0 = performance.now();
  clearInterval(timerIv);
  timerIv = setInterval(() => { $("timer").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s"; }, 100);
}

// ---------------------------------------------------------------- canvas
function clipToViewBox(box, maxHpx) {
  const sv = box.querySelector("svg");
  if (!sv) return;
  const vb = sv.viewBox && sv.viewBox.baseVal;
  if (!vb || !vb.width || !vb.height) return;
  const availW = Math.max(100, box.clientWidth - 20);
  const scale = Math.min(availW / vb.width, maxHpx / vb.height);
  sv.style.width = (vb.width * scale) + "px";
  sv.style.height = (vb.height * scale) + "px";
}

function collectShapes() {
  const svg = $("canvas").querySelector("svg");
  shapes = svg ? Array.from(svg.querySelectorAll(SHAPE_SEL))
      .filter(el => !el.closest(NONRENDER)) : [];
  shapes.forEach((el, i) => el.setAttribute("data-pidx", String(i)));
}

function restoreShapes() {
  const svg = $("canvas").querySelector("svg");
  shapes = [];
  if (!svg) return;
  for (const el of svg.querySelectorAll("[data-pidx]")) {
    if (el.closest(NONRENDER)) continue;
    shapes[parseInt(el.dataset.pidx)] = el;
  }
}

function resetCanvas() {
  selected = new Set(); actions = []; undoStack = []; drag = null;
  $("canvas").innerHTML = trial.stimulus_svg;
  collectShapes();
  bindCanvas();
  clipToViewBox($("canvas"), window.innerHeight * 0.72);
  buildTreePanel();
  refreshSelUI();
}

// 우측 패널: group 조건에서만 시맨틱 그룹 목록 제공.
// path 조건은 그룹 기능 전체(패널 + 더블클릭)가 비활성 — 이것이 실험 조작.
function buildTreePanel() {
  const box = $("treeBox");
  box.innerHTML = "";
  if (trial.cond !== "group") {
    box.innerHTML = '<div class="tree-empty">이 문제에서는 구성 요소 목록이 제공되지 않습니다.<br>캔버스에서 조각을 직접 클릭해 선택하세요.<br>(<b>Ctrl</b>+클릭으로 여러 조각 누적 선택)</div>';
    return;
  }
  const svg = $("canvas").querySelector("svg");
  if (!svg) return;
  const nodes = [];
  const skip = new Set(["defs","clippath","mask","pattern","marker","symbol","filter"]);
  (function walk(el, depth) {
    for (const ch of el.children) {
      const tag = ch.tagName ? ch.tagName.toLowerCase() : "";
      if (skip.has(tag)) continue;
      if (tag === "g") {
        const idxs = Array.from(ch.querySelectorAll("[data-pidx]"))
          .map(s => parseInt(s.dataset.pidx));
        if (idxs.length) nodes.push({
          label: ch.getAttribute("data-label") || ch.getAttribute("id") || "(그룹)",
          depth, idxs,
        });
        walk(ch, depth + 1);
      } else {
        walk(ch, depth);
      }
    }
  })(svg, 0);
  if (!nodes.length) {
    box.innerHTML = '<div class="tree-empty">이 SVG에는 구성 요소 정보가 없습니다.<br>캔버스에서 조각을 직접 클릭해 선택하세요.</div>';
    return;
  }
  for (const n of nodes) {
    const div = document.createElement("div");
    div.className = "tnode";
    div.style.paddingLeft = (6 + n.depth * 14) + "px";
    div.innerHTML = `<span class="chip">그룹</span><strong>${n.label}</strong>` +
                    `<span class="cnt">${n.idxs.length}조각</span>`;
    div.addEventListener("click", (e) => {
      markFirstInteraction();
      const alive = n.idxs.filter(i => shapes[i]);
      if (!alive.length) { toast("이 그룹의 조각이 남아있지 않습니다"); return; }
      setSelection(alive, e.ctrlKey || e.metaKey);
    });
    box.appendChild(div);
  }
}
window.addEventListener("resize", () => { if (trial) clipToViewBox($("canvas"), window.innerHeight * 0.72); });

// ---------------------------------------------------------------- selection
function restoreHighlightBackup(el) {
  if (el.dataset.hl === undefined) return;
  if (el.dataset.oStroke === "__NONE__") el.removeAttribute("stroke");
  else el.setAttribute("stroke", el.dataset.oStroke);
  if (el.dataset.oStrokeW === "__NONE__") el.removeAttribute("stroke-width");
  else el.setAttribute("stroke-width", el.dataset.oStrokeW);
  el.style.stroke = el.dataset.oSStroke || "";
  el.style.strokeWidth = el.dataset.oSStrokeW || "";
  delete el.dataset.hl; delete el.dataset.oStroke; delete el.dataset.oStrokeW;
  delete el.dataset.oSStroke; delete el.dataset.oSStrokeW;
}

function applyHighlight(el, on) {
  if (on) {
    if (el.dataset.hl === undefined) {
      el.dataset.hl = "1";
      el.dataset.oStroke = el.getAttribute("stroke") ?? "__NONE__";
      el.dataset.oStrokeW = el.getAttribute("stroke-width") ?? "__NONE__";
      el.dataset.oSStroke = el.style.stroke || "";
      el.dataset.oSStrokeW = el.style.strokeWidth || "";
    }
    el.setAttribute("stroke", "#2563eb");
    el.setAttribute("stroke-width", "2");
    el.style.stroke = "#2563eb";
    el.style.strokeWidth = "2";
    el.style.filter = "drop-shadow(0 0 3px rgba(37,99,235,.9))";
    el.classList.add("sel");
  } else {
    restoreHighlightBackup(el);
    el.style.filter = "";
    el.classList.remove("sel");
  }
}

function setSelection(idxs, additive) {
  if (!additive) {
    for (const i of selected) if (shapes[i]) applyHighlight(shapes[i], false);
    selected = new Set();
  }
  for (const i of idxs) {
    if (!shapes[i]) continue;
    if (additive && selected.has(i) && idxs.length === 1) {
      selected.delete(i); applyHighlight(shapes[i], false);
    } else {
      selected.add(i); applyHighlight(shapes[i], true);
    }
  }
  refreshSelUI();
}

function clearSelection() { setSelection([], false); }

function refreshSelUI() {
  $("selInfo").textContent = selected.size ? `${selected.size}개 조각 선택됨` : "";
}

// ---------------------------------------------------------------- undo
function snapshot() {
  return { html: $("canvas").innerHTML, actions: actions.slice(), sel: [...selected] };
}
function pushUndoPre(pre) {
  if (pre) { undoStack.push(pre); if (undoStack.length > 30) undoStack.shift(); }
}
function doUndo() {
  const s = undoStack.pop();
  if (!s) { toast("되돌릴 작업이 없습니다"); return; }
  drag = null;
  $("canvas").innerHTML = s.html;
  actions = s.actions;
  restoreShapes();
  bindCanvas();
  clipToViewBox($("canvas"), window.innerHeight * 0.72);
  buildTreePanel();
  selected = new Set(s.sel.filter(i => shapes[i]));
  for (const el of shapes) if (el) applyHighlight(el, false);
  for (const i of selected) applyHighlight(shapes[i], true);
  refreshSelUI();
}

// ---------------------------------------------------------------- interactions
function markFirstInteraction() {
  if (firstMs === null) firstMs = performance.now() - t0;
}

function groupPathsOf(el) {
  const g = el.closest("g");
  if (!g) return null;
  const idxs = [];
  for (const s of g.querySelectorAll("[data-pidx]")) {
    idxs.push(parseInt(s.dataset.pidx));
  }
  return idxs.length ? idxs : null;
}

function bindCanvas() {
  const box = $("canvas");
  const svg = box.querySelector("svg");
  if (!svg) return;

  svg.addEventListener("pointerdown", (e) => {
    markFirstInteraction();
    if (drag) return;
    const el = e.target.closest?.(SHAPE_SEL.replaceAll(" ", ""));
    const shape = (el && el.dataset.pidx !== undefined && !el.closest(NONRENDER)) ? el : null;
    if (!shape) return;
    const idx = parseInt(shape.dataset.pidx);
    if (trial.task === "move" && selected.has(idx) && !(e.ctrlKey || e.metaKey)) {
      const pre = snapshot();
      drag = {
        pre, moved: false, lastDx: 0, lastDy: 0,
        els: [...selected].filter(i => shapes[i]).map(i => {
          const s = shapes[i];
          const m = s.parentNode.getScreenCTM().inverse();
          const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
          return { el: s, i, m, p0: pt.matrixTransform(m),
                   base: s.dataset.baseTransform ?? (s.dataset.baseTransform = s.getAttribute("transform") || ""),
                   ox: parseFloat(s.dataset.offX || "0"), oy: parseFloat(s.dataset.offY || "0") };
        }),
      };
      box.classList.add("dragging");
      svg.setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }
    setSelection([idx], e.ctrlKey || e.metaKey);
  });

  svg.addEventListener("dblclick", (e) => {
    if (trial.cond !== "group") return;   // path 조건: 그룹 선택 기능 없음
    const el = e.target.closest?.(SHAPE_SEL.replaceAll(" ", ""));
    if (!el || el.dataset.pidx === undefined) return;
    const grp = groupPathsOf(el);
    if (grp && grp.length > 1) {
      setSelection(grp, false);
      toast(`묶음 선택: ${grp.length}개 조각`);
    }
  });

  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    for (const d of drag.els) {
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(d.m);
      const dx = p.x - d.p0.x, dy = p.y - d.p0.y;
      if (Math.abs(dx) + Math.abs(dy) > 0.01) drag.moved = true;
      const tx = d.ox + dx, ty = d.oy + dy;
      d.el.setAttribute("transform", `translate(${tx} ${ty})${d.base ? " " + d.base : ""}`);
      d.el.dataset.offX = tx; d.el.dataset.offY = ty;
      drag.lastDx = dx; drag.lastDy = dy;
    }
  });

  const endDrag = () => {
    if (!drag) return;
    if (drag.moved) {
      pushUndoPre(drag.pre);
      actions.push({ type: "move", paths: drag.els.map(d => d.i).sort((a, b) => a - b),
                     dx: +drag.lastDx.toFixed(2), dy: +drag.lastDy.toFixed(2) });
    }
    drag = null;
    box.classList.remove("dragging");
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
}

function doRemove() {
  if (trial.task !== "remove") return;
  if (!selected.size) { toast("먼저 선택하세요"); return; }
  markFirstInteraction();
  pushUndoPre(snapshot());
  const idxs = [...selected].sort((a, b) => a - b);
  for (const i of idxs) shapes[i]?.remove();
  actions.push({ type: "remove", paths: idxs });
  shapes = shapes.map((el, i) => selected.has(i) ? null : el);
  selected = new Set();
  refreshSelUI();
}

// ---------------------------------------------------------------- save
function cleanedSvg() {
  const tmp = $("canvas").querySelector("svg").cloneNode(true);
  tmp.style.width = tmp.style.height = "";
  if (!tmp.getAttribute("style")) tmp.removeAttribute("style");
  tmp.querySelectorAll("[data-pidx]").forEach(el => {
    restoreHighlightBackup(el);
    el.style.filter = "";
    el.classList.remove("sel");
    if (!el.getAttribute("class")) el.removeAttribute("class");
    for (const k of Object.keys(el.dataset)) delete el.dataset[k];
    if (el.getAttribute("style") === "") el.removeAttribute("style");
  });
  return new XMLSerializer().serializeToString(tmp);
}

async function finishTrial(gaveUp) {
  const dur = performance.now() - t0;
  clearInterval(timerIv);
  const body = {
    participant, idx: trial.idx, svg: cleanedSvg(),
    meta: { duration_ms: Math.round(dur),
            first_interaction_ms: firstMs === null ? null : Math.round(firstMs),
            actions, n_actions: actions.length, gave_up: gaveUp },
  };
  const r = await fetch("/api/save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { toast("저장 실패: " + (await r.text()).slice(0, 100)); timerIv = setInterval(() => { $("timer").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s"; }, 100); return; }
  manifest.trials[trial.idx].done = true;
  nextTrial();
}

$("doneBtn").addEventListener("click", () => {
  if (!actions.length && !confirm("아무 편집도 하지 않았습니다. 이대로 제출할까요?")) return;
  finishTrial(false);
});
$("giveupBtn").addEventListener("click", () => {
  if (!confirm("이 문제를 포기하시겠습니까?")) return;
  finishTrial(true);
});
$("delBtn").addEventListener("click", doRemove);
$("undo").addEventListener("click", doUndo);
$("reset").addEventListener("click", () => { if (confirm("이 문제를 처음 상태로 되돌릴까요? (타이머는 계속 갑니다)")) resetCanvas(); });

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "Escape") clearSelection();
  if (e.key === "Delete") doRemove();
  if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); doUndo(); }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


if __name__ == "__main__":
    n_move, n_remove = len(stems_for("move")), len(stems_for("remove"))
    print(f"Human test v2: move {n_move} × 2 + remove {n_remove} × 2 = {2*(n_move+n_remove)} trials/participant")
    print(f"  conditions: {CONDS} (same ours stimulus, UI capability differs)")
    print(f"  targets: {EDIT_GT}")
    print(f"  results → {RESULTS}")
    print(f"Open http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
