"""Edit-target maker on OUR MODEL's outputs (rebuttal human study, take 2).

The human-test target images must come from the SAME svg the participant
edits, so the edit targets are authored here on svg_ours_all outputs
(NOT on the benchmark GT — that was the first attempt, kept in edit_gt/).

Loads the 88 generated SVGs (SSemBench/curated/svg_ours_all) and lets the
author create edited target versions per sample — move / remove (recolor
kept for completeness). Each edit starts from the ORIGINAL generated svg.

UI:
  - Center canvas: interactive SVG. Click = select a single path,
    Ctrl+click = add/remove from selection, Esc = clear selection.
  - Right panel: the model's semantic groups (<g data-label=...> in the
    generated svg). Clicking a group selects ALL its paths at once.
  - move    — drag any selected element; the whole selection translates.
  - remove  — Delete button (or Del key) removes the selection.
  - Save writes to SSemBench/curated/edit_gt_ours/:
        <stem>__<edit>.svg / .png / .json

Run:
    .venv/bin/python edit_gt_maker_ours.py     # http://localhost:8097
"""
from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

_PKG_ROOT = Path(__file__).resolve().parent
SRC = _PKG_ROOT / "SSemBench" / "curated" / "svg_ours_all"
OUT = _PKG_ROOT / "SSemBench" / "curated" / "edit_gt_ours"
EDITS = ("move", "remove", "recolor")
PORT = 8097


def list_files() -> list[Path]:
    return sorted(SRC.glob("*.svg"), key=lambda p: p.stem)


app = FastAPI()


@app.get("/api/files")
def api_files() -> JSONResponse:
    out = []
    for p in list_files():
        done = {e: (OUT / f"{p.stem}__{e}.svg").exists() for e in EDITS}
        out.append({"stem": p.stem, "done": done})
    return JSONResponse({"files": out})


@app.get("/api/file")
def api_file(idx: int) -> JSONResponse:
    files = list_files()
    if idx < 0 or idx >= len(files):
        raise HTTPException(404, "file index out of range")
    svg = files[idx]
    return JSONResponse({
        "idx": idx,
        "n_files": len(files),
        "stem": svg.stem,
        "svg_full": svg.read_text(),
    })


class SaveBody(BaseModel):
    idx: int
    edit: str
    svg: str
    meta: dict


@app.post("/api/save")
def api_save(body: SaveBody) -> JSONResponse:
    files = list_files()
    if body.idx < 0 or body.idx >= len(files):
        raise HTTPException(404, "file index out of range")
    if body.edit not in EDITS:
        raise HTTPException(400, f"unknown edit type: {body.edit}")
    stem = files[body.idx].stem

    # sanity: well-formed XML, and it must still be an <svg> root
    try:
        root = ET.fromstring(body.svg)
    except ET.ParseError as e:
        raise HTTPException(422, f"edited svg is not well-formed XML: {e}")
    if not root.tag.endswith("svg"):
        raise HTTPException(422, "root element is not <svg>")

    # server-side 512px render so the GT target image is canonical
    import cairosvg
    try:
        png = cairosvg.svg2png(bytestring=body.svg.encode("utf-8"),
                               output_width=512, output_height=512,
                               background_color="#ffffff")
    except Exception as e:
        raise HTTPException(422, f"edited svg failed to render: {e}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}__{body.edit}.svg").write_text(body.svg, encoding="utf-8")
    (OUT / f"{stem}__{body.edit}.png").write_bytes(png)
    meta = dict(body.meta)
    meta.update({"stem": stem, "edit": body.edit})
    (OUT / f"{stem}__{body.edit}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return JSONResponse({"ok": True, "saved": f"{stem}__{body.edit}.(svg|png|json)"})


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Edit-GT Maker</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #1f2937; }
  header { position: sticky; top: 0; z-index: 10; background: #1f2937; color: #fff; padding: 8px 14px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 14px; font-weight: 600; }
  header select, header button { font-size: 13px; padding: 5px 9px; border-radius: 4px; border: 1px solid #4b5563; background: #374151; color: #fff; cursor: pointer; }
  header select { max-width: 260px; }
  header button:hover { background: #4b5563; }
  header button.success { background: #16a34a; border-color: #16a34a; }
  header button:disabled { opacity: .4; cursor: not-allowed; }
  .tab { padding: 5px 14px; border-radius: 4px 4px 0 0; }
  .tab.active { background: #2563eb; border-color: #2563eb; font-weight: 700; }
  .badge { font-size: 10px; padding: 1px 6px; border-radius: 8px; background: #16a34a; margin-left: 4px; }
  .progress { font-size: 12px; font-family: ui-monospace, monospace; opacity: .9; }
  .grow { flex: 1; }
  main { display: grid; grid-template-columns: 300px 1fr 320px; gap: 10px; padding: 10px; }
  .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; }
  .panel h3 { margin: 0 0 8px; font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }
  .svg-box { background: repeating-conic-gradient(#f3f4f6 0% 25%, #fff 0% 50%) 50% / 22px 22px; border: 1px solid #eee; border-radius: 4px; display: flex; align-items: center; justify-content: center; min-height: 260px; }
  /* viewport 크기는 JS(clipToViewBox)가 viewBox 비율에 정확히 맞춰 픽셀로 지정 —
     letterbox가 생기지 않아 viewBox 밖 콘텐츠는 화면에서도 잘려 보인다(저장 결과와 동일). */
  .svg-box svg { overflow: hidden; outline: 2px dashed rgba(244, 63, 94, .6); background: #fff; }
  #refBox svg { pointer-events: none; }
  #canvas svg :is(path,rect,circle,ellipse,polygon,polyline,line) { cursor: pointer; }
  #canvas.mode-move svg .sel { cursor: grab; }
  #canvas.dragging svg .sel { cursor: grabbing; }
  .controls { margin-top: 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .controls button { padding: 7px 14px; font-size: 13px; border-radius: 4px; border: 1px solid #d1d5db; background: #fff; cursor: pointer; }
  .controls button:hover { background: #f3f4f6; }
  .swatch { width: 34px; height: 34px; border-radius: 6px; border: 2px solid #d1d5db; cursor: pointer; padding: 0; }
  .swatch:hover { transform: scale(1.12); }
  #delBtn { background: #ef4444; color: #fff; border-color: #ef4444; }
  #delBtn:hover { background: #dc2626; }
  .hint { font-size: 11px; color: #6b7280; margin-top: 6px; line-height: 1.6; }
  .kbd { background: #e5e7eb; border: 1px solid #d1d5db; border-radius: 3px; padding: 0 5px; font-size: 11px; font-family: ui-monospace, monospace; }
  /* tree panel */
  #treeBox { max-height: 72vh; overflow-y: auto; }
  .tnode { display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 4px; font-size: 12px; cursor: pointer; }
  .tnode:hover { background: #eef2ff; }
  .tnode.active { background: #dbeafe; font-weight: 600; }
  .tnode .cnt { font-size: 10px; color: #6b7280; font-family: ui-monospace, monospace; }
  .tnode .chip { font-size: 9px; padding: 1px 5px; border-radius: 8px; }
  .chip.grp { background: #ddd6fe; color: #5b21b6; }
  .chip.leaf { background: #d1fae5; color: #065f46; }
  #selInfo { font-size: 12px; font-family: ui-monospace, monospace; color: #374151; margin-top: 8px; }
  #toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #1f2937; color: #fff; padding: 10px 16px; border-radius: 6px; opacity: 0; transition: opacity .2s; pointer-events: none; z-index: 50; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>Edit-GT Maker — ours 출력 기준</h1>
  <select id="fileSel"></select>
  <button id="prev">◀</button><button id="next">▶</button>
  <span style="width:14px"></span>
  <button class="tab" data-edit="move">move<span id="bMove"></span></button>
  <button class="tab" data-edit="remove">remove<span id="bRemove"></span></button>
  <button class="tab" data-edit="recolor">recolor<span id="bRecolor"></span></button>
  <span class="grow"></span>
  <span class="progress" id="progress"></span>
  <button id="undo">↩ Undo</button>
  <button id="reset">Reset</button>
  <button id="save" class="success">💾 Save</button>
</header>
<main>
  <div class="panel">
    <h3>원본 (참고용)</h3>
    <div class="svg-box" id="refBox"></div>
    <div class="hint" id="editHint"></div>
  </div>
  <div class="panel">
    <h3>편집 캔버스 — <span id="modeLabel"></span></h3>
    <div class="svg-box" id="canvas"></div>
    <div class="controls" id="recolorCtl" style="display:none"></div>
    <div class="controls" id="removeCtl" style="display:none">
      <button id="delBtn">🗑 선택 삭제 (Del)</button>
    </div>
    <div id="selInfo"></div>
    <div class="hint">클릭 = path 선택 · <span class="kbd">Ctrl</span>+클릭 = 누적 선택/해제 · <span class="kbd">Esc</span> = 선택 해제 · 우측 트리 노드 클릭 = 그룹 전체 선택</div>
  </div>
  <div class="panel">
    <h3>모델 시맨틱 그룹 (ours)</h3>
    <div id="treeBox"></div>
  </div>
</main>
<div id="toast"></div>
<script>
const $ = id => document.getElementById(id);
const SHAPE_SEL = "path, rect, circle, ellipse, polygon, polyline, line";
const NONRENDER = "defs, clipPath, mask, pattern, marker, symbol, filter";
const PALETTE = [
  ["빨", "#e53935"], ["주", "#fb8c00"], ["노", "#fdd835"], ["초", "#43a047"],
  ["파", "#1e88e5"], ["남", "#3949ab"], ["보", "#8e24aa"],
];
let files = [], cur = 0, editMode = "move";
let state = null;          // {svg_full, tree, stem, ...}
let shapes = [];           // DOM shape elements in path-index order
let selected = new Set();  // path indices
let actions = [];          // action log for meta
let undoStack = [];        // [{html, actions}]
let activeTreeNode = null;

function toast(m) {
  const t = $("toast"); t.textContent = m; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

// ---------------------------------------------------------------- load
async function loadList() {
  files = (await (await fetch("/api/files")).json()).files;
  const sel = $("fileSel");
  sel.innerHTML = files.map((f, i) => {
    const d = f.done;
    const marks = ["move", "remove", "recolor"].map(e => d[e] ? "✓" : "·").join("");
    return `<option value="${i}">[${i + 1}/${files.length}] ${f.stem}  ${marks}</option>`;
  }).join("");
  sel.value = cur;
}

async function loadFile(idx) {
  if (state && actions.length &&
      !confirm("저장하지 않은 편집이 있습니다. 다른 파일로 이동하면 사라집니다. 이동할까요?")) {
    $("fileSel").value = cur;
    return;
  }
  cur = Math.max(0, Math.min(idx, files.length - 1));
  $("fileSel").value = cur;
  state = await (await fetch(`/api/file?idx=${cur}`)).json();
  $("refBox").innerHTML = state.svg_full;
  resetCanvas();
  applyClipping();
  renderTree();
  updateBadges();
  $("progress").textContent = `${state.stem} — ${shapes.length} paths`;
}

function clipToViewBox(box, maxHpx) {
  // Size the svg viewport to EXACTLY the viewBox aspect ratio so there is
  // no letterbox: anything dragged outside the viewBox disappears on
  // screen, exactly like it will in the saved render.
  const sv = box.querySelector("svg");
  if (!sv) return;
  const vb = sv.viewBox && sv.viewBox.baseVal;
  if (!vb || !vb.width || !vb.height) return;
  const availW = Math.max(100, box.clientWidth - 20);
  const scale = Math.min(availW / vb.width, maxHpx / vb.height);
  sv.style.width = (vb.width * scale) + "px";
  sv.style.height = (vb.height * scale) + "px";
  sv.style.maxWidth = "none";
  sv.style.maxHeight = "none";
}

function applyClipping() {
  clipToViewBox($("canvas"), window.innerHeight * 0.70);
  clipToViewBox($("refBox"), window.innerHeight * 0.32);
}
window.addEventListener("resize", () => { if (state) applyClipping(); });

function collectShapes() {
  // Fresh numbering — ONLY valid on a pristine original svg (resetCanvas).
  const svg = $("canvas").querySelector("svg");
  shapes = svg ? Array.from(svg.querySelectorAll(SHAPE_SEL)).filter(el => !el.closest(NONRENDER)) : [];
  shapes.forEach((el, i) => el.setAttribute("data-pidx", String(i)));
}

function restoreShapes() {
  // Rebuild shapes[] from data-pidx preserved in restored HTML. Removed
  // paths stay as holes so indices keep matching tree.json — renumbering
  // here would silently shift every index after a removed path.
  const svg = $("canvas").querySelector("svg");
  shapes = [];
  if (!svg) return;
  for (const el of svg.querySelectorAll("[data-pidx]")) {
    if (el.closest(NONRENDER)) continue;
    shapes[parseInt(el.dataset.pidx)] = el;
  }
}

function resetCanvas() {
  selected = new Set();
  actions = [];
  undoStack = [];
  activeTreeNode = null;
  drag = null;
  $("canvas").classList.remove("dragging");
  $("canvas").innerHTML = state.svg_full;
  collectShapes();
  bindCanvas();
  applyClipping();
  refreshSelUI();
}

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
    // inline style wins over both presentation attrs and style-attr strokes
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
    if (i < 0 || i >= shapes.length) continue;
    if (additive && selected.has(i) && idxs.length === 1) {
      selected.delete(i); applyHighlight(shapes[i], false);
    } else {
      selected.add(i); applyHighlight(shapes[i], true);
    }
  }
  refreshSelUI();
}

function clearSelection() { setSelection([], false); activeTreeNode = null; markTreeActive(); }

function refreshSelUI() {
  $("selInfo").textContent = selected.size
    ? `선택됨: ${selected.size}개 path [${[...selected].sort((a,b)=>a-b).join(", ")}]`
    : "선택 없음";
}

// ---------------------------------------------------------------- canvas events
let drag = null;  // {pre, moved, lastDx, lastDy, els:[{el, i, m, p0, base, ox, oy}]}

function bindCanvas() {
  const box = $("canvas");
  const svg = box.querySelector("svg");
  if (!svg) return;

  svg.addEventListener("pointerdown", (e) => {
    if (drag) return;  // ignore extra pointers mid-drag
    const el = e.target.closest?.(SHAPE_SEL.replaceAll(" ", ""));
    const shape = (el && el.dataset.pidx !== undefined && !el.closest(NONRENDER)) ? el : null;
    if (!shape) return;
    const idx = parseInt(shape.dataset.pidx);
    // Ctrl+click always toggles selection, even on an already-selected shape
    if (editMode === "move" && selected.has(idx) && !(e.ctrlKey || e.metaKey)) {
      const pre = snapshot();  // pre-mutation snapshot for undo
      drag = {
        pre, moved: false, lastDx: 0, lastDy: 0,
        els: [...selected].filter(i => shapes[i]).map(i => {
          const s = shapes[i];
          // Per-element PARENT frame: an ancestor <g> may carry a CSS/attr
          // transform (e.g. icon_1993 scales .8) — deltas measured in root
          // space would lag the cursor for such shapes.
          const m = s.parentNode.getScreenCTM().inverse();
          const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
          const p0 = pt.matrixTransform(m);
          return { el: s, i, m, p0,
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
    activeTreeNode = null; markTreeActive();
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
      actions.push({ type: "move", paths: drag.els.map(d => d.i).sort((a,b)=>a-b),
                     via: activeTreeNode, dx: +drag.lastDx.toFixed(2), dy: +drag.lastDy.toFixed(2) });
    }
    drag = null;
    box.classList.remove("dragging");
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
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
  $("canvas").classList.remove("dragging");
  $("canvas").innerHTML = s.html;
  actions = s.actions;
  restoreShapes();   // keep original data-pidx numbering (holes for removed)
  bindCanvas();
  selected = new Set(s.sel.filter(i => shapes[i]));
  // re-apply highlight state from markup: strip then re-add cleanly
  for (const el of shapes) if (el) applyHighlight(el, false);
  for (const i of selected) applyHighlight(shapes[i], true);
  refreshSelUI();
}

// ---------------------------------------------------------------- edits
function doRemove() {
  if (!selected.size) { toast("먼저 선택하세요"); return; }
  pushUndoPre(snapshot());
  const idxs = [...selected].sort((a,b)=>a-b);
  for (const i of idxs) shapes[i]?.remove();
  actions.push({ type: "remove", paths: idxs, via: activeTreeNode });
  // reindex remaining shapes so tree selection keeps working on ORIGINAL indices:
  // removed indices leave holes — keep original data-pidx (do NOT recollect).
  shapes = shapes.map((el, i) => selected.has(i) ? null : el);
  selected = new Set();
  refreshSelUI();
  toast(`${idxs.length}개 path 삭제됨`);
}

function doRecolor(color, label) {
  if (!selected.size) { toast("먼저 선택하세요"); return; }
  pushUndoPre(snapshot());
  const idxs = [...selected].sort((a,b)=>a-b);
  const strokeRecolored = [];
  for (const i of idxs) {
    const el = shapes[i];
    if (!el) continue;
    // Stroke-rendered shapes (<line>, or fill:none open paths/arcs) must be
    // recolored via stroke — painting their fill would fill the open
    // interior with a blob (or do nothing at all for <line>).
    const fillNone = el.getAttribute("fill") === "none"
      || /fill\s*:\s*none/.test(el.getAttribute("style") || "")
      || getComputedStyle(el).fill === "none";
    if (el.tagName === "line" || fillNone) {
      // Element is highlighted right now (it's selected) — write the new
      // stroke into the highlight BACKUP so deselect restores to the new
      // color instead of the old one.
      if (el.dataset.hl !== undefined) {
        el.dataset.oStroke = color;
        el.dataset.oSStroke = color;
      } else {
        el.setAttribute("stroke", color);
        el.style.stroke = color;
      }
      strokeRecolored.push(i);
    } else {
      // inline style wins over CSS class rules, presentation attrs, and
      // gradient url() fills — most robust way to guarantee the recolor shows
      el.style.fill = color;
      el.setAttribute("fill", color);
    }
  }
  const act = { type: "recolor", paths: idxs, via: activeTreeNode, color, color_name: label };
  if (strokeRecolored.length) act.stroke_recolored_paths = strokeRecolored;
  actions.push(act);
  toast(`${idxs.length}개 path → ${label} (${color})`
        + (strokeRecolored.length ? ` · ${strokeRecolored.length}개는 stroke 변경` : ""));
}

// ---------------------------------------------------------------- tree panel
// ours 생성물의 <g id data-label="..."> 구조에서 그룹 목록을 만든다.
// (tree.json이 아니라 생성 SVG 자체가 시맨틱 구조의 원천)
function renderTree() {
  const box = $("treeBox");
  box.innerHTML = "";
  const svg = $("canvas").querySelector("svg");
  if (!svg) return;
  const nodes = [];
  const skip = new Set(["defs", "clippath", "mask", "pattern", "marker", "symbol", "filter"]);
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
    box.innerHTML = '<i style="color:#9ca3af;font-size:12px">그룹 정보가 없습니다 — 캔버스에서 직접 선택하세요.</i>';
    return;
  }
  for (const n of nodes) {
    const div = document.createElement("div");
    div.className = "tnode";
    div.style.paddingLeft = (6 + n.depth * 16) + "px";
    div.innerHTML = `<span class="chip grp">grp</span>` +
      `<strong>${n.label}</strong><span class="cnt">${n.idxs.length}p</span>`;
    div.onclick = (e) => {
      const alive = n.idxs.filter(i => shapes[i]);
      if (!alive.length) { toast("이 그룹의 path가 남아있지 않습니다"); return; }
      setSelection(alive, e.ctrlKey || e.metaKey);
      activeTreeNode = n.label;
      markTreeActive(div);
    };
    box.appendChild(div);
  }
}

function markTreeActive(div) {
  $("treeBox").querySelectorAll(".tnode.active").forEach(el => el.classList.remove("active"));
  if (div) div.classList.add("active");
}

// ---------------------------------------------------------------- modes / save
function setMode(m) {
  editMode = m;
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.edit === m));
  $("modeLabel").textContent = m.toUpperCase();
  $("canvas").className = "svg-box mode-" + m;
  $("recolorCtl").style.display = m === "recolor" ? "flex" : "none";
  $("removeCtl").style.display = m === "remove" ? "flex" : "none";
  $("editHint").textContent = {
    move: "선택 후 선택된 도형을 드래그하면 선택 전체가 함께 이동합니다.",
    remove: "선택 후 [선택 삭제] 버튼 또는 Del 키.",
    recolor: "선택 후 팔레트 색을 클릭하면 fill이 교체됩니다.",
  }[m];
}

function cleanedSvg() {
  const tmp = $("canvas").querySelector("svg").cloneNode(true);
  // strip the display-sizing styles clipToViewBox put on the root
  tmp.style.width = tmp.style.height = "";
  tmp.style.maxWidth = tmp.style.maxHeight = "";
  if (!tmp.getAttribute("style")) tmp.removeAttribute("style");
  tmp.querySelectorAll("[data-pidx]").forEach(el => {
    restoreHighlightBackup(el);   // restore highlight-overridden strokes on the clone
    el.style.filter = "";
    el.classList.remove("sel");
    if (!el.getAttribute("class")) el.removeAttribute("class");
    for (const k of Object.keys(el.dataset)) delete el.dataset[k];
    if (el.getAttribute("style") === "") el.removeAttribute("style");
  });
  return new XMLSerializer().serializeToString(tmp);
}

async function save() {
  if (!actions.length) { toast(`${editMode} 편집 내역이 없습니다 — 편집 후 저장하세요`); return; }
  const wrong = actions.filter(a => a.type !== editMode);
  if (wrong.length) {
    if (!confirm(`현재 탭(${editMode})과 다른 편집(${wrong.map(a=>a.type).join(",")})이 섞여 있습니다. 그래도 저장할까요?`)) return;
  }
  const body = {
    idx: cur, edit: editMode, svg: cleanedSvg(),
    meta: { actions, n_actions: actions.length },
  };
  const r = await fetch("/api/save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { toast("저장 실패: " + (await r.text()).slice(0, 120)); return; }
  await loadList();
  // Saved — reset to the pristine original so the next edit type starts
  // clean, and stale actions can't leak into a second save.
  resetCanvas();
  updateBadges();
  toast(`저장됨: ${state.stem}__${editMode} — 캔버스는 원본으로 리셋됨`);
}

function updateBadges() {
  const d = files[cur]?.done || {};
  $("bMove").innerHTML = d.move ? '<span class="badge">✓</span>' : "";
  $("bRemove").innerHTML = d.remove ? '<span class="badge">✓</span>' : "";
  $("bRecolor").innerHTML = d.recolor ? '<span class="badge">✓</span>' : "";
}

// ---------------------------------------------------------------- wiring
document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => {
  if (actions.length && !confirm("저장하지 않은 편집이 있습니다. 탭을 바꾸면 캔버스가 원본으로 리셋됩니다.")) return;
  setMode(b.dataset.edit);
  resetCanvas();
}));
$("fileSel").addEventListener("change", () => loadFile(parseInt($("fileSel").value)));
$("prev").addEventListener("click", () => loadFile(cur - 1));
$("next").addEventListener("click", () => loadFile(cur + 1));
$("reset").addEventListener("click", () => { if (confirm("캔버스를 원본으로 되돌릴까요?")) resetCanvas(); });
$("undo").addEventListener("click", doUndo);
$("save").addEventListener("click", save);
$("delBtn").addEventListener("click", doRemove);

const ctl = $("recolorCtl");
for (const [label, color] of PALETTE) {
  const b = document.createElement("button");
  b.className = "swatch"; b.style.background = color; b.title = `${label} ${color}`;
  b.addEventListener("click", () => doRecolor(color, label));
  ctl.appendChild(b);
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "Escape") clearSelection();
  if (e.key === "Delete" && editMode === "remove") doRemove();
  if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); doUndo(); }
  if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); save(); }
});

(async () => {
  await loadList();
  setMode("move");
  await loadFile(0);
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


if __name__ == "__main__":
    print(f"Edit-GT maker: {len(list_files())} samples from {SRC}")
    print(f"  output: {OUT}")
    print(f"Open http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
