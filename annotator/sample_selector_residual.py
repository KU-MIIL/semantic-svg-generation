"""Sample selector — residual edition.

Same UX as sample_selector.py, but points at sample_residual whose layout has
nested category dirs (sample/Icon, sample/illustration). Reviewer is labeler_3
alone — selections land in sample_residual/_selections/labeler_3.json.

Run:
    .venv/bin/python sample_selector_residual.py
    # serves at http://localhost:8094
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

import visualize_tree as vt
from PIL import Image, ImageDraw

_PKG_ROOT = Path(__file__).resolve().parent
SRC_ROOT = _PKG_ROOT / "sample_residual"

# Composite category labels → filesystem subpath under SRC_ROOT. Slash-free
# labels keep URLs simple.
CAT_PATHS: dict[str, str] = {
    "icon": "icon",
    "illustration": "illustration",
    "emoji": "emoji",
    "sample-Icon": "sample/Icon",
    "sample-illustration": "sample/illustration",
}
CATS = list(CAT_PATHS.keys())
REVIEWERS = ["labeler_3"]
SEL_DIR = SRC_ROOT / "_selections"
PORT = 8094


def cat_dir(cat: str, kind: str) -> Path:
    """Resolve {cat}/{svg|png} to its real path on disk."""
    return SRC_ROOT / CAT_PATHS[cat] / kind


# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------
def list_samples() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for cat in CATS:
        d = cat_dir(cat, "svg")
        if not d.exists():
            continue
        files = [p for p in d.glob("*.svg") if p.stem.isdigit()]
        files.sort(key=lambda p: int(p.stem))
        out.extend((cat, p.stem) for p in files)
    return out


def split_for_reviewers(samples: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """With a single reviewer, everything goes to them. Multi-reviewer code path
    kept for symmetry with sample_selector.py."""
    bucket = {r: [] for r in REVIEWERS}
    if len(REVIEWERS) == 1:
        bucket[REVIEWERS[0]] = list(samples)
        return bucket
    by_cat: dict[str, list[tuple[str, str]]] = {c: [] for c in CATS}
    for cat, stem in samples:
        by_cat[cat].append((cat, stem))
    for cat in CATS:
        lst = by_cat[cat]
        half = (len(lst) + 1) // 2
        bucket[REVIEWERS[0]].extend(lst[:half])
        bucket[REVIEWERS[1]].extend(lst[half:])
    return bucket


# -----------------------------------------------------------------------------
# Selection persistence
# -----------------------------------------------------------------------------
def sel_path(reviewer: str) -> Path:
    return SEL_DIR / f"{reviewer}.json"


def load_selections(reviewer: str) -> set[str]:
    p = sel_path(reviewer)
    if not p.exists():
        return set()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return set(raw.get("selected", []))
    except Exception:
        return set()


def save_selections(reviewer: str, selected: set[str]) -> None:
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    p = sel_path(reviewer)
    body = {
        "reviewer": reviewer,
        "count": len(selected),
        "selected": sorted(selected, key=lambda s: (s.split("/")[0], int(s.split("/")[1]))),
    }
    p.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selection_id(cat: str, stem: str) -> str:
    return f"{cat}/{stem}"


# -----------------------------------------------------------------------------
# Tree visualization
# -----------------------------------------------------------------------------
def render_tree_png(cat: str, stem: str) -> bytes:
    if cat not in CAT_PATHS:
        raise HTTPException(404, f"unknown category: {cat}")
    svg = cat_dir(cat, "svg") / f"{stem}.svg"
    tree_p = cat_dir(cat, "svg") / f"{stem}.tree.json"
    if not svg.exists():
        raise HTTPException(404, f"svg missing: {cat}/{stem}")

    svg_text = svg.read_text()
    tree = None
    if tree_p.exists():
        try:
            tree = json.loads(tree_p.read_text())
        except Exception:
            tree = None

    if tree is None or (isinstance(tree, dict) and tree.get("fail")):
        n_shapes = len(vt.shape_spans(svg_text))
        full = vt.rasterize(svg_text, list(range(n_shapes)), 320)
        W, H = 480, 420
        canvas = Image.new("RGB", (W, H), vt.BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((vt.PAD, 10), f"{cat}/{stem}.svg",
                  fill=(40, 40, 40), font=vt.FONT_TITLE)
        canvas.paste(full, ((W - 320) // 2, 50),
                     full if full.mode == "RGBA" else None)
        if isinstance(tree, dict) and tree.get("fail"):
            draw.rectangle([0, 380, W, 420], fill=(239, 68, 68))
            msg = "✗ MARKED FAILED — semantically tangled"
            bbox = draw.textbbox((0, 0), msg, font=vt.FONT_TITLE)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(((W - tw) // 2, 380 + (40 - th) // 2 - 2),
                      msg, fill=(255, 255, 255), font=vt.FONT_TITLE)
        else:
            draw.rectangle([0, 380, W, 420], fill=(156, 163, 175))
            msg = "(no tree.json — raw SVG)"
            bbox = draw.textbbox((0, 0), msg, font=vt.FONT_TITLE)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(((W - tw) // 2, 380 + (40 - th) // 2 - 2),
                      msg, fill=(255, 255, 255), font=vt.FONT_TITLE)
    else:
        root = vt.build_layout(tree)
        vt.compute_widths(root)
        vt.assign_positions(root, vt.PAD, 0)
        nodes = vt.all_nodes(root)
        W = root.w_subtree + 2 * vt.PAD
        H = (vt.max_depth(root) + 1) * (vt.NODE_H + vt.V_GAP) - vt.V_GAP + 2 * vt.PAD + 30
        canvas = Image.new("RGB", (W, H + 30), vt.BG)
        draw = ImageDraw.Draw(canvas)
        title = f"{cat}/{stem}.svg  —  {len(root.paths)} paths"
        draw.text((vt.PAD, 6), title, fill=(40, 40, 40), font=vt.FONT_TITLE)
        for n in nodes:
            n.y += 30
        vt.draw_edges(draw, root)
        for n in nodes:
            vt.draw_node(canvas, draw, n, svg_text)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI()


@app.get("/api/reviewers")
def api_reviewers() -> JSONResponse:
    return JSONResponse({"reviewers": REVIEWERS})


@app.get("/api/manifest")
def api_manifest(reviewer: str) -> JSONResponse:
    if reviewer not in REVIEWERS:
        raise HTTPException(400, f"unknown reviewer: {reviewer}")
    split = split_for_reviewers(list_samples())
    items = split[reviewer]
    selected = load_selections(reviewer)
    out = [{
        "cat": cat,
        "stem": stem,
        "id": selection_id(cat, stem),
        "selected": selection_id(cat, stem) in selected,
    } for cat, stem in items]
    return JSONResponse({
        "reviewer": reviewer,
        "items": out,
        "total": len(items),
        "selected_count": sum(1 for o in out if o["selected"]),
    })


@app.get("/png/{cat}/{stem}.png")
def serve_png(cat: str, stem: str) -> FileResponse:
    if cat not in CAT_PATHS:
        raise HTTPException(404, f"unknown category: {cat}")
    f = cat_dir(cat, "png") / f"{stem}.png"
    if not f.exists():
        raise HTTPException(404, f"missing: {f}")
    return FileResponse(f, media_type="image/png")


@app.get("/treevis/{cat}/{stem}.png")
def serve_treevis(cat: str, stem: str) -> Response:
    png = render_tree_png(cat, stem)
    return Response(content=png, media_type="image/png")


class ToggleBody(BaseModel):
    reviewer: str
    id: str
    selected: bool


@app.post("/api/toggle")
def api_toggle(body: ToggleBody) -> JSONResponse:
    if body.reviewer not in REVIEWERS:
        raise HTTPException(400, f"unknown reviewer: {body.reviewer}")
    sel = load_selections(body.reviewer)
    if body.selected:
        sel.add(body.id)
    else:
        sel.discard(body.id)
    save_selections(body.reviewer, sel)
    return JSONResponse({"ok": True, "selected_count": len(sel)})


@app.get("/api/selections")
def api_selections(reviewer: str) -> JSONResponse:
    if reviewer not in REVIEWERS:
        raise HTTPException(400, f"unknown reviewer: {reviewer}")
    return JSONResponse({"reviewer": reviewer,
                         "selected": sorted(load_selections(reviewer))})


@app.post("/api/finalize")
def api_finalize() -> JSONResponse:
    """Delete every sample (svg + tree.json + png) NOT in any reviewer's
    selection."""
    kept_ids: set[str] = set()
    for r in REVIEWERS:
        kept_ids |= load_selections(r)
    removed = []
    for cat, stem in list_samples():
        sid = selection_id(cat, stem)
        if sid in kept_ids:
            continue
        for fname in (f"{stem}.svg", f"{stem}.tree.json"):
            fp = cat_dir(cat, "svg") / fname
            if fp.exists():
                fp.unlink()
        png = cat_dir(cat, "png") / f"{stem}.png"
        if png.exists():
            png.unlink()
        removed.append(sid)
    return JSONResponse({"removed": removed, "kept": sorted(kept_ids)})


# -----------------------------------------------------------------------------
# HTML — identical layout to sample_selector.py
# -----------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Sample Selector — Residual</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #1f2937; }
  header { position: sticky; top: 0; background: #1f2937; color: #fff; padding: 10px 14px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; z-index: 10; }
  header h1 { margin: 0; font-size: 14px; font-weight: 600; }
  header select, header button { font-size: 13px; padding: 5px 10px; border-radius: 4px; border: 1px solid #4b5563; background: #374151; color: #fff; cursor: pointer; }
  header button:hover { background: #4b5563; }
  header button.primary { background: #2563eb; border-color: #2563eb; }
  header button.primary:hover { background: #1d4ed8; }
  header button.success { background: #16a34a; border-color: #16a34a; }
  header button.danger { background: #dc2626; border-color: #dc2626; }
  header button.danger:hover { background: #b91c1c; }
  header button:disabled { opacity: 0.4; cursor: not-allowed; }
  .progress { font-size: 12px; opacity: 0.9; font-family: ui-monospace, monospace; }
  .grow { flex: 1; }
  main { padding: 12px; display: grid; grid-template-columns: 280px 1fr; gap: 12px; min-height: calc(100vh - 50px); }
  aside { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; max-height: calc(100vh - 70px); overflow-y: auto; }
  aside h3 { margin: 4px 0 8px; font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
  .list-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 6px; border-radius: 3px; font-size: 12px; font-family: ui-monospace, monospace; cursor: pointer; }
  .list-row:hover { background: #eef2ff; }
  .list-row.active { background: #dbeafe; font-weight: 600; }
  .list-row .badge { font-size: 10px; padding: 1px 5px; border-radius: 8px; }
  .list-row .badge.sel { background: #16a34a; color: #fff; }
  .list-row .badge.skip { background: #e5e7eb; color: #6b7280; }
  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; }
  .panel h3 { margin: 0 0 8px; font-size: 12px; color: #374151; }
  .img-box { background: #fafafa; border: 1px solid #eee; border-radius: 4px; padding: 8px; min-height: 380px; display: flex; align-items: center; justify-content: center; overflow: auto; }
  .img-box img { max-width: 100%; max-height: 600px; object-fit: contain; }
  .controls { display: flex; gap: 8px; align-items: center; padding: 10px; background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; margin-top: 10px; flex-wrap: wrap; }
  .controls button { padding: 8px 16px; font-size: 14px; border-radius: 4px; cursor: pointer; border: 1px solid #d1d5db; background: #fff; }
  .controls button.select { background: #16a34a; color: #fff; border-color: #16a34a; }
  .controls button.skip { background: #f3f4f6; }
  .controls .meta { font-size: 12px; color: #6b7280; font-family: ui-monospace, monospace; }
  .kbd { background: #e5e7eb; border: 1px solid #d1d5db; border-radius: 3px; padding: 1px 6px; font-size: 11px; font-family: ui-monospace, monospace; }
  #toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #1f2937; color: #fff; padding: 10px 16px; border-radius: 6px; opacity: 0; transition: opacity 0.2s; pointer-events: none; z-index: 50; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>Sample Selector — Residual</h1>
  <span>Reviewer:</span>
  <select id="reviewer"></select>
  <button id="prev">◀ Prev</button>
  <button id="next">Next ▶</button>
  <span class="progress" id="progress"></span>
  <span class="grow"></span>
  <span class="progress" id="kept"></span>
  <button id="export" title="Download my selections as JSON">📥 Export JSON</button>
  <button id="finalize" class="danger" title="Delete all samples NOT selected">Finalize</button>
</header>
<main>
  <aside>
    <h3 id="listTitle">Files</h3>
    <div id="list"></div>
  </aside>
  <section>
    <div class="panels">
      <div class="panel">
        <h3>Input PNG (raster)</h3>
        <div class="img-box"><img id="inputImg" alt=""></div>
      </div>
      <div class="panel">
        <h3>Tree.json visualization</h3>
        <div class="img-box"><img id="treeImg" alt=""></div>
      </div>
    </div>
    <div class="controls">
      <button class="select" id="btnSelect">✓ Select (keep)</button>
      <button class="skip" id="btnSkip">Skip</button>
      <span class="meta" id="curMeta"></span>
      <span class="grow"></span>
      <span class="meta">단축키: <span class="kbd">Y</span> 선택, <span class="kbd">N</span> 스킵, <span class="kbd">←</span>/<span class="kbd">→</span> 이동</span>
    </div>
  </section>
</main>
<div id="toast"></div>
<script>
const $ = (id) => document.getElementById(id);
let state = { reviewer: localStorage.getItem("ss_res_reviewer") || "", items: [], idx: 0 };

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2000);
}

async function loadReviewers() {
  const r = await (await fetch("/api/reviewers")).json();
  const sel = $("reviewer");
  sel.innerHTML = '<option value="">— select —</option>'
    + r.reviewers.map(u => `<option value="${u}">${u}</option>`).join("");
  if (state.reviewer && r.reviewers.includes(state.reviewer)) sel.value = state.reviewer;
  else if (r.reviewers.length === 1) {
    sel.value = r.reviewers[0];
    state.reviewer = r.reviewers[0];
    localStorage.setItem("ss_res_reviewer", state.reviewer);
  }
}

async function loadManifest() {
  if (!state.reviewer) { $("list").innerHTML = "<i>reviewer를 선택하세요</i>"; return; }
  const d = await (await fetch(`/api/manifest?reviewer=${encodeURIComponent(state.reviewer)}`)).json();
  state.items = d.items;
  state.idx = Math.min(state.idx, Math.max(0, state.items.length - 1));
  $("listTitle").textContent = `${state.reviewer} — ${d.total} files`;
  renderList();
  renderCurrent();
}

function renderList() {
  const html = state.items.map((it, i) => `
    <div class="list-row ${i === state.idx ? "active" : ""}" data-idx="${i}">
      <span>[${i+1}] ${it.cat}/${it.stem}</span>
      <span class="badge ${it.selected ? "sel" : "skip"}">${it.selected ? "✓" : "—"}</span>
    </div>`).join("");
  $("list").innerHTML = html;
  $("list").querySelectorAll(".list-row").forEach(el => {
    el.addEventListener("click", () => { state.idx = parseInt(el.dataset.idx); renderCurrent(); });
  });
  const selCount = state.items.filter(it => it.selected).length;
  $("kept").textContent = `selected ${selCount}/${state.items.length}`;
  $("progress").textContent = state.items.length
    ? `[${state.idx + 1}/${state.items.length}]` : "(no items)";
}

function renderCurrent() {
  if (!state.items.length) {
    $("inputImg").src = ""; $("treeImg").src = "";
    $("curMeta").textContent = "";
    return;
  }
  const it = state.items[state.idx];
  $("inputImg").src = `/png/${it.cat}/${it.stem}.png?v=${Date.now()}`;
  $("treeImg").src = `/treevis/${it.cat}/${it.stem}.png?v=${Date.now()}`;
  $("curMeta").textContent = `${it.cat}/${it.stem}  — ${it.selected ? "SELECTED" : "skipped"}`;
  $("btnSelect").textContent = it.selected ? "✓ Selected (click to unselect)" : "✓ Select (keep)";
  $("list").querySelectorAll(".list-row").forEach(el => {
    el.classList.toggle("active", parseInt(el.dataset.idx) === state.idx);
  });
  const active = $("list").querySelector(".list-row.active");
  if (active) active.scrollIntoView({ block: "nearest", behavior: "smooth" });
  $("progress").textContent = `[${state.idx + 1}/${state.items.length}]`;
}

async function toggle(forceSelected) {
  if (!state.items.length) return;
  const it = state.items[state.idx];
  const nextSel = (forceSelected === undefined) ? !it.selected : forceSelected;
  const r = await fetch("/api/toggle", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({reviewer: state.reviewer, id: it.id, selected: nextSel}),
  });
  if (!r.ok) { toast("save failed"); return; }
  it.selected = nextSel;
  renderList(); renderCurrent();
}

function nav(delta) {
  if (!state.items.length) return;
  const n = state.items.length;
  state.idx = (state.idx + delta + n) % n;
  renderCurrent();
}

$("reviewer").addEventListener("change", () => {
  state.reviewer = $("reviewer").value;
  localStorage.setItem("ss_res_reviewer", state.reviewer);
  state.idx = 0;
  loadManifest();
});
$("prev").addEventListener("click", () => nav(-1));
$("next").addEventListener("click", () => nav(1));
$("btnSelect").addEventListener("click", () => toggle());
$("btnSkip").addEventListener("click", async () => { await toggle(false); nav(1); });

$("export").addEventListener("click", async () => {
  if (!state.reviewer) { toast("reviewer를 선택하세요"); return; }
  const d = await (await fetch(`/api/selections?reviewer=${encodeURIComponent(state.reviewer)}`)).json();
  const blob = new Blob([JSON.stringify(d, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${state.reviewer}_residual_selections.json`;
  a.click();
});

$("finalize").addEventListener("click", async () => {
  if (!confirm("선택되지 않은 모든 샘플의 svg/tree.json/png를 삭제합니다. 진행할까요?")) return;
  const r = await (await fetch("/api/finalize", {method: "POST"})).json();
  toast(`삭제 ${r.removed.length}개 / 보존 ${r.kept.length}개`);
  await loadManifest();
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft")  { e.preventDefault(); nav(-1); }
  if (e.key === "ArrowRight") { e.preventDefault(); nav(1); }
  if (e.key === "y" || e.key === "Y") { e.preventDefault(); toggle(true);  nav(1); }
  if (e.key === "n" || e.key === "N") { e.preventDefault(); toggle(false); nav(1); }
});

(async () => {
  await loadReviewers();
  if (state.reviewer) await loadManifest();
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


if __name__ == "__main__":
    samples = list_samples()
    by_cat: dict[str, int] = {}
    for c, _ in samples:
        by_cat[c] = by_cat.get(c, 0) + 1
    print(f"Sample selector (residual) at http://localhost:{PORT}")
    print(f"  source: {SRC_ROOT}")
    print(f"  total:  {len(samples)} samples")
    for c, n in by_cat.items():
        print(f"    {c}: {n}")
    print(f"  reviewers: {REVIEWERS}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
