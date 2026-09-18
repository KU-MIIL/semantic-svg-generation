"""Curation selector webui — pick eval-worthy renders from curation_pool.

Run:
    .venv/bin/python curation_selector.py
    # serves at http://localhost:8095

UX (grid, like emoji_selector.py):
  - Reviewer picks name (김태훈 or 박세환). Each gets ~half of every style
    group (interleaved split, so neither reviewer gets only top-ranked rows).
  - Thumbnail GRID, 100 per page. Click a tile to toggle ✓ selection —
    click again to unselect. Every toggle is persisted immediately to
    curation_pool/_selections/<reviewer>.json — no unsaved state.
  - Hover a tile → floating 380px zoom preview (renders are 512px; grid
    tiles are too small to judge quality).
  - Filters: by style group and by selected/unselected. "페이지 전체 ✓/✕"
    batch-toggles the visible page.
  - "Save → renders_emnlp" syncs YOUR assigned subset into
    curation_pool/renders_emnlp/ as SVG SOURCE FILES: selected ids are
    copied from svgs/tier{A,B,C}/<id>.svg, previously saved but
    now-unselected ones are removed. (The grid thumbnails come from
    renders/<id>.png, which are the pipeline's 512px rasterizations of
    exactly these SVGs — same id, 1:1 verified.) The source renders/ and
    svgs/ pools are never modified. The two reviewers' subsets are
    disjoint, so each save only touches that reviewer's files. The header
    shows "저장 안 됨: +N −M" while the current selection differs from what
    Save last wrote.
  - Shortcuts: ←/→ page navigation.

Source data: curation_pool/svgs/tier{A,B,C}/*.svg  (display via
             curation_pool/renders/*.png, metadata via keeps.csv)
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

_PKG_ROOT = Path(__file__).resolve().parent
POOL = _PKG_ROOT / "curation_pool"
RENDERS = POOL / "renders"
SVGS = POOL / "svgs"          # tier{A,B,C}/<id>.svg — what Save actually copies
KEEPS_CSV = POOL / "keeps.csv"
OUT_DIR = POOL / "renders_emnlp"
SEL_DIR = POOL / "_selections"
REVIEWERS = ["김태훈", "박세환"]
PORT = 8095  # 8088/8089/8093/8094 are taken by the labelers/selectors


# -----------------------------------------------------------------------------
# Manifest: all render ids + metadata from keeps.csv, per-reviewer split
# -----------------------------------------------------------------------------
def svg_map() -> dict[str, Path]:
    """id → source SVG path (svgs/tier{A,B,C}/<id>.svg). Stems are unique
    across tiers and match renders/<id>.png 1:1 (verified against keeps.csv)."""
    return {p.stem: p
            for tier_dir in sorted(SVGS.glob("tier*"))
            for p in tier_dir.glob("*.svg")}


def load_manifest() -> list[dict]:
    """All candidates in keeps.csv order, only those whose render PNG exists."""
    rows: list[dict] = []
    with KEEPS_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (RENDERS / f"{r['id']}.png").exists():
                continue
            rows.append({
                "id": r["id"],
                "style": r.get("vlm_style", "") or "unknown",
                "source": r.get("source", ""),
                "license_tier": r.get("license_tier", ""),
                "n_elements": r.get("n_elements", ""),
                "n_colors": r.get("n_colors", ""),
            })
    return rows


def split_for_reviewers(rows: list[dict]) -> dict[str, list[dict]]:
    """Interleave within each style group: even → REVIEWERS[0], odd →
    REVIEWERS[1]. Both reviewers see every style, and neither gets a
    rank-biased half. Deterministic as long as keeps.csv is unchanged."""
    bucket: dict[str, list[dict]] = {r: [] for r in REVIEWERS}
    by_style: dict[str, list[dict]] = {}
    for row in rows:
        by_style.setdefault(row["style"], []).append(row)
    for style in sorted(by_style):
        for i, row in enumerate(by_style[style]):
            bucket[REVIEWERS[i % 2]].append(row)
    return bucket


# -----------------------------------------------------------------------------
# Selection persistence (same scheme as sample_selector.py)
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
    body = {
        "reviewer": reviewer,
        "count": len(selected),
        "selected": sorted(selected),
    }
    sel_path(reviewer).write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    split = split_for_reviewers(load_manifest())
    items = split[reviewer]
    selected = load_selections(reviewer)
    saved = {p.stem for p in OUT_DIR.glob("*.svg")} if OUT_DIR.exists() else set()
    out = [{
        **row,
        "selected": row["id"] in selected,
        "saved": row["id"] in saved,
    } for row in items]
    return JSONResponse({
        "reviewer": reviewer,
        "items": out,
        "total": len(items),
        "selected_count": sum(1 for o in out if o["selected"]),
    })


@app.get("/png/{item_id}.png")
def serve_png(item_id: str) -> FileResponse:
    f = RENDERS / f"{item_id}.png"
    if not f.exists():
        raise HTTPException(404, f"missing: {f}")
    return FileResponse(f, media_type="image/png")


class ToggleBody(BaseModel):
    reviewer: str
    id: str
    selected: bool


@app.post("/api/toggle")
def api_toggle(body: ToggleBody) -> JSONResponse:
    """Toggle persists immediately — re-clicking a selected item unselects it
    and the JSON on disk reflects that at once."""
    if body.reviewer not in REVIEWERS:
        raise HTTPException(400, f"unknown reviewer: {body.reviewer}")
    sel = load_selections(body.reviewer)
    if body.selected:
        sel.add(body.id)
    else:
        sel.discard(body.id)
    save_selections(body.reviewer, sel)
    return JSONResponse({"ok": True, "selected_count": len(sel)})


class ToggleBatchBody(BaseModel):
    reviewer: str
    ids: list[str]
    selected: bool


@app.post("/api/toggle_batch")
def api_toggle_batch(body: ToggleBatchBody) -> JSONResponse:
    """Select/unselect a whole page of ids in one write."""
    if body.reviewer not in REVIEWERS:
        raise HTTPException(400, f"unknown reviewer: {body.reviewer}")
    sel = load_selections(body.reviewer)
    if body.selected:
        sel |= set(body.ids)
    else:
        sel -= set(body.ids)
    save_selections(body.reviewer, sel)
    return JSONResponse({"ok": True, "selected_count": len(sel)})


@app.get("/api/selections")
def api_selections(reviewer: str) -> JSONResponse:
    if reviewer not in REVIEWERS:
        raise HTTPException(400, f"unknown reviewer: {reviewer}")
    return JSONResponse({"reviewer": reviewer,
                         "selected": sorted(load_selections(reviewer))})


class SyncBody(BaseModel):
    reviewer: str


@app.post("/api/sync")
def api_sync(body: SyncBody) -> JSONResponse:
    """Sync this reviewer's assigned subset into renders_emnlp/ as SVGs:
      - selected           → copy svgs/tier{A,B,C}/<id>.svg in
      - assigned+unselected → remove <id>.svg (selection cancelled)
    Legacy <id>.png files from the earlier PNG-based save are cleaned up for
    this reviewer's ids either way. Files belonging to the OTHER reviewer's
    subset are never touched, so both reviewers can save independently.
    Source renders/ and svgs/ are read-only."""
    if body.reviewer not in REVIEWERS:
        raise HTTPException(400, f"unknown reviewer: {body.reviewer}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = split_for_reviewers(load_manifest())
    my_ids = {row["id"] for row in split[body.reviewer]}
    selected = load_selections(body.reviewer) & my_ids
    srcs = svg_map()
    added, removed, skipped = [], [], []
    for item_id in sorted(my_ids):
        dst = OUT_DIR / f"{item_id}.svg"
        # migrate: drop stale PNG from the old PNG-based sync
        legacy_png = OUT_DIR / f"{item_id}.png"
        if legacy_png.exists():
            legacy_png.unlink()
        if item_id in selected:
            src = srcs.get(item_id)
            if src is None:
                skipped.append(item_id)
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
                added.append(item_id)
        else:
            if dst.exists():
                dst.unlink()
                removed.append(item_id)
    total_saved = len(list(OUT_DIR.glob("*.svg")))
    return JSONResponse({
        "ok": True,
        "added": len(added),
        "removed": len(removed),
        "skipped": skipped,          # selected but source svg missing
        "my_saved": len(selected) - len(skipped),
        "total_saved": total_saved,
        "out_dir": str(OUT_DIR),
    })


# -----------------------------------------------------------------------------
# HTML
# -----------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Curation Selector</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #1f2937; }
  header { position: sticky; top: 0; background: #1f2937; color: #fff; padding: 8px 14px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; z-index: 10; }
  header h1 { margin: 0; font-size: 14px; font-weight: 600; }
  header select, header button, header input[type=number] { font-size: 13px; padding: 5px 8px; border-radius: 4px; border: 1px solid #4b5563; background: #374151; color: #fff; cursor: pointer; }
  header input[type=number] { width: 58px; cursor: text; }
  header button:hover { background: #4b5563; }
  header button.success { background: #16a34a; border-color: #16a34a; }
  header button.success:hover { background: #15803d; }
  header button:disabled { opacity: 0.4; cursor: not-allowed; }
  .progress { font-size: 12px; opacity: 0.9; font-family: ui-monospace, monospace; }
  .pendingsync { font-size: 12px; font-family: ui-monospace, monospace; color: #fbbf24; font-weight: 700; }
  .grow { flex: 1; }
  main { padding: 12px; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 8px; }
  .cell { position: relative; background: #fff; border: 3px solid #e5e7eb; border-radius: 8px; padding: 4px; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: border-color .08s, background .08s; }
  .cell img { width: 100%; height: 100%; object-fit: contain; pointer-events: none; }
  .cell .rid { position: absolute; bottom: 2px; left: 2px; right: 2px; font-size: 9px; color: #666; text-align: center; background: rgba(255,255,255,.85); border-radius: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; pointer-events: none; }
  .cell:hover { border-color: #6b7280; }
  .cell.sel { border-color: #16a34a; background: #ecfdf5; }
  .cell.sel::after { content: "✓"; position: absolute; top: 2px; right: 6px; font-size: 18px; color: #16a34a; font-weight: bold; text-shadow: 0 0 3px #fff; }
  #zoom { position: fixed; z-index: 40; pointer-events: none; display: none; background: #fff; border: 1px solid #d1d5db; border-radius: 8px; box-shadow: 0 8px 30px rgba(0,0,0,.25); padding: 8px; }
  #zoom img { width: 380px; height: 380px; object-fit: contain; display: block; }
  #zoom .zmeta { font-size: 11px; color: #4b5563; font-family: ui-monospace, monospace; text-align: center; margin-top: 4px; max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #1f2937; color: #fff; padding: 10px 16px; border-radius: 6px; opacity: 0; transition: opacity 0.2s; pointer-events: none; z-index: 50; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>Curation Selector</h1>
  <select id="reviewer"></select>
  <select id="styleFilter" title="스타일 필터">
    <option value="">style: all</option>
  </select>
  <select id="viewFilter" title="선택 상태 필터">
    <option value="all">view: all</option>
    <option value="selected">view: ✓ selected</option>
    <option value="unselected">view: unselected</option>
  </select>
  <button id="prev">◀</button>
  <span class="progress">Page <input type="number" id="page" min="1" value="1"> / <span id="pages">?</span></span>
  <button id="next">▶</button>
  <button id="selPage" title="이 페이지의 모든 항목 선택">페이지 전체 ✓</button>
  <button id="clrPage" title="이 페이지의 모든 항목 선택 해제">페이지 전체 ✕</button>
  <span class="grow"></span>
  <span class="progress" id="kept"></span>
  <span class="pendingsync" id="pendingSync"></span>
  <button id="export" title="내 선택 목록을 JSON으로 다운로드">📥 JSON</button>
  <button id="save" class="success" title="선택한 항목의 SVG 원본을 renders_emnlp/로 동기화 (선택 해제분은 제거)">💾 Save SVG → renders_emnlp</button>
</header>
<main><div id="grid"></div></main>
<div id="zoom"><img id="zoomImg" alt=""><div class="zmeta" id="zoomMeta"></div></div>
<div id="toast"></div>
<script>
const $ = (id) => document.getElementById(id);
const PAGE_SIZE = 100;
let state = {
  reviewer: localStorage.getItem("cs_reviewer") || "",
  items: [],            // full manifest for this reviewer
  page: 1,
  styleFilter: "",
  viewFilter: "all",
};

function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2500);
}

async function loadReviewers() {
  const r = await (await fetch("/api/reviewers")).json();
  const sel = $("reviewer");
  sel.innerHTML = '<option value="">— 이름 선택 —</option>'
    + r.reviewers.map(u => `<option value="${u}">${u}</option>`).join("");
  if (state.reviewer && r.reviewers.includes(state.reviewer)) sel.value = state.reviewer;
}

async function loadManifest() {
  if (!state.reviewer) { $("grid").innerHTML = "<i>왼쪽 위에서 reviewer를 선택하세요</i>"; return; }
  const d = await (await fetch(`/api/manifest?reviewer=${encodeURIComponent(state.reviewer)}`)).json();
  state.items = d.items;
  // Style filter options with counts
  const styles = {};
  for (const it of state.items) styles[it.style] = (styles[it.style] || 0) + 1;
  const cur = state.styleFilter;
  $("styleFilter").innerHTML = `<option value="">style: all (${state.items.length})</option>`
    + Object.keys(styles).sort().map(s =>
        `<option value="${s}" ${s === cur ? "selected" : ""}>${s} (${styles[s]})</option>`).join("");
  render();
}

function filteredItems() {
  return state.items.filter(it =>
    (!state.styleFilter || it.style === state.styleFilter)
    && (state.viewFilter === "all"
        || (state.viewFilter === "selected" ? it.selected : !it.selected)));
}

function pageItems() {
  const f = filteredItems();
  const pages = Math.max(1, Math.ceil(f.length / PAGE_SIZE));
  state.page = Math.min(Math.max(1, state.page), pages);
  const start = (state.page - 1) * PAGE_SIZE;
  return { f, pages, slice: f.slice(start, start + PAGE_SIZE) };
}

function updateCounts() {
  const selCount = state.items.filter(it => it.selected).length;
  // Pending sync = differences between current selection and what's on disk
  // in renders_emnlp (per manifest's `saved` snapshot, updated after Save).
  const plus  = state.items.filter(it => it.selected && !it.saved).length;
  const minus = state.items.filter(it => !it.selected && it.saved).length;
  $("kept").textContent = `✓ ${selCount} / ${state.items.length}`;
  $("pendingSync").textContent =
    (plus || minus) ? `저장 안 됨: +${plus} −${minus}` : "";
}

function render() {
  const { f, pages, slice } = pageItems();
  $("pages").textContent = pages;
  $("page").max = pages;
  $("page").value = state.page;
  const grid = $("grid");
  grid.innerHTML = "";
  for (const it of slice) {
    const cell = document.createElement("div");
    cell.className = "cell" + (it.selected ? " sel" : "");
    cell.dataset.id = it.id;
    cell.innerHTML = `<img src="/png/${it.id}.png" loading="lazy"><span class="rid">${it.id}</span>`;
    cell.addEventListener("click", () => onCellClick(it, cell));
    cell.addEventListener("mouseenter", (e) => showZoom(it, e));
    cell.addEventListener("mousemove", moveZoom);
    cell.addEventListener("mouseleave", hideZoom);
    grid.appendChild(cell);
  }
  if (!slice.length) grid.innerHTML = "<i>표시할 항목이 없습니다 (필터 확인)</i>";
  updateCounts();
}

async function onCellClick(it, cell) {
  const nextSel = !it.selected;
  cell.classList.toggle("sel", nextSel);   // optimistic update
  const r = await fetch("/api/toggle", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({reviewer: state.reviewer, id: it.id, selected: nextSel}),
  });
  if (!r.ok) { cell.classList.toggle("sel", it.selected); toast("저장 실패"); return; }
  it.selected = nextSel;
  updateCounts();
}

// --- hover zoom -------------------------------------------------------------
function showZoom(it, e) {
  $("zoomImg").src = `/png/${it.id}.png`;
  $("zoomMeta").textContent = `${it.id} · ${it.style} · tier ${it.license_tier} · ${it.n_elements} elems`;
  $("zoom").style.display = "block";
  moveZoom(e);
}
function moveZoom(e) {
  const z = $("zoom");
  const zw = 400, zh = 430;
  let x = e.clientX + 20, y = e.clientY + 20;
  if (x + zw > window.innerWidth)  x = e.clientX - zw - 20;
  if (y + zh > window.innerHeight) y = Math.max(8, window.innerHeight - zh - 8);
  z.style.left = x + "px"; z.style.top = y + "px";
}
function hideZoom() { $("zoom").style.display = "none"; }

// --- page batch select / clear ----------------------------------------------
async function batchPage(selected) {
  const { slice } = pageItems();
  const ids = slice.filter(it => it.selected !== selected).map(it => it.id);
  if (!ids.length) return;
  const r = await fetch("/api/toggle_batch", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({reviewer: state.reviewer, ids, selected}),
  });
  if (!r.ok) { toast("저장 실패"); return; }
  for (const it of slice) if (ids.includes(it.id)) it.selected = selected;
  render();
}

// --- events -------------------------------------------------------------------
$("reviewer").addEventListener("change", () => {
  state.reviewer = $("reviewer").value;
  localStorage.setItem("cs_reviewer", state.reviewer);
  state.page = 1;
  loadManifest();
});
$("styleFilter").addEventListener("change", () => { state.styleFilter = $("styleFilter").value; state.page = 1; render(); });
$("viewFilter").addEventListener("change", () => { state.viewFilter = $("viewFilter").value; state.page = 1; render(); });
$("prev").addEventListener("click", () => { state.page--; render(); window.scrollTo(0, 0); });
$("next").addEventListener("click", () => { state.page++; render(); window.scrollTo(0, 0); });
$("page").addEventListener("change", () => { state.page = parseInt($("page").value) || 1; render(); });
$("selPage").addEventListener("click", () => batchPage(true));
$("clrPage").addEventListener("click", () => batchPage(false));

$("export").addEventListener("click", async () => {
  if (!state.reviewer) { toast("reviewer를 선택하세요"); return; }
  const d = await (await fetch(`/api/selections?reviewer=${encodeURIComponent(state.reviewer)}`)).json();
  const blob = new Blob([JSON.stringify(d, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${state.reviewer}_selections.json`;
  a.click();
});

$("save").addEventListener("click", async () => {
  if (!state.reviewer) { toast("reviewer를 선택하세요"); return; }
  const r = await fetch("/api/sync", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({reviewer: state.reviewer}),
  });
  if (!r.ok) { toast("sync 실패"); return; }
  const d = await r.json();
  // Mark disk state as in-sync with the current selection.
  for (const it of state.items) it.saved = it.selected;
  updateCounts();
  const warn = (d.skipped && d.skipped.length) ? ` ⚠ 원본 svg 없음 ${d.skipped.length}개` : "";
  toast(`SVG 동기화: +${d.added} −${d.removed} (내 저장 ${d.my_saved}개, 폴더 전체 ${d.total_saved}개)${warn}`);
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft")  { e.preventDefault(); state.page--; render(); window.scrollTo(0, 0); }
  if (e.key === "ArrowRight") { e.preventDefault(); state.page++; render(); window.scrollTo(0, 0); }
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
    rows = load_manifest()
    split = split_for_reviewers(rows)
    n_svgs = len(svg_map())
    print(f"Curation selector at http://localhost:{PORT}")
    print(f"  display renders: {RENDERS}  ({len(rows)} pngs)")
    print(f"  source svgs:     {SVGS}  ({n_svgs} svgs)")
    print(f"  output dir:      {OUT_DIR}  (saves .svg)")
    for r in REVIEWERS:
        print(f"    {r}: {len(split[r])} samples")
    print(f"  selections dir: {SEL_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
