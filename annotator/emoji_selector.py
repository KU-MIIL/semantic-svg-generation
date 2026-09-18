"""Grid-based selector webui for filtering the emoji pool down to a kept set.

Run:
    python emoji_selector.py
    # serves at http://localhost:8092

Idea:
  - emoji_pool/{svg,png}      = immutable source (all 668 converted emojis)
  - sample_full/emoji/{svg,png} = the kept selection (what the labeler uses)

UX:
  - 10x10 grid of thumbnails, page navigator, Save button.
  - State per cell:
      SAVED          (currently in sample_full/emoji) -> green ✓
      PENDING_REMOVE (click a SAVED one)              -> red − (will be deleted)
      FREE           (removed earlier, still in pool) -> normal
      PENDING_ADD    (click a FREE one)               -> yellow + (re-add)
  - Click toggles. Save commits: removes deselected from sample_full/emoji,
    re-copies re-selected from emoji_pool. The pool is never modified, so any
    removal is reversible.

All paths are relative to this file (portable: zip & move to any server).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

_PKG_ROOT = Path(__file__).resolve().parent

CATEGORIES = {
    "emoji": {
        "label": "Emoji",
        "src_png": _PKG_ROOT / "emoji_pool" / "png",
        "src_svg": _PKG_ROOT / "emoji_pool" / "svg",
        "dst_png": _PKG_ROOT / "sample_full" / "emoji" / "png",
        "dst_svg": _PKG_ROOT / "sample_full" / "emoji" / "svg",
    },
}

PAGE_SIZE = 100  # 10 x 10 grid
PORT = 8092

_LIST_CACHE: dict[str, list[str]] = {}


def list_stems(cat: str) -> list[str]:
    if cat in _LIST_CACHE:
        return _LIST_CACHE[cat]
    cfg = CATEGORIES[cat]
    pngs = sorted(cfg["src_png"].glob("*.png"),
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    _LIST_CACHE[cat] = [p.stem for p in pngs]
    return _LIST_CACHE[cat]


def saved_stems(cat: str) -> set[str]:
    d: Path = CATEGORIES[cat]["dst_svg"]
    if not d.exists():
        return set()
    return {p.stem for p in d.glob("*.svg")}


app = FastAPI()


@app.get("/api/categories")
def api_categories() -> JSONResponse:
    out = []
    for key, cfg in CATEGORIES.items():
        stems = list_stems(key)
        out.append({
            "key": key,
            "label": cfg["label"],
            "total": len(stems),
            "pages": (len(stems) + PAGE_SIZE - 1) // PAGE_SIZE,
            "saved": len(saved_stems(key)),
        })
    return JSONResponse(out)


@app.get("/api/page/{cat}/{page}")
def api_page(cat: str, page: int) -> JSONResponse:
    if cat not in CATEGORIES:
        raise HTTPException(404, f"unknown category: {cat}")
    stems = list_stems(cat)
    pages_total = (len(stems) + PAGE_SIZE - 1) // PAGE_SIZE
    if page < 1 or page > max(pages_total, 1):
        raise HTTPException(400, f"page out of range (1..{pages_total})")

    start = (page - 1) * PAGE_SIZE
    page_stems = stems[start:start + PAGE_SIZE]
    saved = saved_stems(cat)

    items = [{"stem": s, "state": "saved" if s in saved else "free"}
             for s in page_stems]

    return JSONResponse({
        "cat": cat,
        "page": page,
        "pages": pages_total,
        "start_index": start,
        "total": len(stems),
        "saved_total": len(saved),
        "items": items,
    })


@app.get("/img/{cat}/{stem}.png")
def serve_thumb(cat: str, stem: str) -> FileResponse:
    if cat not in CATEGORIES:
        raise HTTPException(404, f"unknown category: {cat}")
    f = CATEGORIES[cat]["src_png"] / f"{stem}.png"
    if not f.exists():
        raise HTTPException(404, f"missing: {f}")
    return FileResponse(f, media_type="image/png")


class SaveBody(BaseModel):
    cat: str
    add: list[str] = []
    remove: list[str] = []


@app.post("/api/save")
def api_save(body: SaveBody) -> JSONResponse:
    if body.cat not in CATEGORIES:
        raise HTTPException(404, f"unknown category: {body.cat}")
    cfg = CATEGORIES[body.cat]
    cfg["dst_png"].mkdir(parents=True, exist_ok=True)
    cfg["dst_svg"].mkdir(parents=True, exist_ok=True)

    added: list[str] = []
    removed: list[str] = []
    skipped: list[dict] = []

    for s in body.add:
        png = cfg["src_png"] / f"{s}.png"
        svg = cfg["src_svg"] / f"{s}.svg"
        if not svg.exists():
            skipped.append({"stem": s, "reason": "src-svg-missing"})
            continue
        shutil.copy2(svg, cfg["dst_svg"] / f"{s}.svg")
        if png.exists():
            shutil.copy2(png, cfg["dst_png"] / f"{s}.png")
        added.append(s)

    for s in body.remove:
        p_png = cfg["dst_png"] / f"{s}.png"
        p_svg = cfg["dst_svg"] / f"{s}.svg"
        existed = p_png.exists() or p_svg.exists()
        p_png.unlink(missing_ok=True)
        p_svg.unlink(missing_ok=True)
        # also drop a stale labeling sidecar if the emoji is dropped
        (cfg["dst_svg"] / f"{s}.tree.json").unlink(missing_ok=True)
        if existed:
            removed.append(s)

    return JSONResponse({
        "added": added,
        "removed": removed,
        "skipped": skipped,
        "saved_total": len(saved_stems(body.cat)),
    })


HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Emoji Selector</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f5f5f5; }
  header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd; padding: 10px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; z-index: 10; }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  header .stats { color: #666; font-size: 13px; }
  select, button, input[type=number] { font-size: 14px; padding: 6px 10px; border-radius: 4px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
  button.primary { background: #2563eb; color: #fff; border-color: #2563eb; }
  button.primary:disabled { background: #9ca3af; border-color: #9ca3af; cursor: not-allowed; }
  button:hover:not(:disabled) { filter: brightness(0.95); }
  #pending-info { color: #ca8a04; font-weight: 500; }
  #pending-info.has-changes { color: #dc2626; }
  main { padding: 12px; }
  #grid { display: grid; grid-template-columns: repeat(10, 1fr); gap: 6px; max-width: 1200px; margin: 0 auto; }
  .cell { position: relative; background: #fff; border: 2px solid #e5e7eb; border-radius: 6px; padding: 4px; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.1s; }
  .cell img { width: 100%; height: 100%; object-fit: contain; pointer-events: none; }
  .cell .stem { position: absolute; bottom: 2px; left: 2px; right: 2px; font-size: 9px; color: #666; text-align: center; background: rgba(255,255,255,0.85); border-radius: 2px; }
  .cell:hover { border-color: #6b7280; }
  .cell.saved { border-color: #16a34a; background: #ecfdf5; }
  .cell.saved::after { content: "✓"; position: absolute; top: 2px; right: 4px; font-size: 13px; color: #16a34a; font-weight: bold; }
  .cell.pending-add { border-color: #eab308; background: #fef9c3; }
  .cell.pending-add::after { content: "+"; position: absolute; top: 0; right: 4px; font-size: 18px; color: #ca8a04; font-weight: bold; }
  .cell.pending-remove { border-color: #dc2626; background: #fee2e2; }
  .cell.pending-remove::after { content: "−"; position: absolute; top: 0; right: 4px; font-size: 18px; color: #dc2626; font-weight: bold; }
  #toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #1f2937; color: #fff; padding: 10px 18px; border-radius: 6px; opacity: 0; transition: opacity 0.2s; pointer-events: none; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>Emoji Selector</h1>
  <button id="prev">◀ Prev</button>
  <span>Page <input type="number" id="page" min="1" value="1" style="width:56px"> / <span id="pages">?</span></span>
  <button id="next">Next ▶</button>
  <button id="rmPage">이 페이지 전부 제거</button>
  <button id="keepPage">이 페이지 제거취소</button>
  <span class="stats" id="stats"></span>
  <span class="stats" id="pending-info"></span>
  <button id="save" class="primary" disabled>Save</button>
</header>
<main><div id="grid"></div></main>
<div id="toast"></div>
<script>
const $ = (id) => document.getElementById(id);
const grid = $("grid");
let state = { cat: "emoji", page: 1, pages: 1, items: [],
              pendingAdd: new Set(), pendingRemove: new Set() };

function pendingSummary() {
  return { adds: state.pendingAdd.size, rems: state.pendingRemove.size };
}
function refreshPendingUI() {
  const { adds, rems } = pendingSummary();
  const p = $("pending-info");
  if (adds === 0 && rems === 0) {
    p.textContent = "no pending changes"; p.classList.remove("has-changes");
    $("save").disabled = true;
  } else {
    p.textContent = `pending: +${adds} / −${rems}`;
    p.classList.add("has-changes"); $("save").disabled = false;
  }
}
function cellState(stem, base) {
  if (base === "saved") return state.pendingRemove.has(stem) ? "pending-remove" : "saved";
  return state.pendingAdd.has(stem) ? "pending-add" : "free";
}
function render() {
  grid.innerHTML = "";
  for (const it of state.items) {
    const cell = document.createElement("div");
    cell.className = "cell " + cellState(it.stem, it.state);
    cell.dataset.stem = it.stem; cell.dataset.baseState = it.state;
    cell.innerHTML = `<img src="/img/${state.cat}/${it.stem}.png" loading="lazy">`
      + `<span class="stem">${it.stem}</span>`;
    cell.addEventListener("click", onCellClick);
    grid.appendChild(cell);
  }
  refreshPendingUI();
}
function toggle(stem, base) {
  if (base === "saved") {
    state.pendingRemove.has(stem) ? state.pendingRemove.delete(stem)
                                  : state.pendingRemove.add(stem);
  } else {
    state.pendingAdd.has(stem) ? state.pendingAdd.delete(stem)
                               : state.pendingAdd.add(stem);
  }
}
function onCellClick(e) {
  const cell = e.currentTarget;
  toggle(cell.dataset.stem, cell.dataset.baseState);
  cell.className = "cell " + cellState(cell.dataset.stem, cell.dataset.baseState);
  refreshPendingUI();
}
async function loadPage() {
  const r = await fetch(`/api/page/${state.cat}/${state.page}`);
  if (!r.ok) { toast("page load failed"); return; }
  const d = await r.json();
  state.items = d.items; state.pages = d.pages;
  $("pages").textContent = d.pages; $("page").max = d.pages; $("page").value = d.page;
  $("stats").textContent =
    `${d.total} total · ${d.saved_total} kept · idx ${d.start_index}..${d.start_index + d.items.length - 1}`;
  render();
}
function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2500);
}
$("rmPage").addEventListener("click", () => {
  for (const it of state.items) if (it.state === "saved") state.pendingRemove.add(it.stem);
  render();
});
$("keepPage").addEventListener("click", () => {
  for (const it of state.items) state.pendingRemove.delete(it.stem);
  render();
});
async function save() {
  const { adds, rems } = pendingSummary();
  if (adds + rems === 0) return;
  const r = await fetch("/api/save", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ cat: state.cat,
      add: [...state.pendingAdd], remove: [...state.pendingRemove] }),
  });
  if (!r.ok) { toast("save failed"); return; }
  const res = await r.json();
  state.pendingAdd = new Set(); state.pendingRemove = new Set();
  toast(`saved (+${res.added.length} −${res.removed.length}) · kept ${res.saved_total}`);
  await loadPage();
}
$("prev").addEventListener("click", () => { if (state.page > 1) { state.page--; loadPage(); } });
$("next").addEventListener("click", () => { if (state.page < state.pages) { state.page++; loadPage(); } });
$("page").addEventListener("change", () => {
  const n = parseInt($("page").value);
  if (n >= 1 && n <= state.pages) { state.page = n; loadPage(); }
});
$("save").addEventListener("click", save);
window.addEventListener("beforeunload", (e) => {
  const { adds, rems } = pendingSummary();
  if (adds + rems > 0) { e.preventDefault(); e.returnValue = ""; }
});
loadPage();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


if __name__ == "__main__":
    print(f"Emoji selector at http://localhost:{PORT}")
    c = CATEGORIES["emoji"]
    print(f"  pool(src)={c['src_svg'].parent}  ->  kept(dst)={c['dst_svg'].parent}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
