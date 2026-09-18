"""Hierarchical SVG semantic labeler.

Run:
    python _scripts/semantic_labeler.py
    # serves all *.svg under SVG_DIRS at http://localhost:8088

UX:
  - Pick file from dropdown.
  - At each tree level, check paths to group → "Select" (subdividable group)
    or "Finish" (leaf group).
  - "Enter" into a Select-group to subdivide it. "Up" to go back up.
  - "Save" writes nested <g data-semantic="..."> to the SVG and a sidecar
    <name>.tree.json with the full tree.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import io as _io

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

# Tree visualization helpers live in the sibling visualize_tree.py file —
# import lazily inside the handler so a missing PIL/cairo doesn't break the
# main labeler at import time.


# Portable: resolve data dirs relative to this file so the whole folder can be
# zipped and moved to another server without editing absolute paths.
_PKG_ROOT = Path(__file__).resolve().parent
SVG_DIRS = [
    _PKG_ROOT / "sample_full" / "icon" / "svg",
    _PKG_ROOT / "sample_full" / "illustration" / "svg",
    _PKG_ROOT / "sample_full" / "emoji" / "svg",
]
SHAPE_TAGS = ("path", "rect", "circle", "ellipse", "polygon", "polyline", "line")


# -----------------------------------------------------------------------------
# File scanning
# -----------------------------------------------------------------------------
def list_files() -> list[Path]:
    out: list[Path] = []
    for d in SVG_DIRS:
        files = list(d.glob("*.svg"))
        # Skip helper outputs like "183_grouped.svg" — only digit stems.
        files = [p for p in files if p.stem.isdigit()]
        files.sort(key=lambda p: int(p.stem))
        out.extend(files)
    return out


def tree_path(svg: Path) -> Path:
    return svg.with_suffix(".tree.json")


# -----------------------------------------------------------------------------
# Per-user assignment (workload split across labelers)
# -----------------------------------------------------------------------------
_ASSIGN_PATH = _PKG_ROOT / "sample_full" / "_assignments.json"


def load_assignments() -> dict:
    """Return {"<cat>/<stem>": "<user>"} or {} if no assignment file."""
    try:
        raw = json.loads(_ASSIGN_PATH.read_text(encoding="utf-8"))
        return raw.get("assign", {}) if isinstance(raw, dict) else {}
    except Exception:
        return {}


def list_users() -> list[str]:
    try:
        raw = json.loads(_ASSIGN_PATH.read_text(encoding="utf-8"))
        return list(raw.get("users", [])) if isinstance(raw, dict) else []
    except Exception:
        return []


def _file_key(p: Path) -> str:
    """Package-relative posix path, e.g. 'sample_full/icon/svg/239.svg'.
    Unambiguous across sample_full / sample_full_shp (no stem collisions)."""
    try:
        return p.resolve().relative_to(_PKG_ROOT).as_posix()
    except Exception:
        return p.name


def user_files(user: str) -> list[Path]:
    """Files visible/editable by `user`. Empty if no/unknown user. If no
    assignment file exists at all, fall back to the full list (single-user)."""
    files = list_files()
    assign = load_assignments()
    if not assign:
        return files
    user = (user or "").strip()
    if not user:
        return []
    return [p for p in files if assign.get(_file_key(p)) == user]


# -----------------------------------------------------------------------------
# SVG shape utilities
# -----------------------------------------------------------------------------
SHAPE_PATTERN = re.compile(
    r"<(" + "|".join(SHAPE_TAGS) + r")\b([^>]*?)(/>|>([\s\S]*?)</\1>)"
)

# Containers whose inner shapes define clips/masks/templates rather than
# drawable content. Shapes nested in these must be excluded from the path
# index space — otherwise grouping such a shape into a separate node empties
# the clipPath/mask and the element referencing it renders completely blank
# (the icon/239.svg "syringe" bug). They stay in the SVG text untouched.
_NONRENDER_RE = re.compile(
    r"<(defs|clipPath|mask|symbol|pattern|marker)\b[^>]*>[\s\S]*?</\1>",
    re.IGNORECASE,
)


def _nonrender_spans(svg_text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _NONRENDER_RE.finditer(svg_text)]


def _iter_shapes(svg_text: str):
    """Yield shape regex matches that are actual drawable content, skipping
    any nested inside <defs>/<clipPath>/<mask>/<symbol>/<pattern>/<marker>."""
    skip = _nonrender_spans(svg_text)
    for m in SHAPE_PATTERN.finditer(svg_text):
        if any(s <= m.start() < e for s, e in skip):
            continue
        yield m


def parse_shapes(svg_text: str) -> list[dict]:
    out = []
    for i, m in enumerate(_iter_shapes(svg_text)):
        tag = m.group(1)
        attrs = m.group(2).strip()
        out.append({"index": i, "tag": tag, "attrs_preview": attrs[:120]})
    return out


def shape_spans(svg_text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, raw) for every drawable shape element in order.
    Shapes inside <defs>/<clipPath>/<mask> etc. are skipped so the index
    space matches exactly what is painted."""
    return [(m.start(), m.end(), m.group(0)) for m in _iter_shapes(svg_text)]


def svg_with_only(svg_text: str, keep_indices: set[int]) -> str:
    """Return SVG with only specified shape indices retained."""
    pieces = []
    last = 0
    for i, (s, e, raw) in enumerate(shape_spans(svg_text)):
        pieces.append(svg_text[last:s])
        if i in keep_indices:
            pieces.append(raw)
        last = e
    pieces.append(svg_text[last:])
    return "".join(pieces)


_OPEN_TAG_RE = re.compile(r'^(<\w+)([^>]*?)(/?>)')


def _inject_attrs(raw: str, extra: str) -> str:
    """Inject `extra` (e.g. ' opacity="0.12"') into the opening tag of `raw`,
    correctly handling both self-closing (`<path ... />`) and paired
    (`<path ...>...</path>`) forms."""
    m = _OPEN_TAG_RE.match(raw)
    if not m:
        return raw
    name, attrs, closer = m.group(1), m.group(2), m.group(3)
    rest = raw[m.end():]
    return name + attrs + extra + closer + rest


def _strip_attr(raw: str, attr_name: str) -> str:
    """Remove an attribute from the OPENING tag only (don't touch content)."""
    m = _OPEN_TAG_RE.match(raw)
    if not m:
        return raw
    name, attrs, closer = m.group(1), m.group(2), m.group(3)
    rest = raw[m.end():]
    attrs = re.sub(rf'\s*{attr_name}\s*=\s*"[^"]*"', '', attrs)
    return name + attrs + closer + rest


def svg_highlight(svg_text: str, hi_indices: set[int]) -> str:
    """Return SVG with all shapes dimmed to 0.12 opacity, then highlighted
    shapes RE-EMITTED at the end (so they paint on top of dimmed neighbors)
    with a bold blue stroke for guaranteed visibility — even when the
    original path is buried under later painters."""
    spans = shape_spans(svg_text)
    if not spans:
        return svg_text

    out = []
    highlighted_extras: list[str] = []
    last = 0

    for i, (s, e, raw) in enumerate(spans):
        out.append(svg_text[last:s])
        # Dim every shape (replace any opacity, then inject 0.12)
        dim_raw = _strip_attr(raw, "opacity")
        dim_raw = _inject_attrs(dim_raw, ' opacity="0.12"')
        out.append(dim_raw)
        if i in hi_indices:
            hl_raw = _strip_attr(raw, "opacity")
            hl_raw = _strip_attr(hl_raw, "stroke")
            hl_raw = _strip_attr(hl_raw, "stroke-width")
            hl_raw = _inject_attrs(hl_raw, ' opacity="1" stroke="#3b82f6" stroke-width="2"')
            highlighted_extras.append(hl_raw)
        last = e

    remaining = svg_text[last:]
    close_pos = remaining.find("</svg>")
    if highlighted_extras and close_pos >= 0:
        out.append(remaining[:close_pos])
        out.append("\n".join(highlighted_extras))
        out.append("\n")
        out.append(remaining[close_pos:])
    else:
        out.append(remaining)
        if highlighted_extras:
            out.append("\n".join(highlighted_extras))
    return "".join(out)


# -----------------------------------------------------------------------------
# Tree model
# -----------------------------------------------------------------------------
def empty_tree(n_paths: int) -> dict:
    """Root has no children initially; all paths are pending under the root."""
    return {
        "label": "",
        "paths": list(range(n_paths)),
        "children": [],
    }


def is_leaf(node: dict) -> bool:
    """Leaf is determined by KEY PRESENCE, not value:
      - leaf:     no "children" key at all     → terminal, paths are final
      - non-leaf: has "children" key (even [])  → still subdivisible / pending

    An empty `children: []` means "Select group, not yet subdivided" — those
    paths are NOT considered assigned, so save will refuse until the user
    fills it in or converts the group to a Leaf via `→ Leaf`.

    Backwards-compat: legacy `leaf: true` flag still wins if present."""
    if node.get("leaf") is True:
        return True
    if node.get("leaf") is False:
        return False
    return "children" not in node


def collect_leaf_paths(node: dict) -> list[int]:
    if is_leaf(node):
        return list(node.get("paths") or [])
    out = []
    for c in node.get("children") or []:
        out.extend(collect_leaf_paths(c))
    return out


def assigned_in_node(node: dict) -> set[int]:
    """Path indices already covered by some child of this node (any depth)."""
    s: set[int] = set()
    for c in node.get("children") or []:
        s.update(collect_leaf_paths(c))
        if not is_leaf(c):
            for p in c.get("paths") or []:
                s.add(p)
    return s


def pending_paths_in(node: dict) -> list[int]:
    scope = list(node.get("paths") or [])
    assigned = assigned_in_node(node)
    return [p for p in scope if p not in assigned]


def find_node_by_path(tree: dict, path: list[int]) -> dict:
    """Navigate by indices into children list."""
    cur = tree
    for idx in path:
        cur = cur["children"][idx]
    return cur


def fix_node_paths(tree: dict) -> None:
    """Ensure every non-leaf node's `paths` is the union of its children paths.
    Used after structural edits."""
    if tree.get("leaf"):
        return
    for c in tree.get("children") or []:
        fix_node_paths(c)
    # Don't overwrite root's paths (which is the full universe). Only fix
    # non-leaf children: their `paths` reflects scope = union of leaves.
    # We leave it as-is to keep scope stable.


def merge_none_siblings(node: dict) -> None:
    """Recursively merge all leaf children labeled "none" under each node into
    a single "none" leaf. "none" carries no semantic meaning, so splitting it
    across multiple sibling nodes is just noise — collapse them."""
    children = node.get("children")
    if not children:
        return
    none_paths: list[int] = []
    kept: list[dict] = []
    for c in children:
        if is_leaf(c) and c.get("label") == "none":
            none_paths.extend(c.get("paths") or [])
        else:
            kept.append(c)
            merge_none_siblings(c)
    if none_paths:
        merged = sorted(set(none_paths))
        kept.append({"label": "none", "paths": merged})
    node["children"] = kept


def validate(tree: dict, n_paths: int) -> tuple[bool, list[str]]:
    leaves = collect_leaf_paths(tree)
    errors = []
    seen = set()
    for p in leaves:
        if p in seen:
            errors.append(f"path {p} assigned to multiple leaves")
        seen.add(p)
    missing = set(range(n_paths)) - seen
    if missing:
        errors.append(f"paths missing from any leaf: {sorted(missing)}")
    return (not errors), errors


# -----------------------------------------------------------------------------
# Nested SVG output
# -----------------------------------------------------------------------------
def label_to_id(label: str, used: set[str]) -> str:
    """Convert a free-form label to an SVG-safe id (just whitespace → '_').
    No collision suffix — duplicates within a parent are intentionally allowed
    so the user can merge non-contiguous groups later by visual inspection."""
    base = re.sub(r"\s+", "_", label.strip())
    if not base:
        base = "g"
    used.add(base)
    return base


def emit_nested_svg(svg_text: str, tree: dict) -> str:
    """Wrap shape elements with <g id="..."> matching the tree, with proper
    indentation so the nested structure is visually obvious.

    For non-contiguous groups (split for painter's order), each contiguous
    run becomes its own <g> with a unique id (e.g., cap, cap_2, cap_3)."""
    n_shapes = len(shape_spans(svg_text))
    ancestor_labels: list[list[str]] = [[] for _ in range(n_shapes)]

    def visit(node: dict, ancestors: list[str]) -> None:
        cur_ancestors = list(ancestors)
        if node.get("label"):
            cur_ancestors.append(node["label"])
        if node.get("leaf"):
            for p in node.get("paths") or []:
                if 0 <= p < n_shapes:
                    ancestor_labels[p] = list(cur_ancestors)
        else:
            for c in node.get("children") or []:
                visit(c, cur_ancestors)

    visit(tree, [])

    spans = shape_spans(svg_text)
    if not spans:
        return svg_text

    pre = svg_text[:spans[0][0]]
    post = svg_text[spans[-1][1]:]
    pre = pre.rstrip(" \t")
    if not pre.endswith("\n"):
        pre = pre + "\n"
    post = post.lstrip()

    out = [pre]
    # open_stack holds (label, id) for each currently-open group.
    open_stack: list[tuple[str, str]] = []
    # used_stack[k] = set of ids already used among the children of the k-th
    # currently-open group (used_stack[0] = ids among <svg>'s direct children).
    # Length is always len(open_stack) + 1.
    used_stack: list[set[str]] = [set()]
    INDENT = "  "

    def cur_indent() -> str:
        return INDENT * (len(open_stack) + 1)  # +1 because inside <svg>

    def common_prefix_len_labels(a: list[tuple[str, str]], b: list[str]) -> int:
        n = 0
        while n < len(a) and n < len(b) and a[n][0] == b[n]:
            n += 1
        return n

    for i, (_s, _e, raw) in enumerate(spans):
        target = ancestor_labels[i]
        cp = common_prefix_len_labels(open_stack, target)
        # Close groups not in target
        while len(open_stack) > cp:
            open_stack.pop()
            used_stack.pop()
            out.append(INDENT * (len(open_stack) + 1))
            out.append("</g>\n")
        # Open groups in target not yet open
        while len(open_stack) < len(target):
            lbl = target[len(open_stack)]
            # Allocate id within the CURRENT parent's used set (last in stack).
            new_id = label_to_id(lbl, used_stack[-1])
            out.append(INDENT * (len(open_stack) + 1))
            out.append(f'<g id="{new_id}">\n')
            open_stack.append((lbl, new_id))
            used_stack.append(set())
        # Emit shape with current indent
        out.append(cur_indent())
        out.append(raw)
        out.append("\n")

    while open_stack:
        open_stack.pop()
        used_stack.pop()
        out.append(INDENT * (len(open_stack) + 1))
        out.append("</g>\n")

    out.append(post)
    return "".join(out)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def load_tree(svg: Path) -> dict:
    """Load tree.json or build empty tree based on shape count.
    Accepts both new schema (top-level {"root": {...}}) and legacy schema
    (top-level node directly with "label", "leaf", "children")."""
    n = len(parse_shapes(svg.read_text()))
    p = tree_path(svg)
    if p.exists():
        try:
            raw = json.loads(p.read_text())
            # Fail-marker schema: {"fail": true, "paths": [...]}.
            # Treat as a special root with no children — UI surfaces the
            # failure state and disables labeling controls.
            if isinstance(raw, dict) and raw.get("fail") is True:
                return {
                    "label": "",
                    "paths": list(raw.get("paths") or list(range(n))),
                    "children": [],
                    "fail": True,
                }
            # New schema: {"paths": [...], "root": {"children": [...]}}
            if isinstance(raw, dict) and "root" in raw and isinstance(raw["root"], dict):
                root = dict(raw["root"])
                root.setdefault("label", "")
                # New schema keeps the full path universe at the top level;
                # mirror it onto the in-memory tree for compat with existing
                # validation/rendering code.
                if "paths" not in root and "paths" in raw:
                    root["paths"] = list(raw["paths"])
                merge_none_siblings(root)
                return root
            # Legacy schema: top-level is the root node
            return raw
        except Exception:
            pass
    return empty_tree(n)


def _dump_node(node: dict, indent: int = 0, with_label: bool = True) -> str:
    """Pretty-print a tree node. Schema:
      - Leaf:     {"label": "...", "paths": [...]}        — one line, no children
      - Non-leaf: {"label": "...", "paths": [...], "children": [...]}  — multi-line

    If `with_label` is False, the node itself is rendered without a label key
    (used for the root, which is referenced only by the wrapping "root" key).
    """
    pad = "  " * indent
    label = node.get("label", "")
    paths = node.get("paths") or []
    paths_inline = "[" + ", ".join(str(p) for p in paths) + "]"

    if is_leaf(node):
        # One-line leaf
        if with_label:
            return pad + (
                "{"
                f'"label": {json.dumps(label, ensure_ascii=False)}, '
                f'"paths": {paths_inline}'
                "}"
            )
        return pad + "{" + f'"paths": {paths_inline}' + "}"

    children = node.get("children") or []
    lines = [pad + "{"]
    if with_label:
        lines.append(pad + f'  "label": {json.dumps(label, ensure_ascii=False)},')
    lines.append(pad + f'  "paths": {paths_inline},')
    if children:
        lines.append(pad + '  "children": [')
        for i, c in enumerate(children):
            block = _dump_node(c, indent + 2, with_label=True)
            if i < len(children) - 1:
                block += ","
            lines.append(block)
        lines.append(pad + "  ]")
    else:
        lines.append(pad + '  "children": []')
    lines.append(pad + "}")
    return "\n".join(lines)


def _dump_tree(tree: dict) -> str:
    """Top-level wrapper. Emits a top-level `paths` summary (full universe)
    followed by `root` whose body holds children only — no duplicate paths
    on the root node body."""
    paths = list(tree.get("paths") or [])
    paths_inline = "[" + ", ".join(str(p) for p in paths) + "]"
    children = tree.get("children") or []
    lines = ["{", f'  "paths": {paths_inline},', '  "root": {']
    if children:
        lines.append('    "children": [')
        for i, c in enumerate(children):
            block = _dump_node(c, indent=3, with_label=True)
            if i < len(children) - 1:
                block += ","
            lines.append(block)
        lines.append("    ]")
    else:
        lines.append('    "children": []')
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def save_tree_and_svg(svg: Path, tree: dict) -> None:
    """Persist the hierarchy as <file>.tree.json. The SVG itself stays FLAT —
    no <g> wrappers are added. Hierarchy is captured exclusively in tree.json
    so non-contiguous groups (e.g., paths [9, 14, 15] all labeled "cap") can
    be represented faithfully without breaking SVG painter's order."""
    merge_none_siblings(tree)
    tree_path(svg).write_text(_dump_tree(tree) + "\n")
    text = svg.read_text()
    text_no_g = _strip_semantic_g_wrappers(text)
    # Also normalize whitespace: drop empty lines, strip indent that lingered
    # from prior nested saves.
    lines = [ln.strip() for ln in text_no_g.splitlines()]
    lines = [ln for ln in lines if ln != ""]
    candidate = "\n".join(lines) + "\n"
    # Safety net: refuse to write if the result is no longer well-formed XML
    # (downstream tools like cairosvg use strict parsers and will fail).
    # If our strip produced bad XML, leave the on-disk SVG untouched.
    try:
        ET.fromstring(candidate)
    except ET.ParseError as e:
        print(f"[save_tree_and_svg] WARN: strip produced malformed XML for "
              f"{svg.name} ({e}); leaving SVG untouched.")
        return
    svg.write_text(candidate)


def _strip_semantic_g_wrappers(text: str) -> str:
    """Remove <g id|data-semantic="..."> wrappers AND their matching </g> only.
    Stack-based pairing — never strips a </g> belonging to a meaningful group
    (e.g. <g stroke="..."> used for attribute inheritance)."""
    events = []
    for m in re.finditer(r"<g\b[^>]*>|</g\s*>", text):
        s = m.group(0)
        is_close = s.startswith("</")
        is_semantic = (not is_close) and re.search(r'\b(?:id|data-semantic)\s*=', s) is not None
        events.append((m.start(), m.end(), is_close, is_semantic))
    stack = []
    to_strip = set()
    for i, (_s, _e, is_close, is_sem) in enumerate(events):
        if not is_close:
            stack.append((i, is_sem))
        else:
            if not stack:
                continue
            opener_i, opener_is_sem = stack.pop()
            if opener_is_sem:
                to_strip.add(opener_i)
                to_strip.add(i)
    parts, last = [], 0
    for i, (s, e, _ic, _is) in enumerate(events):
        parts.append(text[last:s])
        if i not in to_strip:
            parts.append(text[s:e])
        last = e
    parts.append(text[last:])
    return "".join(parts)


# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI()


@app.get("/api/users")
def api_users() -> JSONResponse:
    return JSONResponse({"users": list_users()})


@app.get("/api/files")
def api_files(user: str = "") -> JSONResponse:
    files = user_files(user)
    out = []
    for p in files:
        n_paths = len(parse_shapes(p.read_text()))
        failed = False
        try:
            t = load_tree(p)
            failed = bool(t.get("fail"))
            ok, _ = validate(t, n_paths)
            assigned = len(set(collect_leaf_paths(t)))
        except Exception:
            ok, assigned = False, 0
        out.append({
            "path": str(p),
            "name": p.name,
            "category": p.parent.parent.name,
            "n_paths": n_paths,
            "n_assigned": assigned,
            "complete": ok and not failed,
            "failed": failed,
            "saved": tree_path(p).exists(),
        })
    return JSONResponse({"files": out})


@app.get("/api/file")
def api_file(idx: int, user: str = "") -> JSONResponse:
    files = user_files(user)
    if idx < 0 or idx >= len(files):
        raise HTTPException(404, "file index out of range")
    svg = files[idx]
    text = svg.read_text()
    shapes = parse_shapes(text)
    return JSONResponse({
        "idx": idx,
        "n_files": len(files),
        "svg_path": str(svg),
        "svg_name": svg.name,
        "category": svg.parent.parent.name,
        "svg_full": text,
        "n_shapes": len(shapes),
        "shapes": [
            {"index": s["index"], "tag": s["tag"], "attrs_preview": s["attrs_preview"]}
            for s in shapes
        ],
        "tree": load_tree(svg),
    })


@app.post("/api/preview")
async def api_preview(body: dict) -> JSONResponse:
    """Return three SVG render strings: only-checked, highlight-checked, only-pending."""
    files = user_files(body.get("user", ""))
    idx = int(body.get("idx", 0))
    if idx < 0 or idx >= len(files):
        raise HTTPException(404, "file index out of range")
    svg = files[idx]
    text = svg.read_text()
    checked = set(int(i) for i in body.get("checked", []))
    return JSONResponse({
        "only_checked": svg_with_only(text, checked),
        "highlight_checked": svg_highlight(text, checked),
    })


@app.post("/api/save")
async def api_save(body: dict) -> JSONResponse:
    files = user_files(body.get("user", ""))
    idx = int(body.get("idx", 0))
    if idx < 0 or idx >= len(files):
        raise HTTPException(404, "file index out of range")
    svg = files[idx]
    tree = body.get("tree", {})
    n_paths = len(parse_shapes(svg.read_text()))
    ok, errors = validate(tree, n_paths)
    if not ok:
        # Refuse to save incomplete trees so disk state stays clean.
        return JSONResponse({"ok": False, "valid": False, "errors": errors}, status_code=422)
    save_tree_and_svg(svg, tree)
    return JSONResponse({"ok": True, "valid": True, "saved_to": str(svg)})


@app.post("/api/treevis")
async def api_treevis(body: dict) -> Response:
    """Render the supplied (in-progress) tree as a hierarchical PNG so the
    user can see the current decomposition without leaving the labeler."""
    files = user_files(body.get("user", ""))
    idx = int(body.get("idx", 0))
    if idx < 0 or idx >= len(files):
        raise HTTPException(404, "file index out of range")
    svg = files[idx]
    text = svg.read_text()
    tree = body.get("tree") or load_tree(svg)

    # Reuse visualize_tree.py's drawing primitives.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import visualize_tree as vt
    from PIL import Image, ImageDraw

    if isinstance(tree, dict) and tree.get("fail"):
        n_shapes = len(parse_shapes(text))
        full = vt.rasterize(text, list(range(n_shapes)), 320)
        W, H = 480, 420
        canvas = Image.new("RGB", (W, H), vt.BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((vt.PAD, 10), f"{svg.parent.parent.name}/{svg.name}",
                  fill=(40, 40, 40), font=vt.FONT_TITLE)
        canvas.paste(full, ((W - 320) // 2, 50), full if full.mode == "RGBA" else None)
        draw.rectangle([0, 380, W, 420], fill=(239, 68, 68))
        msg = "✗ MARKED FAILED — semantically tangled"
        bbox = draw.textbbox((0, 0), msg, font=vt.FONT_TITLE)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((W - tw) // 2, 380 + (40 - th) // 2 - 2), msg,
                  fill=(255, 255, 255), font=vt.FONT_TITLE)
    else:
        root = vt.build_layout(tree)
        vt.compute_widths(root)
        vt.assign_positions(root, vt.PAD, 0)
        nodes = vt.all_nodes(root)
        W = root.w_subtree + 2 * vt.PAD
        H = (vt.max_depth(root) + 1) * (vt.NODE_H + vt.V_GAP) - vt.V_GAP + 2 * vt.PAD + 30
        canvas = Image.new("RGB", (W, H + 30), vt.BG)
        draw = ImageDraw.Draw(canvas)
        title = f"{svg.parent.parent.name}/{svg.name}  —  {len(root.paths)} paths"
        draw.text((vt.PAD, 6), title, fill=(40, 40, 40), font=vt.FONT_TITLE)
        for n in nodes:
            n.y += 30
        vt.draw_edges(draw, root)
        for n in nodes:
            vt.draw_node(canvas, draw, n, text)

    buf = _io.BytesIO()
    canvas.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/fail")
async def api_fail(body: dict) -> JSONResponse:
    """Mark a file as 'fail' — its semantics are too tangled to label cleanly.
    Writes {"fail": true, "paths": [...]} to tree.json without invariant
    checks. The SVG is left flat and untouched. Pass {"unmark": true} to
    remove the fail marker (deletes the sidecar)."""
    files = user_files(body.get("user", ""))
    idx = int(body.get("idx", 0))
    if idx < 0 or idx >= len(files):
        raise HTTPException(404, "file index out of range")
    svg = files[idx]
    tp = tree_path(svg)
    if body.get("unmark"):
        if tp.exists():
            tp.unlink()
        return JSONResponse({"ok": True, "unmarked": True})
    n_paths = len(parse_shapes(svg.read_text()))
    paths_inline = "[" + ", ".join(str(p) for p in range(n_paths)) + "]"
    tp.write_text(
        "{\n"
        f'  "fail": true,\n'
        f'  "paths": {paths_inline}\n'
        "}\n"
    )
    return JSONResponse({"ok": True, "failed": True, "saved_to": str(tp)})


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Hierarchical SVG Labeler</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; margin: 0; background: #f6f7fa; color: #222; }
  header { padding: 10px 16px; background: #222; color: #fff; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 14px; font-weight: 600; margin: 0; }
  header select { padding: 4px 8px; font-size: 12px; min-width: 320px; }
  .file-progress { font-size: 11px; opacity: .8; }
  .main { display: grid; grid-template-columns: 320px 320px 320px 1fr; gap: 10px; padding: 10px; }
  .panel { background: #fff; border: 1px solid #e1e4ea; border-radius: 8px; padding: 10px; }
  .panel h3 { margin: 0 0 8px; font-size: 11px; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
  .svg-box { background: #fafafa; border: 1px solid #eee; border-radius: 6px; padding: 8px; display: flex; justify-content: center; align-items: center; min-height: 320px; }
  .svg-box svg { max-width: 100%; max-height: 360px; }
  button { padding: 5px 10px; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; font-size: 12px; }
  button:hover { background: #f0f0f0; }
  button.primary { background: #3b82f6; color: #fff; border-color: #3b82f6; }
  button.primary:hover { background: #2563eb; }
  button.success { background: #10b981; color: #fff; border-color: #10b981; }
  button.danger { background: #ef4444; color: #fff; border-color: #ef4444; }
  button:disabled { opacity: .4; cursor: not-allowed; }
  input[type="text"] { width: 100%; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
  .breadcrumb { font-size: 13px; padding: 6px 8px; background: #eef; border-radius: 4px; margin-bottom: 8px; font-family: monospace; }
  .breadcrumb a { color: #3730a3; text-decoration: underline; cursor: pointer; }
  .pending, .children { border: 1px solid #eee; border-radius: 4px; padding: 6px; margin-bottom: 8px; max-height: 320px; overflow-y: auto; background: #fafafa; }
  .pending h4, .children h4 { margin: 0 0 6px; font-size: 11px; color: #555; }
  .row { display: flex; align-items: center; gap: 8px; padding: 4px 4px; border-radius: 3px; font-size: 12px; font-family: monospace; }
  .row:hover { background: #e8f1ff; }
  .row input[type="checkbox"] { margin: 0; }
  .row[draggable="true"] { cursor: grab; }
  .row[draggable="true"]:active { cursor: grabbing; }
  .child-row { display: flex; align-items: center; gap: 8px; padding: 5px 6px; font-size: 12px; border-radius: 4px; }
  .child-row:hover { background: #f0f4ff; }
  .child-row.drop-target.drag-over { background: #dcfce7; outline: 2px dashed #16a34a; outline-offset: -2px; }
  .chip { font-size: 10px; padding: 2px 6px; border-radius: 10px; font-family: monospace; }
  .chip.select { background: #ddd6fe; color: #5b21b6; }
  .chip.leaf { background: #d1fae5; color: #065f46; }
  .controls { display: flex; gap: 6px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
  .help { font-size: 11px; color: #666; margin-top: 6px; }
  .warn { background: #fef3c7; border: 1px solid #fbbf24; padding: 6px 8px; border-radius: 4px; font-size: 11px; color: #78350f; margin-top: 6px; }
  .status { font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Hierarchical Labeler</h1>
  <input id="userInput" list="userList" placeholder="이름 입력" autocomplete="off"
    onkeydown="if(event.key==='Enter')setUser()"
    style="padding:4px 8px; border-radius:4px; border:1px solid #888; font-size:13px; width:120px;">
  <datalist id="userList"></datalist>
  <button onclick="setUser()">확인</button>
  <span id="userBadge" style="font-weight:700; color:#fbbf24;"></span>
  <select id="fileSelect" onchange="loadIdx(this.value)"></select>
  <button onclick="prevFile()">‹ Prev file</button>
  <button onclick="nextFile()">Next file ›</button>
  <span class="file-progress" id="fileProgress"></span>
  <span class="file-progress" id="savedCount" title="Files you have saved (tree.json sidecar exists)"></span>
  <span style="flex:1"></span>
  <span class="status" id="status"></span>
  <button onclick="showTreeVis()" title="Render the current tree as a hierarchical image">🌳 Tree</button>
  <button class="danger" onclick="markFail()" id="failBtn" title="Mark this file as too tangled to label cleanly">Fail</button>
  <button class="primary" onclick="save()">Save</button>
</header>
<div id="treevisOverlay" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:200; align-items:center; justify-content:center; padding:20px;" onclick="hideTreeVis(event)">
  <div style="background:#fff; max-width:95vw; max-height:95vh; overflow:auto; border-radius:8px; padding:8px; position:relative;" onclick="event.stopPropagation()">
    <button onclick="hideTreeVis()" style="position:absolute; top:8px; right:8px; background:#222; color:#fff; border:none; border-radius:4px; padding:4px 12px; cursor:pointer; z-index:1;">✕ Close</button>
    <div id="treevisContent" style="min-width:400px; min-height:200px; display:flex; align-items:center; justify-content:center;">Loading…</div>
  </div>
</div>
<div class="main">
  <div class="panel"><h3>Current scope (this node's paths)</h3><div class="svg-box" id="fullBox"></div></div>
  <div class="panel"><h3>Checked preview</h3><div class="svg-box" id="previewBox"></div>
    <div style="margin-top:8px;"><button onclick="checkAll()">Check all pending</button> <button onclick="uncheckAll()">Uncheck</button></div>
  </div>
  <div class="panel"><h3>Hover (highlighted in context)</h3>
    <div class="svg-box" id="hoverBox"></div>
  </div>
  <div class="panel">
    <div class="breadcrumb" id="breadcrumb"></div>
    <div class="pending" id="pendingBox">
      <h4>Pending paths in this level</h4>
      <div id="pendingList"></div>
    </div>
    <div class="children" id="childrenBox">
      <h4>Children created at this level</h4>
      <div id="childrenList"></div>
    </div>
    <div>
      <input type="text" id="labelInput" placeholder="label (e.g. face, body, hair)" autocomplete="off"/>
      <div class="controls">
        <button class="primary" onclick="doSelect()" id="selectBtn">Select (group, divide later)</button>
        <button class="success" onclick="doFinish()" id="finishBtn">Finish (leaf)</button>
        <span style="flex:1"></span>
        <button onclick="goUp()" id="upBtn">↑ Up</button>
      </div>
      <div class="warn" id="warnBox" style="display:none;"></div>
      <div class="help">Check paths in pending → enter label → Select / Finish. Hover a row to highlight in preview.</div>
    </div>
  </div>
</div>

<script>
let files = [];
let curFile = 0;
let curUser = localStorage.getItem('labeler_user') || '';
let state = null;       // {svg_full, n_shapes, shapes, tree, idx, ...}
let cursor = [];        // path of indices into tree.children to current node
let checked = new Set();
let hover = null;       // path index hovered for highlight

async function setUser() {
  const v = (document.getElementById('userInput').value || '').trim();
  if (!v) { alert('이름을 입력하세요.'); return; }
  curUser = v;
  localStorage.setItem('labeler_user', v);
  document.getElementById('userBadge').textContent = '👤 ' + v;
  await bootstrap();
}

async function loadUsers() {
  try {
    const us = (await (await fetch('/api/users')).json()).users || [];
    document.getElementById('userList').innerHTML =
      us.map(u => `<option value="${u}">`).join('');
  } catch (e) {}
}

async function bootstrap() {
  if (!curUser) {
    document.getElementById('fileProgress').textContent =
      '⬅ 먼저 본인 이름을 입력하고 "확인"을 누르세요.';
    return;
  }
  files = (await (await fetch('/api/files?user=' + encodeURIComponent(curUser))).json()).files;
  if (!files.length) {
    document.getElementById('fileSelect').innerHTML = '';
    document.getElementById('savedCount').textContent = '';
    document.getElementById('fileProgress').textContent =
      `"${curUser}" 에게 할당된 파일이 없습니다. 이름을 정확히 입력했는지 확인하세요.`;
    return;
  }
  buildSelect();
  loadIdx(0);
}

async function init() {
  await loadUsers();
  if (curUser) {
    document.getElementById('userInput').value = curUser;
    document.getElementById('userBadge').textContent = '👤 ' + curUser;
  }
  await bootstrap();
}

function buildSelect() {
  const sel = document.getElementById('fileSelect');
  const old = sel.value;
  sel.innerHTML = '';
  files.forEach((f, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    const mark = f.failed ? '✗' : (f.complete ? '✓' : ' ');
    const tail = f.failed ? '  [FAILED]' : `  (${f.n_assigned}/${f.n_paths})`;
    opt.textContent = `[${i+1}/${files.length}] ${mark} ${f.category}/${f.name}${tail} — ${curUser}`;
    sel.appendChild(opt);
  });
  if (old) sel.value = old;
  const nSaved = files.filter(f => f.saved).length;
  const nDone  = files.filter(f => f.complete).length;
  const nFail  = files.filter(f => f.failed).length;
  document.getElementById('savedCount').textContent =
    `💾 ${curUser} — Saved ${nSaved}/${files.length}  (✓${nDone} ✗${nFail})`;
}

async function loadIdx(idx) {
  curFile = parseInt(idx);
  document.getElementById('fileSelect').value = curFile;
  state = await (await fetch('/api/file?idx=' + curFile + '&user=' + encodeURIComponent(curUser))).json();
  cursor = [];
  checked = new Set();
  document.getElementById('fileProgress').textContent =
    `${state.category}/${state.svg_name}  (${state.n_shapes} paths)`;
  await renderScope();
  render();
}

async function renderScope() {
  // Fetch a preview SVG containing only the current cursor node's paths,
  // and use it for the "Current scope" panel + hover-handlers.
  const node = getCursorNode();
  const scope = (node && node.paths) ? node.paths : [];
  const r = await fetch('/api/preview', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx: curFile, checked: scope, user: curUser}),
  });
  const j = await r.json();
  document.getElementById('fullBox').innerHTML = j.only_checked;
  attachGtHoverHandlers();
}

// Walk the GT SVG in DOM, tag each shape with data-pidx in original order,
// add hover/click handlers so user can identify and toggle paths from GT.
function attachGtHoverHandlers() {
  const SHAPE_SEL = 'path, rect, circle, ellipse, polygon, polyline, line';
  const root = document.getElementById('fullBox').querySelector('svg');
  if (!root) { console.warn('[labeler] no svg in fullBox'); return; }
  const allShapes = Array.from(root.querySelectorAll(SHAPE_SEL));
  const shapes = allShapes.filter(el => !el.closest('defs, clipPath, mask, pattern'));
  // The fullBox renders ONLY the current cursor's scope paths in their
  // original SVG order. Map DOM position → original path index using scope.
  const node = getCursorNode();
  const scopePaths = (node && node.paths) ? node.paths.slice().sort((a, b) => a - b) : [];
  console.log('[labeler] attaching hover to', shapes.length, 'shapes; scope=', scopePaths.length);
  shapes.forEach((el, domIdx) => {
    const pIdx = scopePaths[domIdx];
    if (pIdx == null) return;
    el.setAttribute('data-pidx', String(pIdx));
    el.style.cursor = 'pointer';
    el.style.pointerEvents = 'all';
    el.addEventListener('mouseenter', (ev) => {
      el.dataset._origStrokeWidth = el.getAttribute('stroke-width') || '';
      el.dataset._origStroke = el.getAttribute('stroke') || '';
      el.setAttribute('stroke', '#3b82f6');
      el.setAttribute('stroke-width', '2');
      el.style.filter = 'drop-shadow(0 0 4px rgba(59,130,246,.7))';
      showTooltip(ev, pIdx);
      fetchHoverPreview(pIdx);
    });
    el.addEventListener('mousemove', (ev) => moveTooltip(ev));
    el.addEventListener('mouseleave', (ev) => {
      if (el.dataset._origStroke) el.setAttribute('stroke', el.dataset._origStroke);
      else el.removeAttribute('stroke');
      if (el.dataset._origStrokeWidth) el.setAttribute('stroke-width', el.dataset._origStrokeWidth);
      else el.removeAttribute('stroke-width');
      el.style.filter = '';
      hideTooltip();
      fetchHoverPreview(null);
    });
    el.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const pending = pendingOf(getCursorNode());
      if (!pending.includes(pIdx)) return;
      if (checked.has(pIdx)) checked.delete(pIdx); else checked.add(pIdx);
      render();
    });
  });
}

let _tooltipEl = null;
function showTooltip(ev, idx) {
  if (!_tooltipEl) {
    _tooltipEl = document.createElement('div');
    _tooltipEl.id = 'gtTooltip';
    _tooltipEl.style.cssText = 'position:fixed; z-index:99; background:#222; color:#fff; padding:4px 8px; border-radius:4px; font-size:12px; font-family:monospace; pointer-events:none;';
    document.body.appendChild(_tooltipEl);
  }
  const sh = state.shapes[idx];
  const inChecked = checked.has(idx) ? ' [✓ checked]' : '';
  _tooltipEl.textContent = `p${idx} <${sh.tag}>${inChecked}`;
  _tooltipEl.style.display = 'block';
  moveTooltip(ev);
}
function moveTooltip(ev) {
  if (_tooltipEl) {
    _tooltipEl.style.left = (ev.clientX + 12) + 'px';
    _tooltipEl.style.top = (ev.clientY + 12) + 'px';
  }
}
function hideTooltip() {
  if (_tooltipEl) _tooltipEl.style.display = 'none';
}

function getCursorNode() {
  let n = state.tree;
  for (const i of cursor) n = n.children[i];
  return n;
}

function isLeaf(node) {
  // Leaf is determined by KEY PRESENCE: no `children` key → leaf. An empty
  // `children: []` means "Select group, not yet subdivided" → NOT a leaf,
  // so its paths still count as unassigned and save stays disabled.
  if (node.leaf === true) return true;
  if (node.leaf === false) return false;
  return !('children' in node);
}

function leafPathsOf(node) {
  if (isLeaf(node)) return (node.paths || []).slice();
  let out = [];
  for (const c of (node.children || [])) out = out.concat(leafPathsOf(c));
  return out;
}

function assignedInNode(node) {
  const s = new Set();
  for (const c of (node.children || [])) {
    leafPathsOf(c).forEach(p => s.add(p));
    if (!isLeaf(c)) (c.paths || []).forEach(p => s.add(p));
  }
  return s;
}

function pendingOf(node) {
  const scope = node.paths || [];
  const assigned = assignedInNode(node);
  return scope.filter(p => !assigned.has(p));
}

function isFullyAssigned(node) {
  // True when EVERY path in this node's scope ends up in some leaf in the
  // subtree below it. Lets the parent's children-list show a ✓ on Select
  // groups that are fully resolved.
  if (isLeaf(node)) return true;
  const leafSet = new Set(leafPathsOf(node));
  for (const p of (node.paths || [])) {
    if (!leafSet.has(p)) return false;
  }
  return (node.children || []).length > 0;
}

async function fetchCheckedPreview(checkedSet) {
  const r = await fetch('/api/preview', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx: curFile, checked: Array.from(checkedSet), user: curUser}),
  });
  const j = await r.json();
  document.getElementById('previewBox').innerHTML = j.only_checked;
}

async function fetchHoverPreview(hoverIdx) {
  const box = document.getElementById('hoverBox');
  if (hoverIdx == null) {
    box.innerHTML = '<div style="color:#888; font-size:11px;">hover a path…</div>';
    return;
  }
  const r = await fetch('/api/preview', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx: curFile, checked: [hoverIdx], user: curUser}),
  });
  const j = await r.json();
  // Show entire image dimmed with hover path at full opacity, so the user
  // sees WHERE the hovered path sits within the whole icon.
  box.innerHTML = j.highlight_checked;
}

// Backwards-compat shim: any call to fetchPreview(checked, hover) updates both panels.
async function fetchPreview(checkedSet, hoverIdx) {
  await fetchCheckedPreview(checkedSet);
  await fetchHoverPreview(hoverIdx);
}

function render() {
  const node = getCursorNode();
  // Breadcrumb
  const bc = document.getElementById('breadcrumb');
  bc.innerHTML = '';
  const segs = ['Root'];
  for (const i of cursor) {
    let n = state.tree;
    let walk = state.tree;
    for (let j = 0; j <= cursor.indexOf(i); j++) walk = walk.children[cursor[j]];
    // Simpler: just display labels along cursor
  }
  // Build breadcrumb properly
  bc.innerHTML = '';
  const root = document.createElement('a'); root.textContent = 'Root';
  root.onclick = async () => { cursor = []; checked = new Set(); await renderScope(); render(); };
  bc.appendChild(root);
  let n = state.tree;
  for (let k = 0; k < cursor.length; k++) {
    n = n.children[cursor[k]];
    const sep = document.createElement('span'); sep.textContent = ' > ';
    bc.appendChild(sep);
    const a = document.createElement('a'); a.textContent = n.label || '(unnamed)';
    const upto = k;
    a.onclick = async () => { cursor = cursor.slice(0, upto + 1); checked = new Set(); await renderScope(); render(); };
    bc.appendChild(a);
  }
  // Pending list
  const pendingList = document.getElementById('pendingList');
  pendingList.innerHTML = '';
  const pending = pendingOf(node);
  pending.forEach(p => {
    const div = document.createElement('div');
    div.className = 'row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = checked.has(p);
    cb.onchange = () => {
      if (cb.checked) checked.add(p); else checked.delete(p);
      fetchCheckedPreview(checked);
    };
    div.appendChild(cb);
    const label = document.createElement('span');
    const sh = state.shapes[p];
    label.textContent = `[p${p}] <${sh.tag}> ${sh.attrs_preview.slice(0, 60)}`;
    div.appendChild(label);
    div.onmouseenter = () => fetchHoverPreview(p);
    div.onmouseleave = () => fetchHoverPreview(null);
    // Drag-to-leaf: if this row's path is in the checked set, drag the whole
    // checked batch; otherwise drag just this single path. Drop on a leaf
    // child-row appends them to that leaf's paths.
    div.draggable = true;
    div.ondragstart = (ev) => {
      const ids = checked.has(p) ? Array.from(checked) : [p];
      ev.dataTransfer.setData('application/json', JSON.stringify(ids));
      ev.dataTransfer.effectAllowed = 'move';
      div.style.opacity = '0.5';
    };
    div.ondragend = () => { div.style.opacity = ''; };
    pendingList.appendChild(div);
  });
  if (pending.length === 0) {
    pendingList.innerHTML = '<i style="color:#888;">All paths in this level assigned.</i>';
  }
  // Children list
  const childrenList = document.getElementById('childrenList');
  childrenList.innerHTML = '';
  (node.children || []).forEach((c, ci) => {
    const div = document.createElement('div');
    div.className = 'child-row';
    const lbl = document.createElement('strong'); lbl.textContent = c.label || '(unnamed)';
    div.appendChild(lbl);
    const cIsLeaf = isLeaf(c);
    const chip = document.createElement('span'); chip.className = 'chip ' + (cIsLeaf ? 'leaf' : 'select');
    chip.textContent = cIsLeaf ? 'leaf' : 'select';
    div.appendChild(chip);
    // EVERY child row (leaf OR not-yet-finished Select group) is a drop
    // target. Drag any path from the pending list onto a group row and it is
    // added to that group — no need to "Finish"/leaf it first. Move
    // semantics: the path is pulled out of any sibling group so it is never
    // double-assigned.
    div.classList.add('drop-target');
    div.ondragover = (ev) => {
      ev.preventDefault();
      ev.dataTransfer.dropEffect = 'move';
      div.classList.add('drag-over');
    };
    div.ondragleave = () => { div.classList.remove('drag-over'); };
    div.ondrop = (ev) => {
      ev.preventDefault();
      div.classList.remove('drag-over');
      let ids;
      try { ids = JSON.parse(ev.dataTransfer.getData('application/json') || '[]'); }
      catch { ids = []; }
      if (!Array.isArray(ids) || ids.length === 0) return;
      // Accept any dragged path that belongs to this scope's universe.
      const scope = new Set(node.paths || []);
      const accepted = ids.filter(p => scope.has(p));
      if (accepted.length === 0) return;
      // Pull the paths out of any other sibling group in this scope first.
      for (const sib of (node.children || [])) {
        if (sib !== c && Array.isArray(sib.paths)) {
          sib.paths = sib.paths.filter(p => !accepted.includes(p));
        }
      }
      const merged = new Set([...(c.paths || []), ...accepted]);
      c.paths = Array.from(merged).sort((a, b) => a - b);
      accepted.forEach(p => checked.delete(p));
      render();
    };
    // ✓ badge — at a glance the user sees which children are "done".
    // Leaves are terminal so always done; Select groups need their full
    // subtree to be resolved into leaves.
    if (cIsLeaf || isFullyAssigned(c)) {
      const done = document.createElement('span');
      done.textContent = '✓ done';
      done.title = cIsLeaf
        ? 'Leaf — terminal, no further subdivision'
        : 'All paths in this group are assigned to leaves';
      done.style.cssText = 'font-size:10px; padding:2px 7px; border-radius:10px; background:#10b981; color:#fff; font-weight:600;';
      div.appendChild(done);
    } else if (!cIsLeaf) {
      const cPendingCnt = pendingOf(c).length;
      if (cPendingCnt > 0 || (c.children || []).length === 0) {
        const todo = document.createElement('span');
        const total = (c.paths || []).length;
        const assigned = total - cPendingCnt;
        todo.textContent = `${assigned}/${total}`;
        todo.title = `${cPendingCnt} path(s) still pending in this group`;
        todo.style.cssText = 'font-size:10px; padding:2px 7px; border-radius:10px; background:#fbbf24; color:#78350f; font-weight:600;';
        div.appendChild(todo);
      }
    }
    const meta = document.createElement('small'); meta.style.color = '#666';
    meta.textContent = ` ${(c.paths || leafPathsOf(c)).length} paths`;
    div.appendChild(meta);
    // Inline path chips for LEAF children — each chip has a tiny "✕" to
    // remove that single path from this leaf back into the parent's pending.
    // Useful when one of the bundled paths turns out to be unrelated
    // semantically (e.g. a shadow or deco that should go into "none").
    if (cIsLeaf && (c.paths || []).length > 0) {
      const pathsBox = document.createElement('span');
      pathsBox.style.cssText = 'display:inline-flex; flex-wrap:wrap; gap:2px; margin-left:4px;';
      (c.paths || []).slice().sort((a, b) => a - b).forEach(p => {
        const chip = document.createElement('span');
        chip.style.cssText = 'display:inline-flex; align-items:center; background:#e0f2fe; border:1px solid #7dd3fc; border-radius:8px; padding:0 2px 0 5px; font-size:10px; font-family:monospace;';
        chip.textContent = `p${p}`;
        const x = document.createElement('button');
        x.textContent = '✕';
        x.title = `Remove p${p} from "${c.label}" — returns to pending`;
        x.style.cssText = 'background:none; border:none; color:#b91c1c; cursor:pointer; padding:0 2px; font-size:9px; line-height:1;';
        x.onclick = (ev) => {
          ev.stopPropagation();
          c.paths = (c.paths || []).filter(x => x !== p);
          if ((c.paths || []).length === 0) {
            if (!confirm(`"${c.label}" is now empty — delete this leaf?`)) {
              c.paths = [p];  // restore
              return;
            }
            node.children.splice(ci, 1);
          }
          render();
        };
        chip.appendChild(x);
        chip.onmouseenter = () => fetchHoverPreview(p);
        chip.onmouseleave = () => fetchHoverPreview(null);
        pathsBox.appendChild(chip);
      });
      div.appendChild(pathsBox);
    }
    if (!isLeaf(c)) {
      const enter = document.createElement('button');
      enter.textContent = 'Enter →';
      enter.onclick = async () => { cursor.push(ci); checked = new Set(); await renderScope(); render(); };
      div.appendChild(enter);

      // Show "→ Leaf" only when the Select group is "incomplete" — has
      // pending paths or no children yet. If the user has already finished
      // subdividing into leaves (no pending), force-converting back would
      // discard meaningful labels, so we hide the button.
      const cPending = pendingOf(c);
      const hasChildren = (c.children || []).length > 0;
      if (cPending.length > 0 || !hasChildren) {
        const makeLeaf = document.createElement('button');
        makeLeaf.textContent = '→ Leaf';
        makeLeaf.title = 'Convert this Select group into a Leaf (no further subdivision)';
        makeLeaf.style.background = '#d1fae5';
        makeLeaf.style.borderColor = '#065f46';
        makeLeaf.onclick = () => {
          const allPaths = hasChildren ? leafPathsOf(c) : (c.paths || []);
          c.paths = allPaths.slice().sort((a, b) => a - b);
          delete c.leaf;
          delete c.children;
          render();
        };
        div.appendChild(makeLeaf);
      }
    } else {
      const makeSelect = document.createElement('button');
      makeSelect.textContent = '→ Select';
      makeSelect.title = 'Convert this Leaf back into a Select group so you can subdivide it';
      makeSelect.style.background = '#ddd6fe';
      makeSelect.style.borderColor = '#5b21b6';
      makeSelect.onclick = () => {
        delete c.leaf;
        c.children = [];
        render();
      };
      div.appendChild(makeSelect);
    }
    const del = document.createElement('button');
    del.textContent = '✕';
    del.style.color = '#b91c1c';
    del.onclick = () => {
      if (!confirm(`Delete child "${c.label}"? Its paths return to pending.`)) return;
      node.children.splice(ci, 1);
      render();
    };
    div.appendChild(del);
    childrenList.appendChild(div);
  });
  if ((node.children || []).length === 0) {
    childrenList.innerHTML = '<i style="color:#888;">No children yet.</i>';
  }
  // Up button
  document.getElementById('upBtn').disabled = cursor.length === 0;
  const isFailed = !!(state.tree && state.tree.fail);
  // Save gating: all paths must be in some leaf before disk write is allowed.
  const totalLeafPaths = new Set(leafPathsOf(state.tree));
  const allAssigned = totalLeafPaths.size === state.n_shapes && !isFailed;
  const saveBtn = document.querySelector('header button.primary');
  if (saveBtn) {
    saveBtn.disabled = !allAssigned;
    saveBtn.title = isFailed
      ? 'File is marked as failed — unmark first to save a tree'
      : (allAssigned
        ? 'Save tree.json'
        : `Cannot save: ${totalLeafPaths.size}/${state.n_shapes} paths assigned to leaves`);
  }
  const failBtn = document.getElementById('failBtn');
  if (failBtn) {
    failBtn.textContent = isFailed ? 'Unmark Fail' : 'Fail';
    failBtn.style.background = isFailed ? '#fbbf24' : '#ef4444';
    failBtn.style.borderColor = isFailed ? '#d97706' : '#ef4444';
  }
  document.getElementById('fileProgress').textContent = isFailed
    ? `${state.category}/${state.svg_name}  ✗ FAILED — semantically tangled, not labeled`
    : `${state.category}/${state.svg_name}  (${totalLeafPaths.size}/${state.n_shapes} paths assigned)`;
  // Previews
  fetchCheckedPreview(checked);
  fetchHoverPreview(null);
  // Build select
  buildSelect();
}

function checkAll() {
  const node = getCursorNode();
  pendingOf(node).forEach(p => checked.add(p));
  render();
}
function uncheckAll() { checked = new Set(); render(); }

function doSelect() { addChild(false); }
function doFinish() { addChild(true); }

function addChild(asLeaf) {
  let label = document.getElementById('labelInput').value.trim();
  // Leaf groups may be unlabeled — meaningless decoration paths can be
  // bundled together and stored as label "none". Select groups must have
  // a real label since the whole point of Select is "I'll subdivide this
  // semantic later".
  if (!label) {
    if (asLeaf) {
      label = 'none';
    } else {
      alert('Enter a label first (Select groups must be named).');
      return;
    }
  }
  if (checked.size === 0) { alert('Check at least one path.'); return; }
  const node = getCursorNode();
  const paths = Array.from(checked).sort((a, b) => a - b);
  const warn = document.getElementById('warnBox');
  warn.style.display = 'none';
  node.children = node.children || [];
  // Auto-merge new "none" leaves into an existing "none" sibling — meaningless
  // decoration paths should accumulate in a single bucket per scope rather than
  // sprouting one node each.
  if (asLeaf && label === 'none') {
    const existing = node.children.find(c => c.label === 'none' && isLeaf(c));
    if (existing) {
      const merged = new Set([...(existing.paths || []), ...paths]);
      existing.paths = Array.from(merged).sort((a, b) => a - b);
      checked = new Set();
      document.getElementById('labelInput').value = '';
      render();
      return;
    }
  }
  const child = { label, paths };
  if (!asLeaf) child.children = [];   // leaf nodes omit the children key entirely
  node.children.push(child);
  checked = new Set();
  document.getElementById('labelInput').value = '';
  render();
}

async function goUp() { if (cursor.length > 0) { cursor.pop(); checked = new Set(); await renderScope(); render(); } }
function nextFile() { if (curFile < files.length - 1) loadIdx(curFile + 1); }
function prevFile() { if (curFile > 0) loadIdx(curFile - 1); }

async function showTreeVis() {
  const overlay = document.getElementById('treevisOverlay');
  const content = document.getElementById('treevisContent');
  content.innerHTML = '<i style="color:#888;">rendering tree…</i>';
  overlay.style.display = 'flex';
  try {
    const r = await fetch('/api/treevis', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({idx: curFile, tree: state.tree, user: curUser}),
    });
    if (!r.ok) {
      content.innerHTML = `<div style="color:#b91c1c; padding:20px;">Render failed: ${r.status}</div>`;
      return;
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    content.innerHTML = `<img src="${url}" style="max-width:100%; height:auto; display:block;" onload="URL.revokeObjectURL('${url}')"/>`;
  } catch (e) {
    content.innerHTML = `<div style="color:#b91c1c; padding:20px;">Error: ${e}</div>`;
  }
}

function hideTreeVis(ev) {
  if (ev && ev.target && ev.target.id !== 'treevisOverlay' && ev.type !== 'click' && !ev.target.matches('button')) return;
  document.getElementById('treevisOverlay').style.display = 'none';
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const ov = document.getElementById('treevisOverlay');
    if (ov && ov.style.display !== 'none') hideTreeVis();
  }
});

async function markFail() {
  const isFailed = !!(state && state.tree && state.tree.fail);
  if (isFailed) {
    if (!confirm('Unmark this file as failed? The fail marker will be deleted.')) return;
    const r = await fetch('/api/fail', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({idx: curFile, unmark: true, user: curUser}),
    });
    const j = await r.json();
    if (j.ok) {
      files = (await (await fetch('/api/files?user=' + encodeURIComponent(curUser))).json()).files;
      buildSelect();
      await loadIdx(curFile);
    }
    return;
  }
  if (!confirm('Mark this file as FAILED?\n\nUse this when paths share a single visual shape and cannot be cleanly separated by semantic. tree.json will be written as {"fail": true, "paths": [...]} — no leaves required.')) return;
  const r = await fetch('/api/fail', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx: curFile, user: curUser}),
  });
  const j = await r.json();
  if (j.ok) {
    files = (await (await fetch('/api/files?user=' + encodeURIComponent(curUser))).json()).files;
    await loadIdx(curFile);
  }
}

async function save() {
  const status = document.getElementById('status');
  status.textContent = 'saving…'; status.style.color = '#fff';
  const r = await fetch('/api/save', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx: curFile, tree: state.tree, user: curUser}),
  });
  const j = await r.json();
  if (j.ok) {
    status.textContent = 'saved ✓'; status.style.color = '#34d399';
    files = (await (await fetch('/api/files?user=' + encodeURIComponent(curUser))).json()).files;
    buildSelect();
    // Reload current state to reflect any in-place SVG modifications.
    state = await (await fetch('/api/file?idx=' + curFile + '&user=' + encodeURIComponent(curUser))).json();
    document.getElementById('fullBox').innerHTML = state.svg_full;
    render();
  } else {
    status.textContent = 'rejected'; status.style.color = '#f87171';
    alert('Save rejected — every path must be assigned to a leaf first:\n\n' +
          (j.errors || ['unknown error']).join('\n'));
  }
  setTimeout(() => { status.textContent = ''; }, 3000);
}

window.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 's') { e.preventDefault(); save(); }
});

init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


if __name__ == "__main__":
    files = list_files()
    print(f"Found {len(files)} SVGs across {len(SVG_DIRS)} dirs.")
    print("Open http://localhost:8088")
    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="warning")
