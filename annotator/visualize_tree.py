"""Render a tree.json hierarchy as a single PNG: each node shows the
rasterized SVG (only the paths under that node) plus its semantic label.

Outputs (CLI batch mode only — the webui imports this module and uses its
drawing primitives directly, it does not write these files):
  <package>/sample_full/svg_visualized/icon/<stem>.png
  <package>/sample_full/svg_visualized/illustration/<stem>.png

For files marked {"fail": true}, writes a single image with a FAIL banner.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw, ImageFont

# Portable: everything resolves relative to this file so the whole folder
# can be zipped and run on any server. (CLI batch mode only — webui ignores
# these and just uses the drawing helpers.)
_PKG_ROOT = Path(__file__).resolve().parent
SAMPLE_ROOT = _PKG_ROOT / "sample_full"
CATS = [("icon", SAMPLE_ROOT / "icon" / "svg"),
        ("illustration", SAMPLE_ROOT / "illustration" / "svg")]
OUT_ROOT = SAMPLE_ROOT / "svg_visualized"

SHAPE_TAGS = ("path", "rect", "circle", "ellipse", "polygon", "polyline", "line")
SHAPE_PAT = re.compile(
    r"<(" + "|".join(SHAPE_TAGS) + r")\b([^>]*?)(/>|>([\s\S]*?)</\1>)"
)

THUMB = 160                # rasterized node thumbnail size
LABEL_H = 22               # label strip under thumb
NODE_W = THUMB + 12
NODE_H = THUMB + LABEL_H + 12
H_GAP = 16                 # horizontal gap between sibling subtrees
V_GAP = 60                 # vertical gap between levels
PAD = 30
BG = (250, 250, 252)

# Resolve fonts portably: try common names/locations across distros, then
# fall back to PIL's built-in font. No hard-coded absolute path required.
def _load_font(bold: bool, size: int):
    candidates = (
        ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    ) + (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    ) + (["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


FONT = _load_font(True, 13)
FONT_SM = _load_font(False, 11)
FONT_TITLE = _load_font(True, 18)


# Skip shapes nested in clip/mask/template containers so the path index
# space matches semantic_labeler.py exactly (otherwise isolating a node
# empties a clipPath and the whole node renders blank).
_NONRENDER_RE = re.compile(
    r"<(defs|clipPath|mask|symbol|pattern|marker)\b[^>]*>[\s\S]*?</\1>",
    re.IGNORECASE,
)


def shape_spans(svg_text: str) -> list[tuple[int, int, str]]:
    skip = [(m.start(), m.end()) for m in _NONRENDER_RE.finditer(svg_text)]
    out = []
    for m in SHAPE_PAT.finditer(svg_text):
        if any(s <= m.start() < e for s, e in skip):
            continue
        out.append((m.start(), m.end(), m.group(0)))
    return out


def svg_with_only(svg_text: str, keep: set[int]) -> str:
    parts, last = [], 0
    for i, (s, e, raw) in enumerate(shape_spans(svg_text)):
        parts.append(svg_text[last:s])
        if i in keep:
            parts.append(raw)
        last = e
    parts.append(svg_text[last:])
    return "".join(parts)


def rasterize(svg_text: str, paths: list[int], size: int) -> Image.Image:
    keep = set(paths)
    if not keep:
        return Image.new("RGBA", (size, size), (240, 240, 240, 255))
    text = svg_with_only(svg_text, keep)
    try:
        png = cairosvg.svg2png(bytestring=text.encode("utf-8"),
                               output_width=size, output_height=size)
        rgba = Image.open(io.BytesIO(png)).convert("RGBA")
    except Exception:
        return Image.new("RGBA", (size, size), (255, 200, 200, 255))
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def collect_paths(node: dict) -> list[int]:
    """All path indices under a node (including descendants)."""
    if "children" not in node:
        return list(node.get("paths") or [])
    out = []
    for c in node.get("children") or []:
        out.extend(collect_paths(c))
    if not out:
        out = list(node.get("paths") or [])
    return out


def is_leaf(node: dict) -> bool:
    return "children" not in node


# -----------------------------------------------------------------------------
# Tree layout
# -----------------------------------------------------------------------------
class LayoutNode:
    __slots__ = ("data", "label", "paths", "children", "x", "y", "w_subtree")

    def __init__(self, data: dict, label: str):
        self.data = data
        self.label = label
        self.paths = collect_paths(data)
        self.children: list[LayoutNode] = []
        self.x = 0
        self.y = 0
        self.w_subtree = 0


def build_layout(tree: dict) -> LayoutNode:
    """Wrap the tree.json schema into LayoutNodes."""
    root_body = tree.get("root") if "root" in tree else tree
    root = LayoutNode(root_body, "root")
    def walk(parent: LayoutNode, data: dict) -> None:
        for c in data.get("children") or []:
            child = LayoutNode(c, c.get("label", ""))
            parent.children.append(child)
            walk(child, c)
    walk(root, root_body)
    return root


def compute_widths(node: LayoutNode) -> int:
    if not node.children:
        node.w_subtree = NODE_W
        return node.w_subtree
    w = sum(compute_widths(c) for c in node.children) + H_GAP * (len(node.children) - 1)
    node.w_subtree = max(w, NODE_W)
    return node.w_subtree


def assign_positions(node: LayoutNode, x_offset: int, depth: int) -> None:
    node.y = PAD + depth * (NODE_H + V_GAP)
    if not node.children:
        node.x = x_offset + (node.w_subtree - NODE_W) // 2
        return
    cur = x_offset
    for c in node.children:
        assign_positions(c, cur, depth + 1)
        cur += c.w_subtree + H_GAP
    # Center this node above its children
    first = node.children[0]
    last = node.children[-1]
    node.x = (first.x + last.x + NODE_W) // 2 - NODE_W // 2


def all_nodes(root: LayoutNode) -> list[LayoutNode]:
    out = [root]
    for c in root.children:
        out.extend(all_nodes(c))
    return out


def max_depth(node: LayoutNode) -> int:
    if not node.children:
        return 0
    return 1 + max(max_depth(c) for c in node.children)


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------
def draw_node(canvas: Image.Image, draw: ImageDraw.ImageDraw,
              node: LayoutNode, svg_text: str) -> None:
    # Thumbnail
    thumb = rasterize(svg_text, node.paths, THUMB)
    canvas.paste(thumb, (node.x + 6, node.y + 6), thumb if thumb.mode == "RGBA" else None)
    # Border — color codes node type/state at a glance:
    #   gray   = root, blue = Select group, green = labeled leaf,
    #   red    = "none" leaf (intentionally meaningless paths)
    is_leaf_node = not node.children and node.label != "root"
    border = (16, 185, 129) if is_leaf_node else (96, 165, 250)
    if node.label == "root":
        border = (75, 85, 99)
    if node.label == "none":
        border = (220, 38, 38)
    draw.rounded_rectangle(
        [node.x, node.y, node.x + NODE_W, node.y + NODE_H],
        radius=8, outline=border, width=2,
    )
    # Label strip
    label = node.label or "(unnamed)"
    label_y = node.y + NODE_H - LABEL_H
    draw.rectangle([node.x, label_y, node.x + NODE_W, node.y + NODE_H],
                   fill=border)
    text = label
    bbox = draw.textbbox((0, 0), text, font=FONT)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw > NODE_W - 6:
        # truncate
        while tw > NODE_W - 6 and len(text) > 4:
            text = text[:-2]
            bbox = draw.textbbox((0, 0), text + "…", font=FONT)
            tw = bbox[2] - bbox[0]
        text = text + "…"
    draw.text((node.x + (NODE_W - tw) // 2, label_y + (LABEL_H - th) // 2 - 1),
              text, fill=(255, 255, 255), font=FONT)
    # path-count subscript
    cnt = f"{len(node.paths)} paths"
    bbox = draw.textbbox((0, 0), cnt, font=FONT_SM)
    tw2 = bbox[2] - bbox[0]
    draw.text((node.x + NODE_W - tw2 - 6, node.y + 4), cnt,
              fill=(120, 120, 120), font=FONT_SM)


def draw_edges(draw: ImageDraw.ImageDraw, node: LayoutNode) -> None:
    for c in node.children:
        x1 = node.x + NODE_W // 2
        y1 = node.y + NODE_H
        x2 = c.x + NODE_W // 2
        y2 = c.y
        # Elbow line: vertical → horizontal → vertical
        mid_y = (y1 + y2) // 2
        draw.line([(x1, y1), (x1, mid_y)], fill=(180, 180, 200), width=2)
        draw.line([(x1, mid_y), (x2, mid_y)], fill=(180, 180, 200), width=2)
        draw.line([(x2, mid_y), (x2, y2)], fill=(180, 180, 200), width=2)
        draw_edges(draw, c)


def render_tree(svg_path: Path, tree_path: Path, out_path: Path) -> None:
    svg_text = svg_path.read_text()
    raw = json.loads(tree_path.read_text())

    # FAIL marker
    if isinstance(raw, dict) and raw.get("fail") is True:
        # Render full SVG + a banner
        full = rasterize(svg_text, list(range(len(shape_spans(svg_text)))), 320)
        W, H = 480, 420
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((PAD, 10), f"{svg_path.parent.parent.name}/{svg_path.name}",
                  fill=(40, 40, 40), font=FONT_TITLE)
        canvas.paste(full, ((W - 320) // 2, 50), full if full.mode == "RGBA" else None)
        # red FAIL banner
        draw.rectangle([0, 380, W, 420], fill=(239, 68, 68))
        text = "✗ MARKED FAILED — semantically tangled"
        bbox = draw.textbbox((0, 0), text, font=FONT_TITLE)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((W - tw) // 2, 380 + (40 - th) // 2 - 2), text,
                  fill=(255, 255, 255), font=FONT_TITLE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        return

    root = build_layout(raw)
    compute_widths(root)
    assign_positions(root, PAD, 0)
    nodes = all_nodes(root)
    W = root.w_subtree + 2 * PAD
    H = (max_depth(root) + 1) * (NODE_H + V_GAP) - V_GAP + 2 * PAD + 30  # +30 for title

    canvas = Image.new("RGB", (W, H + 30), BG)
    draw = ImageDraw.Draw(canvas)
    title = f"{svg_path.parent.parent.name}/{svg_path.name}  —  {len(root.paths)} paths"
    draw.text((PAD, 6), title, fill=(40, 40, 40), font=FONT_TITLE)

    # shift everything down a bit to make room for title
    for n in nodes:
        n.y += 30

    draw_edges(draw, root)
    for n in nodes:
        draw_node(canvas, draw, n, svg_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    n_done = 0
    n_fail = 0
    for cat, dirp in CATS:
        out_dir = OUT_ROOT / cat
        for tree_p in sorted(dirp.glob("*.tree.json")):
            svg_p = tree_p.with_suffix("").with_suffix(".svg")
            if not svg_p.exists():
                continue
            out_p = out_dir / (svg_p.stem + ".png")
            try:
                render_tree(svg_p, tree_p, out_p)
                # Track fail count for summary
                raw = json.loads(tree_p.read_text())
                if isinstance(raw, dict) and raw.get("fail"):
                    n_fail += 1
                else:
                    n_done += 1
            except Exception as e:
                print(f"  ERR {svg_p.name}: {e}")
    print(f"Rendered {n_done} trees + {n_fail} fail markers → {OUT_ROOT}")


if __name__ == "__main__":
    main()
