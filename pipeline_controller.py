"""Sequential pipeline: decompose → (optional amodal completion) → vectorize.

Runs a single image through the pipeline and dumps every intermediate
artifact for inspection. Decomposer is flat (root → mutually-exclusive
sub-parts at depth=0, no recursion). With ``--amodal-inpaint`` each part
is then run through the Gemini-polygon + FLUX-Fill agent posthoc to
produce amodal per-part SVGs alongside the visible-mask SVGs.

Layout per run:
    runs/<ts>_<stem>/
        input.png
        gemini.json
        segments.json
        overlay.png
        parts/
            p1_<label>.png         RGBA per-part layer
            p1_<label>.svg         vtracer output
            p1_<label>_render.png  sanity render of the SVG
            ...

Usage:
    source ./set_env.sh    # provides GEMINI_API_KEY
    python3 pipeline_controller.py <image_path> [--out-dir runs]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.inpainter import get_agent as _get_amodal_agent
from modules.decomposer import Decomposer, _safe_name
from modules.decomposer.mask_ops import _apply_mask_rgba
from modules.evaluator import (
    ALL_METRICS as EVAL_METRICS,
    evaluate_sample as _evaluate_sample,
)
from modules.vectorizer import VectorizeParams, render_svg, vectorize_layer


def _save_overlay(out_path: Path, img: Image.Image, parts) -> None:
    overlay = img.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for part in parts:
        x, y, w, h = part.bbox_xywh
        if w <= 0 or h <= 0:
            continue
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        draw.text((x + 2, y + 2), f"{part.id} {part.label}", fill="red")
    overlay.save(out_path)


def _flatten_on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[-1] if rgba.mode == "RGBA" else None)
    return bg


def _walk_gemini_bboxes_from_calls(decompose_calls: list) -> list[tuple[str, list]]:
    """Collect ``(label, bbox)`` pairs from the recursive decompose-call log.
    Every per-level Gemini call emits its own components with bboxes, so we
    just flatten across all calls. ``bbox`` is normalized 0-1000 [ymin, xmin,
    ymax, xmax]."""
    out: list[tuple[str, list]] = []
    for call in decompose_calls or []:
        for comp in call.get("components") or []:
            label = str(comp.get("label", "")).strip()
            bbox = comp.get("_bbox")
            if label and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                out.append((label, list(bbox)))
    return out


def _save_gemini_bbox_viz(out_path: Path, source: Image.Image,
                          decompose_calls: list) -> bool:
    """Render the input image with Gemini-supplied ``_bbox`` rectangles drawn
    on top, color-cycled per box. Returns True if the file was written;
    False (no-op) when no Gemini bbox to visualize."""
    bboxes = _walk_gemini_bboxes_from_calls(decompose_calls)
    if not bboxes:
        return False

    canvas = source.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    W, H = canvas.size
    palette = [
        (220, 60, 60), (60, 130, 220), (40, 170, 80), (220, 150, 30),
        (170, 80, 200), (220, 90, 160), (90, 180, 200), (160, 110, 60),
    ]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for i, (label, bbox) in enumerate(bboxes):
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        x0 = max(0, min(W - 1, int(round(xmin * W / 1000.0))))
        y0 = max(0, min(H - 1, int(round(ymin * H / 1000.0))))
        x1 = max(0, min(W, int(round(xmax * W / 1000.0))))
        y1 = max(0, min(H, int(round(ymax * H / 1000.0))))
        if x1 <= x0 or y1 <= y0:
            continue
        color = palette[i % len(palette)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Filled label tag at the box's top-left corner so each box's name
        # stays close to its rectangle even when boxes overlap.
        text = label
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 3
        tag_w, tag_h = tw + pad * 2, th + pad * 2
        # Prefer placing the tag above the box; flip inside if it would clip.
        ty = y0 - tag_h
        if ty < 0:
            ty = y0
        tx = x0
        if tx + tag_w > W:
            tx = max(0, W - tag_w)
        draw.rectangle([tx, ty, tx + tag_w, ty + tag_h], fill=color)
        draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)

    canvas.save(out_path)
    return True


def _font_char_width(font) -> int:
    """Approximate width of a single character in pixels for the given font.
    Used to translate the text column's pixel width into a wrap character
    count for textwrap."""
    try:
        return max(6, int(font.getlength("M")))
    except Exception:
        try:
            return max(6, int(font.getsize("M")[0]))  # type: ignore[attr-defined]
        except Exception:
            return 8


def _render_tree_rows(out_path: Path,
                      rows: list[tuple],
                      *, thumb: int = 112, indent: int = 40,
                      min_row_h: int = 132, text_w: int = 520, pad: int = 16,
                      line_height: int = 18,
                      slot_tags: tuple[str, str] = ("SAM", "recover"),
                      cand_thumb: int = 96) -> None:
    """Render an indented-tree image from prebuilt rows.

    Row tuple shapes (the renderer accepts a mix in the same call):
      6-tuple — ``(caption, thumb_image, depth, parent_row_idx, flagged,
                   recovered)``
      7-tuple — ``(... , pre_recover_thumb)``         pre-recover slot
      8-tuple — ``(... , pre_recover_thumb, candidates)``
                where ``candidates`` is a list of
                ``(tag, PIL.Image, is_winner)`` triples — used by the
                amodal multi-candidate viz to render up to N FLUX
                candidates to the right of the text column. The winner
                gets a red border; the rest stay neutral.

    ``flagged`` paints reject markers in red; ``recovered`` (only honored
    when not flagged) paints SAM bad-recover markers in amber.

    Layout: indented rows with L-shaped connectors; long captions wrap
    in the text column and row heights scale with the wrapped text.
    When ANY row carries a pre-recover thumb, every row's thumb cell is
    widened to ``2*thumb + divider``. When ANY row carries candidates,
    an extra strip on the right widens the canvas by
    ``len(cands) * (cand_thumb + cand_gap)``.
    """
    max_depth = max((r[2] for r in rows), default=0)

    def _unpack(row):
        if len(row) >= 8:
            return (row[0], row[1], row[2], row[3], row[4], row[5],
                    row[6], row[7])
        if len(row) == 7:
            return (row[0], row[1], row[2], row[3], row[4], row[5],
                    row[6], None)
        return (row[0], row[1], row[2], row[3], row[4], row[5],
                None, None)

    any_pre = any(_unpack(r)[6] is not None for r in rows)
    divider = 6
    cell_w = (thumb * 2 + divider) if any_pre else thumb

    # Candidate strip on the right. Sized by the row with the most
    # candidates; rows without candidates leave it empty.
    max_cands = max((len(_unpack(r)[7] or []) for r in rows), default=0)
    cand_gap = 8
    cand_strip_w = (
        max_cands * (cand_thumb + cand_gap) + cand_gap if max_cands else 0
    )
    width = pad * 2 + max_depth * indent + cell_w + 12 + text_w + cand_strip_w

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    try:
        tag_font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        tag_font = font

    char_w = _font_char_width(font)
    chars_per_line = max(30, text_w // char_w)

    # Pre-wrap captions and compute per-row heights so long reasons don't
    # overflow the canvas. Each row is at least min_row_h tall, and grows
    # with the wrapped text. Candidate strip pushes the floor up to
    # cand_thumb + label-tag headroom so the 4 thumbs fit on a single
    # baseline regardless of caption length.
    wrapped_per_row: list[list[str]] = []
    row_heights: list[int] = []
    for row in rows:
        caption = row[0]
        cands = _unpack(row)[7] or []
        lines: list[str] = []
        for paragraph in (caption.splitlines() or [caption]):
            wrapped = textwrap.wrap(
                paragraph, width=chars_per_line, break_long_words=False,
                break_on_hyphens=False,
            ) or [paragraph]
            lines.extend(wrapped)
        text_block_h = len(lines) * line_height
        cand_block_h = (cand_thumb + 20) if cands else 0
        row_h_i = max(min_row_h, thumb + 24, text_block_h + 24,
                      cand_block_h + 12)
        wrapped_per_row.append(lines)
        row_heights.append(row_h_i)

    y_starts: list[int] = []
    y_cur = pad
    for h in row_heights:
        y_starts.append(y_cur)
        y_cur += h
    height = y_cur + pad

    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    def _row_xy(idx):
        depth = _unpack(rows[idx])[2]
        x = pad + depth * indent
        y = y_starts[idx]
        return x, y

    def _paste_thumb(img: Image.Image, slot_x: int, slot_y: int) -> None:
        thumb_img = _flatten_on_white(img.convert("RGBA"))
        thumb_img.thumbnail((thumb, thumb), Image.LANCZOS)
        tx = slot_x + (thumb - thumb_img.width) // 2
        ty = slot_y + (thumb - thumb_img.height) // 2
        canvas.paste(thumb_img, (tx, ty))

    for i, row in enumerate(rows):
        (caption, img, depth, parent_idx, flagged, recovered, pre_thumb,
         candidates) = _unpack(row)
        x, y = _row_xy(i)

        if parent_idx >= 0:
            px, py = _row_xy(parent_idx)
            # Connector targets the LEFT-slot thumb center on both ends:
            # left slot holds the "primary" view (pre-recover when present,
            # the only thumb otherwise) so the tree topology reads the same
            # regardless of whether a row was recovered.
            vx = px + thumb // 2
            py_bottom = py + thumb
            my_y_center = y + thumb // 2
            draw.line([(vx, py_bottom), (vx, my_y_center)],
                      fill=(170, 170, 170), width=2)
            draw.line([(vx, my_y_center), (x, my_y_center)],
                      fill=(170, 170, 170), width=2)

        if flagged:
            outline, border_w = (220, 60, 60), 3
        elif recovered:
            outline, border_w = (230, 150, 30), 3
        else:
            outline, border_w = (210, 210, 210), 1

        if pre_thumb is not None:
            # Left slot: original SAM silhouette (pre-recover).
            _paste_thumb(pre_thumb, x, y)
            draw.rectangle([x, y, x + thumb, y + thumb],
                           outline=(150, 150, 150), width=1)
            # Right slot: recovered result.
            right_x = x + thumb + divider
            _paste_thumb(img, right_x, y)
            draw.rectangle([right_x, y, right_x + thumb, y + thumb],
                           outline=outline, width=border_w)
            # Small corner tags so the columns are self-describing.
            tag_h = 14
            left_tag, right_tag = slot_tags
            left_tw = tag_font.getlength(left_tag) if hasattr(tag_font, "getlength") else 6 * len(left_tag)
            right_tw = tag_font.getlength(right_tag) if hasattr(tag_font, "getlength") else 6 * len(right_tag)
            draw.rectangle(
                [x, y, x + int(left_tw) + 6, y + tag_h], fill=(150, 150, 150),
            )
            draw.text((x + 3, y - 1), left_tag,
                      fill=(255, 255, 255), font=tag_font)
            draw.rectangle(
                [right_x, y, right_x + int(right_tw) + 6, y + tag_h], fill=outline,
            )
            draw.text((right_x + 3, y - 1), right_tag,
                      fill=(255, 255, 255), font=tag_font)
        else:
            _paste_thumb(img, x, y)
            draw.rectangle([x, y, x + thumb, y + thumb],
                           outline=outline, width=border_w)

        if flagged:
            text_fill = (180, 40, 40)
        elif recovered:
            text_fill = (180, 110, 20)
        else:
            text_fill = (20, 20, 20)
        text_x = x + cell_w + 12
        lines = wrapped_per_row[i]
        block_h = len(lines) * line_height
        # Vertically center the wrapped block in the row.
        text_y = y + max(0, (row_heights[i] - block_h) // 2)
        for j, line in enumerate(lines):
            draw.text((text_x, text_y + j * line_height), line,
                      fill=text_fill, font=font)

        # Candidate strip on the right (only on rows that carry FLUX
        # candidates -- normally just the inpainted nodes). Winner gets
        # a red border so the picker decision is visually obvious; the
        # tag (e.g. 'p0') is stamped in the top-left corner of each thumb.
        if candidates:
            strip_x0 = width - pad - cand_strip_w + cand_gap
            for ci, cand in enumerate(candidates):
                tag = str(cand[0])
                cimg = cand[1]
                is_winner = bool(cand[2])
                cx = strip_x0 + ci * (cand_thumb + cand_gap)
                cy = y + (row_heights[i] - cand_thumb) // 2
                # Fallback to a white square if a candidate is missing
                # (e.g. FLUX skipped that prompt due to empty repaint).
                if cimg is None:
                    draw.rectangle(
                        [cx, cy, cx + cand_thumb, cy + cand_thumb],
                        fill=(245, 245, 245),
                        outline=(200, 200, 200), width=1,
                    )
                else:
                    cthumb = _flatten_on_white(cimg.convert("RGBA"))
                    cthumb.thumbnail((cand_thumb, cand_thumb), Image.LANCZOS)
                    ox = cx + (cand_thumb - cthumb.width) // 2
                    oy = cy + (cand_thumb - cthumb.height) // 2
                    canvas.paste(cthumb, (ox, oy))
                cand_outline = (220, 40, 40) if is_winner else (200, 200, 200)
                cand_bw = 3 if is_winner else 1
                draw.rectangle(
                    [cx, cy, cx + cand_thumb, cy + cand_thumb],
                    outline=cand_outline, width=cand_bw,
                )
                # Tag corner (p0/p1/...) — winner gets the red fill,
                # others stay grey so the picker decision still reads on
                # the tag alone if the row is monochrome.
                tag_bg = (220, 40, 40) if is_winner else (140, 140, 140)
                tw = (tag_font.getlength(tag)
                      if hasattr(tag_font, "getlength")
                      else 6 * len(tag))
                draw.rectangle(
                    [cx, cy, cx + int(tw) + 6, cy + 14], fill=tag_bg,
                )
                draw.text((cx + 3, cy - 1), tag,
                          fill=(255, 255, 255), font=tag_font)

    canvas.save(out_path)


def _save_parts_tree(out_path: Path, source: Image.Image, tree_parts) -> None:
    """Default tree viz: each row shows the part's ``layer_image`` (post-mask
    RGBA). Nodes that triggered SAM bad-recover render side-by-side: the
    original SAM silhouette (pre-recover) on the left, the recovered result
    on the right. Segmentation-only — the amodal-inpaint candidate strip
    lives in :func:`_save_parts_tree_amodal`."""
    rows: list[tuple] = []
    rows.append(("input", source.convert("RGBA"), 0, -1, False, False, None))
    src_rgba = source.convert("RGBA")

    def _walk(parts, depth, parent_idx):
        for p in parts:
            verify = (p.extra or {}).get("verify", {})
            src = (p.extra or {}).get("mask_source", "") or ""
            marker = "_bad_recover_"
            recover_tag = (src.split(marker, 1)[1].replace("_", "+")
                           if marker in src else None)
            if p.rejected:
                reason = (verify.get("reason", "")
                          or (p.extra or {}).get("drop_reason", ""))
                cap = (f"{p.id}  {p.label}  [REJECT]  "
                       f"(score={p.score:.2f})  — {reason}")
            else:
                cap = f"{p.id}  {p.label}   (cov={p.icon_coverage:.2f}, score={p.score:.2f})"
                if recover_tag:
                    cap += f"  [RECOVERED:{recover_tag}]"
            # SAM stability score (assess_sam_quality): sb_05 narrow ±0.05
            # IoU, sb_20 wide ±0.20 IoU. bad ← sb_05<0.90 OR sb_20≤0.10.
            # Surface scores on EVERY SAM-detected node so the reader can
            # see why a recover triggered (or didn't).
            sam_q = (p.extra or {}).get("sam_quality")
            if sam_q is not None:
                sb05 = (p.extra or {}).get("sam_quality_sb_05")
                sb20 = (p.extra or {}).get("sam_quality_sb_20")
                parts_tag = [f"SAM={sam_q}"]
                if sb05 is not None:
                    parts_tag.append(f"sb05={sb05:.2f}")
                if sb20 is not None:
                    parts_tag.append(f"sb20={sb20:.2f}")
                if sam_q == "bad":
                    why = (p.extra or {}).get("sam_quality_reason")
                    if why:
                        parts_tag.append(why)
                cap += "  [" + " ".join(parts_tag) + "]"
            idx = len(rows)
            recovered_flag = bool(recover_tag) and not p.rejected
            pre_mask = (p.extra or {}).get("sam_mask_pre_recover")
            pre_thumb = (_apply_mask_rgba(src_rgba, pre_mask)
                         if (recover_tag and pre_mask is not None) else None)
            rows.append((cap, p.layer_image, depth, parent_idx,
                         p.rejected, recovered_flag, pre_thumb))
            if p.children:
                _walk(p.children, depth + 1, idx)

    _walk(tree_parts, 1, 0)
    _render_tree_rows(out_path, rows)


def _save_parts_tree_amodal(out_path: Path, source: Image.Image,
                            tree_parts) -> bool:
    """Tree viz parallel to ``parts_tree``, but for nodes with an amodal
    inpaint result the row shows visible-mask (left) vs amodal fill_rgba
    (right) side by side, and — when the inpaint ran with the multi-
    candidate path — sprouts an extra strip on the right with all N
    FLUX candidates; the pick-judge winner gets a red border so the
    decision is obvious at a glance. Non-inpainted nodes still show
    their regular ``layer_image``. Returns False (no-op) when no node
    had inpaint applied so callers can skip writing the file.

    Tags read ``visible`` / ``amodal``; the ``recovered`` flag is reused
    to paint the amodal-side outline (amber), so it's visually obvious
    which rows the FLUX-Fill pass touched.
    """
    rows: list[tuple] = []
    rows.append(("input", source.convert("RGBA"), 0, -1, False, False, None))
    any_inpaint = [False]

    def _walk(parts, depth, parent_idx):
        for p in parts:
            inpaint = (p.extra or {}).get("inpaint") or {}
            fill = inpaint.get("fill_rgba")
            n_v = int(inpaint.get("n_vertices") or 0)
            skipped = bool(inpaint.get("skipped"))
            parse_err = inpaint.get("parse_error")
            call_err = bool(inpaint.get("call_error"))
            best_tag = inpaint.get("best_tag")
            cand_dict = inpaint.get("candidates") or {}

            has_inpaint = (fill is not None) and (not p.rejected) and (not skipped) and (not call_err)
            if has_inpaint:
                any_inpaint[0] = True

            if p.rejected:
                cap = f"{p.id}  {p.label}  [REJECT]"
            else:
                cap = f"{p.id}  {p.label}"
                if call_err:
                    cap += f"  [INPAINT CALL ERR: {parse_err or '?'}]"
                elif skipped:
                    if inpaint.get("occluded") is False:
                        cap += "  [INPAINT skip (not occluded)]"
                    else:
                        cap += "  [INPAINT SKIPPED (empty repaint mask)]"
                elif fill is not None:
                    if best_tag:
                        cap += f"  [INPAINT verts={n_v} best={best_tag}]"
                    else:
                        cap += f"  [INPAINT verts={n_v}]"
                    if parse_err:
                        cap += f"  parse_err={parse_err[:60]}"
                else:
                    cap += "  (no inpaint entry)"

            if has_inpaint:
                pre_thumb = p.layer_image  # left: visible mask
                main_thumb = fill          # right: amodal fill
            else:
                pre_thumb = None
                main_thumb = p.layer_image

            # Multi-candidate strip: only attach when FLUX actually ran on
            # ≥1 candidate. Sort by prompt-id (p0..p3) so the row reads
            # left-to-right; winner gets a red border via the 3rd tuple
            # element in each candidate triple.
            cand_list: list | None = None
            if has_inpaint and cand_dict:
                cand_list = []
                for tag in sorted(cand_dict.keys()):
                    cand = cand_dict[tag]
                    cimg = (cand.get("fill_rgba")
                            or cand.get("polygon_mask"))
                    cand_list.append((tag, cimg, tag == best_tag))

            idx = len(rows)
            rows.append((cap, main_thumb, depth, parent_idx,
                         p.rejected, has_inpaint, pre_thumb, cand_list))
            if p.children:
                _walk(p.children, depth + 1, idx)

    _walk(tree_parts, 1, 0)
    if not any_inpaint[0]:
        return False
    _render_tree_rows(out_path, rows, slot_tags=("visible", "amodal"))
    return True


def _bbox_rect_layer(input_rgba: Image.Image, bbox_canvas) -> Image.Image:
    """Apply a bbox-rect mask to the original image, returning RGBA where
    pixels inside the bbox are visible and the rest is transparent. Used by
    the Gemini-bbox tree viz to show 'what v4 (bbox-only) would have used'
    per node."""
    arr = np.array(input_rgba.convert("RGBA"))
    H, W = arr.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    if bbox_canvas is not None and len(bbox_canvas) == 4:
        x0, y0, x1, y1 = (int(v) for v in bbox_canvas)
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    arr[..., 3] = np.where(mask, arr[..., 3], 0).astype(np.uint8)
    return Image.fromarray(arr)


def _save_gemini_bbox_crops(out_dir: Path, source: Image.Image,
                            tree_parts) -> int:
    """Save per-node ``debug/bbox_crops/<id>_<label>.png`` = input cropped
    to the node's gemini_bbox. Lets the reviewer eyeball exactly the
    region Gemini routed to SAM for that label. Returns the count
    written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = source.convert("RGB")
    n = 0

    def _slug(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in (s or "")).strip("_")[:40]

    def _walk(parts) -> None:
        nonlocal n
        for p in parts:
            bbox = (p.extra or {}).get("gemini_bbox")
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = [int(round(v)) for v in bbox]
                x0 = max(0, min(x0, rgb.width - 1))
                y0 = max(0, min(y0, rgb.height - 1))
                x1 = max(x0 + 1, min(x1, rgb.width))
                y1 = max(y0 + 1, min(y1, rgb.height))
                crop = rgb.crop((x0, y0, x1, y1))
                crop.save(out_dir / f"{p.id}_{_slug(p.label)}.png")
                n += 1
            if p.children:
                _walk(p.children)

    _walk(tree_parts)
    return n


def _save_gemini_bbox_tree_viz(out_path: Path, source: Image.Image,
                               tree_parts) -> bool:
    """Tree viz parallel to ``parts_tree``, but each row's thumbnail is the
    *Gemini bbox-rect* mask applied to the input — i.e. the silhouette v4
    (bbox-only) would have used before any SAM refinement. Lets the
    reviewer compare Gemini's intent against the post-hoc SAM output side
    by side."""
    rows: list[tuple[str, Image.Image, int, int, bool, bool]] = []
    rows.append(("input", source.convert("RGBA"), 0, -1, False, False))
    src_rgba = source.convert("RGBA")
    any_node = [False]

    def _walk(parts, depth, parent_idx):
        for p in parts:
            bbox = (p.extra or {}).get("gemini_bbox")
            if bbox:
                any_node[0] = True
            thumb_img = _bbox_rect_layer(src_rgba, bbox)
            cap = (f"{p.id}  {p.label}   (d{p.depth}, "
                   f"bbox={list(bbox) if bbox else 'n/a'})")
            idx = len(rows)
            rows.append((cap, thumb_img, depth, parent_idx, p.rejected, False))
            if p.children:
                _walk(p.children, depth + 1, idx)

    _walk(tree_parts, 1, 0)
    if not any_node[0]:
        return False
    _render_tree_rows(out_path, rows)
    return True


def _save_sam_masks_tree_viz(out_path: Path, source: Image.Image,
                             tree_parts) -> bool:
    """Tree viz parallel to ``parts_tree``, but each row's thumbnail is the
    input image masked by the *SAM-replaced* mask only (so the silhouette
    of what SAM produced is the focus). Caption surfaces ``mask_source``
    and ``sam_score`` so it's obvious which nodes were SAM-replaced vs.
    fell back to bbox-rect (when SAM returned no candidates)."""
    rows: list[tuple[str, Image.Image, int, int, bool, bool]] = []
    rows.append(("input", source.convert("RGBA"), 0, -1, False, False))
    src_rgba = source.convert("RGBA")
    any_sam = [False]

    def _walk(parts, depth, parent_idx):
        for p in parts:
            src = (p.extra or {}).get("mask_source")
            score = (p.extra or {}).get("sam_score")
            # v5/v6 emit "sam"; v6.2 adds "judge_*"; v6.3 adds "iter_*".
            if src and (src == "sam" or src.startswith(("judge_", "iter_"))):
                any_sam[0] = True
            if p.mask is not None and p.mask.sum() > 0:
                thumb_img = _apply_mask_rgba(src_rgba, p.mask)
            else:
                thumb_img = Image.new("RGBA", src_rgba.size, (0, 0, 0, 0))
            tag = (f"sam s={score:.2f}" if (src == "sam" and score is not None)
                   else (src or "?"))
            cap = (f"{p.id}  {p.label}   "
                   f"(d{p.depth}, mask_px={int(p.mask.sum()) if p.mask is not None else 0}, "
                   f"{tag})")
            idx = len(rows)
            rows.append((cap, thumb_img, depth, parent_idx, p.rejected, False))
            if p.children:
                _walk(p.children, depth + 1, idx)

    _walk(tree_parts, 1, 0)
    if not any_sam[0]:
        return False
    _render_tree_rows(out_path, rows)
    return True


def _part_summary(p) -> dict:
    d = {
        "id": p.id,
        "label": p.label,
        "score": p.score,
        "bbox_xywh": list(p.bbox_xywh),
        "icon_coverage": p.icon_coverage,
        "depth": p.depth,
        "parent_id": p.parent_id,
        "rejected": p.rejected,
    }
    if p.extra.get("verify"):
        d["verify"] = p.extra["verify"]
    if p.extra.get("fallback_from"):
        d["fallback_from"] = p.extra["fallback_from"]
    # v5: surface post-hoc SAM provenance so segments.json reflects the
    # actual mask origin without spelunking through the pickled tree.
    if "mask_source" in p.extra:
        d["mask_source"] = p.extra["mask_source"]
    if "sam_score" in p.extra:
        d["sam_score"] = p.extra["sam_score"]
    if "sam_quality" in p.extra:
        d["sam_quality"] = p.extra["sam_quality"]
        d["sam_quality_sb_05"] = p.extra.get("sam_quality_sb_05")
        d["sam_quality_sb_20"] = p.extra.get("sam_quality_sb_20")
        if "sam_quality_reason" in p.extra:
            d["sam_quality_reason"] = p.extra["sam_quality_reason"]
    if p.extra.get("gemini_bbox"):
        d["gemini_bbox"] = list(p.extra["gemini_bbox"])
    # v9: cross-tree dedup may rename a child to ``parent label + child``
    # (e.g. ``desk legs``). That intentionally violates the literal H1
    # rule, so we mark it so eval scripts skip H1 for these nodes.
    if p.extra.get("disambiguated"):
        d["disambiguated"] = True
    # v14: judge + inpaint surface so segments.json reflects the
    # per-node interleaved decisions without spelunking through extras.
    if "judge_is_residual" in p.extra:
        d["judge_is_residual"] = bool(p.extra["judge_is_residual"])
    if "judge_needs_inpaint" in p.extra:
        d["judge_needs_inpaint"] = bool(p.extra["judge_needs_inpaint"])
    if "is_background_layer" in p.extra:
        d["is_background_layer"] = bool(p.extra["is_background_layer"])
    if "original_id" in p.extra:
        d["original_id"] = p.extra["original_id"]
    judge = p.extra.get("judge") or {}
    if judge.get("reason"):
        d["judge_reason"] = judge["reason"]
    bg_dbg = judge.get("background") if isinstance(judge, dict) else None
    if isinstance(bg_dbg, dict) and bg_dbg.get("reason"):
        d["judge_background_reason"] = bg_dbg["reason"]
    inpaint = p.extra.get("inpaint") or {}
    if inpaint:
        d["inpaint_applied"] = True
        d["inpaint_n_vertices"] = int(inpaint.get("n_vertices") or 0)
        if inpaint.get("parse_error"):
            d["inpaint_parse_error"] = inpaint["parse_error"]
        if "occluded" in inpaint:
            d["occluded"] = bool(inpaint["occluded"])
    return d


def _tree_summary(p) -> dict:
    d = _part_summary(p)
    if p.children:
        d["children"] = [_tree_summary(c) for c in p.children]
    return d


def _save_segments_json(out_path: Path, parts, tree_parts, meta) -> None:
    payload = {
        "parts": [_part_summary(p) for p in parts],
        "tree": [_tree_summary(p) for p in tree_parts],
        "verifications": meta.get("verifications", []),
        "icon_area": meta["icon_area"],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


_SVG_BODY_RE = __import__("re").compile(r"<svg[^>]*>(.*)</svg>", __import__("re").S)


def _collect_effective_leaves(parts) -> list:
    """DFS-collect accepted parts that have no accepted children.

    Includes accepted-but-childless parts AND accepted parents whose
    children were all rejected (so their pixels still get vectorized).

    Internal-node rejection (a parent rejected by v6.2 judge or v6.3
    iter while it still has accepted children) does NOT cascade — we
    drop the parent's mask but recurse so its accepted descendants are
    still rendered. Pure leaf rejection (rejected with no children) is
    the only case where pixels are dropped entirely.
    """
    out: list = []
    for p in parts:
        if p.rejected:
            if p.children:
                out.extend(_collect_effective_leaves(
                    [c for c in p.children if not c.rejected]
                ))
            continue
        accepted_children = [c for c in p.children if not c.rejected]
        if not accepted_children:
            out.append(p)
        else:
            out.extend(_collect_effective_leaves(accepted_children))
    return out


def _extract_svg_body(svg_path: Path) -> str:
    try:
        text = svg_path.read_text()
    except FileNotFoundError:
        return ""
    m = _SVG_BODY_RE.search(text)
    return m.group(1).strip() if m else ""


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _merge_svgs(merged_svg_path: Path, canvas_size: tuple[int, int],
                items: list[tuple[int, str, str, Path]]) -> Path | None:
    """Stitch a list of SVGs into one composite SVG.

    Each item is `(sort_key, id, label, svg_path)`. Draw order: smallest
    sort_key first (bottom) → largest last (top). Pass ``-inf`` for layers
    that must paint absolutely first (background plates). Returns merged
    path, or None if nothing to merge.
    """
    W, H = canvas_size
    bodies: list[tuple[int, str, str, str]] = []
    for sort_key, iid, lbl, svg_path in items:
        body = _extract_svg_body(svg_path)
        if body:
            bodies.append((sort_key, iid, lbl, body))

    if not bodies:
        return None

    bodies.sort(key=lambda t: t[0])

    # Bake a white background into the merged SVG so the file viewed on its
    # own renders against white (per-part SVGs keep transparency for stacking).
    groups = [
        f'  <rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>'
    ]
    for _, iid, lbl, body in bodies:
        groups.append(
            f'  <g id="{_xml_escape(iid)}" '
            f'data-label="{_xml_escape(lbl)}">\n{body}\n  </g>'
        )

    merged = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        + "\n".join(groups) + "\n"
        "</svg>\n"
    )
    merged_svg_path.write_text(merged)
    return merged_svg_path


def _load_gemini_cache(cache_root: Path, stem: str) -> list[dict] | None:
    """Resolve a per-image gemini.json cache from ``cache_root``.

    Two layouts are tried in order:
      1. ``cache_root/<stem>/gemini.json`` (batch run dir)
      2. ``cache_root/gemini.json`` (single-image run dir; only when the
         cache root itself is the cached image's run dir)
    Returns ``decompose_calls`` list, or ``None`` if no cache found.
    """
    nested = cache_root / stem / "gemini.json"
    flat = cache_root / "gemini.json"
    path = nested if nested.exists() else (flat if flat.exists() else None)
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return payload.get("decompose_calls", []) or []


_PROMPT_IDS = ("p0", "p1", "p2", "p3")


def _run_amodal_inpaint(parts: list, scene_rgb: Image.Image,
                        out_dir: Path, device: str, debug: bool,
                        *,
                        gemini_model: str | None = None,
                        thinking_level: str | None = None,
                        skip_judge: bool = False,
                        polygon_workers: int = 4,
                        seed: int | None = None,
                        prompt_ids: tuple[str, ...] = _PROMPT_IDS) -> dict:
    """Run amodal completion on every non-root, non-rejected part.

    Pipeline:
      Stage A — parallel ``agent.make_mask_multi`` (judge + polygon × N,
                I/O-bound) via a ThreadPoolExecutor with
                ``polygon_workers`` slots. Judge is called once per part;
                only when ``occluded=True`` does it ask Gemini for N
                polygon variants (one per prompt id).
      Stage B — sequential ``agent.fill_from_mask_multi`` per part (FLUX
                × N, GPU-bound) + a final Gemini pick-judge that picks
                the best of the N candidate inpaints.

    Mutates each part's ``extra["inpaint"]`` in place with the winning
    fill (PIL RGBA), numpy bool masks, all N candidate fills (debug
    viz consumes these), the winner's tag, and call metadata. Returns
    aggregate counts for summary.json. Per-part Gemini / FLUX failures
    are caught — the part falls back to its original ``layer_image`` so
    downstream vectorize never breaks.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    agent = _get_amodal_agent(device=device)
    dbg_dir = out_dir / "debug" / "amodal" if debug else None
    if dbg_dir is not None:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    stats = {"n_processed": 0, "n_skipped": 0,
             "n_parse_errors": 0, "n_call_errors": 0,
             "n_occluded_true": 0,
             "n_candidates_per_occluded": len(prompt_ids)}

    targets = [p for p in parts if not p.rejected]
    if not targets:
        return stats

    # Stage A — parallel Gemini judge + polygon-×N calls (I/O-bound).
    workers = max(1, min(polygon_workers, len(targets)))
    t_poly0 = time.time()
    masks: dict[str, dict] = {}
    mask_errors: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_part = {}
        for p in targets:
            future_to_part[pool.submit(
                agent.make_mask_multi, p.layer_image, scene_rgb, p.label,
                prompt_ids=prompt_ids,
                gemini_model=gemini_model,
                thinking_level=thinking_level,
                skip_judge=skip_judge,
            )] = p
        for fut in as_completed(future_to_part):
            p = future_to_part[fut]
            try:
                masks[p.id] = fut.result()
            except Exception as e:
                mask_errors[p.id] = e
    t_poly = time.time() - t_poly0
    print(f"[amodal] polygon-multi stage: {len(masks)}/{len(targets)} ok "
          f"in {t_poly:.1f}s (workers={workers}, "
          f"prompt_ids={list(prompt_ids)})")

    # Stage B — sequential FLUX × N + pick per part.
    for part in targets:
        if part.id in mask_errors:
            e = mask_errors[part.id]
            print(f"[amodal] {part.id} '{part.label}' make_mask_multi "
                  f"failed: {type(e).__name__}: {e}")
            part.extra["inpaint"] = {
                "fill_rgba": part.layer_image,
                "repaint_mask": None, "polygon_mask": None,
                "n_vertices": 0,
                "parse_error": f"{type(e).__name__}: {e}",
                "skipped": True, "call_error": True,
                "candidates": {},
                "best_tag": None,
            }
            stats["n_call_errors"] += 1
            continue
        mk = masks[part.id]
        try:
            out = agent.fill_from_mask_multi(
                part.layer_image, scene_rgb, part.label, mk,
                **({} if seed is None else {"seed": int(seed)}),
                gemini_model=gemini_model,
            )
        except Exception as e:
            print(f"[amodal] {part.id} '{part.label}' FLUX-multi failed: "
                  f"{type(e).__name__}: {e}")
            part.extra["inpaint"] = {
                "fill_rgba": part.layer_image,
                "repaint_mask": None, "polygon_mask": None,
                "n_vertices": 0,
                "parse_error": f"{type(e).__name__}: {e}",
                "skipped": True, "call_error": True,
                "candidates": {},
                "best_tag": None,
            }
            stats["n_call_errors"] += 1
            continue

        rp_arr = np.asarray(out["repaint_mask"], dtype=np.uint8) > 127
        pm_arr = np.asarray(out["polygon_mask"], dtype=np.uint8) > 127
        skipped = not bool(rp_arr.any())
        parse_err = out.get("parse_error")
        best_tag = out.get("best_tag")
        # Flatten per-candidate dict for tree viz + debug consumers.
        # Each ``cand`` carries polygon_mask / repaint_mask (PIL L),
        # fill_rgba / fill_on_white (PIL RGBA / RGB), flux_skipped (bool).
        # ``polygon_usage`` is preserved per candidate so the downstream
        # cost aggregator can sum all N polygon calls — the top-level
        # ``polygon_usage`` field only carries the winner's usage.
        raw_candidates = out.get("candidates") or {}
        candidates: dict[str, dict] = {}
        for tag, cand in raw_candidates.items():
            candidates[tag] = {
                "polygon_mask": cand.get("polygon_mask"),
                "repaint_mask": cand.get("repaint_mask"),
                "fill_rgba": cand.get("fill_rgba"),
                "fill_on_white": cand.get("fill_on_white"),
                "n_vertices": int(cand.get("n_vertices") or 0),
                "parse_error": cand.get("parse_error"),
                "flux_skipped": bool(cand.get("flux_skipped", False)),
                "is_winner": (tag == best_tag),
                "polygon_usage": cand.get("polygon_usage"),
            }

        part.extra["inpaint"] = {
            "fill_rgba": out["fill_rgba"],
            "repaint_mask": rp_arr,
            "polygon_mask": pm_arr,
            "n_vertices": int(out.get("n_vertices") or 0),
            "parse_error": parse_err,
            "skipped": skipped,
            "occluded": bool(out.get("occluded", False)),
            "judge_reason": out.get("judge_reason"),
            # Forward usage from the Gemini calls fill_from_mask_multi
            # wraps (inpaint_judge + inpaint_polygon × N + inpaint_pick).
            "judge_usage": out.get("judge_usage"),
            "polygon_usage": out.get("polygon_usage"),
            "pick_usage": out.get("pick_usage"),
            # Multi-candidate bookkeeping for the tree viz / debug dump.
            "candidates": candidates,
            "best_tag": best_tag,
            "pick_reason": out.get("pick_reason"),
            "pick_mapping": out.get("pick_mapping"),
        }
        stats["n_processed"] += 1
        if skipped:
            stats["n_skipped"] += 1
        if parse_err:
            stats["n_parse_errors"] += 1
        if out.get("occluded"):
            stats["n_occluded_true"] += 1

        if dbg_dir is not None:
            stem = f"{part.id}_{_safe_name(part.label)}"
            # Winner copy (back-compat with the old single-candidate file
            # names so existing analysis scripts keep working).
            out["polygon_mask"].save(dbg_dir / f"{stem}__polygon.png")
            out["repaint_mask"].save(dbg_dir / f"{stem}__repaint.png")
            out["fill_rgba"].save(dbg_dir / f"{stem}__fill_rgba.png")
            out["fill_on_white"].save(dbg_dir / f"{stem}__fill_on_white.png")
            # Per-candidate dump under a sibling subdir so the four
            # variants are easy to compare side by side.
            cand_dir = dbg_dir / "candidates" / stem
            cand_dir.mkdir(parents=True, exist_ok=True)
            for tag, cand in candidates.items():
                win = "_WINNER" if cand["is_winner"] else ""
                pm = cand.get("polygon_mask")
                rp = cand.get("repaint_mask")
                fr = cand.get("fill_rgba")
                fw = cand.get("fill_on_white")
                if pm is not None:
                    pm.save(cand_dir / f"{tag}{win}__polygon.png")
                if rp is not None:
                    rp.save(cand_dir / f"{tag}{win}__repaint.png")
                if fr is not None:
                    fr.save(cand_dir / f"{tag}{win}__fill_rgba.png")
                if fw is not None:
                    fw.save(cand_dir / f"{tag}{win}__fill_on_white.png")

    return stats


def run(image_path: Path, out_dir: Path,
        params: VectorizeParams | None = None,
        debug: bool = False,
        decomposer: Decomposer | None = None,
        eval_models: dict | None = None,
        gemini_cache_root: Path | None = None,
        amodal_inpaint: bool = False,
        amodal_device: str = "cuda:0",
        amodal_skip_judge: bool = False,
        amodal_polygon_workers: int = 4,
        amodal_seed: int | None = None,
        vtracer_modes: tuple[str, ...] | None = None) -> Path:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    image.save(out_dir / "input.png")

    t0 = time.time()
    decomposer = decomposer or Decomposer()
    # Set/clear the Gemini decompose-call replay cache around the call so
    # batch loops install a fresh per-image cache and live calls resume
    # cleanly when a sample has no cache.
    if gemini_cache_root is not None:
        cached_calls = _load_gemini_cache(gemini_cache_root, image_path.stem)
        if cached_calls is None:
            print(f"[gemini-cache] no cache for {image_path.stem}; live call")
        else:
            print(f"[gemini-cache] {image_path.stem}: replaying "
                  f"{len(cached_calls)} cached calls")
        decomposer._gemini.set_replay_cache(cached_calls)
    else:
        decomposer._gemini.set_replay_cache(None)
    debug_dir = out_dir / "debug" if debug else None
    result = decomposer.decompose(image, debug_dir=debug_dir)
    t_decompose = time.time() - t0

    decompose_calls = result.get("decompose_calls", [])
    total_usage: dict[str, int | float] = {
        "prompt_tokens": 0, "candidates_tokens": 0,
        "thoughts_tokens": 0,
        "total_tokens": 0, "cost_usd": 0.0,
    }
    for call in decompose_calls:
        u = call.get("usage_metadata") or {}
        total_usage["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        total_usage["candidates_tokens"] += int(u.get("candidates_tokens", 0) or 0)
        total_usage["thoughts_tokens"] += int(u.get("thoughts_tokens", 0) or 0)
        total_usage["total_tokens"] += int(u.get("total_tokens", 0) or 0)
        total_usage["cost_usd"] += float(u.get("cost_usd", 0.0) or 0.0)
    # Aggregate every Gemini call billed to the inpaint flow across parts.
    # Inpaint-flow billables, in their part-extra slots:
    #   - decomposer residual_judge  → part.extra["judge"]["usage"]
    #   - amodal stage judge         → part.extra["inpaint"]["judge_usage"]
    #   - amodal stage polygon × N   → part.extra["inpaint"]["candidates"]
    #                                                 [pid]["polygon_usage"]
    #   - amodal stage pick-judge    → part.extra["inpaint"]["pick_usage"]
    # The top-level ``polygon_usage`` in ``inpaint`` only carries the
    # winner's usage; we MUST walk ``candidates`` to capture the other
    # N-1 polygon calls. ``pick_usage`` is its own line item.
    inpaint_usage_total: dict[str, int | float] = {
        "prompt_tokens": 0, "candidates_tokens": 0,
        "thoughts_tokens": 0,
        "total_tokens": 0, "cost_usd": 0.0,
        "n_judge_calls": 0, "n_polygon_calls": 0, "n_pick_calls": 0,
    }
    def _accumulate_inpaint(u: dict, *, count_key: str | None = None) -> None:
        inpaint_usage_total["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        inpaint_usage_total["candidates_tokens"] += int(u.get("candidates_tokens", 0) or 0)
        inpaint_usage_total["thoughts_tokens"] += int(u.get("thoughts_tokens", 0) or 0)
        inpaint_usage_total["total_tokens"] += int(u.get("total_tokens", 0) or 0)
        inpaint_usage_total["cost_usd"] += float(u.get("cost_usd", 0.0) or 0.0)
        if count_key is not None:
            inpaint_usage_total[count_key] += 1

    def _walk_inpaint_usages(parts):
        """Yield ``(usage_dict, count_key)`` for every Gemini call that
        should be billed to the inpaint flow. Single source of truth for
        the aggregator — both the pre-amodal and post-amodal passes drive
        through this so they can't drift apart."""
        for part in parts:
            extras = part.extra or {}
            u = (extras.get("judge") or {}).get("usage")
            if u:
                yield u, "n_judge_calls"
            ip = extras.get("inpaint") or {}
            u = ip.get("judge_usage")
            if u:
                yield u, "n_judge_calls"
            for cand in (ip.get("candidates") or {}).values():
                u = (cand or {}).get("polygon_usage")
                if u:
                    yield u, "n_polygon_calls"
            u = ip.get("pick_usage")
            if u:
                yield u, "n_pick_calls"

    for u, count_key in _walk_inpaint_usages(result["parts"]):
        _accumulate_inpaint(u, count_key=count_key)

    (out_dir / "gemini.json").write_text(json.dumps({
        "model": decomposer.gemini_model,
        "n_decompose_calls": len(decompose_calls),
        "total_decompose_usage": total_usage,
        "inpaint_model": getattr(decomposer, "inpaint_model", None),
        "total_inpaint_usage": inpaint_usage_total,
        "decompose_calls": decompose_calls,
    }, indent=2, ensure_ascii=False))

    _save_overlay(out_dir / "overlay.png", result["img_for_vlm"], result["parts"])

    qual_counts = {"good": 0, "bad": 0, "no_info": 0}
    bad_parts: list[tuple[str, str, str]] = []
    for p in result["parts"]:
        q = (p.extra or {}).get("sam_quality")
        if q == "good":
            qual_counts["good"] += 1
        elif q == "bad":
            qual_counts["bad"] += 1
            bad_parts.append((p.id, p.label,
                              p.extra.get("sam_quality_reason", "")))
        else:
            qual_counts["no_info"] += 1
    if qual_counts["good"] or qual_counts["bad"]:
        print(f"[sam_quality] good={qual_counts['good']}  "
              f"bad={qual_counts['bad']}  no_info={qual_counts['no_info']}")
        for pid, lab, reason in bad_parts:
            print(f"  bad: {pid} '{lab}'  ({reason})")
        result["sam_quality_summary"] = qual_counts

    _save_segments_json(
        out_dir / "segments.json",
        result["parts"],
        result["tree_parts"],
        result,
    )

    # Amodal inpaint (optional): per-part Gemini-polygon + FLUX-fill completion
    # on every non-rejected part. Runs AFTER decomposer.decompose(). Results
    # live under part.extra["inpaint"] and feed a parallel vectorize+merge
    # below — they never touch the original layer_image, so merged.svg stays
    # exactly as it was before.
    amodal_stats: dict | None = None
    t_amodal = 0.0
    if amodal_inpaint:
        n_targets = sum(
            1 for p in result["parts"] if not p.rejected
        )
        amodal_model = getattr(decomposer, "inpaint_model", None)
        amodal_level = getattr(decomposer, "inpaint_thinking_level", None)
        print(f"[amodal] running on {n_targets} parts "
              f"(device={amodal_device}, model={amodal_model}, "
              f"thinking={amodal_level}, skip_judge={amodal_skip_judge}) ...")
        scene_rgb = image.convert("RGB")
        t0 = time.time()
        amodal_stats = _run_amodal_inpaint(
            result["parts"], scene_rgb, out_dir, amodal_device, debug,
            gemini_model=amodal_model,
            thinking_level=amodal_level,
            skip_judge=amodal_skip_judge,
            polygon_workers=amodal_polygon_workers,
            seed=amodal_seed,
        )
        t_amodal = time.time() - t0
        print(f"[amodal] processed={amodal_stats['n_processed']} "
              f"skipped={amodal_stats['n_skipped']} "
              f"parse_errors={amodal_stats['n_parse_errors']} "
              f"call_errors={amodal_stats['n_call_errors']} "
              f"({t_amodal:.1f}s)")

        # Re-serialize segments.json so the amodal-pass fields
        # captured by ``_run_amodal_inpaint`` actually make it to disk —
        # the first call happened BEFORE the amodal pass and thus saw
        # an empty ``part.extra["inpaint"]`` for every part.
        if amodal_stats is not None:
            _save_segments_json(
                out_dir / "segments.json",
                result["parts"], result["tree_parts"], result,
            )
            # Same reason: re-aggregate inpaint usage and rewrite
            # gemini.json now that part.extra["inpaint"]["judge_usage"]
            # / ["polygon_usage"] are populated. Without this the totals
            # silently drop every amodal-stage Gemini call.
            for k in ("prompt_tokens", "candidates_tokens",
                      "thoughts_tokens", "total_tokens",
                      "n_judge_calls", "n_polygon_calls",
                      "n_pick_calls"):
                inpaint_usage_total[k] = 0
            inpaint_usage_total["cost_usd"] = 0.0
            for u, count_key in _walk_inpaint_usages(result["parts"]):
                _accumulate_inpaint(u, count_key=count_key)
            (out_dir / "gemini.json").write_text(json.dumps({
                "model": decomposer.gemini_model,
                "n_decompose_calls": len(decompose_calls),
                "total_decompose_usage": total_usage,
                "inpaint_model": getattr(decomposer, "inpaint_model", None),
                "total_inpaint_usage": inpaint_usage_total,
                "decompose_calls": decompose_calls,
            }, indent=2, ensure_ascii=False))

    # Per-part: dump layer PNG, vtracer SVG, rendered PNG.
    # Rejected parts appear in the tree viz but are NOT vectorized.
    # Modes: ``cutout`` (legacy default → unsuffixed filenames) and/or
    # ``stacked`` (``_stacked`` suffix). When ``vtracer_modes`` is None,
    # produce both; when given a single-element tuple, produce only that.
    canvas_size = image.size
    base_vp = params or VectorizeParams()
    modes = tuple(vtracer_modes) if vtracer_modes else ("cutout", "stacked")
    vp_by_mode = {
        m: dataclasses.replace(base_vp, hierarchical=m) for m in modes
    }
    # cutout keeps the legacy filename (``{stem}.svg``); stacked uses a
    # ``_stacked`` suffix so both can coexist in the same parts/ dir.
    def _mode_suffix(mode: str) -> str:
        return "" if mode == "cutout" else f"_{mode}"

    t_vec_total = 0.0
    for part in result["parts"]:
        if part.rejected:
            continue
        stem = f"{part.id}_{_safe_name(part.label)}"
        png_path = parts_dir / f"{stem}.png"
        part.layer_image.save(png_path)
        for mode in modes:
            sfx = _mode_suffix(mode)
            svg_path = parts_dir / f"{stem}{sfx}.svg"
            render_path = parts_dir / f"{stem}{sfx}_render.png"
            t0 = time.time()
            vectorize_layer(png_path, svg_path, vp_by_mode[mode])
            t_vec_total += time.time() - t0
            render_svg(svg_path, render_path, size=canvas_size)

    # Amodal vectorize: same per-part flow, but reads part.extra["inpaint"]
    # ["fill_rgba"]. Parts without an amodal entry (root / rejected / failed
    # call) fall back to layer_image so the _amodal output set stays a
    # superset-compatible mirror of the original.
    t_vec_amodal = 0.0
    if amodal_inpaint:
        for part in result["parts"]:
            if part.rejected:
                continue
            inpaint = (part.extra or {}).get("inpaint") or {}
            fill = inpaint.get("fill_rgba", part.layer_image)
            stem = f"{part.id}_{_safe_name(part.label)}_amodal"
            png_path = parts_dir / f"{stem}.png"
            fill.save(png_path)
            for mode in modes:
                sfx = _mode_suffix(mode)
                svg_path = parts_dir / f"{stem}{sfx}.svg"
                render_path = parts_dir / f"{stem}{sfx}_render.png"
                t0 = time.time()
                vectorize_layer(png_path, svg_path, vp_by_mode[mode])
                t_vec_amodal += time.time() - t0
                render_svg(svg_path, render_path, size=canvas_size)

    # Merge: effective leaves only — flat decompose means every accepted
    # part is its own leaf and contributes one layer to the merged SVG.
    # Z-order follows leaf order (= Gemini decompose response order, which
    # the prompt instructs to emit back-to-front). Residual leaves the v15
    # background judge tagged ``is_background_layer=True`` get a sentinel
    # -inf so they paint ABSOLUTELY first regardless of leaf order — these
    # are the scene background plates (sky, pink circle, sand) that
    # everything else should sit on top of.
    leaves = _collect_effective_leaves(result["tree_parts"])
    for mode in modes:
        sfx = _mode_suffix(mode)
        items: list[tuple[float, str, str, Path]] = []
        for idx, leaf in enumerate(leaves):
            stem = f"{leaf.id}_{_safe_name(leaf.label)}"
            svg_path = parts_dir / f"{stem}{sfx}.svg"
            is_bg = bool((leaf.extra or {}).get("is_background_layer"))
            sort_key = float("-inf") if is_bg else float(idx)
            items.append((sort_key, leaf.id, leaf.label, svg_path))

        merged_svg = out_dir / f"merged{sfx}.svg"
        if _merge_svgs(merged_svg, canvas_size, items):
            merged_png = out_dir / f"merged{sfx}.png"
            render_svg(merged_svg, merged_png, size=canvas_size)
            # cairosvg always writes RGBA; the merged SVG already has a white
            # background baked in, so flattening is just dropping the (all-255)
            # alpha channel — leaves an RGB image with the same visible content.
            _rgb_flatten = Image.open(merged_png)
            if _rgb_flatten.mode != "RGB":
                _flatten_on_white(_rgb_flatten).save(merged_png)
            _rgb_flatten.close()

    # Amodal merge: parallel stitch using each leaf's *_amodal.svg (falls
    # back to the visible SVG when no amodal SVG exists — i.e. a leaf
    # whose inpaint call errored or was skipped). Same z-order as the
    # visible merge so the only difference vs merged.png is which silhouette
    # each layer occupies.
    if amodal_inpaint:
        for mode in modes:
            sfx = _mode_suffix(mode)
            amodal_items: list[tuple[float, str, str, Path]] = []
            for idx, leaf in enumerate(leaves):
                stem = f"{leaf.id}_{_safe_name(leaf.label)}"
                visible_svg = parts_dir / f"{stem}{sfx}.svg"
                amodal_svg = parts_dir / f"{stem}_amodal{sfx}.svg"
                svg_path = amodal_svg if amodal_svg.exists() else visible_svg
                is_bg = bool((leaf.extra or {}).get("is_background_layer"))
                sort_key = float("-inf") if is_bg else float(idx)
                amodal_items.append((sort_key, leaf.id, leaf.label, svg_path))

            merged_amodal_svg = out_dir / f"merged_amodal{sfx}.svg"
            if _merge_svgs(merged_amodal_svg, canvas_size, amodal_items):
                merged_amodal_png = out_dir / f"merged_amodal{sfx}.png"
                render_svg(merged_amodal_svg, merged_amodal_png, size=canvas_size)
                _rgb_flatten = Image.open(merged_amodal_png)
                if _rgb_flatten.mode != "RGB":
                    _flatten_on_white(_rgb_flatten).save(merged_amodal_png)
                _rgb_flatten.close()

    if debug:
        _save_parts_tree(
            out_dir / "parts_tree.png",
            result["img_for_vlm"],
            result["tree_parts"],
        )
        if amodal_inpaint:
            _save_parts_tree_amodal(
                out_dir / "parts_tree_amodal.png",
                result["img_for_vlm"],
                result["tree_parts"],
            )
        _save_gemini_bbox_viz(
            out_dir / "debug" / "gemini_bbox.png",
            result["img_for_vlm"],
            decompose_calls,
        )
        # Tree viz keyed on Gemini bbox-rect mask per node (parallel format
        # to parts_tree, different thumbnail source).
        _save_gemini_bbox_tree_viz(
            out_dir / "debug" / "gemini_bbox_tree.png",
            result["img_for_vlm"],
            result["tree_parts"],
        )
        _save_gemini_bbox_crops(
            out_dir / "debug" / "bbox_crops",
            result["img_for_vlm"],
            result["tree_parts"],
        )
        # Tree viz keyed on the post-hoc SAM mask per node. No-op outside
        # v5 (when no node has mask_source=='sam').
        _save_sam_masks_tree_viz(
            out_dir / "debug" / "sam_masks_tree.png",
            result["img_for_vlm"],
            result["tree_parts"],
        )

    metrics_dict: dict | None = None
    metrics_time = 0.0
    if eval_models is not None and (out_dir / "merged.png").exists():
        # Compare GT (input.png) vs reconstructed (merged.png) — values are
        # rounded to 3 decimals inside `evaluate_sample` so summary.json and
        # metrics.json stay consistent.
        t0 = time.time()
        sm = _evaluate_sample(
            out_dir, eval_models.get("lpips"), eval_models.get("dino"),
            set(EVAL_METRICS),
        )
        metrics_time = time.time() - t0
        metrics_dict = {m: getattr(sm, m) for m in EVAL_METRICS}

    total_cost = (float(total_usage.get("cost_usd", 0.0))
                  + float(inpaint_usage_total.get("cost_usd", 0.0)))
    summary = {
        "image": str(image_path),
        "n_parts": len(result["parts"]),
        "metrics": metrics_dict,
        "time": {
            "decompose_s": t_decompose,
            "vectorize_s": t_vec_total,
            "metrics_s": metrics_time if eval_models is not None else None,
        },
        "cost_usd": {
            "decompose": round(float(total_usage.get("cost_usd", 0.0)), 6),
            "inpaint_judge_polygon": round(
                float(inpaint_usage_total.get("cost_usd", 0.0)), 6,
            ),
            "total": round(total_cost, 6),
        },
        "n_calls": {
            "decompose": len(decompose_calls),
            "judge": int(inpaint_usage_total.get("n_judge_calls", 0)),
            "polygon": int(inpaint_usage_total.get("n_polygon_calls", 0)),
            "pick": int(inpaint_usage_total.get("n_pick_calls", 0)),
        },
    }
    if amodal_inpaint:
        summary["amodal"] = {
            "enabled": True,
            **(amodal_stats or {}),
            "device": amodal_device,
            "seed": amodal_seed,
            "time_inpaint_s": round(t_amodal, 2),
            "time_vectorize_s": round(t_vec_amodal, 2),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_dir


def _expand_inputs(tokens: list[str], limit: int | None) -> list[Path]:
    """Expand file / directory / glob tokens into a sorted list of PNG paths."""
    import glob
    paths: list[Path] = []
    for t in tokens:
        p = Path(t)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.png")))
        elif any(ch in t for ch in "*?["):
            paths.extend(sorted(Path(m) for m in glob.glob(t)))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[warn] input not found: {t}")
    # dedupe preserving order
    seen = set()
    uniq = []
    for p in paths:
        ap = p.resolve()
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(p)
    if limit is not None and limit > 0:
        uniq = uniq[:limit]
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+",
                    help="Image file(s), directory, or glob pattern. "
                         "Directories are expanded to their *.png children.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Single input: run dir. "
                         "Multiple inputs: parent batch dir with per-image subdirs.")
    ap.add_argument("--no-debug", dest="debug", action="store_false",
                    help="Skip debug artifacts. Default is ON: every "
                         "run dumps parts_tree.png, gemini_bbox*.png, "
                         "sam_masks_tree.png, debug/ images, and (in "
                         "batch mode) <batch>/grids/ pages.")
    ap.set_defaults(debug=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of images to process (for batch runs).")
    ap.add_argument("--gemini-model", default=None,
                    help="Override the VLM model id (default: Decomposer's "
                         "built-in default, currently gemini-3-flash-preview). "
                         "Pass a Qwen* id (e.g. Qwen3.6-35B-A3B) to route "
                         "through the vLLM endpoint at $QWEN_BASE_URL "
                         "(default http://localhost:8000/v1) instead.")
    ap.add_argument("--bbox-only", action="store_true",
                    help="Skip SAM3; use Gemini bboxes as rectangular masks "
                         "(clipped to parent). Sanity baseline for measuring "
                         "SAM3's contribution.")
    ap.add_argument("--bbox-sam", action="store_true",
                    help="v5 hybrid: Gemini-only tree (like --bbox-only), then "
                         "post-hoc SAM3 on every non-root node using its "
                         "(label, bbox). Implies --bbox-only + post-hoc SAM.")
    ap.add_argument("--gemini-cache", type=str, default=None,
                    help="Path to a previous run dir whose gemini.json files "
                         "are replayed instead of calling the Gemini API. "
                         "Per-image lookup: <root>/<stem>/gemini.json (batch) "
                         "or <root>/gemini.json (single). Mismatched tree "
                         "context aborts so cache divergence is loud.")
    ap.add_argument("--no-bbox-sam-stability", dest="bbox_sam_stability",
                    action="store_false",
                    help="Disable the v8 successor pipeline (default ON): "
                         "Gemini-only tree, then SAM3 grounded with "
                         "(label, gemini_bbox) on each leaf replaces the "
                         "bbox-rect. ``assess_sam_quality`` (sb_05/sb_20 "
                         "thresholds fit on sweep_v11) tags leaves "
                         "good/bad; bad leaves enter the --bad-recover-* "
                         "path (color recovery is also default ON). "
                         "Auto-disabled when --bbox-sam is set "
                         "(mutually exclusive).")
    ap.set_defaults(bbox_sam_stability=True)
    ap.add_argument("--no-bad-recover-color", dest="bad_recover_color",
                    action="store_false",
                    help="Disable color recovery (default ON when "
                         "bbox-sam-stability is on): exp5 color-palette + "
                         "edge-split pipeline that re-derives the mask "
                         "for leaves flagged sam_quality=='bad' (uses the "
                         "SAM prob map for palette + Sobel edges, "
                         "produces a clean per-colour blob).")
    ap.set_defaults(bad_recover_color=True)
    ap.add_argument("--bad-recover-edge-thresh", type=float, default=0.30,
                    help="Normalized |∇prob| cutoff for the edge-split "
                         "stage of --bad-recover-color (default 0.30).")
    ap.add_argument("--bad-recover-min-split-area", type=int, default=30,
                    help="Drop edge-split fragments smaller than this "
                         "pixel count in --bad-recover-color (default 30).")
    ap.add_argument("--decompose-thinking-budget", type=int, default=0,
                    help="Gemini-3 'thinking' budget for the decompose call "
                         "ONLY (default 0 = off). Set to a positive int (or "
                         "-1 for 'auto') when bbox-tree quality matters more "
                         "than cost — judge/audit/inpaint stay at 0. "
                         "Thinking tokens are billed at the output rate but "
                         "are NOT surfaced in usage_metadata.")
    ap.add_argument("--residual-margin-threshold", type=float, default=0.15,
                    help="Threshold on per-pixel (top1 − top2) sam_prob "
                         "margin used by the residual handler. Pixels "
                         "left uncovered by any leaf are awarded to "
                         "their argmax-winner leaf if their margin is "
                         "≥ this value; the rest collect into a single "
                         "'residual' leaf at root level. Higher = more "
                         "pixels stay in the residual leaf (default 0.15).")
    ap.add_argument("--amodal-inpaint", action="store_true",
                    help="Posthoc: per-part Gemini-polygon + FLUX-Fill amodal "
                         "completion on every non-root, non-rejected node. "
                         "Produces amodal per-part SVGs alongside the "
                         "visible-mask SVGs; the visible-mask merge "
                         "(merged.svg) is unaffected. Requires "
                         "GEMINI_API_KEY and ~15 GB GPU for FLUX.1-Fill-dev.")
    ap.add_argument("--amodal-device", default="cuda:0",
                    help="CUDA device for FLUX.1-Fill-dev (default cuda:0). "
                         "Use a different GPU than SAM3/Qwen if OOM.")
    ap.add_argument("--amodal-seed", type=int, default=None,
                    help="Diffusion seed for the FLUX.1-Fill amodal pass "
                         "(default: the module constant, 42). Vary it to draw "
                         "different completions of the same occluded part -- "
                         "everything else in the pipeline is deterministic, so "
                         "this is the only knob that re-rolls the fill.")
    ap.add_argument("--amodal-polygon-workers", type=int, default=4,
                    help="Parallel ThreadPoolExecutor slots for the "
                         "amodal Gemini polygon stage (default 4). "
                         "Higher = faster but risk Gemini rate-limit. "
                         "Set 1 to disable parallelism.")
    ap.add_argument("--amodal-skip-judge", action="store_true",
                    help="Skip the amodal-pass judge step — call Gemini "
                         "polygon on EVERY non-rejected leaf, then let "
                         "build_repaint_mask (polygon ∧ ¬visible ∧ "
                         "scene_non_white) decide what FLUX repaints. "
                         "Use when you want unconditional amodal "
                         "completion rather than judge-gated occlusion.")
    ap.add_argument("--decompose-mode", choices=("v13", "v15"),
                    default="v15",
                    help="v15 (default): per-node Gemini→SAM→margin-residual"
                         "→residual-only judge→recurse. Reproduces sample10 "
                         "results with no other flags. "
                         "v13 (legacy): flat decompose + post-hoc SAM "
                         "stability + bad-quality color recovery.")
    ap.add_argument("--inpaint-model", default="gemini-3-flash-preview",
                    help="VLM used for v15 residual judge + amodal "
                         "polygon/judge/pick calls (default "
                         "gemini-3-flash-preview). Pass a Qwen* id "
                         "(e.g. Qwen3.6-35B-A3B) to route through the "
                         "vLLM endpoint at $QWEN_BASE_URL instead; add "
                         "new ids to PRICING in modules/gemini_client.py.")
    ap.add_argument("--inpaint-thinking-level",
                    choices=("minimal", "low", "medium", "high"),
                    default="medium",
                    help="Gemini thinking_level for --inpaint-model "
                         "(default medium). Accepts MINIMAL/LOW/MEDIUM/HIGH "
                         "on both gemini-3 and gemini-3.5 families.")
    ap.add_argument("--vtracer-mode", choices=("cutout", "stacked"),
                    default=None,
                    help="vtracer hierarchical mode. When unset (default), "
                         "produce BOTH merged.svg (cutout) and "
                         "merged_stacked.svg. When set, produce only the "
                         "chosen variant: cutout→merged.svg, "
                         "stacked→merged_stacked.svg.")
    args = ap.parse_args()

    image_paths = _expand_inputs(args.images, args.limit)
    if not image_paths:
        print("[error] no input images resolved")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    # --bbox-sam-stability defaults ON; --bbox-sam (legacy v5) is opt-in and
    # mutually exclusive. When the user opts into --bbox-sam, silently flip
    # stability off so the legacy path runs without requiring the user to
    # also pass --no-bbox-sam-stability.
    if args.bbox_sam:
        args.bbox_sam_stability = False
    # Both are supersets of --bbox-only: same Gemini-only tree, but a
    # different post-hoc segmenter on every leaf.
    post_hoc_segmenter = "sam_stability" if args.bbox_sam_stability else None
    decomposer_kwargs: dict = {
        "post_hoc_sam": bool(args.bbox_sam),
        "post_hoc_segmenter": post_hoc_segmenter,
        "bad_recover_color": bool(args.bad_recover_color),
        "bad_recover_edge_thresh": float(args.bad_recover_edge_thresh),
        "bad_recover_min_split_area": int(args.bad_recover_min_split_area),
        "decompose_thinking_budget": int(args.decompose_thinking_budget),
        "residual_margin_threshold": float(args.residual_margin_threshold),
        "decompose_mode": args.decompose_mode,
        "inpaint_model": args.inpaint_model,
        "inpaint_thinking_level": args.inpaint_thinking_level,
        "inpaint_device": args.amodal_device,
    }
    if args.gemini_model:
        decomposer_kwargs["gemini_model"] = args.gemini_model
    shared_decomposer = Decomposer(**decomposer_kwargs)  # load SAM / Gemini once across batch

    eval_models: dict | None = None

    cache_root = Path(args.gemini_cache) if args.gemini_cache else None
    vtracer_modes = (args.vtracer_mode,) if args.vtracer_mode else None

    if len(image_paths) == 1:
        image_path = image_paths[0]
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            out_dir = Path("runs") / f"{ts}_{image_path.stem}"
        result_dir = run(
            image_path, out_dir, debug=args.debug,
            decomposer=shared_decomposer,
            eval_models=eval_models,
            gemini_cache_root=cache_root,
            amodal_inpaint=bool(args.amodal_inpaint),
            amodal_device=args.amodal_device,
            amodal_skip_judge=bool(args.amodal_skip_judge),
            amodal_polygon_workers=int(args.amodal_polygon_workers),
            amodal_seed=args.amodal_seed,
            vtracer_modes=vtracer_modes,
        )
        print(f"[done] artifacts at {result_dir}")
        return

    # Batch mode
    batch_root = Path(args.out_dir) if args.out_dir else Path("runs") / f"{ts}_batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    print(f"[batch] {len(image_paths)} images → {batch_root}")
    index: list[dict] = []
    for i, image_path in enumerate(image_paths, 1):
        out_dir = batch_root / image_path.stem
        print(f"[batch] ({i}/{len(image_paths)}) {image_path.name}")
        try:
            run(image_path, out_dir, debug=args.debug,
                decomposer=shared_decomposer, eval_models=eval_models,
                gemini_cache_root=cache_root,
                amodal_inpaint=bool(args.amodal_inpaint),
                amodal_device=args.amodal_device,
                amodal_skip_judge=bool(args.amodal_skip_judge),
                amodal_polygon_workers=int(args.amodal_polygon_workers),
                amodal_seed=args.amodal_seed,
                vtracer_modes=vtracer_modes)
            index.append({"image": str(image_path), "out": str(out_dir),
                          "status": "ok"})
        except Exception as e:
            print(f"[batch] ERROR on {image_path}: {type(e).__name__}: {e}")
            index.append({"image": str(image_path), "out": str(out_dir),
                          "status": "error", "error": f"{type(e).__name__}: {e}"})
    (batch_root / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False)
    )
    print(f"[batch] done → {batch_root}/index.json")
    # --debug: write paginated 4×1 parts_tree grids to <batch>/grids/ so
    # the whole sweep can be eyeballed without opening 40 windows.
    if args.debug:
        try:
            sys.path.insert(0, str(Path(__file__).parent / "scripts"))
            from viz_trees_grid import generate as _grid_generate
            _grid_generate(batch_root, cols=4, rows=1)
        except Exception as e:
            print(f"[batch] grid viz skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
