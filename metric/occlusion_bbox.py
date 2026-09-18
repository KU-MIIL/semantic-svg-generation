"""Per-anchor occlusion-bbox utilities used as a supplementary score
alongside the union-bbox MSE in :mod:`metric.semantic_flatten_v2` /
:mod:`metric.semantic`.

Two modes are supported:

* ``"occluded"`` (default, V2) — precise. Renders the GT SVG with the
  anchor's drawable paths recoloured to a unique marker color (preserves
  z-order). Pixels visible as the marker in that render are the parts of
  the anchor that *are on top*; pixels where the anchor's amodal mask is
  set but the marker is absent are the parts that other paths cover. The
  bbox tightens to just the genuinely-hidden region.

* ``"contested"`` (legacy, V1) — fast/inclusive. Pixels where the anchor
  and the complement (the rest of the GT) *both* would draw. Over-
  counts visible-on-top overlap as occluded; kept for backwards
  compatibility and ablations.

Pick "occluded" for paper-grade reporting where we want to highlight
methods that recover hidden pixels (inpainting vs. raw tracing).
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np

# NB: ``render_subset_rgb`` is imported lazily inside
# :func:`compute_anchor_occlusion` to avoid a circular import —
# :mod:`metric.semantic` itself imports this module to add the occlusion-
# aware supplementary recall.

# Marker color used by ``render_with_anchor_marker`` for occluded-only
# mask extraction. Pure magenta is rare in real SVG palettes and easy to
# threshold via ``_marker_visible_mask`` even at anti-aliased edges.
_DEFAULT_MARKER_RGB: tuple[float, float, float] = (1.0, 0.0, 1.0)


@dataclass
class AnchorOcclusion:
    """Cached artifacts for one anchor's occluded/contested region.

    ``amodal`` is the anchor's stand-alone render. ``aux_render`` is mode-
    dependent: in ``occluded`` mode it's the marker-color render of the
    full GT (anchor recoloured, others kept), and in ``contested`` mode
    it's the GT complement (all paths except the anchor). ``mask`` is the
    bool H×W mask of pixels counted as "in scope" — truly hidden pixels in
    occluded mode, spatial overlap in contested mode. ``bbox`` is the
    tight (y0, x0, y1, x1) half-open bbox or ``None`` when the mask is
    empty. ``mode`` is ``"occluded"`` or ``"contested"``.
    """
    amodal: np.ndarray
    aux_render: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    mode: str = "occluded"

    # Backwards-compat aliases — older call-sites used these names.
    @property
    def complement(self) -> np.ndarray:
        return self.aux_render

    @property
    def contested_mask(self) -> np.ndarray:
        return self.mask


def _content_mask(arr: np.ndarray, *, white_eps: float = 0.01) -> np.ndarray:
    """Bool H×W mask of non-white pixels (any channel < 1 - eps)."""
    return (arr < (1.0 - white_eps)).any(axis=-1)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """Convert a [0, 1] RGB triple to ``#rrggbb`` hex (rounded)."""
    return "#{:02x}{:02x}{:02x}".format(
        int(round(rgb[0] * 255)),
        int(round(rgb[1] * 255)),
        int(round(rgb[2] * 255)),
    )


def render_with_anchor_marker(
    svg_text: str,
    anchor_paths: Iterable[int],
    *,
    marker_rgb: tuple[float, float, float] = _DEFAULT_MARKER_RGB,
    size: int = 256,
) -> np.ndarray:
    """Render the full SVG with the anchor's drawable paths recoloured to
    ``marker_rgb``; everything else keeps its original fill/stroke.

    z-order is preserved (no paths are removed), so SVG painter's
    algorithm still decides who's on top at each pixel — letting us read
    "visible-anchor" pixels off the marker color directly.

    Patch policy per drawable in the anchor set:
      - ``fill="none"`` → preserved (stroke-only paths stay stroke-only).
      - ``fill="<color>"`` (or default) → set to marker.
      - ``stroke="<color>"`` → set to marker (when explicitly set).
      - ``fill-opacity``/``stroke-opacity``/``opacity`` → forced to 1.

    Returns a float64 RGB array in [0, 1] shape ``(size, size, 3)``,
    composited on a white background (matches :mod:`metric.semantic`).
    """
    import cairosvg
    from PIL import Image as _Image

    # Lazy import to avoid circular dep with ``metric.semantic``.
    from metric.semantic import _DRAWABLE_TAGS, _strip_ns

    a_set = {int(p) for p in anchor_paths}
    marker_hex = _rgb_to_hex(marker_rgb)

    root = ET.fromstring(svg_text)
    idx = [0]

    def patch_drawable(el: ET.Element) -> None:
        fill = (el.get("fill") or "").strip().lower()
        stroke = (el.get("stroke") or "").strip().lower()
        # Preserve fill="none" (stroke-only paths). Otherwise force marker.
        if fill != "none":
            el.set("fill", marker_hex)
        # Only override stroke if it was explicitly set (SVG's default is none).
        if stroke and stroke != "none":
            el.set("stroke", marker_hex)
        # Force fully opaque so the marker is the topmost colour wherever
        # the painter draws.
        el.set("opacity", "1")
        if "fill-opacity" in el.attrib:
            el.set("fill-opacity", "1")
        if "stroke-opacity" in el.attrib:
            el.set("stroke-opacity", "1")

    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            tag = _strip_ns(child.tag)
            if tag in _DRAWABLE_TAGS:
                if idx[0] in a_set:
                    patch_drawable(child)
                idx[0] += 1
            else:
                walk(child)

    walk(root)
    svg_bytes = ET.tostring(root)
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=size,
        output_height=size,
    )
    rgba = _Image.open(BytesIO(png_bytes)).convert("RGBA")
    bg = _Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[-1])
    return np.asarray(bg, dtype=np.float64) / 255.0


def _marker_visible_mask(
    marker_render: np.ndarray,
    *,
    marker_rgb: tuple[float, float, float] = _DEFAULT_MARKER_RGB,
    margin: float = 0.2,
) -> np.ndarray:
    """Bool H×W mask of pixels whose colour is dominated by the marker.

    Handles AA edges where the marker blends with the background or with
    a same-side underlay. Generalises pure-channel markers like magenta
    (1, 0, 1): requires the "high" channels to stay high (> 0.5) and the
    "low" channels to be at least ``margin`` below the minimum of the
    high channels. Falls back to L-infinity distance for arbitrary markers.
    """
    high_idx = [i for i, v in enumerate(marker_rgb) if v > 0.5]
    low_idx = [i for i, v in enumerate(marker_rgb) if v <= 0.5]
    if not (high_idx and low_idx):
        return (np.abs(marker_render - np.asarray(marker_rgb)) < margin).all(
            axis=-1
        )
    chans = (
        marker_render[..., 0],
        marker_render[..., 1],
        marker_render[..., 2],
    )
    high_min = chans[high_idx[0]]
    for h in high_idx[1:]:
        high_min = np.minimum(high_min, chans[h])
    low_max = chans[low_idx[0]]
    for lo in low_idx[1:]:
        low_max = np.maximum(low_max, chans[lo])
    return (high_min > 0.5) & (low_max < (high_min - margin))


def contested_bbox(contested_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight bbox of True pixels as ``(y0, x0, y1, x1)`` half-open.
    Returns ``None`` when the mask is all-False."""
    if not contested_mask.any():
        return None
    ys, xs = np.where(contested_mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def compute_anchor_occlusion(
    gt_text: str,
    anchor_paths: Iterable[int],
    all_paths: Iterable[int],
    *,
    size: int = 256,
    white_eps: float = 0.01,
    amodal_arr: np.ndarray | None = None,
    mode: str = "occluded",
    marker_rgb: tuple[float, float, float] = _DEFAULT_MARKER_RGB,
) -> AnchorOcclusion:
    """Compute the per-anchor occlusion mask + bbox in one of two modes.

    Args:
        gt_text:        GT SVG source.
        anchor_paths:   Path indices belonging to the anchor (P_A).
        all_paths:      All drawable path indices in the GT (used only by
                          the contested mode).
        size:           Render resolution (square).
        white_eps:      Pixels with all RGB ≥ 1 - white_eps are background.
        amodal_arr:     Optional pre-rendered ``render_subset_rgb(gt_text,
                          P_A, size)``. Pass this when the caller already
                          rendered the anchor (saves one rasterization).
        mode:           ``"occluded"`` (default) → marker-color extraction,
                          bbox covers only pixels where the anchor is truly
                          hidden by other layers. ``"contested"`` → legacy
                          V1, bbox covers pixels where anchor and
                          complement spatially overlap (overcount: includes
                          parts where the anchor is on top).
        marker_rgb:     Marker color for occluded-mode rendering. Default
                          pure magenta.

    Returns:
        AnchorOcclusion. ``mask`` and ``bbox`` are mode-specific.
        ``aux_render`` carries the marker render (occluded mode) or the
        complement render (contested mode).
    """
    from metric.semantic import render_subset_rgb  # lazy: see module top

    if mode not in ("occluded", "contested"):
        raise ValueError(
            f"mode={mode!r} not supported; use 'occluded' or 'contested'"
        )

    a_set = {int(p) for p in anchor_paths}
    if amodal_arr is None:
        amodal_arr = render_subset_rgb(gt_text, a_set, size=size)
    amodal_mask = _content_mask(amodal_arr, white_eps=white_eps)

    if mode == "occluded":
        marker_render = render_with_anchor_marker(
            gt_text, a_set, marker_rgb=marker_rgb, size=size,
        )
        visible_mask = _marker_visible_mask(
            marker_render, marker_rgb=marker_rgb,
        )
        # Truly hidden: anchor would draw here, but the marker isn't on top.
        occluded_mask = amodal_mask & ~visible_mask
        return AnchorOcclusion(
            amodal=amodal_arr,
            aux_render=marker_render,
            mask=occluded_mask,
            bbox=contested_bbox(occluded_mask),
            mode="occluded",
        )

    # mode == "contested" — legacy V1
    others = {int(p) for p in all_paths} - a_set
    complement = render_subset_rgb(gt_text, others, size=size)
    c_mask = _content_mask(complement, white_eps=white_eps)
    contested = amodal_mask & c_mask
    return AnchorOcclusion(
        amodal=amodal_arr,
        aux_render=complement,
        mask=contested,
        bbox=contested_bbox(contested),
        mode="contested",
    )


def masked_mse(
    pred_arr: np.ndarray,
    anchor_arr: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean squared error restricted to ``mask`` pixels.

    Works on float64 RGB arrays in [0, 1] and a bool H×W mask. Returns
    NaN when the mask is empty. Unlike bbox-cropped MSE this isolates
    *exactly* the pixels of interest (occluded / contested), so a method
    that only differs from GT outside the mask doesn't contaminate the
    score — and one that only differs inside the mask doesn't get its
    error diluted by easy surrounding pixels.
    """
    if pred_arr.shape != anchor_arr.shape:
        raise ValueError(
            f"pred and anchor arrays must share shape, got "
            f"{pred_arr.shape} vs {anchor_arr.shape}"
        )
    if not mask.any():
        return float("nan")
    diff_sq = (pred_arr.astype(np.float64) - anchor_arr.astype(np.float64)) ** 2
    per_pixel = diff_sq.mean(axis=-1)
    return float(per_pixel[mask].mean())


def crop_to_fixed_bbox(
    anchor_arr: np.ndarray,
    pred_arr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop both arrays to ``bbox`` and square-pad with white.

    Mirrors the square-padding convention in
    :func:`metric.semantic_flatten_v2._crop_to_union_bbox` so the MSE
    numbers are comparable across bbox modes. Half-open ``bbox`` =
    ``(y0, x0, y1, x1)``.
    """
    if anchor_arr.shape != pred_arr.shape:
        raise ValueError(
            f"anchor and pred arrays must share shape, got "
            f"{anchor_arr.shape} vs {pred_arr.shape}"
        )
    y0, x0, y1, x1 = bbox
    a = anchor_arr[y0:y1, x0:x1]
    p = pred_arr[y0:y1, x0:x1]
    ch, cw = a.shape[:2]
    if ch == 0 or cw == 0:
        return a, p
    if ch == cw:
        return a, p
    side = max(ch, cw)
    oy = (side - ch) // 2
    ox = (side - cw) // 2
    pad_a = np.ones((side, side, 3), dtype=a.dtype)
    pad_p = np.ones((side, side, 3), dtype=p.dtype)
    pad_a[oy:oy + ch, ox:ox + cw] = a
    pad_p[oy:oy + ch, ox:ox + cw] = p
    return pad_a, pad_p


__all__ = [
    "AnchorOcclusion",
    "compute_anchor_occlusion",
    "contested_bbox",
    "crop_to_fixed_bbox",
    "masked_mse",
    "render_with_anchor_marker",
]
