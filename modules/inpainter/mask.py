"""Rasterize the Gemini polygon into a binary mask and derive the FLUX
repaint mask (= polygon AND NOT visible AND scene-non-white).

These two helpers run on the CPU between the Gemini polygon call and the
FLUX inference. Keeping them pure-numpy makes it easy to swap the
upstream polygon source (e.g. for an ablation) without touching FLUX —
the only contract FLUX cares about is ``build_repaint_mask``'s output.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .imaging import WHITE_THRESHOLD

# Repaint-mask dilation in pixels. 0 = exact polygon ∧ ¬visible ∧
# scene-non-white. A small positive value blurs the boundary by that many
# pixels so FLUX doesn't leave a hairline seam at the visible/repaint
# join; in practice the v08 FLUX prompt already blends cleanly so we ship
# at 0. If you bump it, the dilation is clipped back to the union of
# visible-or-non-white pixels so FLUX never paints over true background.
DILATE_PX = 0


def polygon_to_mask(polygon: list[tuple[int, int]], W: int, H: int) -> np.ndarray:
    """Fill the closed polygon into a uint8 (0/255) mask of size (H, W).

    Returns an all-zero mask when ``polygon`` has fewer than 3 vertices
    (the same degenerate-input contract as :func:`predict_complete_polygon`,
    which also emits an empty list on parse failure).
    """
    import cv2
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def build_repaint_mask(polygon_mask: np.ndarray, target_alpha: np.ndarray,
                       scene_rgb: Image.Image) -> np.ndarray:
    """Pixels FLUX is allowed to repaint — the genuinely occluded region.

    Three-way AND of:
      ``polygon_mask``        — Gemini's predicted full silhouette of the
                                 part (rasterized polygon).
      ``NOT target_alpha>0``  — exclude pixels already drawn in this
                                 part's RGBA cutout; FLUX must not
                                 overpaint the visible content.
      ``scene_rgb non-white`` — the occluder has to actually exist in
                                 IMAGE 1; if the polygon spills onto
                                 white background, that's the model
                                 hallucinating beyond the scene and we
                                 don't want FLUX to draw over white.

    Optionally dilated by ``DILATE_PX`` (then re-clipped to the
    visible-or-non-white region so the dilation can't escape the scene).
    Returns uint8 (0/255) — the array FLUX consumes as its ``mask_image``.
    """
    import cv2
    visible = target_alpha > 0
    raw = (polygon_mask > 0) & ~visible
    inp = np.array(scene_rgb.convert("RGB"))
    nonwhite = ~(inp >= WHITE_THRESHOLD).all(axis=2)
    raw = raw & nonwhite
    if DILATE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_PX * 2 + 1, DILATE_PX * 2 + 1))
        raw = cv2.dilate(raw.astype(np.uint8), k, 1).astype(bool)
        raw = raw & (nonwhite | visible)
    return raw.astype(np.uint8) * 255
