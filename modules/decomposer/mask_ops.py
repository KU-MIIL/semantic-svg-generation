"""Pure image/mask utilities used across the decomposition pipeline."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _rgba_on_white(img_rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
    bg.paste(img_rgba, mask=img_rgba.split()[-1])
    return bg


def _clean_mask(
    mask: np.ndarray,
    close_kernel: int = 3,
    smooth_epsilon_px: float = 1.5,
    small_hole_ratio: float = 0.02,
) -> np.ndarray:
    """Fill sub-threshold holes and smooth boundaries via contour extraction."""
    import cv2

    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return mask
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    out = np.zeros_like(m)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    def _smooth(c):
        if smooth_epsilon_px > 0 and len(c) >= 4:
            return cv2.approxPolyDP(c, smooth_epsilon_px, True)
        return c

    outer_area: dict[int, float] = {}
    for i, h in enumerate(hierarchy):
        if h[3] == -1:
            c = _smooth(contours[i])
            cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
            outer_area[i] = float(cv2.contourArea(contours[i]))

    for i, h in enumerate(hierarchy):
        parent = h[3]
        if parent == -1:
            continue
        parent_area = outer_area.get(parent, 0.0)
        hole_area = float(cv2.contourArea(contours[i]))
        if parent_area > 0 and hole_area / parent_area >= small_hole_ratio:
            c = _smooth(contours[i])
            cv2.drawContours(out, [c], -1, 0, thickness=cv2.FILLED)

    return out.astype(bool)


def _apply_mask_rgba(img_rgba: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(img_rgba.convert("RGBA"))
    arr[..., 3] = np.where(mask, arr[..., 3], 0).astype(np.uint8)
    return Image.fromarray(arr)


def _bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
