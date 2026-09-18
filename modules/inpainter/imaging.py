"""RGBA / RGB conversion helpers shared by the polygon, mask, and FLUX
inference steps of the inpaint pipeline.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

WHITE_THRESHOLD = 245       # >= this on all channels => "white/empty"


def rgba_on_white(rgba: Image.Image) -> Image.Image:
    """Composite an RGBA layer over white. Used wherever Gemini / FLUX
    needs an opaque scene with no alpha channel."""
    arr = np.array(rgba.convert("RGBA"))
    rgb = arr[..., :3].astype(np.float32)
    a = arr[..., 3:4].astype(np.float32) / 255.0
    out = rgb * a + 255.0 * (1.0 - a)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGB")


def rgb_to_rgba_white_alpha(rgb: Image.Image,
                            thr: int = WHITE_THRESHOLD) -> Image.Image:
    """Re-introduce an alpha channel from an RGB image by treating any
    near-white pixel as transparent. Used to convert FLUX's RGB output
    back into the project's RGBA Part convention."""
    arr = np.array(rgb.convert("RGB"))
    is_white = (arr >= thr).all(axis=2)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = arr
    rgba[..., 3] = np.where(is_white, 0, 255)
    return Image.fromarray(rgba, "RGBA")
