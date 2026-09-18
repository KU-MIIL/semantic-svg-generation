"""Structural similarity via scikit-image."""
from __future__ import annotations

import numpy as np


def compute(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM over float32 RGB arrays in [0, 1]. Higher = better (typ. [0, 1])."""
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


__all__ = ["compute"]
