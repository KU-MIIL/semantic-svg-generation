"""Mean squared error between two RGB image arrays."""
from __future__ import annotations

import numpy as np


def compute(a: np.ndarray, b: np.ndarray) -> float:
    """MSE over float32 RGB arrays in [0, 1]. Lower = better."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float((diff * diff).mean())


__all__ = ["compute"]
