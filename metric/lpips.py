"""LPIPS perceptual distance via the ``lpips`` pypi package.

The pypi ``lpips`` module is imported inside :func:`load` — no collision
with this file's dotted path ``metric.lpips`` because ``import lpips``
resolves against top-level ``sys.path``, not the containing package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class LpipsModel:
    net: Any
    device: str


def load(device: str = "cpu", net: str = "alex") -> LpipsModel:
    """Instantiate the LPIPS network (default AlexNet backbone)."""
    import lpips as lpips_pkg

    print(f"[lpips] loading net={net} on {device}")
    model = lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return LpipsModel(net=model, device=device)


def _to_tensor(img: Image.Image, device: str):
    """PIL RGB → torch.Tensor shape (1, 3, H, W) in [-1, 1]."""
    import torch

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)                    # HWC → CHW
    t = torch.from_numpy(arr).unsqueeze(0)          # (1, 3, H, W)
    t = t * 2.0 - 1.0
    return t.to(device)


def compute(model: LpipsModel, a: Image.Image, b: Image.Image) -> float:
    """LPIPS distance between two PIL RGB images. Lower = more similar."""
    import torch

    ta = _to_tensor(a, model.device)
    tb = _to_tensor(b, model.device)
    with torch.no_grad():
        d = model.net(ta, tb)
    return float(d.item())


__all__ = ["LpipsModel", "load", "compute"]
