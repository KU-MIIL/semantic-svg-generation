"""DINOv2 similarity: mean-pooled patch features, cosine → [0, 1].

Mirrors the reference ``compute_dion.py``: ``features = last_hidden_state.mean(dim=1)``
then cosine similarity normalized as ``(cos + 1) / 2`` so scores land in [0, 1]
(1.0 = identical direction, 0.0 = opposite).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class DinoModel:
    processor: Any
    model: Any
    device: str


def load(device: str = "cpu", model_id: str = "facebook/dinov2-base") -> DinoModel:
    """Load a DINO/DINOv2 image encoder and its processor."""
    from transformers import AutoImageProcessor, AutoModel

    print(f"[dino] loading {model_id} on {device}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return DinoModel(processor=processor, model=model, device=device)


def _features(model: DinoModel, img: Image.Image):
    import torch

    rgb = img if img.mode == "RGB" else img.convert("RGB")
    inputs = model.processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.model(**inputs)
    return out.last_hidden_state.mean(dim=1)                  # (1, D)


def compute(model: DinoModel, a: Image.Image, b: Image.Image) -> float:
    """Mean-pooled cosine similarity, normalized to [0, 1]. Higher = more similar."""
    import torch

    f1 = _features(model, a)
    f2 = _features(model, b)
    cos = torch.nn.CosineSimilarity(dim=1)(f1, f2).item()
    return (cos + 1.0) / 2.0


__all__ = ["DinoModel", "load", "compute"]
