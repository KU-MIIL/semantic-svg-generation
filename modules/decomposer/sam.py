"""SAM3 plumbing: determinism, cached model loading, single-shot segmentation.

The CUBLAS env var that pins deterministic algorithms is set in this package's
``__init__.py`` — it must run before any ``import torch`` in this module."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


_SAM_CACHE: dict[str, tuple] = {}
_DETERMINISM_APPLIED = False


def _apply_determinism(seed: int = 0) -> None:
    """Pin SAM3 inference to be reproducible across runs.

    SAM3's forward pass has no sampling, so the run-to-run jitter we see is
    GPU/cuDNN nondeterminism (reduction order in bfloat16) bumping pixels
    across the score/mask thresholds. Toggle deterministic algorithms +
    seed everything; ``warn_only=True`` so any op without a deterministic
    kernel falls back instead of crashing the pipeline.
    """
    global _DETERMINISM_APPLIED
    if _DETERMINISM_APPLIED:
        return
    import random
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    _DETERMINISM_APPLIED = True


def _get_sam(sam_model: str, device: str, dtype):
    if sam_model in _SAM_CACHE:
        return _SAM_CACHE[sam_model]
    _apply_determinism()
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained(sam_model, torch_dtype=dtype).to(device).eval()
    proc = Sam3Processor.from_pretrained(sam_model)
    _SAM_CACHE[sam_model] = (model, proc)
    return model, proc


def _sam3_segment(
    model, processor, img: Image.Image, label: str, device: str,
    score_threshold: float = 0.3, mask_threshold: float = 0.5,
    input_box: list[int] | None = None,
) -> list[tuple[float, list[int], np.ndarray]]:
    """Run SAM3 with a text prompt. When ``input_box`` ([x0, y0, x1, y1] in
    canvas pixels) is provided it is fed as a grounding bbox alongside the
    text — used to disambiguate sibling instances of the same kind of object
    that text alone collapses (e.g., two hammers)."""
    import torch

    proc_kwargs: dict[str, Any] = {
        "images": img, "text": label, "return_tensors": "pt",
    }
    if input_box is not None:
        proc_kwargs["input_boxes"] = [[list(input_box)]]
    inputs = processor(**proc_kwargs).to(device)
    if "input_boxes" in inputs:
        # The processor builds input_boxes as float32, but the model weights
        # are bfloat16 on CUDA — mismatch crashes the geometry encoder. Cast
        # to the model's parameter dtype so both sides agree.
        model_dtype = next(model.parameters()).dtype
        inputs["input_boxes"] = inputs["input_boxes"].to(dtype=model_dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    H, W = img.size[1], img.size[0]
    result = processor.post_process_instance_segmentation(
        outputs, threshold=score_threshold, mask_threshold=mask_threshold,
        target_sizes=[(H, W)],
    )[0]
    return [
        (float(s), [int(x) for x in b.tolist()], m.cpu().numpy().astype(bool))
        for s, b, m in zip(result["scores"], result["boxes"], result["masks"])
    ]


def assess_sam_quality(prob_map: np.ndarray) -> dict:
    """Classify a SAM probability map as good vs bad quality.

    Empirically derived from 522 (label, bbox) rows in sweep_v11
    (experiments/sam_stability/build_criteria_v4): sb is the only signal
    that discriminates user-flagged bad masks from the rest; logit_max,
    peak count and high_conf are all redundant given sb.

        bad   ←  sb_05 < 0.90   OR   sb_20 ≤ 0.10
        good  ←  else

    sb_05 (narrow ±0.05) catches most fragmented masks; sb_20 (wide ±0.20)
    catches small-object cases where the mask is too small for the narrow
    perturbation to register but its boundary still collapses under a
    larger one. Returns dict with quality/sb_05/sb_20/reason."""
    pmap = prob_map.astype(np.float64)
    union05 = int((pmap > 0.45).sum())
    inter05 = int((pmap > 0.55).sum())
    sb_05 = (inter05 / union05) if union05 > 0 else 0.0
    union20 = int((pmap > 0.30).sum())
    inter20 = int((pmap > 0.70).sum())
    sb_20 = (inter20 / union20) if union20 > 0 else 0.0
    reasons: list[str] = []
    if sb_05 < 0.90:
        reasons.append("sb_05<0.90")
    if sb_20 <= 0.10:
        reasons.append("sb_20<=0.10")
    return {
        "quality": "bad" if reasons else "good",
        "sb_05": float(sb_05),
        "sb_20": float(sb_20),
        "reason": "+".join(reasons),
    }


def _sam3_segment_with_prob(
    model, processor, img: Image.Image, label: str, device: str,
    score_threshold: float = 0.3, mask_threshold: float = 0.5,
    input_box: list[int] | None = None,
) -> list[dict]:
    """SAM3 grounded segmentation that exposes the per-pixel prob map.

    Bypasses ``post_process_instance_segmentation`` so callers can run
    custom downstream analysis on the soft probability (e.g.
    :func:`assess_sam_quality`, color-palette / peak-threshold recovery).
    Each output row: ``{"score", "bbox", "mask", "prob"}``.
    """
    import torch

    proc_kwargs: dict[str, Any] = {
        "images": img, "text": label, "return_tensors": "pt",
    }
    if input_box is not None:
        proc_kwargs["input_boxes"] = [[list(input_box)]]
    inputs = processor(**proc_kwargs).to(device)
    if "input_boxes" in inputs:
        model_dtype = next(model.parameters()).dtype
        inputs["input_boxes"] = inputs["input_boxes"].to(dtype=model_dtype)
    with torch.inference_mode():
        outputs = model(**inputs)

    H, W = img.size[1], img.size[0]
    pred_logits = outputs.pred_logits[0]            # (Q,)
    pred_masks = outputs.pred_masks[0]              # (Q, h, w) logits
    pred_boxes = outputs.pred_boxes[0]              # (Q, 4) normalized xyxy
    presence_logits = outputs.presence_logits       # (B, 1) or None

    scores = pred_logits.sigmoid()
    if presence_logits is not None:
        scores = scores * presence_logits[0].sigmoid()
    mask_probs = pred_masks.sigmoid().to(torch.float32)

    keep = scores > score_threshold
    scores = scores[keep]
    mask_probs = mask_probs[keep]
    boxes = pred_boxes[keep]
    if scores.numel() == 0:
        return []

    boxes = boxes * torch.tensor(
        [W, H, W, H], device=boxes.device, dtype=boxes.dtype,
    )
    mask_probs = torch.nn.functional.interpolate(
        mask_probs.unsqueeze(0), size=(H, W),
        mode="bilinear", align_corners=False,
    ).squeeze(0)

    out: list[dict] = []
    for s, b, p in zip(scores, boxes, mask_probs):
        prob_np = p.cpu().numpy().astype(np.float32)
        out.append({
            "score": float(s),
            "bbox": [int(x) for x in b.tolist()],
            "mask": (prob_np > mask_threshold).astype(bool),
            # Per-pixel sigmoid'd probability. Same array we just thresholded
            # to make the mask; carried out so callers can visualize the soft
            # boundary (heatmap) instead of only the binary cut.
            "prob": prob_np,
        })
    out.sort(key=lambda d: -d["score"])
    return out
