"""Recovery strategies for SAM masks flagged as bad by ``assess_sam_quality``.

Two algorithms, both invoked AFTER ``assess_sam_quality`` has tagged the
mask ``sam_quality == "bad"`` (sb_05 / sb_20 thresholds, fit on sweep_v11):

1. ``recover_via_color_palette`` — exp5: bbox-crop the prob+RGB, sample
   colours from SAM∩bbox, broadcast palette inside bbox, label 4-conn
   components, drop components disjoint from SAM, then split by Sobel
   ``|∇prob|`` and keep only the split fragments that still touch SAM.
2. ``recover_via_peak_threshold`` — find the two largest peaks of the
   prob density via gaussian KDE; threshold the mask at the **mean** of
   the two peak x-positions instead of 0.5. The mean is moved on top of
   or below 0.5 entirely by where the peaks landed, so the rule is just
   ``mask = prob > τ*`` with τ* = peaks_mean.

Both functions return a full-canvas bool mask of the same shape as ``prob``.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label as _cc_label, sobel
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


# ---------- shared primitives (ported from experiments/improve_bad/exp5)

_CC_STRUCT_4 = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool,
)


def _rgb_to_keys(rgb: np.ndarray) -> np.ndarray:
    """Pack (H, W, 3) uint8 → (H, W) int32 = R<<16 | G<<8 | B for fast set ops."""
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r << 16) | (g << 8) | b


def _extract_exact_color_mask(
    prob: np.ndarray, rgb: np.ndarray, bbox: list[int],
    *, sam_tau: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (out_mask, sam_mask, n_unique_colors), all full-canvas; both
    masks are non-zero only inside ``bbox``."""
    H, W = prob.shape
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)

    out_mask = np.zeros((H, W), dtype=bool)
    sam_mask = np.zeros((H, W), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return out_mask, sam_mask, 0

    prob_c = prob[y0:y1, x0:x1]
    rgb_c = rgb[y0:y1, x0:x1]
    sam_c = prob_c > sam_tau
    if not sam_c.any():
        return out_mask, sam_mask, 0

    keys_c = _rgb_to_keys(rgb_c)
    palette = np.unique(keys_c[sam_c])
    out_c = np.isin(keys_c, palette)

    sam_mask[y0:y1, x0:x1] = sam_c
    out_mask[y0:y1, x0:x1] = out_c
    return out_mask, sam_mask, int(palette.size)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n = _cc_label(mask, structure=_CC_STRUCT_4)
    return labels.astype(np.int32), int(n)


def _keep_components_overlapping(
    labels: np.ndarray, n: int, anchor: np.ndarray,
) -> tuple[np.ndarray, int]:
    if n == 0:
        return labels, 0
    keep_ids = np.unique(labels[anchor])
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0:
        return np.zeros_like(labels), 0
    remap = np.zeros(n + 1, dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids, start=1):
        remap[int(old_id)] = new_id
    return remap[labels], int(keep_ids.size)


def _prob_gradient_magnitude(prob: np.ndarray) -> np.ndarray:
    gx = sobel(prob.astype(np.float32), axis=1, mode="reflect")
    gy = sobel(prob.astype(np.float32), axis=0, mode="reflect")
    return np.hypot(gx, gy)


def _normalized_prob_gradient(prob: np.ndarray) -> np.ndarray:
    g = _prob_gradient_magnitude(prob)
    mx = float(g.max())
    if mx > 1e-9:
        g = g / mx
    return np.clip(g, 0.0, 1.0)


def _split_components_by_edges(
    final_mask: np.ndarray, prob: np.ndarray,
    *, edge_thresh: float, min_area: int,
) -> tuple[np.ndarray, int]:
    if not final_mask.any():
        return np.zeros_like(final_mask, dtype=np.int32), 0
    edges = _normalized_prob_gradient(prob) > edge_thresh
    split_input = final_mask & ~edges
    labels, n = _label_components(split_input)
    if n == 0:
        return labels, 0
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    keep = np.where(areas[1:] >= min_area)[0] + 1
    if keep.size == 0:
        return np.zeros_like(labels), 0
    remap = np.zeros(n + 1, dtype=np.int32)
    for new_id, old_id in enumerate(keep, start=1):
        remap[int(old_id)] = new_id
    return remap[labels], int(keep.size)


def _components_overlapping_anchor(
    labels: np.ndarray, n: int, anchor: np.ndarray,
) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    ids = np.unique(labels[anchor])
    ids = ids[ids > 0]
    if ids.size:
        out[ids.astype(np.int64) - 1] = True
    return out


# ---------- public recovery entry points

def recover_via_color_palette(
    prob: np.ndarray, rgb: np.ndarray, bbox: list[int],
    *, sam_tau: float = 0.5, edge_thresh: float = 0.30,
    min_split_area: int = 30,
) -> tuple[np.ndarray, dict]:
    """Run the exp5 color-recovery pipeline on a bad SAM mask.

    Args:
        prob: SAM-output probability map, full canvas (H, W).
        rgb:  Input image as (H, W, 3) uint8 numpy array, full canvas.
        bbox: [x0, y0, x1, y1] in canvas px (xyxy).
        sam_tau: SAM threshold used to sample the palette (default 0.5).
        edge_thresh: Sobel |∇prob| cutoff (normalized) for edge split.
        min_split_area: drop edge-split fragments smaller than this.

    Returns:
        (mask, info) where mask is full-canvas bool, info dict reports
        intermediate counts for diagnostics.
    """
    out_mask, sam_mask, n_palette = _extract_exact_color_mask(
        prob, rgb, bbox, sam_tau=sam_tau,
    )
    labels_all, n_all = _label_components(out_mask)
    labels, n_kept = _keep_components_overlapping(labels_all, n_all, sam_mask)
    final_mask = labels > 0
    split_labels, n_split = _split_components_by_edges(
        final_mask, prob,
        edge_thresh=edge_thresh, min_area=min_split_area,
    )
    overlap = _components_overlapping_anchor(split_labels, n_split, sam_mask)
    if n_split > 0 and overlap.any():
        keep_ids = np.where(overlap)[0] + 1
        merged = np.isin(split_labels, keep_ids)
    else:
        # Edge split removed everything (or the prob map has no strong
        # internal edges). Fall back to the pre-split colour mask.
        merged = final_mask
    info = {
        "n_palette_colors": int(n_palette),
        "out_mask_pixels": int(out_mask.sum()),
        "n_components_total": int(n_all),
        "n_components_kept": int(n_kept),
        "n_edge_split": int(n_split),
        "n_edge_split_overlap": int(overlap.sum()),
        "final_pixels": int(merged.sum()),
        "edge_thresh": float(edge_thresh),
        "min_split_area": int(min_split_area),
    }
    return merged, info


def _find_density_peaks(
    prob: np.ndarray, *, n_points: int = 1024,
) -> list[float]:
    """Locate prob density peaks via gaussian KDE. Returns ascending x
    positions of local maxima found by ``scipy.signal.find_peaks``."""
    vals = prob.ravel().astype(np.float64)
    if vals.size < 2 or float(vals.std()) == 0.0:
        return []
    kde = gaussian_kde(vals)
    xs = np.linspace(0.0, 1.0, n_points)
    ys = kde(xs)
    peak_idx, _ = find_peaks(ys)
    return [float(xs[i]) for i in peak_idx]


def recover_via_peak_threshold(
    prob: np.ndarray, *, side: str = "high",
    sam_tau: float = 0.5, n_points: int = 1024,
) -> tuple[np.ndarray, dict]:
    """Pick the prob-density peak nearest to ``sam_tau`` on the requested
    side and use it as the binarization threshold.

    Args:
        side: ``"high"`` selects the peak whose x position is just above
              ``sam_tau`` (i.e. the part-pixel peak in a bimodal map).
              ``"low"`` selects the peak just below ``sam_tau`` (the
              background peak). The two modes give different cuts:
              ``"high"`` shrinks the SAM mask, ``"low"`` enlarges it.
        sam_tau: Reference cutoff (default 0.5). The selected peak must
              sit strictly on the requested side of this value.
        n_points: KDE x-resolution.

    Returns:
        (mask, info) — info reports ``tau_star`` (the chosen threshold,
        either the peak x or the ``sam_tau`` fallback) plus all detected
        peaks for inspection. If no peak exists on the requested side,
        falls back to ``mask = prob > sam_tau`` with ``side: "fallback"``.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    peaks = _find_density_peaks(prob, n_points=n_points)
    info: dict = {
        "n_peaks": len(peaks),
        "peaks": peaks,
        "sam_tau": float(sam_tau),
        "side_request": side,
    }
    if side == "high":
        candidates = [p for p in peaks if p > sam_tau + 1e-9]
    else:
        candidates = [p for p in peaks if p < sam_tau - 1e-9]
    if not candidates:
        info["tau_star"] = float(sam_tau)
        info["side"] = "fallback"
        return prob > sam_tau, info
    # Closest to sam_tau on the requested side.
    chosen = min(candidates, key=lambda p: abs(p - sam_tau))
    info["tau_star"] = float(chosen)
    info["side"] = side
    return prob > chosen, info


__all__ = [
    "recover_via_color_palette",
    "recover_via_peak_threshold",
]
