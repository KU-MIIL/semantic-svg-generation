# ============================================================================
# semantic_flatten_v2.py — anchor-based recall metric (V2.3, dual-judge).
# judge = MSE  → score = MSE on (GT ∪ pred) bbox crop  ("mse_mse" pair)
# judge = DINO → score = DINO on (GT ∪ pred) bbox crop ("dino_dino" pair)
# Both run independent greedy selections per anchor, returning a separate
# FlattenRecallResult each.
#
# Per-anchor processing inside ``compute(...)`` is dispatched via a
# ``ThreadPoolExecutor`` (``max_workers`` kw, default
# ``min(2, cpu_count(), N)`` — the measured sweet spot; cairo lock
# contention regresses past ~4 threads). Pred per-path ink masks used
# by the IoU prefilter are precomputed once per ``compute`` call and
# shared across anchors, cutting the per-anchor render budget. DINO is
# opt-in: leave ``dino_model=None`` and ``compute_pairs`` returns only
# the ``"mse_mse"`` pair.
# ============================================================================

"""Anchor-based recall metric: greedy path selection per GT anchor, then
score the final selection on the union of the GT-anchor bbox and the
selected pred-paths bbox.

For each non-root, non-"none" GT node (= anchor) we:

  1. Render the anchor's path-subset against a white background.
  2. Greedy-select pred paths (forward pass + alternating add/remove rounds
     until convergence) keeping a candidate iff it strictly reduces the
     judge "distance" against the anchor. Two judges are supported:
        * ``"mse"``  — union-bbox MSE distance (lower better).
        * ``"dino"`` — union-bbox DINO distance = 1 - cosine sim (lower
                       better; cheap-ish but ~10-50× slower than MSE).
     Candidates are prefiltered by ink-mask IoU > 1e-9 (the V2.1 setup
     that keeps only pred paths sharing ≥1 pixel with the anchor mask).
  3. Compute the **union bbox** = smallest rectangle that contains both
     the GT anchor's non-white pixels AND the selected pred-paths'
     non-white pixels. Crop both renders to that bbox (square-padded
     with white) and score with the SAME metric used as the judge:
        * judge=mse  → ``score_mse``  ∈ [0, ?); lower better.
        * judge=dino → ``score_dino`` ∈ [0, 1]; higher better.

The union bbox prevents two failure modes the V2.1 anchor-only crop had:
(a) small anchor inflated by surrounding white when DINO compares mostly
empty backgrounds; (b) pred drawing far outside the anchor goes
unscored when the crop ignores its location.

``compute(...)``         — one judge (single pair) per call.
``compute_pairs(...)``   — runs both judges, returns
                            ``{"mse_mse": ..., "dino_dino": ...}``.

For predictions that already carry a tree.json grouping, use
:mod:`metric.semantic` for the direct N×M comparison instead (also
exposes the same two pairs).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from metric import dino as dino_metric
from metric import mse as mse_metric
from metric.occlusion_bbox import (
    compute_anchor_occlusion,
    masked_mse,
)
from metric.semantic import (
    Node,
    count_drawables,
    flatten_tree,
    render_subset_rgb,
)


@dataclass
class AnchorRecall:
    """Greedy selection outcome + final-score breakdown for one anchor.

    ``judge_*`` are full-frame MSE distances used internally by greedy
    selection (lower better). ``score_dino`` and ``score_mse`` are the
    final per-anchor scores computed on the (GT ∪ pred) bbox crop —
    DINO is a similarity in [0, 1] (higher better), MSE is a distance
    (lower better).
    """
    gt_index: int                       # index in the flattened GT node list
    gt_label: str
    gt_depth: int
    gt_paths: tuple[int, ...]
    selected_paths: list[int]           # pred path indices kept by greedy
    judge_initial: float                # MSE(white, anchor) — full frame
    judge_final: float                  # MSE(render(selected), anchor) — full frame
    score_baseline_dino: float          # dino_sim(white, anchor) on anchor bbox crop
    score_baseline_mse: float           # MSE(white, anchor) on anchor bbox crop
    score_dino: float                   # dino_sim on (GT ∪ pred) bbox crop
    score_mse: float                    # MSE on (GT ∪ pred) bbox crop
    # Occlusion-aware extras (V2.4). bbox = contested region of the anchor
    # (overlap with the rest of the GT). NaN/None when the anchor has no
    # contested pixels (the entire silhouette is uncovered).
    occlusion_bbox: tuple[int, int, int, int] | None = None
    score_mse_occluded: float = float("nan")

    @property
    def judge_improvement(self) -> float:
        """How much greedy reduced full-frame MSE distance. Larger = better."""
        return self.judge_initial - self.judge_final

    @property
    def dino_uplift(self) -> float:
        """DINO sim gain over the white-baseline floor (higher = better)."""
        return self.score_dino - self.score_baseline_dino

    @property
    def mse_uplift(self) -> float:
        """MSE reduction from the white-baseline floor (higher = better)."""
        return self.score_baseline_mse - self.score_mse

    @classmethod
    def from_dict(cls, d: dict) -> "AnchorRecall":
        """Reconstruct from :meth:`FlattenRecallResult.to_dict` entries.
        Drops the computed properties (judge_improvement, *_uplift) since
        they're recomputed on access."""
        return cls(
            gt_index=int(d["gt_index"]),
            gt_label=str(d["gt_label"]),
            gt_depth=int(d["gt_depth"]),
            gt_paths=tuple(d["gt_paths"]),
            selected_paths=list(d["selected_paths"]),
            judge_initial=float(d["judge_initial"]),
            judge_final=float(d["judge_final"]),
            score_baseline_dino=float(d["score_baseline_dino"]),
            score_baseline_mse=float(d["score_baseline_mse"]),
            score_dino=float(d["score_dino"]),
            score_mse=float(d["score_mse"]),
            occlusion_bbox=(
                tuple(d["occlusion_bbox"])  # type: ignore[arg-type]
                if d.get("occlusion_bbox") is not None else None
            ),
            score_mse_occluded=float(
                d.get("score_mse_occluded", float("nan"))
            ),
        )


@dataclass
class FlattenRecallResult:
    """End-to-end result for one (gt_svg, pred_svg, judge_metric) call."""
    n_gt: int                           # number of anchors evaluated
    p_pred: int                         # number of drawable primitives in pred
    judge_metric: str = "mse"           # "mse" or "dino" — what drove greedy
    bbox_padding: float = 0.0           # ratio padding applied around the union bbox
    iou_prefilter: float = 1e-9         # candidate path IoU prefilter (V2.1 default)
    anchors: list[AnchorRecall] = field(default_factory=list)
    recall_dino: float = float("nan")   # mean(score_dino) — populated when judge=="dino"
    recall_mse: float = float("nan")    # mean(score_mse)  — populated when judge=="mse"
    # Occlusion-aware aggregates (V2.4). Mean restricted to anchors with a
    # non-empty contested bbox; NaN when no such anchor exists.
    recall_mse_occluded: float = float("nan")
    n_anchors_with_occlusion: int = 0

    @property
    def recall(self) -> float:
        """Mean of the score that matches ``judge_metric``: ``recall_mse``
        when judge=='mse' (lower better), ``recall_dino`` when
        judge=='dino' (higher better)."""
        return self.recall_mse if self.judge_metric == "mse" else self.recall_dino

    def to_dict(self) -> dict:
        return {
            "n_gt": self.n_gt,
            "p_pred": self.p_pred,
            "judge_metric": self.judge_metric,
            "bbox_padding": self.bbox_padding,
            "iou_prefilter": self.iou_prefilter,
            "recall_dino": self.recall_dino,
            "recall_mse": self.recall_mse,
            "recall_mse_occluded": self.recall_mse_occluded,
            "n_anchors_with_occlusion": self.n_anchors_with_occlusion,
            "anchors": [
                {
                    "gt_index": a.gt_index,
                    "gt_label": a.gt_label,
                    "gt_depth": a.gt_depth,
                    "gt_paths": list(a.gt_paths),
                    "selected_paths": list(a.selected_paths),
                    "judge_initial": a.judge_initial,
                    "judge_final": a.judge_final,
                    "judge_improvement": a.judge_improvement,
                    "score_baseline_dino": a.score_baseline_dino,
                    "score_baseline_mse": a.score_baseline_mse,
                    "score_dino": a.score_dino,
                    "score_mse": a.score_mse,
                    "dino_uplift": a.dino_uplift,
                    "mse_uplift": a.mse_uplift,
                    "occlusion_bbox": (
                        list(a.occlusion_bbox)
                        if a.occlusion_bbox is not None else None
                    ),
                    "score_mse_occluded": a.score_mse_occluded,
                }
                for a in self.anchors
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FlattenRecallResult":
        """Reconstruct from :meth:`to_dict` output. Used to skip the
        expensive greedy selection when a cached per-sample JSON exists
        for the same (pred_svg, gt_svg, size, judge_metric) tuple. The
        cache is sufficient for downstream visualization or rescoring
        because greedy paths + per-anchor scores are deterministic given
        the inputs."""
        return cls(
            n_gt=int(d["n_gt"]),
            p_pred=int(d["p_pred"]),
            judge_metric=str(d.get("judge_metric", "mse")),
            bbox_padding=float(d.get("bbox_padding", 0.0)),
            iou_prefilter=float(d.get("iou_prefilter", 1e-9)),
            anchors=[AnchorRecall.from_dict(a) for a in d.get("anchors", [])],
            recall_dino=float(d.get("recall_dino", float("nan"))),
            recall_mse=float(d.get("recall_mse", float("nan"))),
            recall_mse_occluded=float(
                d.get("recall_mse_occluded", float("nan"))
            ),
            n_anchors_with_occlusion=int(d.get("n_anchors_with_occlusion", 0)),
        )


def _arr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(
        (np.clip(arr, 0.0, 1.0) * 255 + 0.5).astype("uint8"), "RGB")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _content_bbox(arr: np.ndarray,
                  *, white_eps: float = 0.01) -> tuple[int, int, int, int] | None:
    """Return the tight bbox of non-white pixels in ``arr`` as
    ``(y0, x0, y1, x1)`` (half-open). Returns None when the array is
    entirely white."""
    mask = (arr < (1.0 - white_eps)).any(axis=-1)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _union_bbox(a_box: tuple[int, int, int, int] | None,
                b_box: tuple[int, int, int, int] | None,
                ) -> tuple[int, int, int, int] | None:
    """Smallest rectangle containing both bboxes. Returns None when both
    are None (both arrays empty)."""
    if a_box is None and b_box is None:
        return None
    if a_box is None:
        return b_box
    if b_box is None:
        return a_box
    return (min(a_box[0], b_box[0]), min(a_box[1], b_box[1]),
            max(a_box[2], b_box[2]), max(a_box[3], b_box[3]))


def _crop_to_union_bbox(anchor_arr: np.ndarray, pred_arr: np.ndarray,
                        *, white_eps: float = 0.01,
                        padding_ratio: float = 0.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Crop both arrays to the union of their non-white bboxes, then
    pad to a square with white fill (preserves aspect ratio for the
    DINO backbone). The union bbox forces scoring to see every region
    where EITHER side drew content — small anchors aren't inflated by
    surrounding white, and pred ink drawn outside the anchor isn't
    ignored.

    Both arrays MUST share the same H×W shape. Returns the cropped+
    padded versions of ``(anchor, pred)``. If both arrays are all-white,
    returns them unchanged (degenerate edge case).
    """
    if anchor_arr.shape != pred_arr.shape:
        raise ValueError(
            f"anchor and pred arrays must share shape, got "
            f"{anchor_arr.shape} vs {pred_arr.shape}"
        )
    a_box = _content_bbox(anchor_arr, white_eps=white_eps)
    p_box = _content_bbox(pred_arr, white_eps=white_eps)
    u = _union_bbox(a_box, p_box)
    if u is None:
        return anchor_arr, pred_arr
    H, W = anchor_arr.shape[:2]
    y0, x0, y1, x1 = u
    if padding_ratio > 0.0:
        h, w = y1 - y0, x1 - x0
        py = max(1, int(round(h * padding_ratio)))
        px = max(1, int(round(w * padding_ratio)))
        y0, y1 = max(0, y0 - py), min(H, y1 + py)
        x0, x1 = max(0, x0 - px), min(W, x1 + px)
    a = anchor_arr[y0:y1, x0:x1]
    p = pred_arr[y0:y1, x0:x1]
    ch, cw = a.shape[:2]
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


def _score_pair(pred_arr: np.ndarray, anchor_arr: np.ndarray,
                dino_model: Any,
                *, padding_ratio: float = 0.05) -> tuple[float, float]:
    """Compute (DINO cosine similarity, MSE distance) on the union
    bbox crop. DINO ∈ [0, 1] (clamped), higher better. MSE ≥ 0, lower
    better.

    When ``dino_model`` is ``None`` the DINO forward pass is skipped and
    the DINO score comes back as ``NaN`` — useful when the caller only
    cares about the MSE-MSE pair (judge=mse + score=mse) and wants to
    avoid the GPU cost.
    """
    anchor_use, pred_use = _crop_to_union_bbox(
        anchor_arr, pred_arr, padding_ratio=padding_ratio)
    if dino_model is None:
        dino = float("nan")
    else:
        dino = _clamp01(float(dino_metric.compute(
            dino_model, _arr_to_pil(pred_use), _arr_to_pil(anchor_use))))
    mse = float(mse_metric.compute(pred_use, anchor_use))
    return dino, mse


def _union_bbox_mse(pred_arr: np.ndarray, anchor_arr: np.ndarray) -> float:
    """MSE on the (GT ∪ pred) content bbox crop — same crop used by the
    final score, no padding. Used as the greedy judge so selection
    directly optimises the scored quantity."""
    a, p = _crop_to_union_bbox(anchor_arr, pred_arr, padding_ratio=0.0)
    return float(mse_metric.compute(p, a))


def _union_bbox_dino_dist(pred_arr: np.ndarray, anchor_arr: np.ndarray,
                          dino_model: Any) -> float:
    """1 - DINOv2 cosine similarity on the (GT ∪ pred) content bbox crop
    (no padding). Distance ∈ [0, 1] (lower better) so it plugs into the
    same "minimise judge" greedy loop as MSE. Pays a GPU forward pass per
    call — ~10-50× slower than ``_union_bbox_mse``."""
    a, p = _crop_to_union_bbox(anchor_arr, pred_arr, padding_ratio=0.0)
    sim = float(dino_metric.compute(
        dino_model, _arr_to_pil(p), _arr_to_pil(a)))
    return 1.0 - _clamp01(sim)


def _greedy_select_for_anchor(pred_text: str, anchor_img: np.ndarray,
                              P: int, size: int,
                              *,
                              judge_fn,
                              max_rounds: int = 10,
                              iou_prefilter: float = 1e-9,
                              pred_path_masks: list[np.ndarray] | None = None,
                              ) -> tuple[list[int], float, float, np.ndarray]:
    """Forward pass + iterative add/remove rounds until convergence.

    Phase 1 — Forward pass: walk path indices in document order. For each
        candidate p, compute judge_fn(render(selected ∪ {p}), anchor);
        keep p iff that strictly reduces the running distance.

    Phase 2 — Alternating add/remove rounds: repeatedly try
        (a) adding each currently-unselected path (re-evaluation in the
            context of the now-different selected set); keep iff dist drops.
        (b) removing each currently-selected path; drop iff dist drops.
        After each round, if the selected set is unchanged, declare
        convergence and stop. Cap: ``max_rounds`` (default 10).

    ``judge_fn(pred_arr, anchor_arr) -> float`` is the scalar distance
    driving selection — typically ``_union_bbox_mse`` (cheap) or a
    closure over ``_union_bbox_dino_dist`` (slow, GPU). It MUST return
    smaller-is-better; both supported judges do.

    Candidate set is restricted to pred paths whose ink-mask IoU with the
    anchor mask exceeds ``iou_prefilter`` (default 1e-9, the V2.1 setup
    that keeps only candidates sharing ≥1 pixel with the anchor). When
    ``pred_path_masks`` is provided (one boolean H×W ink mask per pred
    path index), the IoU prefilter reuses those masks instead of
    re-rendering each pred path per anchor — anchor-independent work
    that ``compute(...)`` hoists out of the per-anchor loop.

    Returns ``(selected_paths, dist_initial, dist_final, final_render)``.
    """
    white = np.ones_like(anchor_img)

    candidate_paths: list[int]
    if iou_prefilter > 0.0:
        anchor_mask = (anchor_img < 0.99).any(axis=-1)
        ious: list[tuple[float, int]] = []
        for p in range(P):
            if pred_path_masks is not None:
                r_mask = pred_path_masks[p]
            else:
                r = render_subset_rgb(pred_text, {p}, size=size)
                r_mask = (r < 0.99).any(axis=-1)
            inter = float((anchor_mask & r_mask).sum())
            union = float((anchor_mask | r_mask).sum())
            iou = inter / union if union > 0 else 0.0
            ious.append((iou, p))
        candidate_paths = sorted(p for iou, p in ious if iou > iou_prefilter)
    else:
        candidate_paths = list(range(P))

    universe = set(candidate_paths)

    init_dist = judge_fn(white, anchor_img)
    cur_dist = init_dist
    cur_selected: list[int] = []
    cur_render = white

    # Phase 1: forward
    for p in candidate_paths:
        tentative = cur_selected + [p]
        tentative_render = render_subset_rgb(pred_text, set(tentative), size=size)
        new_dist = judge_fn(tentative_render, anchor_img)
        if new_dist < cur_dist:
            cur_selected = tentative
            cur_render = tentative_render
            cur_dist = new_dist

    # Phase 2: alternating add/remove rounds
    for _round in range(max_rounds):
        prev_set = set(cur_selected)

        # (a) add probes
        unselected = sorted(universe - prev_set)
        for p in unselected:
            tentative = sorted(set(cur_selected) | {p})
            tentative_render = render_subset_rgb(pred_text, set(tentative), size=size)
            new_dist = judge_fn(tentative_render, anchor_img)
            if new_dist < cur_dist:
                cur_selected = tentative
                cur_render = tentative_render
                cur_dist = new_dist

        # (b) remove probes — iterate snapshot so list mutations are safe
        for p in list(cur_selected):
            if p not in cur_selected:
                continue                         # may have been removed earlier in round
            tentative = [x for x in cur_selected if x != p]
            tentative_render = render_subset_rgb(pred_text, set(tentative), size=size)
            new_dist = judge_fn(tentative_render, anchor_img)
            if new_dist < cur_dist:
                cur_selected = tentative
                cur_render = tentative_render
                cur_dist = new_dist

        if set(cur_selected) == prev_set:
            break    # converged: no single add/remove improves further

    return cur_selected, init_dist, cur_dist, cur_render


def _process_anchor(
    i: int, node: Node,
    *,
    gt_text: str, pred_text: str,
    P: int, size: int,
    judge_fn,
    dino_model: Any,
    bbox_padding: float,
    iou_prefilter: float,
    gt_all_paths: set[int],
    compute_occlusion: bool,
    occlusion_mode: str,
    pred_path_masks: list[np.ndarray] | None = None,
) -> AnchorRecall:
    """One-anchor worker: render anchor, score baseline, run greedy if
    P > 0, then assemble the final ``AnchorRecall``. Independent of all
    other anchors → safe to run concurrently from a thread pool.

    When ``compute_occlusion`` is True, also computes the anchor's
    occlusion mask (precise via marker-color extraction in ``occluded``
    mode, or the broader spatial-overlap version in ``contested`` mode)
    and the supplementary MSE restricted to those pixels. Cost: one extra
    ``render_subset_rgb`` (or marker render) per anchor.

    The supplementary score uses *masked MSE* (average squared error over
    the mask pixels only), not bbox-cropped MSE — so the precise mode
    isolates exactly the hidden pixels instead of averaging over a
    rectangular crop that includes both hidden and visible pixels.
    """
    anchor_img = render_subset_rgb(gt_text, set(node.paths), size=size)
    white = np.ones_like(anchor_img)
    score_baseline_dino, score_baseline_mse = _score_pair(
        white, anchor_img, dino_model, padding_ratio=bbox_padding,
    )

    if compute_occlusion:
        # Anchor-independent of pred; reused for the final score and viz.
        # amodal_arr=anchor_img skips a redundant rasterisation.
        occl = compute_anchor_occlusion(
            gt_text, set(node.paths), gt_all_paths,
            size=size, amodal_arr=anchor_img, mode=occlusion_mode,
        )
        occl_bbox = occl.bbox
        occl_mask = occl.mask
    else:
        occl_bbox = None
        occl_mask = None

    if P == 0:
        judge_init = judge_fn(white, anchor_img)
        # No pred drawables → score is baseline white vs anchor.
        if occl_mask is not None and occl_mask.any():
            score_mse_occl = masked_mse(white, anchor_img, occl_mask)
        else:
            score_mse_occl = float("nan")
        return AnchorRecall(
            gt_index=i, gt_label=node.label, gt_depth=node.depth,
            gt_paths=node.paths,
            selected_paths=[],
            judge_initial=judge_init, judge_final=judge_init,
            score_baseline_dino=score_baseline_dino,
            score_baseline_mse=score_baseline_mse,
            score_dino=score_baseline_dino,
            score_mse=score_baseline_mse,
            occlusion_bbox=occl_bbox,
            score_mse_occluded=score_mse_occl,
        )

    selected, judge_init, judge_final, final_render = (
        _greedy_select_for_anchor(
            pred_text=pred_text, anchor_img=anchor_img,
            P=P, size=size,
            judge_fn=judge_fn,
            iou_prefilter=iou_prefilter,
            pred_path_masks=pred_path_masks,
        )
    )
    score_dino, score_mse = _score_pair(
        final_render, anchor_img, dino_model, padding_ratio=bbox_padding,
    )
    if occl_mask is not None and occl_mask.any():
        score_mse_occl = masked_mse(final_render, anchor_img, occl_mask)
    else:
        score_mse_occl = float("nan")
    return AnchorRecall(
        gt_index=i, gt_label=node.label, gt_depth=node.depth,
        gt_paths=node.paths,
        selected_paths=selected,
        judge_initial=judge_init, judge_final=judge_final,
        score_baseline_dino=score_baseline_dino,
        score_baseline_mse=score_baseline_mse,
        score_dino=score_dino,
        score_mse=score_mse,
        occlusion_bbox=occl_bbox,
        score_mse_occluded=score_mse_occl,
    )


def compute(gt_svg: str | Path, gt_tree: dict,
            pred_svg: str | Path,
            *,
            size: int = 256,
            include_none: bool = False,
            gt_min_depth: int = 0,
            judge_metric: str = "mse",
            bbox_padding: float = 0.0,
            iou_prefilter: float = 1e-9,
            max_workers: int | None = None,
            dino_model: Any = None,
            compute_occlusion: bool = False,
            occlusion_mode: str = "occluded") -> FlattenRecallResult:
    """Anchor-based recall metric (V2.3 setup) — one judge per call.

    Judge: union-bbox ``judge_metric`` ('mse' or 'dino') drives greedy
    path selection per GT anchor. Score: the SAME metric, computed on the
    union of the GT anchor's content bbox and the selected pred-paths'
    content bbox (square-padded for the DINO backbone).

    To get both ("mse_mse" + "dino_dino") pairs in one call, use
    :func:`compute_pairs` instead.

    Args:
        gt_svg, gt_tree:    GT side — SVG path + tree.json dict.
        pred_svg:           Flat pred SVG (no tree expected/used).
        size:               Render resolution per anchor (square).
        include_none:       If True, ``"none"`` filler GT nodes also act as
                              anchors. Default False.
        gt_min_depth:       Skip GT nodes whose depth is below this cutoff.
        judge_metric:       Greedy judge: ``"mse"`` (cheap, lower better)
                              or ``"dino"`` (GPU, 1 - cos sim, lower better).
                              Score uses the same metric on the final
                              selection.
        bbox_padding:       Margin around the union bbox before square
                              padding (fraction of bbox dims). Default 0.0
                              (tight union bbox, no extra margin).
        iou_prefilter:      Restrict greedy candidates to pred paths whose
                              ink-mask IoU with the anchor mask exceeds
                              this threshold. Default 1e-9 keeps only
                              candidates that share at least one pixel
                              with the anchor (V2.1 production setup).
                              Set to ``0.0`` to disable the filter.
        max_workers:        Per-anchor ``ThreadPoolExecutor`` size. ``None``
                              (default) picks ``min(2, os.cpu_count(), N)``
                              — measured sweet spot across small/large
                              samples; threads ≥4 regress on P-heavy SVGs
                              due to cairo C-library lock contention. Set
                              to ``1`` to force the sequential code path
                              (deterministic profiling); push to 4+
                              explicitly only after profiling a given
                              workload.
        dino_model:         Pre-loaded DINOv2 model. ``None`` (default) →
                              ``score_baseline_dino`` and ``score_dino`` come
                              back as NaN; only the MSE side is meaningful.
                              Required when ``judge_metric='dino'``. Load
                              once via ``metric.dino.load(device='cuda')``.
        compute_occlusion:  When True, also computes per-anchor occlusion
                              mask + ``score_mse_occluded``. Adds one
                              extra render per anchor (marker or
                              complement, depending on mode). Off by
                              default.
        occlusion_mode:     ``"occluded"`` (default, precise) uses marker-
                              colour rendering to isolate only the pixels
                              of the anchor that are *truly* covered by
                              other layers. ``"contested"`` (legacy, V1)
                              counts any anchor pixel that spatially
                              overlaps another group — overcounts since
                              it also flags anchor-on-top pixels.

    Edge cases:
        - GT has no anchors (N=0)    → recall = NaN, no anchors emitted.
        - Pred has no drawables (P=0) → each anchor's score_* equals
          its baseline (white vs anchor on the anchor's own bbox crop);
          selected_paths empty.
    """
    if judge_metric not in ("mse", "dino"):
        raise ValueError(
            f"judge_metric={judge_metric!r} not supported; "
            f"use 'mse' or 'dino'."
        )
    if judge_metric == "dino" and dino_model is None:
        raise ValueError("dino_model is required when judge_metric='dino'")

    if judge_metric == "mse":
        judge_fn = _union_bbox_mse
    else:
        def judge_fn(pred_arr, anchor_arr):
            return _union_bbox_dino_dist(pred_arr, anchor_arr, dino_model)

    gt_text = Path(gt_svg).read_text()
    pred_text = Path(pred_svg).read_text()

    gt_nodes = flatten_tree(gt_tree, include_none=include_none,
                            min_depth=gt_min_depth)
    N = len(gt_nodes)
    P = count_drawables(pred_text)
    # GT-wide drawable index set used to compute each anchor's complement
    # render for the contested-bbox supplementary metric. Counting on the
    # GT (not pred) makes the complement = "all other GT paths". Skipped
    # when occlusion scoring is off — saves a per-call count_drawables.
    gt_all_paths: set[int] = (
        set(range(count_drawables(gt_text))) if compute_occlusion else set()
    )

    if N == 0:
        return FlattenRecallResult(
            n_gt=0, p_pred=P, judge_metric=judge_metric,
            bbox_padding=bbox_padding, iou_prefilter=iou_prefilter,
            anchors=[], recall_dino=float("nan"), recall_mse=float("nan"),
            recall_mse_occluded=float("nan"), n_anchors_with_occlusion=0,
        )

    if max_workers is None:
        # Sweet-spot cap = 2: measured across small/large samples, threads
        # ≥4 regress on P-heavy SVGs due to lock contention in the cairo
        # C library underneath CairoSVG. Callers that want to push harder
        # can pass max_workers explicitly.
        effective_workers = min(2, os.cpu_count() or 1, N)
    else:
        effective_workers = max(1, min(max_workers, N))

    # Precompute pred per-path ink masks once. The IoU prefilter is
    # anchor-independent (it compares each pred path's ink mask to the
    # anchor mask), so rendering each pred path P times for N anchors
    # was wasted work. Hoist into a single P-sweep shared by all anchors.
    pred_path_masks: list[np.ndarray] | None = None
    if P > 0 and iou_prefilter > 0.0:
        def _render_pred_mask(p: int) -> np.ndarray:
            r = render_subset_rgb(pred_text, {p}, size=size)
            return (r < 0.99).any(axis=-1)

        if effective_workers == 1:
            pred_path_masks = [_render_pred_mask(p) for p in range(P)]
        else:
            with ThreadPoolExecutor(
                max_workers=effective_workers,
                thread_name_prefix="sfv2-prefilter",
            ) as pool:
                pred_path_masks = list(pool.map(_render_pred_mask, range(P)))

    def _run(pair: tuple[int, Node]) -> AnchorRecall:
        return _process_anchor(
            pair[0], pair[1],
            gt_text=gt_text, pred_text=pred_text,
            P=P, size=size, judge_fn=judge_fn,
            dino_model=dino_model,
            bbox_padding=bbox_padding,
            iou_prefilter=iou_prefilter,
            gt_all_paths=gt_all_paths,
            compute_occlusion=compute_occlusion,
            occlusion_mode=occlusion_mode,
            pred_path_masks=pred_path_masks,
        )

    if effective_workers == 1:
        anchors: list[AnchorRecall] = [
            _run((i, node)) for i, node in enumerate(gt_nodes)
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="sfv2-anchor",
        ) as pool:
            anchors = list(pool.map(_run, list(enumerate(gt_nodes))))

    # Populate only the recall_* field that matches judge_metric (the
    # "paired" score). The cross field is left NaN so callers can't
    # accidentally read a non-paired score and misinterpret it.
    if judge_metric == "mse":
        recall_mse = float(np.mean([a.score_mse for a in anchors]))
        recall_dino = float("nan")
    else:
        recall_dino = float(np.mean([a.score_dino for a in anchors]))
        recall_mse = float("nan")

    # Occlusion-aware aggregate: nanmean across anchors with a non-empty
    # contested bbox. Anchors with no occlusion are excluded from the
    # mean rather than counted as 0, so the metric strictly answers
    # "how well do we recover hidden pixels on occluded anchors?".
    occl_scores = [a.score_mse_occluded for a in anchors
                   if a.occlusion_bbox is not None]
    if occl_scores:
        recall_mse_occluded = float(np.nanmean(occl_scores))
    else:
        recall_mse_occluded = float("nan")

    return FlattenRecallResult(
        n_gt=N, p_pred=P, judge_metric=judge_metric,
        bbox_padding=bbox_padding, iou_prefilter=iou_prefilter,
        anchors=anchors,
        recall_dino=recall_dino,
        recall_mse=recall_mse,
        recall_mse_occluded=recall_mse_occluded,
        n_anchors_with_occlusion=len(occl_scores),
    )


def compute_pairs(gt_svg: str | Path, gt_tree: dict,
                  pred_svg: str | Path,
                  *,
                  size: int = 256,
                  include_none: bool = False,
                  gt_min_depth: int = 0,
                  bbox_padding: float = 0.0,
                  iou_prefilter: float = 1e-9,
                  max_workers: int | None = None,
                  dino_model: Any = None,
                  compute_occlusion: bool = False,
                  occlusion_mode: str = "occluded") -> dict[str, FlattenRecallResult]:
    """Run available judge pairs, returning one ``FlattenRecallResult`` each.

    DINO is opt-in: when ``dino_model`` is ``None`` (default) only the
    ``"mse_mse"`` pair is computed and the returned dict has a single key.
    Pass a loaded ``DinoModel`` (via ``metric.dino.load(device="cuda")``)
    to also get the ``"dino_dino"`` pair. The MSE judge converges fast;
    the DINO judge is GPU-bound and ~10-50× slower.

    ``max_workers`` is forwarded to each inner :func:`compute` call (peak
    thread count is ``max_workers``, not ``2 × max_workers``, because the
    two judges run sequentially).
    """
    pairs: dict[str, FlattenRecallResult] = {
        "mse_mse": compute(
            gt_svg, gt_tree, pred_svg,
            size=size, include_none=include_none, gt_min_depth=gt_min_depth,
            judge_metric="mse", bbox_padding=bbox_padding,
            iou_prefilter=iou_prefilter, max_workers=max_workers,
            dino_model=dino_model, compute_occlusion=compute_occlusion,
            occlusion_mode=occlusion_mode,
        ),
    }
    if dino_model is not None:
        pairs["dino_dino"] = compute(
            gt_svg, gt_tree, pred_svg,
            size=size, include_none=include_none, gt_min_depth=gt_min_depth,
            judge_metric="dino", bbox_padding=bbox_padding,
            iou_prefilter=iou_prefilter, max_workers=max_workers,
            dino_model=dino_model, compute_occlusion=compute_occlusion,
            occlusion_mode=occlusion_mode,
        )
    return pairs


__all__ = [
    "AnchorRecall",
    "FlattenRecallResult",
    "compute",
    "compute_pairs",
]
