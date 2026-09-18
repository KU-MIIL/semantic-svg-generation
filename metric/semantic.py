from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from metric import dino as dino_metric
from metric import mse as mse_metric
from metric.occlusion_bbox import (
    compute_anchor_occlusion,
    masked_mse,
)


_DRAWABLE_TAGS = {
    "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
}
# Containers that DEFINE geometry instead of painting it. Their children are
# referenced by url(#id) and never drawn in document order, so counting them
# shifts every later index. The camera-ready annotation excludes them, and
# including them put the two out of step on 8 of 203 samples — worst case
# illustration_openclipart-209134 at 35 vs 27, an 8-index drift that would
# have silently pointed every GT group at the wrong shapes.
_NON_RENDERED_CONTAINERS = {
    "defs", "clipPath", "mask", "pattern", "marker", "symbol",
}
_SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG_NS)

# Judge-score pairs supported in :class:`SemanticResult.per_metric`.
# Each pair builds an N×M judge matrix to pick matches; the score is the
# union-bbox distance at the matched pair under the score metric:
#   "mse_mse"   — judge = MSE,  score = MSE   (score read from judge matrix)
#   "dino_dino" — judge = DINO, score = DINO  (score read from judge matrix)
#   "mse_dino"  — judge = MSE,  score = DINO  (DINO computed only at the
#                                              N+M matched pairs — no full
#                                              N×M DINO matrix)
ALL_METRICS: tuple[str, ...] = ("mse_mse", "dino_dino", "mse_dino")


@dataclass
class Node:
    label: str
    paths: tuple[int, ...]
    depth: int


@dataclass
class Match:
    gt_index: int
    gt_label: str
    pred_index: int
    pred_label: str
    distance: float


@dataclass
class MetricScores:
    """Per-direction aggregates and the matches that produced them.

    For each pair (mse_mse or dino_dino):
      - ``recall`` / ``precision`` are the raw union-bbox distances at the
        matched pairs (lower better; MSE ≥ 0, DINO ∈ [0, 1]).
      - ``recall_sim`` / ``precision_sim`` map distance → similarity via
        ``clamp(1 - distance, 0, 1)``. For DINO this recovers cosine
        similarity exactly; for MSE it's a heuristic that's near-1 for
        good matches and degrades smoothly.
      - ``f1`` is the harmonic mean of the two sim values.
    """
    recall: float                     # mean union-bbox distance, GT → matched pred
    precision: float                  # mean union-bbox distance, Pred → matched gt
    recall_sim: float                 # clamp(1 - recall, 0, 1)
    precision_sim: float              # clamp(1 - precision, 0, 1)
    f1: float                         # 2 * P_sim * R_sim / (P_sim + R_sim)
    recall_matches: list[Match] = field(default_factory=list)     # one per GT node
    precision_matches: list[Match] = field(default_factory=list)  # one per Pred node
    # Occlusion-aware supplementary recall (mse_mse only). NaN when:
    # - the pair isn't mse_mse, or
    # - no GT node has a non-empty contested bbox.
    recall_occluded: float = float("nan")
    n_anchors_with_occlusion: int = 0


@dataclass
class SemanticResult:
    n_gt: int
    m_pred: int
    per_metric: dict[str, MetricScores] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_gt": self.n_gt,
            "m_pred": self.m_pred,
            "per_metric": {
                name: {
                    "recall": s.recall,
                    "precision": s.precision,
                    "recall_sim": s.recall_sim,
                    "precision_sim": s.precision_sim,
                    "f1": s.f1,
                    "recall_occluded": s.recall_occluded,
                    "n_anchors_with_occlusion": s.n_anchors_with_occlusion,
                    "recall_matches": [m.__dict__ for m in s.recall_matches],
                    "precision_matches": [m.__dict__ for m in s.precision_matches],
                }
                for name, s in self.per_metric.items()
            },
        }


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def flatten_tree(tree: dict, *, include_none: bool = False,
                 min_depth: int = 0) -> list[Node]:
    out: list[Node] = []
    root = tree.get("root", tree)

    def walk(node: dict, depth: int) -> None:
        if not isinstance(node, dict):
            return
        for child in node.get("children", []) or []:
            label = str(child.get("label", "")).strip()
            paths = tuple(int(p) for p in child.get("paths", []) or [])
            if (paths and depth >= min_depth
                    and (include_none or label.lower() != "none")):
                out.append(Node(label=label, paths=paths, depth=depth))
            walk(child, depth + 1)

    walk(root, 0)
    return out


def render_subset_rgb(svg_text: str, keep_indices: set[int],
                      size: int = 256) -> np.ndarray:
    """Render the SVG with only drawable primitives at ``keep_indices`` visible.

    Drawables outside ``keep_indices`` are removed from the document; their
    parent containers (``<g>`` with transforms etc.) are left intact so the
    surviving primitives still inherit any necessary coordinate transforms.
    The result is composited on white (matching :mod:`modules.evaluator`).

    Returns a float64 RGB array of shape ``(size, size, 3)`` in [0, 1].
    """
    import cairosvg

    root = ET.fromstring(svg_text)
    idx = [0]

    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            tag = _strip_ns(child.tag)
            if tag in _NON_RENDERED_CONTAINERS:
                continue
            if tag in _DRAWABLE_TAGS:
                if idx[0] not in keep_indices:
                    parent.remove(child)
                idx[0] += 1
            else:
                walk(child)

    walk(root)

    svg_bytes = ET.tostring(root)
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=size,
        output_height=size,
    )
    rgba = Image.open(BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[-1])
    return np.asarray(bg, dtype=np.float64) / 255.0


def count_drawables(svg_text: str) -> int:
    """Number of drawable primitives in the SVG, in document order.

    Uses the same DFS + ``_DRAWABLE_TAGS`` convention as
    :func:`render_subset_rgb` so the indices line up with what
    ``flatten_tree`` and tree.json's ``paths`` arrays refer to.
    """
    root = ET.fromstring(svg_text)
    idx = [0]

    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            tag = _strip_ns(child.tag)
            if tag in _NON_RENDERED_CONTAINERS:
                continue
            if tag in _DRAWABLE_TAGS:
                idx[0] += 1
            else:
                walk(child)

    walk(root)
    return idx[0]


def _arr_to_pil(arr: np.ndarray) -> Image.Image:
    """Float64 RGB in [0, 1] → PIL RGB uint8 (for DINO)."""
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")


def _content_bbox(arr: np.ndarray,
                  *, white_eps: float = 0.01) -> tuple[int, int, int, int] | None:
    """Tight bbox of non-white pixels as ``(y0, x0, y1, x1)`` half-open.
    Returns None when the array is all white."""
    mask = (arr < (1.0 - white_eps)).any(axis=-1)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _union_bbox_crop(gt_arr: np.ndarray, pred_arr: np.ndarray,
                     *, white_eps: float = 0.01
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Crop both renders to the union of their non-white bboxes, square-pad
    with white. Mirrors
    :func:`metric.semantic_flatten_v2._crop_to_union_bbox` (no padding
    margin). The union ensures the scoring metric sees every region where
    EITHER side drew content — small GT groups aren't inflated by
    surrounding white, and pred ink drawn outside the GT bbox isn't
    ignored. No resize — DINOv2's processor handles the 256/224 pipeline
    internally; MSE works at the crop resolution directly.
    """
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(
            f"gt and pred arrays must have same shape, got "
            f"{gt_arr.shape} vs {pred_arr.shape}"
        )
    a_box = _content_bbox(gt_arr, white_eps=white_eps)
    p_box = _content_bbox(pred_arr, white_eps=white_eps)
    if a_box is None and p_box is None:
        return gt_arr, pred_arr
    if a_box is None:
        u = p_box
    elif p_box is None:
        u = a_box
    else:
        u = (min(a_box[0], p_box[0]), min(a_box[1], p_box[1]),
             max(a_box[2], p_box[2]), max(a_box[3], p_box[3]))
    y0, x0, y1, x1 = u
    g = gt_arr[y0:y1, x0:x1]
    p = pred_arr[y0:y1, x0:x1]
    ch, cw = g.shape[:2]
    if ch == cw:
        return g, p
    side = max(ch, cw)
    oy = (side - ch) // 2
    ox = (side - cw) // 2
    pad_g = np.ones((side, side, 3), dtype=g.dtype)
    pad_p = np.ones((side, side, 3), dtype=p.dtype)
    pad_g[oy:oy + ch, ox:ox + cw] = g
    pad_p[oy:oy + ch, ox:ox + cw] = p
    return pad_g, pad_p


def _node_renders(svg_text: str, nodes: list[Node], size: int
                  ) -> list[np.ndarray]:
    """Render every node as a full-frame array (white background)."""
    return [render_subset_rgb(svg_text, set(n.paths), size=size) for n in nodes]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pair_distance(metric: str, gt_arr: np.ndarray, pred_arr: np.ndarray,
                   dino_model: Any | None = None) -> float:
    """Union-bbox distance between one GT node render and one pred node
    render. ``metric`` selects:
      ``"mse"``  → MSE on the cropped pair (lower better).
      ``"dino"`` → 1 - DINOv2 cosine sim on the cropped pair (∈[0,1]).
    """
    gt_crop, pred_crop = _union_bbox_crop(gt_arr, pred_arr)
    if metric == "mse":
        return float(mse_metric.compute(pred_crop, gt_crop))
    if metric == "dino":
        if dino_model is None:
            raise ValueError("dino_model is required for metric='dino'")
        sim = float(dino_metric.compute(
            dino_model, _arr_to_pil(pred_crop), _arr_to_pil(gt_crop)))
        return 1.0 - _clamp01(sim)
    raise ValueError(f"unknown metric {metric!r}; use 'mse' or 'dino'")


def _build_distance_matrix(metric: str,
                           gt_full: list[np.ndarray],
                           pred_full: list[np.ndarray],
                           dino_model: Any | None = None) -> np.ndarray:
    """N×M union-bbox distance matrix for the requested metric."""
    N, M = len(gt_full), len(pred_full)
    D = np.empty((N, M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            D[i, j] = _pair_distance(metric, gt_full[i], pred_full[j],
                                     dino_model=dino_model)
    return D


def _empty_metric_result(*, n_gt: int, m_pred: int,
                         metrics: Iterable[str]) -> SemanticResult:
    """Edge-case results when N==0 or M==0 — replicated per metric so
    callers always see the requested keys."""
    nan = float("nan")
    out: dict[str, MetricScores] = {}
    for m in metrics:
        if n_gt == 0:
            out[m] = MetricScores(recall=nan, precision=nan,
                                  recall_sim=nan, precision_sim=nan, f1=nan)
        elif m_pred == 0:
            out[m] = MetricScores(recall=1.0, precision=nan,
                                  recall_sim=0.0, precision_sim=nan, f1=nan)
    return SemanticResult(n_gt=n_gt, m_pred=m_pred, per_metric=out)


# Map pair key -> (judge_metric, score_metric). The judge metric drives
# the N×M matrix used for argmin matching; the score metric is the
# distance reported at the matched pair. When judge==score, the score
# is read directly off the judge matrix; otherwise it's computed on the
# matched (i, j) only.
_PAIR_JUDGE_SCORE: dict[str, tuple[str, str]] = {
    "mse_mse":   ("mse",  "mse"),
    "dino_dino": ("dino", "dino"),
    "mse_dino":  ("mse",  "dino"),
}


def _pair_needs_dino(pair_key: str) -> bool:
    j, s = _PAIR_JUDGE_SCORE[pair_key]
    return "dino" in (j, s)


def compute(gt_svg: str | Path, gt_tree: dict,
            pred_svg: str | Path, pred_tree: dict,
            *,
            size: int = 256,
            metrics: Iterable[str] = ALL_METRICS,
            dino_model: Any | None = None,
            lpips_model: Any | None = None,
            compute_occlusion: bool = False,
            occlusion_mode: str = "occluded") -> SemanticResult:
    """Group-by-group N×M matching with the V2.3 dual-pair setup.

    For each requested pair in ``metrics`` (default = both):
      1. Build the N×M union-bbox distance matrix using the pair's metric
         (``mse_mse`` → MSE; ``dino_dino`` → 1 - cosine sim).
      2. ``recall_j[i]  = argmin_j D[i, j]``  (each GT node → best pred).
      3. ``precision_i[j] = argmin_i D[i, j]`` (each pred node → best GT).
      4. ``recall`` / ``precision`` = mean of matched-pair distances.
      5. ``recall_sim`` / ``precision_sim`` = clamp(1 - distance, 0, 1).
      6. ``f1`` = harmonic mean of the two sim values.

    The judge (matrix metric) and the score (matrix-value at matched pair)
    are the same metric — mirroring :mod:`metric.semantic_flatten_v2`'s
    per-anchor judge-score contract.

    Args:
        gt_svg, gt_tree:    GT side — SVG path + tree.json dict.
        pred_svg, pred_tree: Pred side — same schema.
        size:               Render resolution per group (square).
        metrics:            Subset of :data:`ALL_METRICS`. Default = both
                              ``("mse_mse", "dino_dino")``.
        dino_model:         Pre-loaded DINOv2 model — required iff
                              ``"dino_dino"`` is in ``metrics``.
        lpips_model:        Accepted for backward compatibility; ignored.

    Edge cases:
      - GT empty: all scores NaN.
      - Pred empty: recall=1.0; precision and F1 NaN. Replicated per pair.
    """
    del lpips_model  # accepted for caller compat; no longer used

    metrics = tuple(dict.fromkeys(metrics))  # preserve order, dedup
    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        raise ValueError(f"unknown metric pair(s): {unknown}. "
                         f"Supported: {list(ALL_METRICS)}")
    dino_pairs = [m for m in metrics if _pair_needs_dino(m)]
    if dino_pairs and dino_model is None:
        raise ValueError(f"metrics include {dino_pairs} which need DINO but "
                         f"dino_model is None — load via "
                         f"metric.dino.load(device=...)")

    gt_text = Path(gt_svg).read_text()
    pred_text = Path(pred_svg).read_text()

    gt_nodes = flatten_tree(gt_tree)
    N = len(gt_nodes)
    if N == 0:
        return _empty_metric_result(n_gt=0, m_pred=0, metrics=metrics)

    gt_full = _node_renders(gt_text, gt_nodes, size)

    # Per-GT-node contested bbox (anchor-side only — independent of pred).
    # Reused across all (gt, pred) pairs to produce a second MSE matrix
    # restricted to the occluded region. Anchors with no contested overlap
    # carry bbox=None and are excluded from the occlusion mean. Skipped
    # entirely when ``compute_occlusion`` is off — saves N complement
    # renders per call.
    gt_occlusions: list = []
    if compute_occlusion:
        gt_all_paths: set[int] = set(range(count_drawables(gt_text)))
        gt_occlusions = [
            compute_anchor_occlusion(
                gt_text, set(n.paths), gt_all_paths,
                size=size, amodal_arr=gt_full[i], mode=occlusion_mode,
            )
            for i, n in enumerate(gt_nodes)
        ]

    pred_nodes = flatten_tree(pred_tree)
    M = len(pred_nodes)
    if M == 0:
        return _empty_metric_result(n_gt=N, m_pred=0, metrics=metrics)

    pred_full = _node_renders(pred_text, pred_nodes, size)

    per_metric: dict[str, MetricScores] = {}
    for pair_key in metrics:
        judge_m, score_m = _PAIR_JUDGE_SCORE[pair_key]
        D = _build_distance_matrix(judge_m, gt_full, pred_full,
                                   dino_model=dino_model)
        recall_j = D.argmin(axis=1)
        precision_i = D.argmin(axis=0)
        if judge_m == score_m:
            recall_dists = [float(D[i, int(recall_j[i])]) for i in range(N)]
            precision_dists = [float(D[int(precision_i[j]), j])
                               for j in range(M)]
        else:
            # Compute the score metric only at the matched indices. Cache
            # by (i, j) since recall and precision can land on the same
            # pair (e.g. when a single match is mutually best).
            _score_cache: dict[tuple[int, int], float] = {}

            def _score_at(i: int, j: int) -> float:
                key = (i, j)
                if key not in _score_cache:
                    _score_cache[key] = _pair_distance(
                        score_m, gt_full[i], pred_full[j],
                        dino_model=dino_model,
                    )
                return _score_cache[key]

            recall_dists = [_score_at(i, int(recall_j[i])) for i in range(N)]
            precision_dists = [_score_at(int(precision_i[j]), j)
                               for j in range(M)]
        recall = float(np.mean(recall_dists))
        precision = float(np.mean(precision_dists))
        recall_sim = _clamp01(1.0 - recall)
        precision_sim = _clamp01(1.0 - precision)
        denom = precision_sim + recall_sim
        f1 = (2.0 * precision_sim * recall_sim / denom) if denom > 0 else 0.0
        recall_matches = [
            Match(gt_index=i, gt_label=gt_nodes[i].label,
                  pred_index=int(recall_j[i]),
                  pred_label=pred_nodes[int(recall_j[i])].label,
                  distance=recall_dists[i])
            for i in range(N)
        ]
        precision_matches = [
            Match(gt_index=int(precision_i[j]),
                  gt_label=gt_nodes[int(precision_i[j])].label,
                  pred_index=j, pred_label=pred_nodes[j].label,
                  distance=precision_dists[j])
            for j in range(M)
        ]
        # Occlusion-aware supplementary recall: for each GT row i with a
        # non-empty contested bbox, MSE between that row's anchor render
        # and the recall-matched pred, cropped to the bbox. Computed only
        # for the mse_mse pair — DINO doesn't fit a tiny crop well. Off
        # entirely when ``compute_occlusion`` is False (gt_occlusions=[]).
        occl_recall = float("nan")
        n_occl = 0
        if compute_occlusion and pair_key == "mse_mse" and gt_occlusions:
            occl_vals: list[float] = []
            for i in range(N):
                mask = gt_occlusions[i].mask
                if not mask.any():
                    continue
                j = int(recall_j[i])
                occl_vals.append(masked_mse(pred_full[j], gt_full[i], mask))
            n_occl = len(occl_vals)
            if occl_vals:
                occl_recall = float(np.mean(occl_vals))

        per_metric[pair_key] = MetricScores(
            recall=recall, precision=precision,
            recall_sim=recall_sim, precision_sim=precision_sim, f1=f1,
            recall_matches=recall_matches,
            precision_matches=precision_matches,
            recall_occluded=occl_recall,
            n_anchors_with_occlusion=n_occl,
        )

    return SemanticResult(n_gt=N, m_pred=M, per_metric=per_metric)


def compute_score_only(gt_svg: str | Path, gt_tree: dict,
                       pred_svg: str | Path, pred_tree: dict,
                       *,
                       recall_matches: list,
                       precision_matches: list,
                       score_metric: str,
                       size: int = 256,
                       dino_model: Any | None = None) -> MetricScores:
    """Re-score a pre-computed set of matches with a different metric.

    Reuses the (gt_index, pred_index) pairs from a prior :func:`compute`
    run (e.g. from a cached mse_mse result) and recomputes the union-bbox
    score at those pairs under ``score_metric``. Avoids rebuilding the
    N×M judge matrix — only N+M renders + score evaluations.

    Args:
        recall_matches:    list of :class:`Match` or dicts with at least
                            ``gt_index``, ``pred_index``. Length == N
                            (one per GT node, in GT order).
        precision_matches: same shape, length == M (one per pred node).
        score_metric:       ``"mse"`` or ``"dino"``.

    Returns a :class:`MetricScores` with ``recall_occluded = NaN`` and
    ``n_anchors_with_occlusion = 0`` (occlusion is not re-evaluated here).
    """
    if score_metric not in ("mse", "dino"):
        raise ValueError(f"score_metric must be 'mse' or 'dino', got "
                         f"{score_metric!r}")
    if score_metric == "dino" and dino_model is None:
        raise ValueError("score_metric='dino' but dino_model is None")

    gt_nodes = flatten_tree(gt_tree)
    pred_nodes = flatten_tree(pred_tree)
    N, M = len(gt_nodes), len(pred_nodes)
    if N == 0 or M == 0:
        nan = float("nan")
        return MetricScores(
            recall=(1.0 if N > 0 and M == 0 else nan),
            precision=nan,
            recall_sim=(0.0 if N > 0 and M == 0 else nan),
            precision_sim=nan, f1=nan,
        )

    def _idx(m, key):
        return int(m[key] if isinstance(m, dict) else getattr(m, key))

    if len(recall_matches) != N:
        raise ValueError(f"recall_matches has {len(recall_matches)} entries, "
                         f"expected N={N}")
    if len(precision_matches) != M:
        raise ValueError(f"precision_matches has {len(precision_matches)} "
                         f"entries, expected M={M}")

    gt_text = Path(gt_svg).read_text()
    pred_text = Path(pred_svg).read_text()
    gt_full = _node_renders(gt_text, gt_nodes, size)
    pred_full = _node_renders(pred_text, pred_nodes, size)

    _cache: dict[tuple[int, int], float] = {}

    def _score(i: int, j: int) -> float:
        key = (i, j)
        if key not in _cache:
            _cache[key] = _pair_distance(score_metric, gt_full[i], pred_full[j],
                                         dino_model=dino_model)
        return _cache[key]

    recall_dists: list[float] = []
    out_recall_matches: list[Match] = []
    for i, m in enumerate(recall_matches):
        gi = _idx(m, "gt_index")
        pj = _idx(m, "pred_index")
        if gi != i:
            raise ValueError(f"recall_matches[{i}].gt_index={gi}, expected {i}")
        d = _score(gi, pj)
        recall_dists.append(d)
        out_recall_matches.append(
            Match(gt_index=gi, gt_label=gt_nodes[gi].label,
                  pred_index=pj, pred_label=pred_nodes[pj].label,
                  distance=d),
        )

    precision_dists: list[float] = []
    out_precision_matches: list[Match] = []
    for j, m in enumerate(precision_matches):
        gi = _idx(m, "gt_index")
        pj = _idx(m, "pred_index")
        if pj != j:
            raise ValueError(f"precision_matches[{j}].pred_index={pj}, "
                             f"expected {j}")
        d = _score(gi, pj)
        precision_dists.append(d)
        out_precision_matches.append(
            Match(gt_index=gi, gt_label=gt_nodes[gi].label,
                  pred_index=pj, pred_label=pred_nodes[pj].label,
                  distance=d),
        )

    recall = float(np.mean(recall_dists))
    precision = float(np.mean(precision_dists))
    recall_sim = _clamp01(1.0 - recall)
    precision_sim = _clamp01(1.0 - precision)
    denom = precision_sim + recall_sim
    f1 = (2.0 * precision_sim * recall_sim / denom) if denom > 0 else 0.0
    return MetricScores(
        recall=recall, precision=precision,
        recall_sim=recall_sim, precision_sim=precision_sim, f1=f1,
        recall_matches=out_recall_matches,
        precision_matches=out_precision_matches,
    )


__all__ = [
    "ALL_METRICS",
    "Node", "Match", "MetricScores", "SemanticResult",
    "flatten_tree", "render_subset_rgb", "count_drawables",
    "compute", "compute_score_only",
]
