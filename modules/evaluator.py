"""Pixel-wise evaluator: compare ``input.png`` against ``merged.png`` per sample.

Walks a batch directory produced by :mod:`pipeline_controller`, computes
MSE / SSIM / LPIPS / DINO for every sample that has both ``input.png`` and
``merged.png``, writes a per-sample ``metrics.json`` into each sample dir,
and emits ``<batch>/metrics_index.json`` with per-sample rows + aggregates.

RGBA inputs are composited on a white background (out = rgb*a + 1*(1-a))
before any metric runs, so the arbitrary RGB values stored in transparent
regions don't leak into the scores. Images keep their native resolution;
the merged side is resized to match the input side only if they differ.

Usage:
    python3 -m modules.evaluator runs/20260421_120006_batch
    python3 -m modules.evaluator runs/20260421_120006_batch --metrics mse,ssim --limit 2
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from metric import dino as dino_metric
from metric import lpips as lpips_metric
from metric import mse as mse_metric
from metric import ssim as ssim_metric


ALL_METRICS = ("mse", "ssim", "lpips", "dino")
METRIC_ROUND_PLACES = 4


def _round_metric(v):
    """Round a metric value to ``METRIC_ROUND_PLACES`` decimals; pass through
    ``None`` unchanged so error/missing entries stay distinguishable from 0."""
    return round(v, METRIC_ROUND_PLACES) if isinstance(v, (int, float)) else v


@dataclass
class SampleMetrics:
    sample: str
    status: str                                # "ok" | "missing_merged" | "error"
    mse: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    dino: float | None = None
    errors: dict[str, str] | None = None
    time_s: float | None = None
    # Failure-mode flags sourced from <sample_dir>/meta.json (written by
    # scripts/vlm_vectorize.py). For non-VLM pipelines these stay False.
    no_compile: bool = False                   # SVG was substituted by void <svg></svg>
    post_processed: bool = False               # clean_svg() was used (whether successful or void fallback)


def _pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _composite_on_white(img: Image.Image) -> Image.Image:
    """RGBA → RGB on white via alpha compositing: out = rgb*a + 1*(1-a).

    PIL's ``Image.new + paste(mask=alpha)`` implements exactly this formula,
    which matches the reference ``compute_ssim_lpips_mse.py`` / ``compute_dion.py``
    scripts. A bare ``.convert("RGB")`` would drop alpha without blending,
    leaving arbitrary RGB values in transparent regions to dominate the metrics.
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[-1])
    return bg


def _prepare_pair(input_path: Path, merged_path: Path
                  ) -> tuple[np.ndarray, np.ndarray, Image.Image, Image.Image]:
    """Load both images, composite on white, align sizes.

    Returns ``(arr_a, arr_b, pil_a, pil_b)``: ``arr_*`` are float64 RGB in
    [0, 1] with shape ``(H, W, 3)``; ``pil_*`` are PIL RGB at the same size.
    The merged side is resized to the input side only if sizes differ.
    """
    pil_a = _composite_on_white(Image.open(input_path))
    pil_b = _composite_on_white(Image.open(merged_path))
    if pil_b.size != pil_a.size:
        pil_b = pil_b.resize(pil_a.size, Image.LANCZOS)
    arr_a = np.asarray(pil_a, dtype=np.float64) / 255.0
    arr_b = np.asarray(pil_b, dtype=np.float64) / 255.0
    return arr_a, arr_b, pil_a, pil_b


def _safe(fn, *args, errors: dict[str, str], name: str):
    """Run ``fn(*args)``; on exception, record in ``errors`` and return None."""
    try:
        return fn(*args)
    except Exception as e:
        errors[name] = f"{type(e).__name__}: {e}"
        print(f"[warn] {name} failed: {errors[name]}")
        return None


def evaluate_sample(sample_dir: Path,
                    lpips_model: lpips_metric.LpipsModel | None,
                    dino_model: dino_metric.DinoModel | None,
                    metrics_filter: set[str],
                    merged_name: str = "merged.png") -> SampleMetrics:
    """Run all (filtered) metrics on one sample directory."""
    name = sample_dir.name
    input_path = sample_dir / "input.png"
    merged_path = sample_dir / merged_name

    # Pull failure-mode flags from the generation step's meta.json (if any).
    # Surfaced into both SampleMetrics and aggregates so cross-model
    # comparisons can show that "model X scored well only because we dropped
    # 88% of its outputs" vs "model Y scored worse because failures were
    # included as white-image penalties".
    no_compile = False
    post_processed = False
    meta_path = sample_dir / "meta.json"
    if meta_path.is_file():
        try:
            m = json.loads(meta_path.read_text())
            no_compile = bool(m.get("no_compile"))
            post_processed = bool(m.get("post_processed"))
        except (ValueError, OSError):
            pass

    if not input_path.is_file():
        return SampleMetrics(sample=name, status="error",
                             errors={"prepare": "missing input.png"},
                             no_compile=no_compile,
                             post_processed=post_processed)
    if not merged_path.is_file():
        return SampleMetrics(sample=name, status="missing_merged",
                             no_compile=no_compile,
                             post_processed=post_processed)

    t0 = time.time()
    errors: dict[str, str] = {}
    try:
        arr_a, arr_b, pil_a, pil_b = _prepare_pair(input_path, merged_path)
    except Exception as e:
        return SampleMetrics(sample=name, status="error",
                             errors={"prepare": f"{type(e).__name__}: {e}"},
                             time_s=time.time() - t0,
                             no_compile=no_compile,
                             post_processed=post_processed)

    result = SampleMetrics(sample=name, status="ok",
                           no_compile=no_compile,
                           post_processed=post_processed)

    if "mse" in metrics_filter:
        result.mse = _round_metric(_safe(mse_metric.compute, arr_a, arr_b,
                                         errors=errors, name="mse"))
    if "ssim" in metrics_filter:
        result.ssim = _round_metric(_safe(ssim_metric.compute, arr_a, arr_b,
                                          errors=errors, name="ssim"))
    if "lpips" in metrics_filter and lpips_model is not None:
        result.lpips = _round_metric(_safe(
            lpips_metric.compute, lpips_model, pil_a, pil_b,
            errors=errors, name="lpips"))
    if "dino" in metrics_filter and dino_model is not None:
        result.dino = _round_metric(_safe(
            dino_metric.compute, dino_model, pil_a, pil_b,
            errors=errors, name="dino"))

    result.errors = errors or None
    result.time_s = time.time() - t0

    (sample_dir / "metrics.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False)
    )
    return result


def _aggregate(samples: list[SampleMetrics]) -> dict:
    """Mean of each metric over status=='ok' samples, plus failure-mode
    ratios (matches StarVector's ``ratio_non_compiling`` /
    ``ratio_post_processed``).

    NaN values get skipped alongside None — under failure-inclusive scoring
    a metric backend may emit NaN on a degenerate input (e.g. SSIM on a
    constant white image), and we don't want a single NaN to poison the mean.
    """
    agg: dict = {}
    ok = [s for s in samples if s.status == "ok"]
    for m in ALL_METRICS:
        vals = [
            getattr(s, m) for s in ok
            if getattr(s, m) is not None and not math.isnan(getattr(s, m))
        ]
        if not vals:
            continue
        agg[m] = _round_metric(float(statistics.fmean(vals)))

    if ok:
        agg["ratio_non_compiling"] = _round_metric(
            sum(s.no_compile for s in ok) / len(ok)
        )
        agg["ratio_post_processed"] = _round_metric(
            sum(s.post_processed for s in ok) / len(ok)
        )
    return agg


def evaluate_batch(batch_dir: Path,
                   device: str | None = None,
                   metrics_filter: set[str] | None = None,
                   limit: int | None = None,
                   merged_name: str = "merged.png") -> dict:
    """Iterate sample subdirs under ``batch_dir`` and compute metrics."""
    batch_dir = Path(batch_dir)
    metrics_filter = metrics_filter or set(ALL_METRICS)
    device = _pick_device(device)
    print(f"[eval] batch={batch_dir} device={device} "
          f"metrics={sorted(metrics_filter)}")

    lpips_model = lpips_metric.load(device=device) if "lpips" in metrics_filter else None
    dino_model = dino_metric.load(device=device) if "dino" in metrics_filter else None

    sample_dirs = sorted(p for p in batch_dir.iterdir() if p.is_dir())
    if limit is not None and limit > 0:
        sample_dirs = sample_dirs[:limit]

    samples: list[SampleMetrics] = []
    for i, sd in enumerate(sample_dirs, 1):
        sm = evaluate_sample(sd, lpips_model, dino_model, metrics_filter,
                             merged_name=merged_name)
        samples.append(sm)
        summary_bits = [f"status={sm.status}"]
        for m in ALL_METRICS:
            v = getattr(sm, m)
            if v is not None:
                summary_bits.append(f"{m}={v:.3f}")
        print(f"[eval] ({i}/{len(sample_dirs)}) {sm.sample}  " + "  ".join(summary_bits))

    n_ok = sum(1 for s in samples if s.status == "ok")
    n_missing = sum(1 for s in samples if s.status == "missing_merged")
    n_error = sum(1 for s in samples if s.status == "error")

    payload = {
        "batch_dir": str(batch_dir),
        "device": device,
        "metrics": sorted(metrics_filter),
        "n_total": len(samples),
        "n_ok": n_ok,
        "n_missing_merged": n_missing,
        "n_error": n_error,
        "aggregates": _aggregate(samples),
        "samples": [asdict(s) for s in samples],
    }

    index_path = batch_dir / "metrics_index.json"
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[done] metrics_index at {index_path}  "
          f"(ok={n_ok} missing={n_missing} error={n_error})")
    return payload


def main():
    ap = argparse.ArgumentParser(
        description="Pixel-wise evaluator for a vectorizer batch run."
    )
    ap.add_argument("batch_dir",
                    help="Path to a batch directory, e.g. runs/20260421_120006_batch")
    ap.add_argument("--device", default=None,
                    help="cuda | cpu (default: auto-detect).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of samples to evaluate.")
    ap.add_argument("--metrics", default=",".join(ALL_METRICS),
                    help="Comma-separated subset of metrics to run "
                         f"(default: all of {','.join(ALL_METRICS)}).")
    ap.add_argument("--merged-name", default="merged.png",
                    help="Filename inside each sample dir to compare against "
                         "input.png (default: merged.png for pipeline runs; "
                         "use output.png for vlm_vectorize / vtracer batches).")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        ap.error(f"not a directory: {batch_dir}")

    requested = {m.strip() for m in args.metrics.split(",") if m.strip()}
    unknown = requested - set(ALL_METRICS)
    if unknown:
        ap.error(f"unknown metric(s): {sorted(unknown)}. "
                 f"Supported: {list(ALL_METRICS)}")

    evaluate_batch(batch_dir, device=args.device,
                   metrics_filter=requested, limit=args.limit,
                   merged_name=args.merged_name)


if __name__ == "__main__":
    main()
