#!/usr/bin/env python3
"""Evaluate your own image->SVG method on the Semantic-SVG Benchmark.

This is the main entry point for the benchmark. Point it at a directory of your
method's SVG predictions (one ``<id>.svg`` per benchmark sample) and it reports
every metric from the paper, overall and per category.

    # 1. fetch the benchmark (once)
    python scripts/prepare_benchmark.py                 # -> ./benchmark

    # 2. produce one SVG per sample, named by the benchmark id, e.g.
    #    my_preds/emoji_161.svg, my_preds/openclipart__189889.svg, ...
    #    (ids and their input PNGs are in ./benchmark — see the README)

    # 3. score
    python evaluate.py --pred-dir my_preds --name MyMethod

Metrics (see the README for definitions):

  pixel      MSE  (down, lower better)   pixel MSE of your render vs the GT render
             DINO (up,   higher better)  DINOv2 cosine similarity, whole image

  grouping   Precision / Recall / F1 (up) of your SVG's semantic grouping vs the
             GT tree. Your grouping is read from the SVG's <g> nesting; a flat
             SVG (no groups) scores as a single group. Reported for both the
             MSE-matched and DINO-matched variants.

  anchor     Recall (up, DINO) / distance (down, MSE): grouping-free per-object
             recall. Needs no <g> structure, so it is defined for any SVG.

Only ``huggingface_hub`` + ``cairosvg`` + ``torch`` are needed. DINO metrics load
DINOv2 on the GPU; pass ``--no-dino`` to skip them (MSE-side only, CPU-friendly).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metric import mse as mse_metric
from metric import semantic
from metric import semantic_flatten_v2 as anchor_metric
from modules.vectorizer import render_svg
from scripts.pred_tree_from_groups import build_tree_from_groups

RENDER_TIMEOUT_S = 60.0
EMPTY_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>'


# --------------------------------------------------------------------------- #
# rendering (isolated + timed: a malformed prediction must not wedge the run)  #
# --------------------------------------------------------------------------- #
def _render_child(svg: str, png: str, size, q) -> None:
    try:
        render_svg(Path(svg), Path(png), size=size)
        q.put("ok")
    except Exception as e:  # noqa: BLE001
        q.put(f"{type(e).__name__}: {e}")


def _render_pred(pred_svg: Path, out_png: Path, size) -> bool:
    """Render one prediction under a wall-clock cap. Returns True if it failed
    and a white penalty tile was written instead (cairosvg can hang forever on
    malformed markup and does not release the GIL, so we fork + kill)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_render_child, args=(str(pred_svg), str(out_png), size, q))
    p.start()
    p.join(RENDER_TIMEOUT_S)
    if p.is_alive():
        p.terminate(); p.join(5)
        Image.new("RGB", size, (255, 255, 255)).save(out_png)
        return True
    try:
        res = q.get_nowait()
    except Exception:
        res = "no result"
    if res == "ok" and out_png.is_file():
        return False
    Image.new("RGB", size, (255, 255, 255)).save(out_png)
    return True


def _on_white(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[-1])
    return bg


def _mean(vals):
    xs = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(np.mean(xs)) if xs else float("nan")


# --------------------------------------------------------------------------- #
def _load_manifest(gt_root: Path):
    names = [l.strip() for l in (gt_root / "basenames.txt").read_text().splitlines() if l.strip()]
    cats = {}
    mf = gt_root / "manifest.json"
    if mf.is_file():
        for e in json.loads(mf.read_text()):
            cats[e["basename"]] = e.get("category") or e["basename"].rsplit("_", 1)[0]
    return names, cats


def _score_pixel(gt_png: Path, pred_svg: Path, work: Path, dino_model):
    gt = _on_white(Image.open(gt_png))
    rendered = work / f"{pred_svg.stem}.png"
    penalty = _render_pred(pred_svg, rendered, gt.size)
    pr = _on_white(Image.open(rendered))
    if pr.size != gt.size:
        pr = pr.resize(gt.size, Image.LANCZOS)
    a = np.asarray(gt, np.float64) / 255.0
    b = np.asarray(pr, np.float64) / 255.0
    out = {"mse": float(mse_metric.compute(a, b)), "penalty": penalty}
    if dino_model is not None:
        from metric import dino as dino_metric
        out["dino"] = float(dino_metric.compute(dino_model, gt, pr))
    return out


def _score_grouping(gt_svg: Path, gt_tree: dict, pred_svg: Path, size, dino_model, empty_svg: Path):
    try:
        tree, _ = build_tree_from_groups(pred_svg.read_text())
    except Exception:  # malformed SVG won't parse
        tree = None
    unparseable = tree is None
    if unparseable:  # score against a blank single-group canvas (see eval_semantic_all)
        pred_svg = empty_svg
        tree = {"root": {"children": [{"label": "__blank__", "paths": [0]}]}}
    metrics = ("mse_mse", "dino_dino") if dino_model is not None else ("mse_mse",)
    res = semantic.compute(gt_svg, gt_tree, pred_svg, tree, size=size,
                           metrics=metrics, dino_model=dino_model)
    out = {"unparseable": unparseable}
    for key in metrics:
        s = res.per_metric[key]
        out[key] = {"precision": s.precision_sim, "recall": s.recall_sim, "f1": s.f1}
    return out


def _score_anchor(gt_svg: Path, gt_tree: dict, pred_svg: Path, size, workers, dino_model, empty_svg: Path):
    try:
        semantic.count_drawables(pred_svg.read_text())
        use = pred_svg
        unparseable = False
    except Exception:
        use = empty_svg
        unparseable = True
    pairs = anchor_metric.compute_pairs(gt_svg, gt_tree, use, size=size,
                                        bbox_padding=0.05, max_workers=workers,
                                        dino_model=dino_model)
    out = {"unparseable": unparseable, "recall_mse": pairs["mse_mse"].recall_mse}
    if dino_model is not None:
        out["recall_dino"] = 0.0 if unparseable else pairs["dino_dino"].recall_dino
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True,
                    help="directory of your predictions, one <id>.svg per sample")
    ap.add_argument("--gt-root", type=Path, default=_REPO / "benchmark",
                    help="benchmark root from scripts/prepare_benchmark.py (default: ./benchmark)")
    ap.add_argument("--name", default="method", help="label for this method in the report")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write results (default: <pred-dir>/_eval)")
    ap.add_argument("--metrics", default="all",
                    help="comma list of {pixel,grouping,anchor} or 'all' (default)")
    ap.add_argument("--size", type=int, default=256, help="semantic render size (default 256)")
    ap.add_argument("--workers", type=int, default=4, help="anchor-recall threads (default 4)")
    ap.add_argument("--no-dino", action="store_true",
                    help="skip all DINO metrics (no GPU / model download needed)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N samples")
    args = ap.parse_args()

    which = {"pixel", "grouping", "anchor"} if args.metrics == "all" \
        else {m.strip() for m in args.metrics.split(",") if m.strip()}
    if not args.gt_root.is_dir():
        sys.exit(f"[error] benchmark not found at {args.gt_root}. "
                 f"Run: python scripts/prepare_benchmark.py --out {args.gt_root}")
    if not args.pred_dir.is_dir():
        sys.exit(f"[error] --pred-dir {args.pred_dir} is not a directory")

    names, cats = _load_manifest(args.gt_root)
    if args.limit:
        names = names[:args.limit]
    out_dir = args.out_dir or (args.pred_dir / "_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_rendered"; work.mkdir(exist_ok=True)
    empty_svg = out_dir / "_empty.svg"; empty_svg.write_text(EMPTY_SVG)

    dino_model = None
    if not args.no_dino:
        try:
            import torch
            from metric import dino as dino_metric
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[load] DINOv2 on {dev}")
            dino_model = dino_metric.load(device=dev)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not load DINO ({type(e).__name__}: {e}); "
                  f"continuing with MSE-side metrics only. Use --no-dino to silence.")
            dino_model = None

    print(f"[eval] {args.name}: {len(names)} samples  metrics={sorted(which)}  "
          f"dino={'on' if dino_model is not None else 'off'}")

    rows, t0 = [], time.time()
    n_missing = 0
    for i, b in enumerate(names, 1):
        gt_svg = args.gt_root / "svg" / f"{b}.svg"
        gt_png = args.gt_root / "png" / f"{b}.png"
        pred = args.pred_dir / f"{b}.svg"
        row = {"basename": b, "category": cats.get(b, "?")}
        if not pred.is_file():
            row["status"] = "missing"
            n_missing += 1
            rows.append(row)
            continue
        row["status"] = "ok"
        gt_tree = json.loads((args.gt_root / "svg" / f"{b}.tree.json").read_text())
        if "pixel" in which and gt_png.is_file():
            px = _score_pixel(gt_png, pred, work, dino_model)
            row["mse"] = px["mse"]; row["pixel_penalty"] = px["penalty"]
            if "dino" in px:
                row["dino"] = px["dino"]
        if "grouping" in which:
            g = _score_grouping(gt_svg, gt_tree, pred, args.size, dino_model, empty_svg)
            row["group_unparseable"] = g["unparseable"]
            for k in ("mse_mse", "dino_dino"):
                if k in g:
                    row[f"group_{k}_f1"] = g[k]["f1"]
                    row[f"group_{k}_precision"] = g[k]["precision"]
                    row[f"group_{k}_recall"] = g[k]["recall"]
        if "anchor" in which:
            a = _score_anchor(gt_svg, gt_tree, pred, args.size, args.workers, dino_model, empty_svg)
            row["anchor_recall_mse"] = a["recall_mse"]
            if "recall_dino" in a:
                row["anchor_recall_dino"] = a["recall_dino"]
        rows.append(row)
        if i % 10 == 0 or i == len(names):
            print(f"  [{i}/{len(names)}]  {time.time()-t0:.0f}s", flush=True)

    _report(args.name, rows, which, dino_model is not None, out_dir, n_missing)


def _agg(rows, key):
    return _mean([r[key] for r in rows if key in r])


def _report(name, rows, which, has_dino, out_dir, n_missing):
    scored = [r for r in rows if r.get("status") == "ok"]
    summary = {"method": name, "n": len(scored), "n_missing": n_missing,
               "overall": {}, "per_category": {}}

    def collect(subset):
        d = {}
        if "pixel" in which:
            d["mse"] = _agg(subset, "mse")
            if has_dino:
                d["dino"] = _agg(subset, "dino")
        if "grouping" in which:
            d["grouping_f1"] = _agg(subset, "group_dino_dino_f1" if has_dino else "group_mse_mse_f1")
            d["grouping_precision"] = _agg(subset, "group_dino_dino_precision" if has_dino else "group_mse_mse_precision")
            d["grouping_recall"] = _agg(subset, "group_dino_dino_recall" if has_dino else "group_mse_mse_recall")
        if "anchor" in which:
            if has_dino:
                d["anchor_recall_dino"] = _agg(subset, "anchor_recall_dino")
            d["anchor_recall_mse_dist"] = _agg(subset, "anchor_recall_mse")
        return d

    summary["overall"] = collect(scored)
    cats = sorted({r["category"] for r in scored})
    for c in cats:
        summary["per_category"][c] = collect([r for r in scored if r["category"] == c])

    # CSV (per-sample) + JSON (summary)
    if rows:
        fields = sorted({k for r in rows for k in r})
        with (out_dir / "per_sample.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    o = summary["overall"]
    print(f"\n{'='*64}\n  {name}   (n={len(scored)}, missing={n_missing})\n{'='*64}")
    if "pixel" in which:
        line = f"  pixel      MSE(↓) {o['mse']:.4f}"
        if has_dino:
            line += f"    DINO(↑) {o['dino']:.4f}"
        print(line)
    if "grouping" in which:
        tag = "DINO" if has_dino else "MSE"
        print(f"  grouping   F1(↑) {o['grouping_f1']:.4f}   "
              f"P(↑) {o['grouping_precision']:.4f}   R(↑) {o['grouping_recall']:.4f}   [{tag}-scored]")
    if "anchor" in which:
        line = "  anchor     "
        if has_dino:
            line += f"Recall/DINO(↑) {o['anchor_recall_dino']:.4f}   "
        line += f"Recall/MSE-dist(↓) {o['anchor_recall_mse_dist']:.4f}"
        print(line)
    print(f"\n  per-category F1/MSE by {', '.join(cats)}  ->  {out_dir/'summary.json'}")
    print(f"  per-sample rows                          ->  {out_dir/'per_sample.csv'}")


if __name__ == "__main__":
    main()
