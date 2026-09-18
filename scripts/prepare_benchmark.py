#!/usr/bin/env python3
"""Download the Semantic-SVG benchmark from HuggingFace into the local layout
the evaluation scripts expect.

The benchmark is published (evaluation-only, ``test`` split, 203 samples) at

    https://huggingface.co/datasets/KU-MIIL/Semantic-SVG-Benchmark

as ``svg/<id>.svg`` + ``trees/<id>.tree.json`` (+ ``manifest.csv``). This script
fetches those, renders a reference ``png/<id>.png`` for the pixel metrics, and
writes ``basenames.txt`` + ``manifest.json`` so that both

    scripts/eval_semantic_all.py        (grouping / anchor-recall, reads svg/*)
    scripts/eval_mse_dino_all_methods.py (MSE / DINO, reads png/*)

run against it with no further arguments (their ``--gt-root`` defaults to the
``benchmark/`` dir produced here).

The sample id is used verbatim as the on-disk stem, so predictions must be named
by the same id, e.g. ``paper_results/<method>/all/<id>.svg``.

    python3 scripts/prepare_benchmark.py                 # -> ./benchmark
    python3 scripts/prepare_benchmark.py --out /data/ssvg --size 448

Only ``huggingface_hub`` (download) and ``cairosvg`` (render, via
``modules.vectorizer``) are required; ``datasets``/``pyarrow`` are not.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules.vectorizer import render_svg

HF_REPO = "KU-MIIL/Semantic-SVG-Benchmark"
# Coarse category (emoji / icon / illustration) per sample. This is the axis the
# paper's per-category tables break down on; it is NOT one of the columns the HF
# dataset publishes (the web-curated samples carry a fine `source` such as
# openclipart/twemoji, which does not map 1:1 to the coarse class), so the
# annotation is shipped in-repo and merged into the local manifest here.
CATEGORIES_FILE = _REPO_ROOT / "scripts" / "benchmark_categories.json"


def _download(repo: str, out_snapshot: Path) -> Path:
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(out_snapshot),
        allow_patterns=["svg/*", "trees/*", "manifest.csv"],
    )
    return Path(local)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "benchmark",
                    help="output benchmark root (default: <repo>/benchmark)")
    ap.add_argument("--repo", default=HF_REPO,
                    help=f"HF dataset repo id (default: {HF_REPO})")
    ap.add_argument("--size", type=int, default=448,
                    help="reference PNG render size in px (default: 448, square)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="HF download cache / snapshot dir "
                         "(default: <out>/_hf_snapshot)")
    ap.add_argument("--skip-render", action="store_true",
                    help="only fetch svg/tree + write manifest, do not render PNGs")
    args = ap.parse_args()

    out: Path = args.out
    snapshot = args.cache_dir or (out / "_hf_snapshot")
    (out / "svg").mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(parents=True, exist_ok=True)

    print(f"[prepare] downloading {args.repo} -> {snapshot}")
    src = _download(args.repo, snapshot)

    # Source of truth for the sample set + per-sample metadata.
    manifest_csv = src / "manifest.csv"
    with manifest_csv.open() as f:
        rows = list(csv.DictReader(f))
    ids = [r["id"] for r in rows]
    print(f"[prepare] {len(ids)} samples in manifest.csv")

    categories = {}
    if CATEGORIES_FILE.is_file():
        categories = json.loads(CATEGORIES_FILE.read_text())
    else:
        print(f"[warn] {CATEGORIES_FILE.name} missing — category will fall back to `source`")

    manifest_out: list[dict] = []
    missing_svg: list[str] = []
    n_rendered = 0
    for i, r in enumerate(rows, 1):
        sid = r["id"]
        src_svg = src / "svg" / f"{sid}.svg"
        src_tree = src / "trees" / f"{sid}.tree.json"
        if not src_svg.is_file() or not src_tree.is_file():
            missing_svg.append(sid)
            continue

        dst_svg = out / "svg" / f"{sid}.svg"
        dst_tree = out / "svg" / f"{sid}.tree.json"
        dst_svg.write_bytes(src_svg.read_bytes())
        # trees/<id>.tree.json lives next to the svg here, the layout the eval
        # scripts glob (<gt-root>/svg/<id>.tree.json).
        dst_tree.write_bytes(src_tree.read_bytes())

        if not args.skip_render:
            dst_png = out / "png" / f"{sid}.png"
            if not dst_png.is_file():
                try:
                    render_svg(dst_svg, dst_png, size=(args.size, args.size))
                    n_rendered += 1
                except Exception as e:  # noqa: BLE001 - report and continue
                    print(f"  [render-fail] {sid}: {type(e).__name__}: {e}")

        category = categories.get(sid) or (r.get("source") or "")
        entry = {"basename": sid, "category": category}
        # carry the numeric/string metadata columns through verbatim
        for k in ("source", "origin", "upstream_art", "upstream_license",
                  "n_paths", "n_objects", "n_nodes", "tree_depth", "has_gradient"):
            if k in r:
                entry[k] = r[k]
        manifest_out.append(entry)

        if i % 25 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] fetched"
                  + ("" if args.skip_render else f", {n_rendered} rendered"))

    (out / "basenames.txt").write_text(
        "\n".join(e["basename"] for e in manifest_out) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False) + "\n")

    if missing_svg:
        print(f"[warn] {len(missing_svg)} ids had no svg/tree on the hub: "
              f"{missing_svg[:5]}{' ...' if len(missing_svg) > 5 else ''}")

    import collections
    by_cat = collections.Counter(e["category"] for e in manifest_out)
    print(f"[done] {len(manifest_out)} samples -> {out}")
    print(f"       basenames.txt, manifest.json, svg/, png/")
    print(f"       by category: {dict(by_cat)}")
    print(f"\nRun the evaluation with:")
    print(f"  python3 scripts/eval_semantic_all.py --methods Ours "
          f"--gt-root {out} --out-dir paper_results/eval_semantic")
    print(f"  python3 scripts/eval_mse_dino_all_methods.py "
          f"--methods Ours --gt-root {out}")


if __name__ == "__main__":
    main()
