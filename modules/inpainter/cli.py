"""CLI entry point for one-off inpaint completion runs.

    GEMINI_API_KEY=... python -m modules.inpainter.cli \
        --part part.png --scene scene.png --label sledgehammer \
        --device cuda:0 --outdir out/

Writes ``<stem>__mask.png``, ``<stem>__fill_rgba.png``, and
``<stem>__fill_on_white.png`` to ``--outdir``. Useful for sanity-
checking the polygon + FLUX path on a single (part, scene, label) pair
without driving the full SVGAgent pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

from .agent import SEED, InpaintAgent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    help="SAM part cutout (RGBA png)")
    ap.add_argument("--scene", required=True,
                    help="full original scene (RGB png)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default="inpaint_out")
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    part = Image.open(args.part).convert("RGBA")
    scene = Image.open(args.scene).convert("RGB")
    agent = InpaintAgent(device=args.device)
    r = agent.inpaint(part, scene, args.label, seed=args.seed)
    stem = Path(args.part).stem
    r["repaint_mask"].save(out / f"{stem}__mask.png")
    r["fill_rgba"].save(out / f"{stem}__fill_rgba.png")
    r["fill_on_white"].save(out / f"{stem}__fill_on_white.png")
    print(f"[{stem}] verts={r['n_vertices']} "
          f"err={r['parse_error']} -> {out}/")


if __name__ == "__main__":
    main()
