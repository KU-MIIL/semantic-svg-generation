"""Thin wrapper around the ``vtracer`` Python binding.

Each per-part RGBA layer goes through :func:`vectorize_layer`, which writes
an SVG next to the input PNG. The pipeline controller is responsible for
deciding what to do with the resulting files (render back, compose, etc.).

Install:
    pip install vtracer
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class VectorizeParams:
    colormode: str = "color"           # "color" | "binary"
    # "stacked" is the paper's setting and the original default. Commit
    # e76e2f1 ("multi-candidate sampling ... VLM vectorizer") silently flipped
    # it to "cutout", which is a different baseline entirely: cutout removes
    # overlapped area from lower layers, so its paths line up with group
    # boundaries and the post-hoc optimal grouping metric scores it ~2.4x
    # better. scripts/vtracer_baseline.py took this default, so the whole
    # camera-ready vtracer row was measured in the wrong mode. Keep it
    # "stacked"; pass hierarchical= explicitly if you want cutout.
    hierarchical: str = "stacked"      # "stacked" | "cutout"
    mode: str = "spline"               # "spline" | "polygon" | "none"
    filter_speckle: int = 4
    color_precision: int = 6
    layer_difference: int = 16
    corner_threshold: int = 60
    length_threshold: float = 4.0
    max_iterations: int = 10
    splice_threshold: int = 45
    path_precision: int = 3


def vectorize_layer(
    png_path: Path,
    svg_path: Path,
    params: VectorizeParams | None = None,
) -> Path:
    """Run vtracer on ``png_path``; write SVG to ``svg_path``; return ``svg_path``.

    Transparent pixels are preserved — vtracer respects the alpha channel
    and won't emit paths for fully-transparent regions.
    """
    import vtracer

    p = params or VectorizeParams()
    png_path = Path(png_path)
    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    vtracer.convert_image_to_svg_py(
        str(png_path),
        str(svg_path),
        colormode=p.colormode,
        hierarchical=p.hierarchical,
        mode=p.mode,
        filter_speckle=p.filter_speckle,
        color_precision=p.color_precision,
        layer_difference=p.layer_difference,
        corner_threshold=p.corner_threshold,
        length_threshold=p.length_threshold,
        max_iterations=p.max_iterations,
        splice_threshold=p.splice_threshold,
        path_precision=p.path_precision,
    )
    return svg_path


def render_svg(svg_path: Path, png_path: Path, size: tuple[int, int] | None = None) -> Path:
    """Sanity-render an SVG back to PNG so the caller can diff against the source."""
    import cairosvg

    svg_path = Path(svg_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if size is not None:
        kwargs["output_width"], kwargs["output_height"] = size
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), **kwargs)
    return png_path


__all__ = ["VectorizeParams", "vectorize_layer", "render_svg"]
