"""Gemini "P3" shape-inference polygon: given the full scene and the
SAM-cutout layer, ask Gemini-3 for the *complete* outline of the part
(including the occluded portion).

Locked after a prompt/mask ablation — direct base64-PNG mask responses
were unstable/expensive, polygon-of-coords is light, full-res, and
robust. The polygon is rasterized to a binary mask downstream by
:mod:`modules.inpainter.mask`.
"""
from __future__ import annotations

import json
import os
import re

from PIL import Image

# from . import _nothink  # noqa: F401  (disable Gemini CoT — must precede gemini)
from .imaging import rgba_on_white

GEMINI_MODEL = "gemini-3-flash-preview"

# Four polygon-prompt variants exposed via :data:`POLYGON_PROMPTS`
# (``p0|p1|p2|p3``; vertex range 30-50). Each prompt nudges Gemini to
# reason differently; the downstream Gemini pick-judge then picks the
# best of the N candidate inpaints. ``P3_POLYGON_PROMPT`` remains as a
# back-compat alias pointing at ``p0`` (the historical default).

# p0 (baseline) — no category prior, pure visual reasoning. Matches
# the legacy single-prompt path bit-for-bit.
_P0_PROMPT = """You are given two images:
  IMAGE 1: the full original scene.
  IMAGE 2: a cropped layer showing ONLY the part labeled "{label}"
           (white = transparent / not part of the layer). It is
           INCOMPLETE because other scene elements occlude it.
Coordinates: 0..1000 normalized over IMAGE 1 (origin top-left).

Infer the complete true shape of "{label}" as it exists in IMAGE 1,
reasoning ONLY from the visible evidence in IMAGE 2 (visible contour
curvature, proportions, symmetry) together with where occluders hide it
in IMAGE 1. Make NO assumption about object category or any typical /
textbook shape -- derive the shape from what is actually shown. Output
ONE closed polygon of that complete shape that:
- passes through ALL visible pixels of "{label}",
- continues the visible contour smoothly where it is smooth and with
  corners where it has corners,
- fills ONLY the genuinely occluded gaps, never exceeding a visible
  natural endpoint or entering background.
Single closed boundary, 30-50 vertices, short segments on curves.

Output ONLY a JSON object (no markdown):
{{"polygon": [[x1,y1], [x2,y2], ..., [xN,yN]]}}"""

# p1 — category-aware: actively uses domain knowledge of "{label}".
_P1_PROMPT = """You are given two images:
  IMAGE 1: the full original scene.
  IMAGE 2: a cropped layer showing ONLY the part labeled "{label}"
           (white = transparent / not part of the layer). It is
           INCOMPLETE because other scene elements occlude it.
Coordinates: 0..1000 normalized over IMAGE 1 (origin top-left).

Predict the full silhouette of "{label}" as a typical flat vector
illustration of that object would appear in IMAGE 1. Use BOTH:
  (a) the visible part in IMAGE 2 -- its actual contour, proportions,
      orientation, and where it is interrupted by an occluder,
  (b) common knowledge of what a "{label}" looks like as a clean
      illustration -- its overall outline and proportions.
Anchor the silhouette to the visible pixels (never deviate from them).
For occluded gaps, prefer the shape most consistent with (a) AND (b).
Never extend into background or past a visible natural endpoint.
Single closed boundary, 30-50 vertices, short segments on curves.

Output ONLY a JSON object (no markdown):
{{"polygon": [[x1,y1], [x2,y2], ..., [xN,yN]]}}"""

# p2 — structured reasoning: explicit VISIBLE / OCCLUDED two-step.
_P2_PROMPT = """You are given two images:
  IMAGE 1: the full original scene.
  IMAGE 2: a cropped layer showing ONLY the part labeled "{label}"
           (white = transparent / not part of the layer). It is
           INCOMPLETE because other scene elements occlude it.
Coordinates: 0..1000 normalized over IMAGE 1 (origin top-left).

Predict the complete outline of "{label}" as it exists in IMAGE 1,
including the parts hidden by occluders.

Reason in TWO steps before answering:
  STEP 1 -- VISIBLE: identify the contour of "{label}" that is fully
    visible in IMAGE 2. Note its curvature, corners, symmetry, and the
    natural endpoints where the silhouette is interrupted by another
    layer drawn on top.
  STEP 2 -- OCCLUDED: locate the regions in IMAGE 1 where another
    element covers "{label}". Estimate the most plausible continuation
    of the silhouette into those regions, using contour smoothness,
    symmetry, and proportional cues from STEP 1. Do not invent shape
    beyond what the visible evidence supports.

Then output ONE closed polygon that:
- passes through ALL visible pixels of "{label}",
- continues smoothly into the occluded regions,
- never enters background pixels or extends past visible endpoints.
Single closed boundary, 30-50 vertices, short segments on curves.

Output ONLY a JSON object (no markdown):
{{"polygon": [[x1,y1], [x2,y2], ..., [xN,yN]]}}"""

# p3 — geometric strict: MUST / NOT rules, terser.
_P3_PROMPT = """You are given two images:
  IMAGE 1: the full original scene.
  IMAGE 2: a cropped layer showing the part labeled "{label}", cut out
           on a white background. White = not part of the layer. The
           layer is INCOMPLETE because other elements occlude it in
           IMAGE 1.
Coordinates: 0..1000 normalized over IMAGE 1 (origin top-left).

Reconstruct the complete silhouette of "{label}" as a SINGLE closed
polygon. Rules:
  - The polygon MUST enclose every visible pixel of "{label}" from
    IMAGE 2 -- no visible pixel may fall outside the polygon.
  - The polygon must NOT extend into regions of IMAGE 1 that are
    clearly background (white / empty space) or that belong to layers
    other than the occluders of "{label}".
  - Closures across occluded gaps should be the simplest contour
    consistent with the visible silhouette's curvature and endpoint
    tangents.
  - Place more vertices on curved regions, fewer on straight runs.
Single closed boundary, 30-50 vertices.

Output ONLY a JSON object (no markdown):
{{"polygon": [[x1,y1], [x2,y2], ..., [xN,yN]]}}"""

POLYGON_PROMPTS: dict[str, str] = {
    "p0": _P0_PROMPT,
    "p1": _P1_PROMPT,
    "p2": _P2_PROMPT,
    "p3": _P3_PROMPT,
}

# Back-compat alias: older callers import P3_POLYGON_PROMPT directly.
P3_POLYGON_PROMPT = POLYGON_PROMPTS["p0"]


# Cached Gemini client. Lazy-init so importing this module doesn't try
# to read GEMINI_API_KEY at import time.
_GEMINI_CLIENT = None


def _gemini(model: str | None = None):
    """Return a client suitable for ``model``. ``Qwen*`` model names are
    routed to the vLLM shim in :mod:`modules.qwen_client`; ``gpt-*`` ids
    hit :mod:`modules.openai_client`; ``claude-*`` ids hit
    :mod:`modules.claude_client`; everything else uses the live Gemini
    API. Callers pass the per-call model so judge / polygon / picker
    can stay model-agnostic."""
    from modules.qwen_client import is_qwen_model, get_client as _qwen_client
    if is_qwen_model(model):
        return _qwen_client()
    from modules.openai_client import is_gpt_model, get_client as _openai_client
    if is_gpt_model(model):
        return _openai_client()
    from modules.claude_client import is_claude_model, get_client as _claude_client
    if is_claude_model(model):
        return _claude_client()
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _GEMINI_CLIENT = genai.Client(api_key=key)
    return _GEMINI_CLIENT


def _strip_json_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip().strip("`").strip()


def predict_complete_polygon(scene_rgb: Image.Image,
                             part_rgba: Image.Image,
                             label: str,
                             *,
                             model: str | None = None,
                             thinking_level: str | None = None,
                             prompt_id: str = "p0",
                             prompt_template: str | None = None,
                             ) -> tuple[list[tuple[int, int]], dict]:
    """Ask Gemini for the complete outline of ``label`` in scene coords.

    ``model`` overrides the default ``GEMINI_MODEL`` (e.g. ``gemini-3-flash-preview``).
    ``thinking_level`` (``minimal|low|medium|high``) routes through the
    Gemini-3 ``thinking_level`` knob — when set, the ``medium`` default
    is suppressed for this call so both params don't collide on the wire.

    ``prompt_id`` selects one of :data:`POLYGON_PROMPTS` (``p0|p1|p2|p3``).
    Pass ``prompt_template`` (must contain a ``{label}`` placeholder) to
    override completely; when set it wins over ``prompt_id``.

    Returns ``(polygon_pixels, debug)``. ``polygon_pixels`` is a list of
    ``(x, y)`` integer tuples in IMAGE-1 pixel space; empty list when the
    model fails to parse or emits < 3 vertices. ``debug`` carries the
    raw text, vertex count, prompt id, and any ``parse_error``.
    """
    from modules.qwen_client import is_qwen_model
    from modules.openai_client import is_gpt_model
    from modules.claude_client import is_claude_model
    use_qwen = is_qwen_model(model)
    use_gpt = is_gpt_model(model)
    use_claude = is_claude_model(model)
    if not (use_qwen or use_gpt or use_claude):
        from google.genai import types as _t

    if prompt_template is None:
        if prompt_id not in POLYGON_PROMPTS:
            raise ValueError(
                f"unknown prompt_id={prompt_id!r}; "
                f"expected one of {list(POLYGON_PROMPTS)}")
        tpl = POLYGON_PROMPTS[prompt_id]
    else:
        tpl = prompt_template

    part_on_white = rgba_on_white(part_rgba)
    use_model = model or GEMINI_MODEL
    call_kwargs: dict = dict(
        model=use_model,
        contents=[tpl.format(label=label), scene_rgb, part_on_white],
    )
    if thinking_level and not (use_qwen or use_gpt or use_claude):
        call_kwargs["config"] = _t.GenerateContentConfig(
            thinking_config=_t.ThinkingConfig(
                thinking_level=thinking_level.upper(),
                include_thoughts=False,
            ),
        )
    elif thinking_level and (use_gpt or use_claude):
        # Match the Gemini thinking_level on the GPT/Claude shims (Gemini
        # config is a google.genai-only type; the shims take a plain hint).
        call_kwargs["reasoning"] = thinking_level
    from modules.gemini_client import _usage_from_response, _log_usage
    from dataclasses import asdict
    resp = _gemini(use_model).models.generate_content(**call_kwargs)
    usage = _usage_from_response(resp, use_model, "inpaint_polygon")
    _log_usage(usage)
    raw = resp.text or ""
    text = _strip_json_fence(raw)
    dbg: dict = {"label": label, "raw": raw, "usage": asdict(usage),
                 "prompt_id": prompt_id if prompt_template is None else "custom"}

    poly_norm: list = []
    try:
        poly_norm = json.loads(text).get("polygon", [])
    except json.JSONDecodeError as e:
        m = re.search(r"\[\s*\[[^\]]+\]\s*(?:,\s*\[[^\]]+\])*\s*\]", text)
        if m:
            try:
                poly_norm = json.loads(m.group(0))
            except json.JSONDecodeError:
                dbg["parse_error"] = str(e)
                return [], dbg
        else:
            dbg["parse_error"] = str(e)
            return [], dbg

    if not poly_norm or not isinstance(poly_norm, list):
        dbg["parse_error"] = "polygon empty or not a list"
        return [], dbg

    W, H = scene_rgb.size
    # Per-model coordinate convention. Verified by recall (poly ∩ visible /
    # |visible|) on sample 109.png:
    #   gemini-3.5-flash         : returns [y, x] (bbox-style), need swap.
    #                              recall_direct=0.62 vs recall_swap=0.65
    #                              on 4 leaves; swap wins on 3/4 cases.
    #   gemini-3-flash-preview   : returns [x, y] as prompt asks, no swap.
    #                              recall_direct=0.93 vs recall_swap=0.13
    #                              on 4 leaves; direct wins on 4/4 cases.
    # When neither model matches we default to direct (prompt-literal).
    swap_yx = bool(use_model and "gemini-3.5" in use_model)
    pix: list[tuple[int, int]] = []
    for pt in poly_norm:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            continue
        try:
            if swap_yx:
                y, x = float(pt[0]), float(pt[1])
            else:
                x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        x = max(0.0, min(1000.0, x))
        y = max(0.0, min(1000.0, y))
        pix.append((int(round(x / 1000.0 * W)),
                    int(round(y / 1000.0 * H))))
    if len(pix) < 3:
        dbg["parse_error"] = f"too few vertices ({len(pix)})"
        return [], dbg
    dbg["n_vertices"] = len(pix)
    return pix, dbg
