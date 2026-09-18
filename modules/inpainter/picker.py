"""Gemini pick-judge: choose the best of N FLUX inpaint candidates.

Given the original scene, the SAM cutout, and several FLUX inpaint
candidates, ask Gemini which candidate best covers and redraws the
OCCLUDED region of the part. The composite step inside the agent keeps
the visible RGB intact, so candidates differ ONLY in how they filled
the occluded portion -- and that is what the judge evaluates.

Per call we deterministically shuffle the candidate-to-letter mapping
(seeded by ``label`` + tags) to remove letter-position bias while
staying reproducible across runs.

Public entry point: :func:`pick_best`. Returns the chosen tag plus the
mapping and the raw judge response for debugging.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from typing import Mapping, Sequence

from PIL import Image

from .imaging import rgba_on_white
from .polygon import GEMINI_MODEL, _gemini, _strip_json_fence


def _maybe_nothink(model: str | None) -> None:
    from modules.qwen_client import is_qwen_model
    from modules.openai_client import is_gpt_model
    from modules.claude_client import is_claude_model
    if is_qwen_model(model) or is_gpt_model(model) or is_claude_model(model):
        return
    from . import _nothink  # noqa: F401  (disable Gemini CoT — must precede gemini)

PICK_PROMPT = """You are picking the best inpainting result for a vector-graphic part.

INPUTS
- IMAGE 1 = original scene. The part labeled "{label}" exists somewhere in
  it but is partially OCCLUDED by other layers drawn on top.
- IMAGE 2 = the SAM cutout of the "{label}" layer on white. This is the
  partial / visible portion of the part (the rest is hidden in the scene).
- IMAGES 3..N = candidate inpainting results (A, B, C, D, ...) that
  attempted to reconstruct the FULL "{label}" by filling in the OCCLUDED
  region. The visible region was preserved from IMAGE 2 in all of them,
  so candidates differ ONLY in how they filled the occluded portion.

MOST IMPORTANT CRITERION (this dominates)
- Did the candidate COVER and DRAW the OCCLUDED region well? That is,
  the part of "{label}" that was missing in IMAGE 2 must be recovered
  as a sensible, complete shape that extends the visible silhouette
  naturally and FILLS the gap. A candidate that leaves the occluded
  region empty, broken, or only partially filled is worse than one
  that fully and plausibly reconstructs it. Pick the candidate that
  most COMPLETELY recovers the hidden portion of the "{label}".

Secondary criteria (only used to break ties between candidates that
already cover the occluded region equally well):
- It should not hallucinate unrelated objects/faces in the filled region.
- Color and edge style should roughly match the visible portion.
- It should not bleed far into regions that were obviously background.

OUTPUT -- JSON only, no markdown:
{{"best": "{letters}",
 "reason": "one short sentence naming the visible evidence (focus on occluded coverage)"}}"""


def _shuffled_mapping(label: str, tags: Sequence[str]) -> dict[str, str]:
    """Per-call deterministic shuffle of letters -> tags.

    Seed = hash(label, *tags) so the same (label, tag-set) always
    produces the same letter mapping across runs.
    """
    tags = list(tags)
    rng = random.Random(hash(("pick_judge", label, tuple(tags))) & 0xffffffff)
    rng.shuffle(tags)
    letters = [chr(ord("A") + i) for i in range(len(tags))]
    return dict(zip(letters, tags))


def pick_best(scene_rgb: Image.Image,
              part_rgba: Image.Image,
              candidates: Mapping[str, Image.Image],
              label: str,
              *,
              model: str | None = None,
              thinking_level: str | None = "medium",
              ) -> dict:
    """Pick the best of N candidate inpaints.

    Parameters
    ----------
    scene_rgb : PIL.Image (RGB) -- original scene.
    part_rgba : PIL.Image (RGBA) -- SAM cutout of the part.
    candidates : dict tag -> RGBA inpaint result. Order does not matter
        (we shuffle internally). Typical use: ``{"p0": fill_p0, ...}``.
    label : the part label string used to format the prompt.
    model : Gemini model id; defaults to project default
        (:data:`modules.inpainter.polygon.GEMINI_MODEL`).
    thinking_level : ``minimal|low|medium|high`` or ``None`` to bypass.
        Default ``medium`` -- the ablation showed this materially
        improves occluded-coverage judgement.

    Returns
    -------
    dict::

        {
          "best_tag":   str,           # winning candidate tag
          "best_letter": str,          # letter as Gemini saw it
          "mapping":    dict letter -> tag,
          "reason":     str,           # short reason from Gemini
          "raw":        str,           # raw response text
          "parse_error": str | None,
          "usage":      dict | None,   # token / cost telemetry
        }
    """
    from modules.qwen_client import is_qwen_model
    from modules.openai_client import is_gpt_model
    from modules.claude_client import is_claude_model

    if not candidates:
        raise ValueError("candidates must not be empty")
    if len(candidates) > 26:
        raise ValueError("pick_best supports up to 26 candidates (A..Z)")

    tags = list(candidates.keys())
    mapping = _shuffled_mapping(label, tags)
    letters = list(mapping.keys())
    letters_alt = " | ".join(f'"{L}"' for L in letters)

    sam_img = rgba_on_white(part_rgba)
    ordered_imgs = [rgba_on_white(candidates[mapping[L]]) for L in letters]

    prompt = PICK_PROMPT.format(label=label, letters=letters_alt)
    contents = [prompt, scene_rgb, sam_img, *ordered_imgs]

    use_model = model or GEMINI_MODEL
    _maybe_nothink(use_model)
    call_kwargs: dict = dict(model=use_model, contents=contents)
    if thinking_level and not (is_qwen_model(use_model) or is_gpt_model(use_model)
                                or is_claude_model(use_model)):
        from google.genai import types as _t
        call_kwargs["config"] = _t.GenerateContentConfig(
            thinking_config=_t.ThinkingConfig(
                thinking_level=thinking_level.upper(),
                include_thoughts=False,
            ),
        )
    elif thinking_level and (is_gpt_model(use_model) or is_claude_model(use_model)):
        # Match the Gemini thinking_level on the GPT/Claude shims.
        call_kwargs["reasoning"] = thinking_level

    # Log token usage to the shared telemetry sink (same as polygon /
    # judge). _usage_from_response is lazy-imported to avoid pulling
    # gemini_client at module import time.
    from modules.gemini_client import _usage_from_response, _log_usage
    resp = _gemini(use_model).models.generate_content(**call_kwargs)
    usage = _usage_from_response(resp, use_model, "inpaint_pick")
    _log_usage(usage)
    raw = resp.text or ""
    text = _strip_json_fence(raw)

    parse_error: str | None = None
    best_letter = ""
    reason = ""
    try:
        parsed = json.loads(text)
        best_letter = str(parsed.get("best", "")).strip().upper()
        reason = str(parsed.get("reason", ""))[:240]
    except json.JSONDecodeError as e:
        m = re.search(r'"best"\s*:\s*"?([A-Z])"?', text, flags=re.IGNORECASE)
        if m:
            best_letter = m.group(1).upper()
        else:
            parse_error = str(e)

    if best_letter not in mapping:
        # fail-safe: default to letter A (= first shuffled tag) so caller
        # always receives a usable result.
        parse_error = parse_error or (
            f"best={best_letter!r} not in mapping {mapping}")
        best_letter = letters[0]

    return {
        "best_tag": mapping[best_letter],
        "best_letter": best_letter,
        "mapping": mapping,
        "reason": reason,
        "raw": raw,
        "parse_error": parse_error,
        "usage": asdict(usage),
    }


__all__ = ["pick_best", "PICK_PROMPT"]
