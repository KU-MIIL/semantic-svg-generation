"""Gemini judge: residual / occlusion gate for a per-part layer.

v14 combines two binary calls into one:
  * ``is_residual``  — does IMAGE 2 form a coherent named sub-shape, or
                       is it just scatter / fragments?
  * ``occluded``     — only when ``is_residual=false``: is the layer
                        occluded by another scene element such that a
                        non-trivial chunk is hidden in IMAGE 2?

Decoupling residual from occlusion lets the v14 decompose pipeline reject
residual leaves outright (no inpaint, no recursion) while still routing
genuinely occluded parts through polygon + FLUX.

Shape of the call mirrors :func:`predict_complete_polygon` — same images,
same model/thinking_level controls — so the agent can drive both with one
set of knobs.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .imaging import WHITE_THRESHOLD, rgba_on_white
from .polygon import GEMINI_MODEL, _gemini, _strip_json_fence


def _maybe_nothink(model: str | None) -> None:
    """Install the google.genai monkeypatch only when we're actually
    going to call Gemini. Qwen / GPT / Claude paths mustn't try to
    import google.genai (the env may not even have it installed)."""
    from modules.qwen_client import is_qwen_model
    from modules.openai_client import is_gpt_model
    from modules.claude_client import is_claude_model
    if is_qwen_model(model) or is_gpt_model(model) or is_claude_model(model):
        return
    from . import _nothink  # noqa: F401  (must precede the gemini call)


@dataclass
class SiblingInfo:
    """One row of the geometric prior table passed to the occlusion judge.

    Built by :func:`pipeline_controller._compute_sibling_overlaps` from the
    closed list of same-parent peers; the judge prompt instructs Gemini to
    pick occluder names ONLY from the ``label`` column so hallucinated
    occluders are structurally impossible.
    """
    label: str
    part_id: str
    iou: float
    target_coverage_by_sibling: float    # |target ∩ sib| / |target|
    sibling_coverage_by_target: float    # |target ∩ sib| / |sib|
    adjacency_pixels: int                # |dilate1(target) ∩ sib − target ∩ sib|
    rel_pos: str                         # "above"/"below"/"left"/"right"/"contains"/...
    bbox_xywh: tuple[int, int, int, int]


@dataclass
class JudgeOutput:
    """Structured return of :func:`judge_needs_inpaint`.

    ``debug`` carries the raw text, parse warnings/errors, pixel context,
    and a ``needs_inpaint`` mirror of ``occluded`` so downstream code that
    still reads the legacy key keeps working.
    """
    is_residual: bool
    occluded: bool                       # mirror of legacy needs_inpaint
    occluding_siblings: list[str] = field(default_factory=list)
    occluder_labels: list[str] = field(default_factory=list)
    description: str | None = None
    reason: str = ""
    debug: dict = field(default_factory=dict)

    def __iter__(self):
        # Back-compat: legacy callers expect ``(needs_inpaint, is_residual, debug)``.
        yield self.occluded
        yield self.is_residual
        yield self.debug


def _make_thinking_cfg(thinking_level: str | None,
                       thinking_budget: int | None):
    """Return a GenerateContentConfig with the appropriate thinking knob.

    gemini-3-flash takes ``thinking_budget`` (int: 0=off / -1=auto / +int);
    gemini-3.5-flash takes ``thinking_level`` ("minimal"/"low"/"medium"/
    "high"). When the caller passes ``thinking_budget`` it wins — the
    integer form is the post-3.5 API surface and is unambiguous. Returns
    ``None`` when neither knob is set so the call uses model defaults.
    """
    from google.genai import types as _t
    if thinking_budget is not None:
        return _t.GenerateContentConfig(
            thinking_config=_t.ThinkingConfig(
                thinking_budget=int(thinking_budget),
                include_thoughts=False,
            ),
        )
    if thinking_level:
        return _t.GenerateContentConfig(
            thinking_config=_t.ThinkingConfig(
                thinking_level=thinking_level.upper(),
                include_thoughts=False,
            ),
        )
    return None

JUDGE_PROMPT = """You decide up to FOUR things about a vector-graphic layer.

INPUTS
- IMAGE 1: the full original scene.
- IMAGE 2: a single layer labeled "{label}", cut out on white.
  White pixels = transparent / not part of the layer. IMAGE 2 may be
  incomplete because in IMAGE 1 some of this layer is hidden under
  another part drawn on top of it.

PART PIXEL CONTEXT
- Visible pixels of THIS layer in IMAGE 2: {part_visible_px}
- Total color (non-white) pixels of the FULL scene in IMAGE 1: {scene_color_px}
- This layer covers about {ratio_pct:.2f}% of the scene's color area.
{sibling_block}{parent_block}
DECISION A — is_residual
  Set TRUE when IMAGE 2 is scatter / dot fragments / thin slivers /
  leftover pixels with no coherent shape — i.e. there is no single
  named sub-part a designer would draw to produce this silhouette.
  Set FALSE when IMAGE 2 forms a meaningful coherent sub-shape (even
  if partially occluded).

DECISION B — occluded (evaluate ONLY when is_residual=false)
  Set TRUE when ALL of:
    (a) IMAGE 1 shows clear visual evidence that this layer is occluded
        by another scene element — IMAGE 2's silhouette has a noticeable
        chunk cut out by something drawn in front of it.
    (b) The hidden region is non-trivial: rough rule, adding it would
        need pixels comparable to >= 5% of the layer's currently visible
        area. Thin antialiased edges and sub-pixel noise do NOT count.
  Set FALSE otherwise (already complete / in-front of everything that
  touches it / supposed occlusion is just edge noise).

Be CONSERVATIVE on occluded: prefer false unless the gap is clearly
visible. When is_residual=true, ALWAYS set occluded=false (we drop
the residual; we don't reconstruct it).

DECISION C — occluding_siblings (evaluate ONLY when occluded=true)
  Output a JSON array of SIBLING NAMES (the 'name' column of the table
  above) for every sibling that paints OVER this layer. Closed set: only
  those exact names are valid, NEVER invent new names and NEVER pick the
  parent layer. Use the geometric prior:
    - High `sib->tgt` with a visibly cut-out silhouette in IMAGE 1 ⇒
      strong occluder candidate.
    - High `iou` with symmetric coverage ⇒ probably a redundant pair,
      not an occluder.
    - `adj_px > 0` with `sib->tgt ≈ 0` ⇒ they only touch, not occlude.
  Order entries most-occluding first. Empty array `[]` is required when
  occluded=false; it is also acceptable when occluded=true and you
  cannot identify the occluder from the provided list (the prompt fails
  open to FLUX in that case).
{decision_d_block}
OUTPUT — JSON only, no markdown:
{{"is_residual": true | false, "occluded": true | false, \
"occluding_siblings": [<sibling names>], "reason": \
"one short sentence naming the visible evidence"{description_field}}}"""


# Optional DECISION D — fine-grained description. Only injected when the
# caller asks for it (``make_description=True``); piggy-backed on the same
# call to keep total latency around one Gemini round trip.
_DECISION_D_BLOCK = """
DECISION D — description (evaluate ONLY when occluded=true)
  Output a single English sentence describing the COMPLETE (un-occluded)
  '{label}' as a clean vector graphic: dominant colors (use what IMAGE 2
  shows, extrapolate texture/edges from the visible portion), overall
  shape, and characteristic features. ~20 words, concrete, no meta
  language ("the image shows ...") and NO mention of occluders or
  background. Example for a 'shovel': "A sturdy metallic shovel with a
  curved, deep gray scoop and smooth, reflective silver handle." When
  occluded=false, set description to null.
"""


def _format_sibling_table(siblings: list["SiblingInfo"] | None) -> str:
    """Render siblings as a fixed-width text table prompt block.

    Returns an empty string when ``siblings`` is None/empty so the prompt
    falls back to per-part-only judgement (matches the pre-v2 behaviour).
    """
    if not siblings:
        return ""
    rows = [
        ("name", "iou", "sib->tgt", "tgt->sib", "adj_px", "rel_pos", "bbox"),
        ("----", "---", "--------", "--------", "------", "-------", "----"),
    ]
    for s in siblings[:8]:
        x, y, w, h = s.bbox_xywh
        rows.append((
            s.label[:22],
            f"{s.iou:.3f}",
            f"{s.target_coverage_by_sibling:.3f}",
            f"{s.sibling_coverage_by_target:.3f}",
            str(int(s.adjacency_pixels)),
            s.rel_pos,
            f"({x},{y},{w},{h})",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(7)]
    lines = [
        " | ".join(c.ljust(w) for c, w in zip(row, widths))
        for row in rows
    ]
    body = "\n  ".join(lines)
    return (
        "\nSIBLING OVERLAP TABLE (closed set; pick occluders ONLY from "
        "the 'name' column):\n"
        "  iou       = Jaccard of sibling vs target masks\n"
        "  sib->tgt  = fraction of THIS layer's visible pixels covered by sibling\n"
        "  tgt->sib  = fraction of the SIBLING's pixels covered by this layer\n"
        "  adj_px    = boundary-touching pixels (1-px dilation overlap)\n"
        "  rel_pos   = sibling centroid relative to target centroid\n"
        f"  {body}\n"
    )


def _format_parent_block(parent_label: str | None,
                         parent_coverage: float | None) -> str:
    """Per-layer note about the parent (NOT an occluder candidate).

    The parent of a part is the layer the part sits ON TOP OF. Children
    paint over their parent, so the parent should NEVER appear as an
    occluder. We surface it only so the model is aware the visible mask
    is bounded by the parent shape.
    """
    if not parent_label:
        return ""
    cov = "" if parent_coverage is None else f" (covers ~{parent_coverage:.2f} of target)"
    return (
        f"PARENT LAYER (NOT an occluder candidate): '{parent_label}'{cov}\n"
    )


def judge_needs_inpaint(scene_rgb: Image.Image,
                        part_rgba: Image.Image,
                        label: str,
                        *,
                        siblings: list[SiblingInfo] | None = None,
                        parent_label: str | None = None,
                        parent_coverage: float | None = None,
                        make_description: bool = False,
                        model: str | None = None,
                        thinking_level: str | None = None,
                        thinking_budget: int | None = None,
                        ) -> JudgeOutput:
    """Ask Gemini whether ``label`` is residual / needs inpainting, and
    (when ``siblings`` is provided) which named siblings occlude it.

    When ``make_description=True`` the same call additionally produces a
    fine-grained one-sentence description of the complete object — used
    downstream as the FLUX inpaint prompt instead of the bare ``label``.

    Returns a :class:`JudgeOutput`. Legacy callers can still unpack it as
    ``(needs_inpaint, is_residual, debug)`` because :class:`JudgeOutput`
    implements ``__iter__`` to yield those three fields. On parse failure
    the function fails open on ``occluded`` (defaults True) but reports
    ``occluding_siblings=[]`` and ``description=None`` — so downstream
    gracefully degrades to the original bare-label FLUX call.
    """
    part_arr = np.array(part_rgba.convert("RGBA"))
    part_visible_px = int((part_arr[..., 3] > 0).sum())
    scene_arr = np.array(scene_rgb.convert("RGB"))
    scene_color_px = int((~(scene_arr >= WHITE_THRESHOLD).all(axis=2)).sum())
    ratio_pct = 100.0 * part_visible_px / max(scene_color_px, 1)

    sibling_block = _format_sibling_table(siblings)
    parent_block = _format_parent_block(parent_label, parent_coverage)
    decision_d = (
        _DECISION_D_BLOCK.format(label=label) if make_description else ""
    )
    description_field = ', "description": "..." | null' if make_description else ""

    part_on_white = rgba_on_white(part_rgba)
    prompt = JUDGE_PROMPT.format(
        label=label,
        part_visible_px=part_visible_px,
        scene_color_px=scene_color_px,
        ratio_pct=ratio_pct,
        sibling_block=sibling_block,
        parent_block=parent_block,
        decision_d_block=decision_d,
        description_field=description_field,
    )
    use_model = model or GEMINI_MODEL
    _maybe_nothink(use_model)
    from modules.qwen_client import is_qwen_model
    from modules.openai_client import is_gpt_model
    from modules.claude_client import is_claude_model
    call_kwargs: dict = dict(
        model=use_model,
        contents=[prompt, scene_rgb, part_on_white],
    )
    if not (is_qwen_model(use_model) or is_gpt_model(use_model)
            or is_claude_model(use_model)):
        cfg = _make_thinking_cfg(thinking_level, thinking_budget)
        if cfg is not None:
            call_kwargs["config"] = cfg
    elif thinking_level and (is_gpt_model(use_model)
                             or is_claude_model(use_model)):
        # Match the Gemini thinking_level on the GPT/Claude shims.
        call_kwargs["reasoning"] = thinking_level
    from modules.gemini_client import _usage_from_response, _log_usage
    resp = _gemini(use_model).models.generate_content(**call_kwargs)
    usage = _usage_from_response(resp, use_model, "inpaint_judge")
    _log_usage(usage)
    raw = resp.text or ""
    text = _strip_json_fence(raw)
    valid_labels: list[str] = (
        [s.label for s in siblings] if siblings else []
    )
    label_to_id: dict[str, str] = (
        {s.label: s.part_id for s in siblings} if siblings else {}
    )
    dbg: dict = {
        "label": label, "raw": raw,
        "part_visible_px": part_visible_px,
        "scene_color_px": scene_color_px,
        "part_scene_ratio_pct": round(ratio_pct, 3),
        "usage": asdict(usage),
        "n_siblings_in_prompt": len(siblings) if siblings else 0,
        "description_requested": bool(make_description),
    }

    def _validate_occluders(raw_list: list) -> tuple[list[str], list[str], list[str]]:
        """Drop names not in the closed sibling set. Returns
        (kept_ids, kept_labels, dropped_names)."""
        kept_ids: list[str] = []
        kept_labels: list[str] = []
        dropped: list[str] = []
        if not raw_list:
            return kept_ids, kept_labels, dropped
        seen: set[str] = set()
        for item in raw_list:
            name = str(item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            if name in label_to_id:
                kept_ids.append(label_to_id[name])
                kept_labels.append(name)
            else:
                dropped.append(name)
        return kept_ids, kept_labels, dropped

    try:
        parsed = json.loads(text)
        is_residual = bool(parsed.get("is_residual", False))
        # Accept either the new "occluded" key or the legacy
        # "needs_inpaint" alias (Gemini occasionally falls back to the
        # older name even after the prompt asks for the new one).
        if "occluded" in parsed:
            occluded = bool(parsed.get("occluded"))
        else:
            occluded = bool(parsed.get("needs_inpaint", True))
        if is_residual:
            occluded = False
        raw_occluders = parsed.get("occluding_siblings") or []
        if not isinstance(raw_occluders, list):
            raw_occluders = []
        kept_ids, kept_labels, dropped = _validate_occluders(raw_occluders)
        if not occluded:
            kept_ids, kept_labels = [], []
        description = parsed.get("description")
        if description is not None:
            description = str(description).strip() or None
        reason = str(parsed.get("reason", ""))[:240]
        dbg.update({
            "is_residual": is_residual,
            "occluded": occluded,
            "needs_inpaint": occluded,  # legacy mirror
            "occluding_sibling_ids": kept_ids,
            "occluding_siblings": kept_labels,
            "dropped_names": dropped,
            "description": description,
            "reason": reason,
        })
        return JudgeOutput(
            is_residual=is_residual,
            occluded=occluded,
            occluding_siblings=kept_ids,
            occluder_labels=kept_labels,
            description=description,
            reason=reason,
            debug=dbg,
        )
    except json.JSONDecodeError as e:
        # Salvage from broken JSON: regex each flag independently.
        m_res = re.search(r'"is_residual"\s*:\s*(true|false)', text,
                          flags=re.IGNORECASE)
        m_occ = re.search(
            r'"(?:occluded|needs_inpaint)"\s*:\s*(true|false)',
            text, flags=re.IGNORECASE,
        )
        m_list = re.search(
            r'"occluding_siblings"\s*:\s*(\[[^\]]*\])',
            text, flags=re.IGNORECASE,
        )
        if m_res or m_occ or m_list:
            is_residual = (m_res.group(1).lower() == "true") if m_res else False
            occluded = (m_occ.group(1).lower() == "true") if m_occ else True
            if is_residual:
                occluded = False
            raw_occluders: list = []
            if m_list:
                try:
                    raw_occluders = json.loads(m_list.group(1))
                except json.JSONDecodeError:
                    raw_occluders = []
            kept_ids, kept_labels, dropped = _validate_occluders(raw_occluders)
            if not occluded:
                kept_ids, kept_labels = [], []
            dbg.update({
                "parse_warning": f"salvaged from broken JSON: {e}",
                "is_residual": is_residual,
                "occluded": occluded,
                "needs_inpaint": occluded,
                "occluding_sibling_ids": kept_ids,
                "occluding_siblings": kept_labels,
                "dropped_names": dropped,
                "description": None,
                "reason": "",
            })
            return JudgeOutput(
                is_residual=is_residual,
                occluded=occluded,
                occluding_siblings=kept_ids,
                occluder_labels=kept_labels,
                description=None,
                reason="",
                debug=dbg,
            )
        # Total parse failure → fail-open on occluded, empty occluders
        # so fill_from_mask falls back to bare-label FLUX.
        dbg.update({
            "parse_error": str(e),
            "is_residual": False,
            "occluded": True,
            "needs_inpaint": True,
            "occluding_sibling_ids": [],
            "occluding_siblings": [],
            "dropped_names": [],
            "description": None,
            "reason": "(judge parse failed; defaulting to inpaint)",
        })
        return JudgeOutput(
            is_residual=False,
            occluded=True,
            occluding_siblings=[],
            occluder_labels=[],
            description=None,
            reason="(judge parse failed; defaulting to inpaint)",
            debug=dbg,
        )


RESIDUAL_PROMPT = """You decide ONE thing about a vector-graphic layer.

INPUTS
- IMAGE 1: the full original scene (for context).
- IMAGE 2: a single layer labeled "{label}", cut out on white.
  White pixels = transparent / not part of the layer.

PIXEL CONTEXT
- Visible pixels of THIS layer in IMAGE 2: {part_visible_px}
- Total color (non-white) pixels of the FULL scene in IMAGE 1: {scene_color_px}
- This layer covers about {ratio_pct:.2f}% of the scene's color area.

DECISION — is_residual
  Set TRUE only when IMAGE 2 is GARBAGE leftover that a designer
  would never put on a layer:
    • Scattered dots / noise pixels with no shape.
    • Thin edge slivers / partial outlines tracing the boundary of
      a sibling part.
    • Broken fragments of a single shape already represented by
      siblings.

  Set FALSE — this layer is REAL — when IMAGE 2 is any coherent
  shape, INCLUDING:
    • A coherent large fill with CUTOUTS where foreground siblings
      sit on top of it. This is the SCENE BACKGROUND
      (sky / circular plate / sand ground / wall). The fact that
      it has notches cut into it does NOT make it residual — it
      makes it the back layer of the painter's stack.
    • A partially-occluded coherent sub-shape.

The phrase "leftover background partition with cutouts" describes a
REAL background layer, NOT residual. Background layers paint FIRST
and other parts paint on top of them; their characteristic look is
"big fill with foreground-shaped holes". Classify them as is_residual
= FALSE so the downstream stage-2 judge can decide z-order.

PRIORITY HINTS (apply BEFORE the qualitative judgement):
  • If the layer covers ≥ 15% of the scene's color area, it is
    almost certainly a real layer, not slivers. Default FALSE.
  • If the layer is mostly thin lines / scattered dots covering
    < 5% of the scene, it is almost certainly residual. Default TRUE.

Be CONSERVATIVE: when uncertain, FALSE (keep). Dropping a real layer
is irreversible; keeping a borderline layer just adds one more
sibling.

OUTPUT — JSON only, no markdown:
{{"is_residual": true | false, "reason": "one short sentence naming the visible evidence"}}"""


def judge_residual_only(scene_rgb: Image.Image,
                        part_rgba: Image.Image,
                        label: str,
                        *,
                        model: str | None = None,
                        thinking_level: str | None = None,
                        thinking_budget: int | None = None,
                        ) -> tuple[bool, dict]:
    """Residual-only gate for v15 tree build.

    Cheaper variant of :func:`judge_needs_inpaint` — only asks the
    is_residual question, no occlusion / inpaint analysis. Used per
    sibling during v15's per-node tree build to drop pure-residual
    nodes before recursion. The full occlusion judge runs LATER on
    accepted leaves only.

    Returns ``(is_residual, debug)``. Fail-closed on parse error
    (``is_residual=False`` → keep the node) so a flaky judge call
    never silently drops a real part.
    """
    from modules.gemini_client import _usage_from_response, _log_usage

    part_arr = np.array(part_rgba.convert("RGBA"))
    part_visible_px = int((part_arr[..., 3] > 0).sum())
    scene_arr = np.array(scene_rgb.convert("RGB"))
    scene_color_px = int((~(scene_arr >= WHITE_THRESHOLD).all(axis=2)).sum())
    ratio_pct = 100.0 * part_visible_px / max(scene_color_px, 1)

    part_on_white = rgba_on_white(part_rgba)
    prompt = RESIDUAL_PROMPT.format(
        label=label,
        part_visible_px=part_visible_px,
        scene_color_px=scene_color_px,
        ratio_pct=ratio_pct,
    )
    use_model = model or GEMINI_MODEL
    _maybe_nothink(use_model)
    from modules.qwen_client import is_qwen_model
    from modules.openai_client import is_gpt_model
    from modules.claude_client import is_claude_model
    call_kwargs: dict = dict(
        model=use_model,
        contents=[prompt, scene_rgb, part_on_white],
    )
    if not (is_qwen_model(use_model) or is_gpt_model(use_model)
            or is_claude_model(use_model)):
        cfg = _make_thinking_cfg(thinking_level, thinking_budget)
        if cfg is not None:
            call_kwargs["config"] = cfg
    elif thinking_level and (is_gpt_model(use_model)
                             or is_claude_model(use_model)):
        # Match the Gemini thinking_level on the GPT/Claude shims.
        call_kwargs["reasoning"] = thinking_level
    resp = _gemini(use_model).models.generate_content(**call_kwargs)
    usage = _usage_from_response(resp, use_model, "residual_judge")
    _log_usage(usage)
    raw = resp.text or ""
    text = _strip_json_fence(raw)
    dbg: dict = {
        "label": label, "raw": raw,
        "part_visible_px": part_visible_px,
        "scene_color_px": scene_color_px,
        "part_scene_ratio_pct": round(ratio_pct, 3),
        "usage": asdict(usage),
    }
    try:
        parsed = json.loads(text)
        is_residual = bool(parsed.get("is_residual", False))
        dbg["is_residual"] = is_residual
        dbg["reason"] = str(parsed.get("reason", ""))[:240]
        return is_residual, dbg
    except json.JSONDecodeError as e:
        m = re.search(r'"is_residual"\s*:\s*(true|false)', text,
                      flags=re.IGNORECASE)
        if m:
            is_residual = m.group(1).lower() == "true"
            dbg["parse_warning"] = f"salvaged from broken JSON: {e}"
            dbg["is_residual"] = is_residual
            dbg["reason"] = ""
            return is_residual, dbg
        dbg["parse_error"] = str(e)
        dbg["is_residual"] = False
        dbg["reason"] = "(residual judge parse failed; defaulting to keep)"
        return False, dbg


BACKGROUND_PROMPT = """You decide ONE thing about a vector-graphic layer
that has ALREADY been confirmed as a coherent (non-residual) sub-shape.

INPUTS
- IMAGE 1: the full original scene (for context).
- IMAGE 2: the layer labeled "{label}", cut out on white. White pixels
  are transparent / not part of the layer.

PIXEL CONTEXT
- Visible pixels of THIS layer in IMAGE 2: {part_visible_px}
- Total color (non-white) pixels of the FULL scene in IMAGE 1: {scene_color_px}
- This layer covers about {ratio_pct:.2f}% of the scene's color area.

DECISION — is_background
  Set TRUE when IMAGE 2 is the BACKGROUND of the scene — a single
  coherent fill that sits BEHIND every other named part (sky, circular
  background plate, sand / desert ground, snowfield, wall, sea, etc.).
  The visual hallmark is cutouts where foreground sibling parts sit
  on top of it. A true background must paint FIRST in the layer stack
  (z-order = 0).

  Set FALSE when IMAGE 2 is a normal foreground / mid-layer part that
  should paint in its mask-area order with the other siblings. A
  normal part may be partially occluded but it is not the canvas-wide
  base that other parts rest on top of.

A background part typically:
  - covers ≥ 20% of the scene's color area (see PIXEL CONTEXT above)
  - has cutouts where other parts sit on top of it
  - has no occluders for itself (nothing sits behind it)

PRIORITY HINT: if the layer covers ≥ 20% AND has visible cutouts for
foreground siblings, it is almost certainly the background — default
TRUE. Smaller layers with no cutout pattern are usually mid-layer
parts — default FALSE.

Be CONSERVATIVE: when uncertain, FALSE (normal z-order). A wrong
"background" tag would hide a foreground layer behind everything
else.

OUTPUT — JSON only, no markdown:
{{"is_background": true | false, "reason": "one short sentence naming the visible evidence"}}"""


def judge_background_layer(scene_rgb: Image.Image,
                           part_rgba: Image.Image,
                           label: str,
                           *,
                           model: str | None = None,
                           thinking_level: str | None = None,
                           thinking_budget: int | None = None,
                           ) -> tuple[bool, dict]:
    """Background-layer gate for v15 tree build. Called only AFTER
    ``judge_residual_only`` confirmed the layer is NOT residual.

    Returns ``(is_background, debug)``. When True, the caller marks
    ``part.extra["is_background_layer"] = True`` so the downstream
    merge can hoist this leaf to z-order=0 (paint first).
    Fail-closed on parse error (``is_background=False`` → normal
    z-order) so a flaky judge never silently re-orders the scene.
    """
    from modules.gemini_client import _usage_from_response, _log_usage

    part_arr = np.array(part_rgba.convert("RGBA"))
    part_visible_px = int((part_arr[..., 3] > 0).sum())
    scene_arr = np.array(scene_rgb.convert("RGB"))
    scene_color_px = int((~(scene_arr >= WHITE_THRESHOLD).all(axis=2)).sum())
    ratio_pct = 100.0 * part_visible_px / max(scene_color_px, 1)

    part_on_white = rgba_on_white(part_rgba)
    prompt = BACKGROUND_PROMPT.format(
        label=label,
        part_visible_px=part_visible_px,
        scene_color_px=scene_color_px,
        ratio_pct=ratio_pct,
    )
    use_model = model or GEMINI_MODEL
    _maybe_nothink(use_model)
    from modules.qwen_client import is_qwen_model
    from modules.openai_client import is_gpt_model
    from modules.claude_client import is_claude_model
    call_kwargs: dict = dict(
        model=use_model,
        contents=[prompt, scene_rgb, part_on_white],
    )
    if not (is_qwen_model(use_model) or is_gpt_model(use_model)
            or is_claude_model(use_model)):
        cfg = _make_thinking_cfg(thinking_level, thinking_budget)
        if cfg is not None:
            call_kwargs["config"] = cfg
    elif thinking_level and (is_gpt_model(use_model)
                             or is_claude_model(use_model)):
        # Match the Gemini thinking_level on the GPT/Claude shims.
        call_kwargs["reasoning"] = thinking_level
    resp = _gemini(use_model).models.generate_content(**call_kwargs)
    usage = _usage_from_response(resp, use_model, "background_judge")
    _log_usage(usage)
    raw = resp.text or ""
    text = _strip_json_fence(raw)
    dbg: dict = {
        "label": label, "raw": raw,
        "part_visible_px": part_visible_px,
        "scene_color_px": scene_color_px,
        "part_scene_ratio_pct": round(ratio_pct, 3),
        "usage": asdict(usage),
    }
    try:
        parsed = json.loads(text)
        is_background = bool(parsed.get("is_background", False))
        dbg["is_background"] = is_background
        dbg["reason"] = str(parsed.get("reason", ""))[:240]
        return is_background, dbg
    except json.JSONDecodeError as e:
        m = re.search(r'"is_background"\s*:\s*(true|false)', text,
                      flags=re.IGNORECASE)
        if m:
            is_background = m.group(1).lower() == "true"
            dbg["parse_warning"] = f"salvaged from broken JSON: {e}"
            dbg["is_background"] = is_background
            dbg["reason"] = ""
            return is_background, dbg
        dbg["parse_error"] = str(e)
        dbg["is_background"] = False
        dbg["reason"] = "(background judge parse failed; defaulting to normal z-order)"
        return False, dbg


__all__ = ["judge_needs_inpaint", "JUDGE_PROMPT",
           "judge_residual_only", "RESIDUAL_PROMPT",
           "judge_background_layer", "BACKGROUND_PROMPT",
           "JudgeOutput", "SiblingInfo"]
