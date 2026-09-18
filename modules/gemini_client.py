"""Thin wrapper around google-genai with token/cost accounting.

Owns the single Gemini call site the pipeline uses — ``decompose_one_level``
(hierarchical part decomposition) — along with its prompt template and the
JSON extraction helpers.

Every successful call prints a one-line usage report to stdout and returns a
``usage_metadata`` dict that callers embed into their result records. No
aggregation is done here — each call stands alone.

Pricing is hardcoded in :data:`PRICING` (USD per 1M tokens). Update there if
Google changes published prices.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image


DECOMPOSE_ONE_LEVEL_PROMPT = """You decide whether to decompose a region of a flat-color illustration
into named sub-parts. {parent_context}

══════════════════════════════════════════════════════════════════════
DECIDE: is_atomic = true | false

ATOMIC IS THE DEFAULT. The pipeline already paints the parent's mask;
only split when each child would be drawn on its OWN LAYER by a
human designer. Visual fragments sharing one fill are NOT separate
layers.

Choose ATOMIC unless ALL THREE hold:
  1. The region contains 2+ semantically NAMED sub-parts (each can
     carry a plain noun-phrase a designer would put on a layer).
  2. EACH sub-part occupies ≥ 25% of the parent region's bbox area.
     This is a HARD floor. Tiny decorations (logos, buttons,
     stickers, individual pedals/seats/handles on a vehicle,
     small accessories) are NOT separate parts — they belong to
     the parent's fill. If any candidate child would be < 25%
     area, the whole split is wrong → ATOMIC.
  3. The sub-parts visibly OCCLUDE each other (one silhouette has a
     notch cut by the other, so the inpaint merge needs separate
     layers to reconstruct each).

DEPTH-2 GATE — depth-2 is RARE.

When parent_label is non-empty you are at depth 2. ATOMIC is almost
always the right answer here — the dataset has roughly 1 depth-2
case in 15. Only mark non-atomic if EVERY check above holds AND the
split is obvious to ANY designer reading the parent shape. Any
uncertainty → ATOMIC.

Depth ≥ 3 is BLOCKED by code — never attempt.

══════════════════════════════════════════════════════════════════════
ALWAYS ATOMIC (no exceptions, at any depth):
  • Single-fill natural shapes (sun, moon, cloud, fire, smoke, water,
    leaf, flower, rainbow band, individual star, single mushroom,
    single mountain, single tree).
  • Tools / man-made objects (laptop, phone, hammer, bottle, lamp,
    trophy, spray bottle, magnifying glass, target, lightbulb,
    sunglasses, kettle, microphone, vase, basket, knife, bicycle).
    Decorations on their surface (logo, label, sticker, screen
    content, key rows, individual pedals/seats/handles) are part
    of the tool's fill — NEVER separate parts.
  • ALL body parts including HEAD (head, face, hand, foot, eye, ear,
    nose, mouth, hair, lip, neck). NEVER split a head into face
    features. Characters / persons / animals are ATOMIC as a single
    body — do not split into head/body/legs.
  • Repeated identical small shapes (spikes, gems, dots, stripes,
    petals, rays, sparkles, buttons). Keep as one plural entry OR
    omit entirely. Do not enumerate per instance.
  • Pure stroke / outline / shading / highlight / shadow. Never split.

NEVER EMIT placeholder labels: "background", "none", "remainder",
"residual", "rest", "leftover", "other", "misc", "outline", "stroke",
"highlight", "shadow", "shading", "fill". The parent mask absorbs
unnamed pixels — do not invent catch-alls.

══════════════════════════════════════════════════════════════════════
Z-ORDER — HARD RULE (inpaint merge depends on this).

List `components` BACK-to-FRONT (painter's order; index 0 deepest):
  • Background / container goes FIRST. Foreground / overlay LAST.
  • Occlusion decides: if A is partially hidden by B, A comes
    before B. Whichever silhouette has a notch cut into it is BEHIND.
  • Frame / outline around a fill goes FIRST.
  • Container + contents: the container goes FIRST, contents LAST.
  • Repeated instances side-by-side without overlap — natural reading
    order (left → right) is fine.

══════════════════════════════════════════════════════════════════════
LABEL + BBOX RULES:
  • label = plain noun phrase. No parent tokens, no catch-all.
  • _bbox = tight [ymin, xmin, ymax, xmax] in 0-1000 normalized
    coords of the FULL ORIGINAL image.
  • Sibling bboxes describe DISJOINT regions (may touch, must not
    strongly overlap). EXCEPT container + contents at the same level
    (jail + prisoner, sunglasses + face) — overlap OK, z-order
    disambiguates.

══════════════════════════════════════════════════════════════════════
OUTPUT — valid JSON only, no markdown:
{
  "is_atomic": <true|false>,
  "components": [{"label": "...", "_bbox": [ymin, xmin, ymax, xmax]}, ...]
}

══════════════════════════════════════════════════════════════════════
EXAMPLES

[Root, scene with sun + two clouds (multi-entity, all atomic).]
{
  "is_atomic": false,
  "components": [
    {"label": "sun",         "_bbox": [120, 700, 320, 920]},
    {"label": "left cloud",  "_bbox": [400,  80, 600, 420]},
    {"label": "right cloud", "_bbox": [380, 540, 580, 880]}
  ]
}

[Atomic, focus on "laptop" — single tool. The logo on the lid is
decoration on the laptop's surface, NOT a separate part.]
{"is_atomic": true, "components": []}

[Atomic, focus on "head" — face / eyes / mouth / hair are NEVER
separated. Head is one named part.]
{"is_atomic": true, "components": []}

[Atomic, focus on any sub-region with parent_label non-empty —
default answer at depth 2 is ATOMIC.]
{"is_atomic": true, "components": []}
"""


def _bbox_iou_norm(a: list | tuple, b: list | tuple) -> float:
    """IoU of two [ymin, xmin, ymax, xmax] rectangles in any consistent
    coord system. Returns 0.0 for degenerate / non-overlapping inputs."""
    ay0, ax0, ay1, ax1 = (float(v) for v in a)
    by0, bx0, by1, bx1 = (float(v) for v in b)
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    if iy1 <= iy0 or ix1 <= ix0:
        return 0.0
    inter = (iy1 - iy0) * (ix1 - ix0)
    area_a = max(0.0, (ay1 - ay0)) * max(0.0, (ax1 - ax0))
    area_b = max(0.0, (by1 - by0)) * max(0.0, (bx1 - bx0))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _format_tree_so_far(
    tree_so_far: list[dict] | None,
    focus_bbox: list | tuple | None,
) -> str:
    """Render the partial decomposition tree (other subtrees already
    decomposed) into a bullet list for prompt injection. Each entry is
    ``{"label": str, "bbox": [ymin, xmin, ymax, xmax]}`` in normalized
    0-1000 coords (same frame as ``focus_bbox``). For each entry we
    also emit its IoU with the current FOCUS rectangle — high IoU is
    the signal that the model would be claiming someone else's region
    under a new label. Returns "" when ``tree_so_far`` is empty.
    """
    entries = [e for e in (tree_so_far or [])
               if isinstance(e, dict) and e.get("label")
               and isinstance(e.get("bbox"), (list, tuple))
               and len(e["bbox"]) == 4]
    if not entries:
        return ""
    has_focus = (focus_bbox is not None
                 and isinstance(focus_bbox, (list, tuple))
                 and len(focus_bbox) == 4)
    lines: list[str] = []
    for e in entries:
        lab = str(e["label"]).strip()
        bb = [int(round(float(v))) for v in e["bbox"]]
        if has_focus:
            iou = _bbox_iou_norm(focus_bbox, bb)
            lines.append(
                f"  - '{lab}' @ [ymin={bb[0]}, xmin={bb[1]}, "
                f"ymax={bb[2]}, xmax={bb[3]}] · IoU with focus = "
                f"{iou * 100:.0f}%"
            )
        else:
            lines.append(
                f"  - '{lab}' @ [ymin={bb[0]}, xmin={bb[1]}, "
                f"ymax={bb[2]}, xmax={bb[3]}]"
            )
    label_list = ", ".join(f'"{e["label"]}"' for e in entries)
    return (
        "\nPARTIAL TREE SO FAR — other subtrees already decomposed by "
        "earlier recursion. Labels and bboxes here belong to nodes "
        "OUTSIDE this focus's subtree. Your children must respect them "
        "(H4 below):\n"
        + "\n".join(lines)
        + f"\nForbidden labels (cross-tree): [{label_list}]."
    )


def _build_decompose_context(
    parent_label: str,
    focus_bbox: list | tuple | None = None,
    depth: int = 0,
    ancestor_labels: list[str] | None = None,
    max_depth: int | None = None,
    sibling_labels: list[str] | None = None,
    tree_so_far: list[dict] | None = None,
) -> str:
    """Adapt the prompt's contextual sentence.

    Three modes:
      - root (parent_label empty): "FULL SCENE".
      - sub-region with focus_bbox: full-original-image input, with the
        focus rectangle named in 0-1000 coords. Caller passes the parent's
        Gemini bbox raw [ymin, xmin, ymax, xmax]. Per-call directives:
          • the current ``depth`` and (when given) ``max_depth`` so the
            model knows how much budget is left;
          • the full ``ancestor_labels`` path so the model can't re-emit
            a label that already sits above us (the "cap → highlights →
            left mushroom" cascade);
          • a token-level no-repeat directive on parent_label tokens so
            children stay plain ("head", not "sledgehammer head");
          • a leaf hint when parent_label is a singular leaf-level
            label (face, nose, stem …) — ATOMIC is usually correct.
      - sub-region without focus_bbox (legacy crop mode, e.g. SAM3 path):
        keep the previous "previously-segmented part" wording.
    """
    pl = (parent_label or "").strip()
    tree_block = _format_tree_so_far(tree_so_far, focus_bbox)
    if not pl:
        return ("It is the FULL SCENE. Decompose into top-level objects "
                "whenever the scene contains 2+ named objects (or 2+ "
                "instances of the same object — use positional or color "
                "disambiguation). If the scene is a single object, "
                "decompose it into its semantic sub-parts at the level "
                "a designer would label. Order `components` BACK-to-FRONT "
                "(painter's order) — background / occluded objects first, "
                "foreground / occluding objects last."
                + tree_block)
    if focus_bbox is not None and len(focus_bbox) == 4:
        ymin, xmin, ymax, xmax = (int(round(float(v))) for v in focus_bbox)
        # Build a per-call no-repeat directive. Forbid the FULL parent
        # label string from appearing in child labels (kills "sledgehammer
        # head" / "spray bottle body"); also forbid the parent's last
        # token as a standalone word ("hammer" inside "claw hammer" still
        # blocks "hammer head"). Critically, we DO NOT forbid every
        # parent token — that turned "eyes" → "left eye / right eye"
        # into "eyes" → "left / right" because the model dropped the
        # singular "eye" too. Singular forms of a plural parent are
        # always OK.
        no_repeat = ""
        if pl:
            last = pl.lower().rsplit(" ", 1)[-1] if pl.strip() else ""
            no_repeat = (
                f" Do NOT include the full phrase \"{pl}\" inside any "
                f"child label (e.g. NOT \"{pl} head\" — just \"head\")."
            )
            if last and last != pl.lower() and len(last) >= 3:
                no_repeat += (
                    f" Also do not use the standalone word \"{last}\" "
                    f"as a token in a child label."
                )
            # Positive guidance: plural-parent → singular-child is fine.
            if pl.lower().endswith("s") and len(pl) >= 3:
                singular_hint = pl.lower().rstrip("s")
                no_repeat += (
                    f" (Children may still use the singular form "
                    f"\"{singular_hint}\" — e.g. \"left {singular_hint}\", "
                    f"\"right {singular_hint}\" — that is the desired "
                    f"naming.)"
                )

        # Tree-path / ancestor exclusion. ``ancestor_labels`` is the
        # chain from root down to (but excluding) the current parent.
        # ``sibling_labels`` is the labels already emitted at the same
        # level (siblings of the current node). We forbid both —
        # otherwise the model emits e.g. "cap → highlights → left
        # mushroom" (ancestor cycle), or "body → ears + face" when
        # "ears"/"face" are already body's siblings at root.
        ancestors = [a for a in (ancestor_labels or [])
                     if isinstance(a, str) and a.strip()]
        siblings = [s for s in (sibling_labels or [])
                    if isinstance(s, str) and s.strip()]
        path_line = ""
        forbid_line = ""
        if ancestors or siblings:
            path_str = " → ".join(ancestors + [pl])
            path_line = f" Tree path (root → here): {path_str}."
            forbidden_tokens = []
            forbidden_labels = []
            seen = set()
            common_modifiers = {
                "left", "right", "top", "bottom",
                "front", "back", "upper", "lower",
            }
            # Ancestor tokens go in as TOKEN forbids (also block prefixed
            # variants — "cap" blocks "cap shadow").
            for lab in ancestors:
                for t in lab.lower().split():
                    if t and t not in seen and t not in common_modifiers:
                        seen.add(t)
                        forbidden_tokens.append(t)
            # Sibling labels go in as FULL-LABEL forbids only — the
            # token-level forbid would over-block legitimate children
            # of a different parent (e.g. "head" and "hand" both have
            # token "h"; we don't want to block "hand" because "head"
            # is a sibling).
            for lab in siblings:
                if lab.lower() not in seen:
                    seen.add(lab.lower())
                    forbidden_labels.append(lab)
            parts = []
            if forbidden_tokens:
                parts.append(
                    "tokens [" + ", ".join(f'"{t}"' for t in forbidden_tokens)
                    + "] (ancestors — any label containing these tokens is "
                    "forbidden)"
                )
            if forbidden_labels:
                parts.append(
                    "labels [" + ", ".join(f'"{t}"' for t in forbidden_labels)
                    + "] (already emitted as siblings; do not duplicate)"
                )
            if parts:
                forbid_line = (
                    " FORBIDDEN CHILD LABELS — children here may NOT use "
                    + "; ".join(parts) + "."
                )

        depth_line = ""
        if max_depth is not None:
            depth_line = (
                f" Current depth = {depth} (max {int(max_depth)}). "
                f"Children would be at depth {depth + 1}."
            )
        else:
            depth_line = f" Current depth = {depth}."

        return (
            f"It is the FULL ORIGINAL IMAGE. FOCUS REGION = bbox "
            f"[ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}] in "
            f"normalized 0-1000 image coords, labeled \"{pl}\".{depth_line}"
            f"{path_line} Decompose ONLY the contents inside that "
            f"rectangle — ignore everything outside. Sub-parts' bboxes "
            f"use the SAME 0-1000 frame (coords of the original image)."
            f"{forbid_line}{no_repeat}{tree_block}"
        )
    return (f"It is a previously-segmented part labeled \"{pl}\" "
            f"(non-part pixels appear white). Decide whether \"{pl}\" can "
            f"itself be decomposed further into 2+ semantic sub-parts.")


# USD per 1,000,000 tokens: (input_price, output_price).
# Verify against https://ai.google.dev/gemini-api/docs/pricing when Google
# updates public prices. Unknown models fall back to 0 and print a warning.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro":   (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    # Standard tier (paid), text/image/video input, output includes thinking
    # tokens. Source: https://ai.google.dev/gemini-api/docs/pricing
    "gemini-3-flash-preview":   (0.50, 3.00),
    # gemini-3.5-flash: placeholder mirrors gemini-2.5-flash; verify when
    # Google publishes 3.5 pricing.
    "gemini-3.5-flash":         (0.30, 2.50),
    # Self-hosted on vLLM; no per-token cost. Listed so the PRICING
    # warning doesn't spam stdout when --inpaint-model / --gemini-model
    # points at Qwen.
    "Qwen3.6-35B-A3B":          (0.0, 0.0),
    "Qwen/Qwen3.6-35B-A3B":     (0.0, 0.0),
    # Gemma served through the Gemini API. Free tier at time of writing, but
    # listed so the PRICING warning stays quiet and the cost column is explicit.
    "gemma-4-31b-it":           (0.0, 0.0),
    "gemma-4-26b-a4b-it":       (0.0, 0.0),
    # OpenAI GPT-5 family. USD per 1M tokens (input, output).
    # Source: https://openai.com/api/pricing (verify when tier changes).
    "gpt-5":                    (1.25, 10.00),
    "gpt-5-mini":               (0.25, 2.00),
    "gpt-5-nano":               (0.05, 0.40),
    # Anthropic Claude family. USD per 1M tokens (input, output).
    # Source: https://platform.claude.com/docs/en/about-claude/pricing
    "claude-haiku-4-5":         (1.00, 5.00),
    "claude-sonnet-4-6":        (3.00, 15.00),
    "claude-opus-4-7":          (5.00, 25.00),
    "claude-opus-4-8":          (5.00, 25.00),
}

_WARNED_UNKNOWN_MODELS: set[str] = set()


@dataclass
class GeminiUsage:
    model: str
    call_kind: str                 # "decompose_one_level"
    prompt_tokens: int
    candidates_tokens: int
    thoughts_tokens: int           # Gemini 2.5+ "thinking" tokens; billed
                                   # at the output rate but reported in a
                                   # separate ``thoughts_token_count`` field
                                   # (not folded into candidates_token_count).
    total_tokens: int
    cost_usd: float


_CLIENT = None


def _get_client(model: str | None = None):
    """Return a client suitable for ``model``.

    Routes on the model name: ``Qwen*`` → vLLM-backed shim from
    :mod:`modules.qwen_client`, ``gpt-*`` → OpenAI Chat Completions shim
    from :mod:`modules.openai_client`, everything else → the live Gemini
    client. The Gemini client is cached process-wide (``_CLIENT``) since
    the upstream call sites only ever use one Gemini model per process.
    The Qwen / OpenAI shims have their own internal singletons.
    """
    from modules.qwen_client import is_qwen_model, get_client as _qwen_client
    if is_qwen_model(model):
        return _qwen_client()
    from modules.openai_client import is_gpt_model, get_client as _openai_client
    if is_gpt_model(model):
        return _openai_client()
    from modules.claude_client import is_claude_model, get_client as _claude_client
    if is_claude_model(model):
        return _claude_client()
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _extract_json_object(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("no JSON object in response")
    parsed = json.loads(text[s : e + 1])
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object at top level")
    return parsed


# Catch-all-style words to strip from a decompose response.
# Keeps H2 satisfied even when the model slips one through.
_CATCHALL_WORDS = frozenset({
    "none", "background", "remainder", "residual", "outline",
    "rest", "leftover", "other", "misc", "etc", "fill",
})


def _is_catchall_label(label: str) -> bool:
    """True when label looks like a catch-all (single word from the
    catch-all set, optionally preceded by an article)."""
    k = (label or "").strip().lower()
    if not k:
        return True
    if k in _CATCHALL_WORDS:
        return True
    # 'a none' / 'the background' etc.
    parts = k.split()
    if parts[0] in {"a", "an", "the"} and len(parts) >= 2 and parts[1] in _CATCHALL_WORDS:
        return True
    return False


def _strip_parent_tokens(child_label: str, parent_label: str) -> str:
    """Remove parent-label tokens from a child label so H1 is enforced
    even when the model slipped (e.g. parent='sledgehammer',
    child='sledgehammer head' → 'head').

    The cleanup walks parent tokens, strips them from the child as
    whole-word matches, collapses extra spaces, and returns the
    trimmed result. If the cleanup would leave an empty string the
    original label is kept (caller decides whether to drop the node).
    """
    if not child_label or not parent_label:
        return (child_label or "").strip()
    pl = parent_label.strip().lower()
    cl = child_label.strip()
    # Drop the entire parent phrase first.
    cl_low = cl.lower()
    if pl and pl in cl_low:
        cl = (cl[:cl_low.index(pl)] + cl[cl_low.index(pl) + len(pl):]).strip()
    # Then drop each parent token individually as whole word.
    for tok in pl.split():
        if not tok:
            continue
        # whole-word removal, case-insensitive
        cl_tokens = cl.split()
        cl_tokens = [t for t in cl_tokens if t.strip().lower() != tok]
        cl = " ".join(cl_tokens).strip()
    # Collapse runs of whitespace.
    cl = " ".join(cl.split())
    return cl


def _price_for(model: str) -> tuple[float, float]:
    if model in PRICING:
        return PRICING[model]
    if model not in _WARNED_UNKNOWN_MODELS:
        print(f"[gemini] WARN: no pricing entry for '{model}'; "
              f"cost_usd will be 0. Add it to PRICING in modules/gemini_client.py.")
        _WARNED_UNKNOWN_MODELS.add(model)
    return (0.0, 0.0)


def _usage_from_response(resp: Any, model: str, call_kind: str) -> GeminiUsage:
    meta = getattr(resp, "usage_metadata", None)
    pt = int(getattr(meta, "prompt_token_count", 0) or 0)
    ct = int(getattr(meta, "candidates_token_count", 0) or 0)
    # ``thoughts_token_count`` is the Gemini-2.5+ "thinking" pass. Billed at
    # the output rate. Older SDKs / 2.0-flash don't report it — getattr
    # default of 0 keeps the math safe in that case.
    th = int(getattr(meta, "thoughts_token_count", 0) or 0)
    tt_attr = getattr(meta, "total_token_count", None)
    tt = int(tt_attr) if tt_attr is not None else (pt + ct + th)
    in_price, out_price = _price_for(model)
    cost = pt / 1_000_000 * in_price + (ct + th) / 1_000_000 * out_price
    return GeminiUsage(
        model=model, call_kind=call_kind,
        prompt_tokens=pt, candidates_tokens=ct,
        thoughts_tokens=th,
        total_tokens=tt, cost_usd=cost,
    )


def _zero_usage(model: str, call_kind: str) -> GeminiUsage:
    return GeminiUsage(model=model, call_kind=call_kind,
                       prompt_tokens=0, candidates_tokens=0,
                       thoughts_tokens=0,
                       total_tokens=0, cost_usd=0.0)


def _log_usage(u: GeminiUsage) -> None:
    print(f"[gemini] {u.call_kind:<15s} model={u.model:<20s} "
          f"in={u.prompt_tokens:<6d} out={u.candidates_tokens:<5d} "
          f"think={u.thoughts_tokens:<5d} "
          f"total={u.total_tokens:<6d} cost=${u.cost_usd:.6f}")


# Transient server-side failures we retry rather than propagate. Under an
# 8-way sharded batch the API returns these often enough to matter: without a
# retry the amodal stage drops that part and the sample silently ships without
# its inpainted completion, which is invisible in the output but changes the
# numbers.
_RETRYABLE_MARKERS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
    "500", "INTERNAL", "DEADLINE_EXCEEDED", "Deadline expired",
    "overloaded", "Timeout", "timed out",
)
_MAX_ATTEMPTS = int(os.environ.get("GEMINI_MAX_ATTEMPTS", "5"))


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(m.lower() in text.lower() for m in _RETRYABLE_MARKERS)


def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff with jitter: ~2s, 4s, 8s, 16s (capped at 30s)."""
    time.sleep(min(30.0, 2.0 ** (attempt + 1)) * (0.5 + random.random()))


class GeminiClient:
    """Stateful wrapper that knows the model name and routes every call
    through :meth:`_generate` so usage logging is always applied.

    Callers receive ``usage_metadata`` as a plain dict (``asdict(GeminiUsage)``)
    and are expected to embed it into the same record they already persist
    (``decompose_calls[i]`` for ``decompose_one_level``).
    """

    def __init__(self, model: str = "gemini-3-flash-preview",
                 decompose_thinking_budget: int = 0):
        self.model = model
        # Gemini-3's "thinking" pass is off by default for every call
        # (``thinking_budget=0``) to keep cost predictable. The decompose
        # path is the most reasoning-heavy call — hierarchical part
        # naming + bbox layout — so it gets its own opt-in slot.
        # Setting ``decompose_thinking_budget`` > 0 (or -1 for "auto")
        # routes ONLY ``decompose_one_level`` through a higher budget;
        # judge/audit/inpaint calls stay at 0. Costs scale roughly
        # linearly with the budget (thinking tokens are billed as
        # output and NOT surfaced in ``candidates_tokens``), so leave
        # this off when running large batches.
        self.decompose_thinking_budget = int(decompose_thinking_budget)
        # Optional replay queue for ``decompose_one_level``. When set, calls
        # consume cached entries (matched by parent_label + focus_bbox) in
        # DFS order instead of hitting the API. See ``set_replay_cache``.
        self._replay_calls: list[dict] | None = None
        self._replay_idx: int = 0

    def set_replay_cache(self, decompose_calls: list[dict] | None) -> None:
        """Install a cached ``decompose_calls`` list (from a previous run's
        ``gemini.json``) so ``decompose_one_level`` returns cached results
        in DFS order. Pass ``None`` to clear and resume live API calls."""
        self._replay_calls = list(decompose_calls) if decompose_calls else None
        self._replay_idx = 0


    def _consume_replay(self, parent_label: str,
                        focus_bbox: list | tuple | None
                        ) -> tuple[bool, list[dict], str | None, str, dict]:
        assert self._replay_calls is not None
        if self._replay_idx >= len(self._replay_calls):
            raise RuntimeError(
                f"gemini replay exhausted at idx {self._replay_idx}; "
                f"tree must have diverged from cached run"
            )
        call = self._replay_calls[self._replay_idx]
        self._replay_idx += 1
        cached_pl = call.get("parent_label", "") or ""
        if cached_pl != (parent_label or ""):
            raise RuntimeError(
                f"gemini replay mismatch at idx {self._replay_idx - 1}: "
                f"cached parent_label={cached_pl!r} != live {parent_label!r}"
            )
        cached_fb = call.get("focus_bbox")
        live_fb = list(focus_bbox) if focus_bbox is not None else None
        cached_fb_norm = list(cached_fb) if cached_fb is not None else None
        if cached_fb_norm != live_fb:
            raise RuntimeError(
                f"gemini replay mismatch at idx {self._replay_idx - 1}: "
                f"cached focus_bbox={cached_fb_norm} != live {live_fb}"
            )
        is_atomic = bool(call.get("is_atomic"))
        components = call.get("components", []) or []
        parse_error = call.get("parse_error")
        raw = call.get("raw_response", "") or ""
        usage = call.get("usage_metadata") or asdict(
            _zero_usage(self.model, "decompose_one_level [replay]")
        )
        if not isinstance(usage, dict):
            usage = asdict(usage)
        return is_atomic, components, parse_error, raw, usage

    def _generate(self, contents: list, call_kind: str,
                  *, model: str | None = None,
                  thinking_budget: int | None = None,
                  max_output_tokens: int | None = None,
                  ) -> tuple[str, GeminiUsage]:
        """Common path: call generate_content, extract text + usage, log.

        ``model`` lets a caller override the per-instance default for one
        call (used by v7 audit which wants gemini-2.5-pro while the rest
        of the pipeline stays on the lighter decompose model).

        ``thinking_budget`` defaults to 0 (Gemini-3's "thinking" pass off).
        Thinking tokens are billed at the standard output rate but are
        NOT reported in ``candidates_tokens`` in the response — they show
        up only on Google's billing dashboard. The decompose path may
        opt in via ``self.decompose_thinking_budget``; other call sites
        keep the safe-by-default 0 unless they pass a value explicitly.

        ``max_output_tokens`` caps a single response. The pipeline's own calls
        emit short JSON and leave it unset, but open-ended generation (asking a
        model for a whole SVG) can degenerate into a repetition loop — one
        observed sample ran to 65k output tokens, $0.20 for one image — so
        those callers should always pass a cap.
        """
        use_model = model or self.model
        # Qwen / GPT paths skip the Gemini-only ``GenerateContentConfig``
        # build — their shims ignore ``config`` and constructing it
        # requires importing ``google.genai`` (which we don't want to
        # demand when the whole run stays on a non-Gemini backend).
        from modules.qwen_client import is_qwen_model
        from modules.openai_client import is_gpt_model
        from modules.claude_client import is_claude_model
        client = _get_client(use_model)
        if (is_qwen_model(use_model) or is_gpt_model(use_model)
                or is_claude_model(use_model)):
            kwargs = {"model": use_model, "contents": contents}
        else:
            from google.genai import types
            budget = 0 if thinking_budget is None else int(thinking_budget)
            # Gemma is served through the same endpoint but rejects a thinking
            # budget outright ("Thinking budget is not supported for this
            # model.", HTTP 400), so it gets the config without one. Its
            # behaviour already matches thinking_budget=0, which is this
            # pipeline's default for every call.
            cfg_kwargs: dict[str, Any] = {"temperature": 0.0}
            if not use_model.lower().startswith("gemma"):
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=budget)
            if max_output_tokens:
                cfg_kwargs["max_output_tokens"] = int(max_output_tokens)
            cfg = types.GenerateContentConfig(**cfg_kwargs)
            kwargs = {"model": use_model, "contents": contents, "config": cfg}

        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = client.models.generate_content(**kwargs)
                break
            except Exception as e:
                if attempt == _MAX_ATTEMPTS - 1 or not _is_retryable(e):
                    raise
                print(f"[gemini] {call_kind} retry {attempt + 1}/"
                      f"{_MAX_ATTEMPTS - 1} after {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
                _sleep_backoff(attempt)

        usage = _usage_from_response(resp, use_model, call_kind)
        _log_usage(usage)
        return (resp.text or "").strip(), usage

    def decompose_one_level(
        self,
        img: Image.Image,
        parent_label: str = "",
        focus_bbox: list | tuple | None = None,
        depth: int = 0,
        ancestor_labels: list[str] | None = None,
        max_depth: int | None = None,
        sibling_labels: list[str] | None = None,
        tree_so_far: list[dict] | None = None,
    ) -> tuple[bool, list[dict], str | None, str, dict]:
        """Ask Gemini to decompose ONE region into mutually-exclusive
        sub-parts (or mark it atomic).

        ``focus_bbox`` (0-1000 [ymin, xmin, ymax, xmax]) lets the caller
        pass the FULL original image and tell Gemini which rectangle to
        decompose without cropping. When omitted, falls back to the
        legacy crop-based prompt (caller already cropped non-region
        pixels to white).

        ``depth`` (caller-supplied, root=0) and ``ancestor_labels``
        (the chain root → … → parent, EXCLUDING the parent itself)
        are woven into the prompt so the model knows where it is in
        the tree and which labels it must NOT re-emit as children.
        Together they kill the "cap → highlights → left mushroom"
        cascade where descendants re-discover ancestor labels.

        Returns ``(is_atomic, components, parse_error, raw_text, usage)``.

        ``components`` is a list of dicts with keys ``label`` and ``_bbox``
        ([ymin, xmin, ymax, xmax] in 0-1000 normalized). On API/parse
        failure, ``is_atomic`` is forced to True (treat as leaf) so the
        caller naturally stops recursing rather than fabricating bad
        children.
        """
        if self._replay_calls is not None:
            return self._consume_replay(parent_label, focus_bbox)
        prompt = DECOMPOSE_ONE_LEVEL_PROMPT.replace(
            "{parent_context}",
            _build_decompose_context(
                parent_label, focus_bbox,
                depth=depth, ancestor_labels=ancestor_labels,
                max_depth=max_depth,
                sibling_labels=sibling_labels,
                tree_so_far=tree_so_far,
            ),
        )
        try:
            text, usage = self._generate(
                [img, prompt], "decompose_one_level",
                thinking_budget=self.decompose_thinking_budget,
            )
        except Exception as e:
            usage = _zero_usage(self.model, "decompose_one_level")
            return True, [], f"decompose_api_error: {type(e).__name__}: {e}", "", asdict(usage)

        try:
            parsed = _extract_json_object(text)
            is_atomic = bool(parsed.get("is_atomic", True))
            comps_raw = parsed.get("components", []) or []
            components: list[dict] = []
            for c in comps_raw:
                if not isinstance(c, dict):
                    continue
                label = str(c.get("label", "")).strip()
                if not label:
                    continue
                components.append({
                    "label": label,
                    "_bbox": c.get("_bbox"),
                })
            return is_atomic, components, None, text, asdict(usage)
        except Exception as ex:
            return (True, [],
                    f"decompose_parse_error: {type(ex).__name__}: {ex}",
                    text, asdict(usage))


__all__ = [
    "GeminiClient", "GeminiUsage", "PRICING",
    "DECOMPOSE_ONE_LEVEL_PROMPT",
]
