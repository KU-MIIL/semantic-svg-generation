"""Inpaint completion agent: Gemini-polygon mask + FLUX.1-Fill-dev inpaint.

Pipeline per ``inpaint()`` call:
    part RGBA (SAM cutout) + scene RGB + label
      → :func:`judge_needs_inpaint` (Gemini binary gate)        ── new
      → if no  : return original part, no polygon / FLUX
      → if yes :
          :func:`predict_complete_polygon` (Gemini P3 shape)
        + :func:`polygon_to_mask` + :func:`build_repaint_mask`
        + FLUX.1-Fill-dev with the v08 vector-graphic prompt
        + composite: keep original visible pixels, alpha = visible | repaint

FLUX is heavy (~35 GB bf16 fully resident) so the pipe is lazy-loaded on the first
``inpaint()`` call. Use :func:`modules.inpainter.client.get_agent` for the
singleton entry point — the bare class here is suitable for tests / CLI.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from .imaging import rgba_on_white, rgb_to_rgba_white_alpha
from .judge import SiblingInfo, judge_needs_inpaint
from .mask import build_repaint_mask, polygon_to_mask
from .picker import pick_best
from .polygon import POLYGON_PROMPTS, predict_complete_polygon

FLUX_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_SIZE = 1024            # FLUX.1-Fill-dev native res
GUIDANCE = 10.0
STEPS = 30
SEED = 42

# v08 = FLUX inpaint prompt (best in the prompt sweep)
V08_FLUX_PROMPT = ("reconstruct the full {label} as a single clean "
                   "vector graphic image, continue the existing colors "
                   "and edges")


class InpaintAgent:
    """Judge → polygon → FLUX inpaint pipeline in one :meth:`inpaint` call.

    State: the FLUX.1-Fill-dev pipe (~35 GB bf16, or ~26 GB peak with
    SVGAGENT_FLUX_OFFLOAD=1) is lazy-loaded on the
    first call to :meth:`inpaint` with a non-empty repaint mask and kept
    around for the rest of the process — there is no release method, so
    the usual pattern is one shared instance per CUDA device (use
    :func:`modules.inpainter.client.get_agent` for that).

    Two entry points:
      * :meth:`make_mask` — judge + polygon + repaint mask only (no GPU).
                            Useful for tests, prompt sweeps, or for
                            inspecting what FLUX would have been given.
      * :meth:`inpaint`   — the full pipeline. Internally calls
                            :meth:`make_mask` and routes the result
                            through FLUX when the repaint mask is non-
                            empty; otherwise returns the original part
                            untouched.

    All Gemini knobs (``gemini_model``, ``thinking_level``,
    ``judge_level``, ``skip_judge``) are kwargs on both methods. FLUX is
    pinned via the module-level constants (``FLUX_MODEL``, ``FLUX_SIZE``,
    ``GUIDANCE``, ``STEPS``, ``V08_FLUX_PROMPT`` — locked after a prompt
    sweep, see :mod:`modules.inpainter.polygon` for the polygon-side
    lockdown notes).
    """

    def __init__(self, device: str = "cuda:0",
                 flux_model: str = FLUX_MODEL):
        self.device = device
        self.flux_model = flux_model
        self._pipe = None

    # -- FLUX loader -------------------------------------------------------
    def _pipe_(self):
        if self._pipe is None:
            from diffusers import FluxFillPipeline
            print(f"[inpaint] loading {self.flux_model} "
                  f"(bf16) on {self.device} ...")
            pipe = FluxFillPipeline.from_pretrained(
                self.flux_model, torch_dtype=torch.bfloat16)
            # Fully resident, FLUX.1-Fill-dev needs ~35 GB (12B transformer +
            # T5-XXL, both bf16) -- not the ~15 GB this module's docstring used
            # to claim. On a card shared with another job that does not fit, and
            # the OOM surfaces as a caught per-part exception: every layer
            # silently falls back to its visible mask and the run looks fine
            # while no amodal completion happened at all. Set
            # SVGAGENT_FLUX_OFFLOAD=1 to keep the weights in host RAM and stage
            # one component at a time (peak ~26 GB, ~2x slower, same math).
            if os.environ.get("SVGAGENT_FLUX_OFFLOAD", "").lower() in (
                    "1", "true", "yes"):
                print("[inpaint] model CPU offload enabled "
                      "(SVGAGENT_FLUX_OFFLOAD)")
                pipe.enable_model_cpu_offload(device=self.device)
            else:
                pipe = pipe.to(self.device)
            self._pipe = pipe
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    # -- Gemini-only step (no GPU) -----------------------------------------
    def make_mask(self, part_rgba: Image.Image, scene_rgb: Image.Image,
                  label: str,
                  *,
                  siblings: list[SiblingInfo] | None = None,
                  parent_label: str | None = None,
                  parent_coverage: float | None = None,
                  description_mode: bool = False,
                  gemini_model: str | None = None,
                  thinking_level: str | None = None,
                  judge_level: str | None = None,
                  skip_judge: bool = False) -> dict:
        """Run judge → polygon → mask (no FLUX). Returns dict with
        ``polygon_mask`` (PIL L), ``repaint_mask`` (PIL L), ``n_vertices``,
        ``parse_error``, plus the judge outcome (``occluded`` /
        ``needs_inpaint``, ``occluding_siblings``, ``description``,
        ``judge_reason``, pixel context).

        When the judge says ``occluded=false`` polygon is skipped and
        both masks come back empty so the caller can short-circuit FLUX.

        ``siblings`` / ``parent_label`` / ``parent_coverage`` feed the
        closed-set occlusion gate (v2). ``description_mode=True`` asks
        the judge to also emit a fine-grained description sentence that
        :meth:`fill_from_mask` then uses as the FLUX prompt label.

        ``gemini_model`` / ``thinking_level`` drive the polygon call;
        ``judge_level`` (defaults to ``thinking_level``) drives the judge
        call. Pass ``skip_judge=True`` to bypass the judge entirely.
        """
        W, H = scene_rgb.size
        judge_dbg: dict = {}
        is_residual = False
        occluding_sibling_ids: list[str] = []
        occluding_siblings: list[str] = []
        description: str | None = None
        if skip_judge:
            needs = True
        else:
            jo = judge_needs_inpaint(
                scene_rgb, part_rgba, label,
                siblings=siblings,
                parent_label=parent_label,
                parent_coverage=parent_coverage,
                make_description=description_mode,
                model=gemini_model,
                thinking_level=judge_level or thinking_level,
            )
            needs = jo.occluded
            is_residual = jo.is_residual
            judge_dbg = jo.debug
            occluding_sibling_ids = list(jo.occluding_siblings)
            occluding_siblings = list(jo.occluder_labels)
            description = jo.description

        # Common judge-surfaced fields included in both the short-circuit
        # and full-mask returns so downstream consumers can read the same
        # keys regardless of whether FLUX runs.
        common_judge_fields = {
            "is_residual": is_residual,
            "occluded": needs,
            "occluding_sibling_ids": occluding_sibling_ids,
            "occluding_siblings": occluding_siblings,
            "description": description,
            "judge_reason": judge_dbg.get("reason"),
            "judge_parse_warning": judge_dbg.get("parse_warning"),
            "judge_parse_error": judge_dbg.get("parse_error"),
            "judge_dropped_names": judge_dbg.get("dropped_names", []),
            "part_visible_px": judge_dbg.get("part_visible_px"),
            "scene_color_px": judge_dbg.get("scene_color_px"),
            "part_scene_ratio_pct": judge_dbg.get("part_scene_ratio_pct"),
            "n_siblings_in_prompt":
                judge_dbg.get("n_siblings_in_prompt", 0),
            "description_requested":
                judge_dbg.get("description_requested", False),
            "judge_usage": judge_dbg.get("usage"),
        }

        if not needs:
            empty = np.zeros((H, W), np.uint8)
            return {
                "polygon_mask": Image.fromarray(empty, "L"),
                "repaint_mask": Image.fromarray(empty, "L"),
                "n_vertices": 0,
                "parse_error": judge_dbg.get("parse_error"),
                "needs_inpaint": False,
                "polygon_usage": None,
                **common_judge_fields,
            }

        poly, dbg = predict_complete_polygon(
            scene_rgb, part_rgba, label,
            model=gemini_model, thinking_level=thinking_level,
        )
        pm = (polygon_to_mask(poly, W, H) if poly
              else np.zeros((H, W), np.uint8))
        rp = build_repaint_mask(
            pm, np.array(part_rgba.convert("RGBA"))[..., 3], scene_rgb,
        )
        return {
            "polygon_mask": Image.fromarray(pm, "L"),
            "repaint_mask": Image.fromarray(rp, "L"),
            "n_vertices": dbg.get("n_vertices", 0),
            "parse_error": dbg.get("parse_error"),
            "needs_inpaint": True,
            "polygon_usage": dbg.get("usage"),
            **common_judge_fields,
        }

    # -- FLUX-only step (consumes a precomputed mask) ----------------------
    def fill_from_mask(self, part_rgba: Image.Image,
                       scene_rgb: Image.Image,
                       label: str, mk: dict,
                       seed: int = SEED) -> dict:
        """Run FLUX using a precomputed mask dict (output of
        :meth:`make_mask`). Splitting this out lets callers parallelize
        the Gemini polygon calls across parts and batch the FLUX side
        independently. Returns the same shape as :meth:`inpaint`.

        Skips FLUX when ``repaint_mask`` is empty — ``fill_rgba`` falls
        back to the original ``part_rgba``, same contract as
        :meth:`inpaint`.
        """
        part_rgba = part_rgba.convert("RGBA")
        scene_rgb = scene_rgb.convert("RGB")
        rp_img = mk["repaint_mask"]
        rp = np.array(rp_img) > 127
        tgt = np.array(part_rgba)
        vis = tgt[..., 3] > 0
        cond = rgba_on_white(part_rgba)

        # When the description judge produced a fine-grained sentence,
        # feed it into the FLUX prompt slot instead of the bare label —
        # the V08 template's "as a single clean vector graphic image,
        # continue the existing colors and edges" suffix still anchors
        # the style.
        prompt_label = mk.get("description") or label

        if rp.sum() == 0:
            fill_rgba = part_rgba
            flux_prompt_label = None
        else:
            pipe = self._pipe_()
            gen = torch.Generator(device=self.device).manual_seed(seed)
            res = pipe(
                prompt=V08_FLUX_PROMPT.format(label=prompt_label),
                image=cond.resize((FLUX_SIZE, FLUX_SIZE), Image.LANCZOS),
                mask_image=rp_img.resize((FLUX_SIZE, FLUX_SIZE),
                                         Image.NEAREST),
                height=FLUX_SIZE, width=FLUX_SIZE,
                guidance_scale=GUIDANCE, num_inference_steps=STEPS,
                max_sequence_length=512, generator=gen,
            ).images[0]
            frgb = res.convert("RGB").resize(part_rgba.size, Image.LANCZOS)
            arr = np.array(rgb_to_rgba_white_alpha(frgb))
            arr[vis, :3] = tgt[vis, :3]
            allowed = vis | rp
            arr[..., 3] = np.where(
                allowed, np.maximum(arr[..., 3], tgt[..., 3]),
                0).astype(np.uint8)
            fill_rgba = Image.fromarray(arr, "RGBA")
            flux_prompt_label = prompt_label

        return {
            "fill_rgba": fill_rgba,
            "fill_on_white": rgba_on_white(fill_rgba),
            "repaint_mask": rp_img,
            "polygon_mask": mk["polygon_mask"],
            "n_vertices": mk["n_vertices"],
            "parse_error": mk["parse_error"],
            "needs_inpaint": mk.get("needs_inpaint", True),
            "is_residual": mk.get("is_residual", False),
            "occluded": mk.get("occluded", mk.get("needs_inpaint", True)),
            "occluding_sibling_ids":
                list(mk.get("occluding_sibling_ids") or []),
            "occluding_siblings":
                list(mk.get("occluding_siblings") or []),
            "description": mk.get("description"),
            "flux_prompt_label": flux_prompt_label,
            "judge_reason": mk.get("judge_reason"),
            "judge_parse_warning": mk.get("judge_parse_warning"),
            "judge_parse_error": mk.get("judge_parse_error"),
            "judge_dropped_names": list(mk.get("judge_dropped_names") or []),
            "n_siblings_in_prompt": mk.get("n_siblings_in_prompt", 0),
            "description_requested":
                mk.get("description_requested", False),
            "part_visible_px": mk.get("part_visible_px"),
            "scene_color_px": mk.get("scene_color_px"),
            "part_scene_ratio_pct": mk.get("part_scene_ratio_pct"),
            "judge_usage": mk.get("judge_usage"),
            "polygon_usage": mk.get("polygon_usage"),
        }

    # -- Full pipeline -----------------------------------------------------
    def inpaint(self, part_rgba: Image.Image, scene_rgb: Image.Image,
                label: str, seed: int = SEED,
                *,
                siblings: list[SiblingInfo] | None = None,
                parent_label: str | None = None,
                parent_coverage: float | None = None,
                description_mode: bool = False,
                gemini_model: str | None = None,
                thinking_level: str | None = None,
                judge_level: str | None = None,
                skip_judge: bool = False) -> dict:
        """Run the full judge → polygon → FLUX pipeline. Returns dict with
        ``fill_rgba`` (PIL RGBA), ``fill_on_white`` (PIL RGB),
        ``repaint_mask`` (PIL L), ``polygon_mask`` (PIL L), ``n_vertices``,
        ``parse_error``, plus the judge fields surfaced by
        :meth:`make_mask` (``occluded`` / ``needs_inpaint``,
        ``occluding_siblings``, ``description``, ``judge_reason``, …).

        When the judge says ``occluded=false`` OR the repaint mask is
        empty (no occluder pixels), FLUX is skipped and ``fill_rgba`` is
        the original ``part_rgba``.
        """
        mk = self.make_mask(
            part_rgba, scene_rgb, label,
            siblings=siblings,
            parent_label=parent_label,
            parent_coverage=parent_coverage,
            description_mode=description_mode,
            gemini_model=gemini_model, thinking_level=thinking_level,
            judge_level=judge_level, skip_judge=skip_judge,
        )
        return self.fill_from_mask(part_rgba, scene_rgb, label, mk, seed=seed)

    # -- Multi-prompt Gemini step (no GPU) ---------------------------------
    def make_mask_multi(self, part_rgba: Image.Image,
                        scene_rgb: Image.Image, label: str,
                        *,
                        prompt_ids: tuple[str, ...] = ("p0", "p1", "p2", "p3"),
                        siblings: list[SiblingInfo] | None = None,
                        parent_label: str | None = None,
                        parent_coverage: float | None = None,
                        description_mode: bool = False,
                        gemini_model: str | None = None,
                        thinking_level: str | None = "medium",
                        judge_level: str | None = None,
                        skip_judge: bool = False) -> dict:
        """Run judge ONCE then polygon × N (one per ``prompt_id``).

        The occlusion judge is shared across all N polygon variants --
        the judge inputs (scene, part) don't depend on the polygon
        prompt, so calling it once saves N-1 Gemini calls.

        Returns a dict with the common judge-surfaced fields at the top
        level plus a ``candidates`` sub-dict keyed by ``prompt_id``::

            {
              "needs_inpaint": bool,
              "occluded": bool, "is_residual": bool,
              "occluding_siblings", "description", ...
              "judge_usage", "judge_reason", ...
              "candidates": {
                  "p0": {polygon_mask, repaint_mask, n_vertices,
                         parse_error, polygon_usage},
                  "p1": {...}, ...
              },
            }

        When the judge says ``occluded=false`` ``candidates`` comes back
        empty so the caller can skip FLUX entirely.
        """
        unknown = [p for p in prompt_ids if p not in POLYGON_PROMPTS]
        if unknown:
            raise ValueError(
                f"unknown prompt_ids={unknown}; "
                f"expected subset of {list(POLYGON_PROMPTS)}")

        W, H = scene_rgb.size
        judge_dbg: dict = {}
        is_residual = False
        occluding_sibling_ids: list[str] = []
        occluding_siblings: list[str] = []
        description: str | None = None
        if skip_judge:
            needs = True
        else:
            jo = judge_needs_inpaint(
                scene_rgb, part_rgba, label,
                siblings=siblings,
                parent_label=parent_label,
                parent_coverage=parent_coverage,
                make_description=description_mode,
                model=gemini_model,
                thinking_level=judge_level or thinking_level,
            )
            needs = jo.occluded
            is_residual = jo.is_residual
            judge_dbg = jo.debug
            occluding_sibling_ids = list(jo.occluding_siblings)
            occluding_siblings = list(jo.occluder_labels)
            description = jo.description

        common_judge_fields = {
            "is_residual": is_residual,
            "occluded": needs,
            "occluding_sibling_ids": occluding_sibling_ids,
            "occluding_siblings": occluding_siblings,
            "description": description,
            "judge_reason": judge_dbg.get("reason"),
            "judge_parse_warning": judge_dbg.get("parse_warning"),
            "judge_parse_error": judge_dbg.get("parse_error"),
            "judge_dropped_names": judge_dbg.get("dropped_names", []),
            "part_visible_px": judge_dbg.get("part_visible_px"),
            "scene_color_px": judge_dbg.get("scene_color_px"),
            "part_scene_ratio_pct": judge_dbg.get("part_scene_ratio_pct"),
            "n_siblings_in_prompt":
                judge_dbg.get("n_siblings_in_prompt", 0),
            "description_requested":
                judge_dbg.get("description_requested", False),
            "judge_usage": judge_dbg.get("usage"),
        }

        if not needs:
            return {
                "needs_inpaint": False,
                "candidates": {},
                **common_judge_fields,
            }

        target_alpha = np.array(part_rgba.convert("RGBA"))[..., 3]
        candidates: dict[str, dict] = {}
        for pid in prompt_ids:
            poly, dbg = predict_complete_polygon(
                scene_rgb, part_rgba, label,
                model=gemini_model, thinking_level=thinking_level,
                prompt_id=pid,
            )
            pm = (polygon_to_mask(poly, W, H) if poly
                  else np.zeros((H, W), np.uint8))
            rp = build_repaint_mask(pm, target_alpha, scene_rgb)
            candidates[pid] = {
                "polygon_mask": Image.fromarray(pm, "L"),
                "repaint_mask": Image.fromarray(rp, "L"),
                "n_vertices": dbg.get("n_vertices", 0),
                "parse_error": dbg.get("parse_error"),
                "polygon_usage": dbg.get("usage"),
            }

        return {
            "needs_inpaint": True,
            "candidates": candidates,
            **common_judge_fields,
        }

    # -- Multi-prompt FLUX + pick-judge -----------------------------------
    def fill_from_mask_multi(self, part_rgba: Image.Image,
                             scene_rgb: Image.Image,
                             label: str, mk: dict,
                             seed: int = SEED,
                             *,
                             gemini_model: str | None = None,
                             pick_level: str | None = "medium") -> dict:
        """Run FLUX × N (one per candidate in ``mk["candidates"]``) then
        ask :func:`pick_best` which result is best.

        ``mk`` is the output of :meth:`make_mask_multi`. Returns a dict
        shaped like :meth:`fill_from_mask` for the winner (so the caller
        can drop it into ``part.extra["inpaint"]`` unchanged), plus
        ``candidates`` with every fill kept for debugging and
        ``best_tag`` / ``pick_*`` describing the pick decision.

        When ``mk["needs_inpaint"]`` is False or every repaint mask is
        empty FLUX and the picker are both skipped and the original part
        is returned untouched.
        """
        part_rgba = part_rgba.convert("RGBA")
        scene_rgb = scene_rgb.convert("RGB")
        W, H = scene_rgb.size
        candidates = dict(mk.get("candidates") or {})
        prompt_label = mk.get("description") or label

        def _empty_short_circuit(reason: str) -> dict:
            empty_L = Image.fromarray(np.zeros((H, W), np.uint8), "L")
            return {
                "fill_rgba": part_rgba,
                "fill_on_white": rgba_on_white(part_rgba),
                "repaint_mask": empty_L,
                "polygon_mask": empty_L,
                "n_vertices": 0,
                "parse_error": None,
                "needs_inpaint": mk.get("needs_inpaint", False),
                "is_residual": mk.get("is_residual", False),
                "occluded": mk.get("occluded", False),
                "occluding_sibling_ids":
                    list(mk.get("occluding_sibling_ids") or []),
                "occluding_siblings":
                    list(mk.get("occluding_siblings") or []),
                "description": mk.get("description"),
                "flux_prompt_label": None,
                "judge_reason": mk.get("judge_reason"),
                "judge_parse_warning": mk.get("judge_parse_warning"),
                "judge_parse_error": mk.get("judge_parse_error"),
                "judge_dropped_names":
                    list(mk.get("judge_dropped_names") or []),
                "n_siblings_in_prompt": mk.get("n_siblings_in_prompt", 0),
                "description_requested":
                    mk.get("description_requested", False),
                "part_visible_px": mk.get("part_visible_px"),
                "scene_color_px": mk.get("scene_color_px"),
                "part_scene_ratio_pct": mk.get("part_scene_ratio_pct"),
                "judge_usage": mk.get("judge_usage"),
                "polygon_usage": None,
                "best_tag": None,
                "best_letter": None,
                "pick_mapping": None,
                "pick_reason": reason,
                "pick_raw": "",
                "pick_parse_error": None,
                "pick_usage": None,
                "candidates": {},
            }

        if not mk.get("needs_inpaint") or not candidates:
            return _empty_short_circuit("(skipped -- judge said no inpaint)")

        tgt = np.array(part_rgba)
        vis = tgt[..., 3] > 0
        cond = rgba_on_white(part_rgba)
        prompt_ids = list(candidates.keys())

        any_repaint = False
        for pid in prompt_ids:
            rp_img = candidates[pid]["repaint_mask"]
            rp = np.array(rp_img) > 127
            if rp.sum() == 0:
                candidates[pid]["fill_rgba"] = part_rgba
                candidates[pid]["fill_on_white"] = rgba_on_white(part_rgba)
                candidates[pid]["flux_skipped"] = True
                continue
            any_repaint = True
            pipe = self._pipe_()
            gen = torch.Generator(device=self.device).manual_seed(seed)
            res = pipe(
                prompt=V08_FLUX_PROMPT.format(label=prompt_label),
                image=cond.resize((FLUX_SIZE, FLUX_SIZE), Image.LANCZOS),
                mask_image=rp_img.resize((FLUX_SIZE, FLUX_SIZE),
                                         Image.NEAREST),
                height=FLUX_SIZE, width=FLUX_SIZE,
                guidance_scale=GUIDANCE, num_inference_steps=STEPS,
                max_sequence_length=512, generator=gen,
            ).images[0]
            frgb = res.convert("RGB").resize(part_rgba.size, Image.LANCZOS)
            arr = np.array(rgb_to_rgba_white_alpha(frgb))
            arr[vis, :3] = tgt[vis, :3]
            allowed = vis | rp
            arr[..., 3] = np.where(
                allowed, np.maximum(arr[..., 3], tgt[..., 3]),
                0).astype(np.uint8)
            fill_rgba = Image.fromarray(arr, "RGBA")
            candidates[pid]["fill_rgba"] = fill_rgba
            candidates[pid]["fill_on_white"] = rgba_on_white(fill_rgba)
            candidates[pid]["flux_skipped"] = False

        if not any_repaint:
            # All candidates had empty repaint masks -- FLUX did nothing,
            # no point asking the picker. Same contract as the empty-mask
            # short-circuit above but keep ``candidates`` so debug viz
            # still has the per-prompt polygons.
            out = _empty_short_circuit(
                "(all candidates had empty repaint masks)")
            out["candidates"] = candidates
            return out

        pick = pick_best(
            scene_rgb, part_rgba,
            {pid: candidates[pid]["fill_rgba"] for pid in prompt_ids},
            label,
            model=gemini_model, thinking_level=pick_level,
        )
        best_tag = pick["best_tag"]
        winner = candidates[best_tag]
        flux_prompt_label = (None if winner.get("flux_skipped")
                             else prompt_label)

        return {
            "fill_rgba": winner["fill_rgba"],
            "fill_on_white": winner["fill_on_white"],
            "repaint_mask": winner["repaint_mask"],
            "polygon_mask": winner["polygon_mask"],
            "n_vertices": winner["n_vertices"],
            "parse_error": winner["parse_error"],
            "needs_inpaint": mk.get("needs_inpaint", True),
            "is_residual": mk.get("is_residual", False),
            "occluded": mk.get("occluded", mk.get("needs_inpaint", True)),
            "occluding_sibling_ids":
                list(mk.get("occluding_sibling_ids") or []),
            "occluding_siblings":
                list(mk.get("occluding_siblings") or []),
            "description": mk.get("description"),
            "flux_prompt_label": flux_prompt_label,
            "judge_reason": mk.get("judge_reason"),
            "judge_parse_warning": mk.get("judge_parse_warning"),
            "judge_parse_error": mk.get("judge_parse_error"),
            "judge_dropped_names":
                list(mk.get("judge_dropped_names") or []),
            "n_siblings_in_prompt": mk.get("n_siblings_in_prompt", 0),
            "description_requested":
                mk.get("description_requested", False),
            "part_visible_px": mk.get("part_visible_px"),
            "scene_color_px": mk.get("scene_color_px"),
            "part_scene_ratio_pct": mk.get("part_scene_ratio_pct"),
            "judge_usage": mk.get("judge_usage"),
            "polygon_usage": winner.get("polygon_usage"),
            "best_tag": best_tag,
            "best_letter": pick["best_letter"],
            "pick_mapping": pick["mapping"],
            "pick_reason": pick["reason"],
            "pick_raw": pick["raw"],
            "pick_parse_error": pick["parse_error"],
            "pick_usage": pick.get("usage"),
            "candidates": candidates,
        }

    # -- Batched FLUX (no pick) -------------------------------------------
    def fill_from_mask_batch(self, part_rgba: Image.Image,
                             scene_rgb: Image.Image,
                             label: str, mk: dict,
                             seed: int = SEED) -> dict:
        """Like ``fill_from_mask_multi`` but runs every non-empty candidate
        through FLUX in a SINGLE forward pass (true batch=N) and does NOT
        call the pick-judge. The async runner schedules ``pick_best`` as a
        fire-and-forget task on the event loop instead.

        Return shape matches ``fill_from_mask_multi`` for the per-candidate
        ``candidates`` sub-dict (each entry gets ``fill_rgba`` /
        ``fill_on_white`` / ``flux_skipped``), and surfaces the shared
        judge / mk fields at the top level. The picker-related keys
        (``best_tag``, ``pick_*``) are left as ``None`` -- the caller must
        fill them in after pick_best returns.
        """
        part_rgba = part_rgba.convert("RGBA")
        scene_rgb = scene_rgb.convert("RGB")
        W, H = scene_rgb.size
        candidates = dict(mk.get("candidates") or {})
        prompt_label = mk.get("description") or label

        def _empty_short_circuit(reason: str) -> dict:
            empty_L = Image.fromarray(np.zeros((H, W), np.uint8), "L")
            return {
                "fill_rgba": part_rgba,
                "fill_on_white": rgba_on_white(part_rgba),
                "repaint_mask": empty_L,
                "polygon_mask": empty_L,
                "n_vertices": 0,
                "parse_error": None,
                "needs_inpaint": mk.get("needs_inpaint", False),
                "is_residual": mk.get("is_residual", False),
                "occluded": mk.get("occluded", False),
                "occluding_sibling_ids":
                    list(mk.get("occluding_sibling_ids") or []),
                "occluding_siblings":
                    list(mk.get("occluding_siblings") or []),
                "description": mk.get("description"),
                "flux_prompt_label": None,
                "judge_reason": mk.get("judge_reason"),
                "judge_parse_warning": mk.get("judge_parse_warning"),
                "judge_parse_error": mk.get("judge_parse_error"),
                "judge_dropped_names":
                    list(mk.get("judge_dropped_names") or []),
                "n_siblings_in_prompt": mk.get("n_siblings_in_prompt", 0),
                "description_requested":
                    mk.get("description_requested", False),
                "part_visible_px": mk.get("part_visible_px"),
                "scene_color_px": mk.get("scene_color_px"),
                "part_scene_ratio_pct": mk.get("part_scene_ratio_pct"),
                "judge_usage": mk.get("judge_usage"),
                "polygon_usage": None,
                "best_tag": None,
                "best_letter": None,
                "pick_mapping": None,
                "pick_reason": reason,
                "pick_raw": "",
                "pick_parse_error": None,
                "pick_usage": None,
                "candidates": {},
                "active_pids": [],
            }

        if not mk.get("needs_inpaint") or not candidates:
            return _empty_short_circuit("(skipped -- judge said no inpaint)")

        tgt = np.array(part_rgba)
        vis = tgt[..., 3] > 0
        cond = rgba_on_white(part_rgba)
        prompt_ids = list(candidates.keys())

        # Partition: non-empty masks go to the FLUX batch, empty ones
        # short-circuit to the original part (same as the sequential path).
        active_pids: list[str] = []
        for pid in prompt_ids:
            rp = np.array(candidates[pid]["repaint_mask"]) > 127
            if rp.sum() == 0:
                candidates[pid]["fill_rgba"] = part_rgba
                candidates[pid]["fill_on_white"] = rgba_on_white(part_rgba)
                candidates[pid]["flux_skipped"] = True
            else:
                active_pids.append(pid)

        if not active_pids:
            out = _empty_short_circuit(
                "(all candidates had empty repaint masks)")
            out["candidates"] = candidates
            return out

        # Single FLUX forward with batch = len(active_pids).
        pipe = self._pipe_()
        cond_resized = cond.resize((FLUX_SIZE, FLUX_SIZE), Image.LANCZOS)
        images = [cond_resized for _ in active_pids]
        masks_resized = [
            candidates[pid]["repaint_mask"].resize(
                (FLUX_SIZE, FLUX_SIZE), Image.NEAREST)
            for pid in active_pids
        ]
        prompts = [V08_FLUX_PROMPT.format(label=prompt_label)
                   for _ in active_pids]
        gen = torch.Generator(device=self.device).manual_seed(seed)
        res_list = pipe(
            prompt=prompts, image=images, mask_image=masks_resized,
            height=FLUX_SIZE, width=FLUX_SIZE,
            guidance_scale=GUIDANCE, num_inference_steps=STEPS,
            max_sequence_length=512, generator=gen,
        ).images

        # Composite each result back into part-sized RGBA, identical math
        # to the per-pid path in fill_from_mask_multi.
        for pid, res in zip(active_pids, res_list):
            rp_img = candidates[pid]["repaint_mask"]
            rp = np.array(rp_img) > 127
            frgb = res.convert("RGB").resize(part_rgba.size, Image.LANCZOS)
            arr = np.array(rgb_to_rgba_white_alpha(frgb))
            arr[vis, :3] = tgt[vis, :3]
            allowed = vis | rp
            arr[..., 3] = np.where(
                allowed, np.maximum(arr[..., 3], tgt[..., 3]),
                0).astype(np.uint8)
            fill_rgba = Image.fromarray(arr, "RGBA")
            candidates[pid]["fill_rgba"] = fill_rgba
            candidates[pid]["fill_on_white"] = rgba_on_white(fill_rgba)
            candidates[pid]["flux_skipped"] = False

        # No pick here -- runner will call pick_best on
        # {pid: candidates[pid]["fill_rgba"] for pid in active_pids} and
        # then patch best_tag / pick_* into the returned dict.
        return {
            "fill_rgba": None,            # patched after pick_best
            "fill_on_white": None,        # patched after pick_best
            "repaint_mask": None,         # patched after pick_best
            "polygon_mask": None,         # patched after pick_best
            "n_vertices": 0,              # patched after pick_best
            "parse_error": None,          # patched after pick_best
            "needs_inpaint": mk.get("needs_inpaint", True),
            "is_residual": mk.get("is_residual", False),
            "occluded": mk.get("occluded", mk.get("needs_inpaint", True)),
            "occluding_sibling_ids":
                list(mk.get("occluding_sibling_ids") or []),
            "occluding_siblings":
                list(mk.get("occluding_siblings") or []),
            "description": mk.get("description"),
            "flux_prompt_label": prompt_label,
            "judge_reason": mk.get("judge_reason"),
            "judge_parse_warning": mk.get("judge_parse_warning"),
            "judge_parse_error": mk.get("judge_parse_error"),
            "judge_dropped_names": list(mk.get("judge_dropped_names") or []),
            "n_siblings_in_prompt": mk.get("n_siblings_in_prompt", 0),
            "description_requested":
                mk.get("description_requested", False),
            "part_visible_px": mk.get("part_visible_px"),
            "scene_color_px": mk.get("scene_color_px"),
            "part_scene_ratio_pct": mk.get("part_scene_ratio_pct"),
            "judge_usage": mk.get("judge_usage"),
            "polygon_usage": None,        # winner's polygon_usage filled
            "best_tag": None,             # patched after pick_best
            "best_letter": None,
            "pick_mapping": None,
            "pick_reason": None,
            "pick_raw": "",
            "pick_parse_error": None,
            "pick_usage": None,
            "candidates": candidates,
            "active_pids": active_pids,
        }

    # -- Multi-prompt full pipeline (judge + N polygons + N FLUX + pick) --
    def inpaint_multi(self, part_rgba: Image.Image, scene_rgb: Image.Image,
                      label: str, seed: int = SEED,
                      *,
                      prompt_ids: tuple[str, ...] = ("p0", "p1", "p2", "p3"),
                      siblings: list[SiblingInfo] | None = None,
                      parent_label: str | None = None,
                      parent_coverage: float | None = None,
                      description_mode: bool = False,
                      gemini_model: str | None = None,
                      thinking_level: str | None = "medium",
                      judge_level: str | None = "medium",
                      pick_level: str | None = "medium",
                      skip_judge: bool = False) -> dict:
        """One-shot convenience wrapper: make_mask_multi + fill_from_mask_multi.

        See :meth:`make_mask_multi` and :meth:`fill_from_mask_multi` for
        the return shape and short-circuit behavior.
        """
        mk = self.make_mask_multi(
            part_rgba, scene_rgb, label,
            prompt_ids=prompt_ids,
            siblings=siblings,
            parent_label=parent_label,
            parent_coverage=parent_coverage,
            description_mode=description_mode,
            gemini_model=gemini_model, thinking_level=thinking_level,
            judge_level=judge_level, skip_judge=skip_judge,
        )
        return self.fill_from_mask_multi(
            part_rgba, scene_rgb, label, mk, seed=seed,
            gemini_model=gemini_model, pick_level=pick_level,
        )
