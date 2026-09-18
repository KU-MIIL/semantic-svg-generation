"""v15 — tree-only build (no inpaint).

Per-node loop (depth-first), each level executes:

  1. Gemini ``decompose_one_level`` on the original scene + focus_bbox.
  2. For each child component: SAM3 grounded with (label, bbox) +
     optional color-recovery when ``assess_sam_quality`` flags it bad.
  3. ``_v15_assign_sibling_residual_pixels`` — uncovered pixels are
     awarded to the margin-winner sibling (top1−top2 ≥ τ); the
     remaining low-margin leftover is returned for step 4.
  4. Single ``residual`` leaf emitted from the leftover + judged ONCE
     by ``judge_residual_only``. Named siblings (Gemini-labeled) are
     trusted — no per-sibling judge call. Restores v13 residual
     semantics (1 judge call per level instead of N).
  5. Recurse on accepted named children only; the residual leaf never
     recurses. Stop at ``depth >= max_depth`` or Gemini atomic.

Inpaint is **deliberately out of scope** in v15 — the design separates
tree structure (semantics) from leaf reconstruction (inpaint). The
post-tree inpaint step will run as a separate pass on accepted leaves
only, dramatically cutting inpaint call count vs v14 (which inpainted
on every accepted node during build).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .mask_ops import _apply_mask_rgba, _bbox_xywh, _rgba_on_white
from .part import Part
from .run import _DecomposeRun
from .sam import (
    _sam3_segment_with_prob, assess_sam_quality,
)
from .sam_recovery import recover_via_color_palette
from .tree_utils import _bbox_from_gemini, _safe_name


class _DecomposeRunV15(_DecomposeRun):
    """v15 tree-only build — inherits every v13 helper, overrides ``run``
    and the build loop with a per-node Gemini/SAM flow + single
    residual-leaf judge per level (v13-style; named siblings trusted).
    No inpaint, no FLUX, no composite scene."""

    def __init__(self, decomposer, image: Image.Image,
                 debug_dir: Any | None = None,
                 max_depth: int = 2):
        super().__init__(decomposer, image, debug_dir, max_depth=max_depth)
        self._sam_model = None
        self._sam_proc = None

    def run(self) -> dict[str, Any]:
        self._prepare_image()
        self._sam_model, self._sam_proc = self.cfg._ensure_loaded()
        if self._sam_model is None or self._sam_proc is None:
            raise RuntimeError(
                "v15 requires SAM3; cfg._ensure_loaded() returned None."
            )

        self._scene_rgb_np_root = np.asarray(
            self.img_for_vlm.convert("RGB"), dtype=np.uint8,
        )

        tree_parts = self._build_v15(
            parent_mask=None,
            parent_id=None,
            parent_label="",
            depth=0,
            parent_bbox_raw=None,
            ancestor_labels=[],
            sibling_labels=[],
        )
        if not tree_parts:
            tree_parts = [self._build_root_atomic_part()]

        # Post-loop — the residual leaf + judge already ran per-level
        # inside _build_v15 (v13-style), so no top-level residual pass
        # is needed here. Remaining helpers are dedup / collapse only.
        tree_parts = self._apply_cross_tree_mask_dedup(tree_parts)
        tree_parts = self._apply_single_child_collapse(tree_parts,
                                                       parent_label="root")
        self._apply_label_dedupe(tree_parts)
        self._cleanup_sam_prob(tree_parts)
        self._cleanup_pre_recover(tree_parts)
        self._renumber_background_to_p0(tree_parts)
        self._flatten(tree_parts)

        return {
            "tree_parts": tree_parts,
            "parts": self.flat_parts,
            "decompose_calls": self.decompose_calls,
            "img_rgba": self.img_rgba,
            "img_for_vlm": self.img_for_vlm,
            "icon_area": self.icon_area,
        }

    # ------------------------------------------------------------------
    # Per-level build (no inpaint)
    # ------------------------------------------------------------------

    def _build_v15(self,
                   parent_mask: np.ndarray | None,
                   parent_id: str | None,
                   parent_label: str,
                   depth: int,
                   parent_bbox_raw: list | tuple | None = None,
                   ancestor_labels: list[str] | None = None,
                   sibling_labels: list[str] | None = None) -> list[Part]:
        if depth >= self.max_depth:
            return []

        ancestor_labels = list(ancestor_labels or [])
        sibling_labels = list(sibling_labels or [])

        # v15 always uses the original scene + focus_bbox (no composite
        # since no inpaint runs inside the build).
        scene_on_white = self.img_for_vlm
        scene_rgb_np = self._scene_rgb_np_root

        # ── Step A: Gemini decompose one level
        tree_so_far_snapshot = list(self._tree_so_far)
        is_atomic, components = self._decompose_level(
            scene_on_white, parent_label, depth, parent_id,
            focus_bbox=parent_bbox_raw,
            ancestor_labels=ancestor_labels,
            sibling_labels=sibling_labels,
            tree_so_far=tree_so_far_snapshot,
        )
        if is_atomic or not components:
            return []

        parent_area = (int(parent_mask.sum()) if parent_mask is not None
                       else self.W * self.H)

        # ── Step B: per-child SAM grounded + optional color-recovery
        nodes: list[Part] = []
        for comp in components:
            label = str(comp.get("label", "")).strip()
            if not label:
                continue
            bbox_raw = comp.get("_bbox")
            gemini_bbox = _bbox_from_gemini(bbox_raw, self.W, self.H)
            if gemini_bbox is None:
                print(f"    skip '{label}' (depth {depth}) — invalid bbox")
                continue

            sam_out = self._sam_for_child(
                scene_on_white, scene_rgb_np, label, gemini_bbox, parent_mask,
            )
            mask = sam_out["mask"]
            if mask is None or not mask.any():
                print(f"    skip '{label}' (depth {depth}) — empty SAM mask")
                continue

            mask_area = int(mask.sum())
            cov_of_parent = mask_area / max(parent_area, 1)
            cov = float((mask & self.content_mask).sum()) / max(self.icon_area, 1)
            self.counter += 1
            pid = f"p{self.counter}"
            layer = _apply_mask_rgba(self.img_rgba, mask)
            part = Part(
                id=pid, label=label, bbox_xywh=_bbox_xywh(mask),
                score=sam_out["sam_score"], icon_coverage=cov, mask=mask,
                layer_image=layer, depth=depth, parent_id=parent_id,
            )
            part.extra["gemini_bbox"] = gemini_bbox
            part.extra["gemini_bbox_raw"] = list(bbox_raw) if bbox_raw else None
            part.extra["coverage_of_parent"] = cov_of_parent
            part.extra["mask_source"] = sam_out["mask_source"]
            if sam_out["prob"] is not None:
                part.extra["sam_prob"] = sam_out["prob"]
                part.extra["sam_score"] = sam_out["sam_score"]
            if sam_out["sam_quality"] is not None:
                q = sam_out["sam_quality"]
                part.extra["sam_quality"] = q["quality"]
                part.extra["sam_quality_sb_05"] = q["sb_05"]
                part.extra["sam_quality_sb_20"] = q["sb_20"]
                if q["reason"]:
                    part.extra["sam_quality_reason"] = q["reason"]
            if sam_out["recover_info"] is not None:
                part.extra["recover_info"] = sam_out["recover_info"]
            # Keep pre-recover silhouette around for debug dump after the
            # full part is built — caller dumps PNGs under debug/sam_recover.
            if sam_out["mask_pre_recover"] is not None:
                part.extra["sam_mask_pre_recover"] = sam_out["mask_pre_recover"]
            self._dump_sam_recover_debug(part)
            nodes.append(part)

        if not nodes:
            return []

        # ── Step C: sibling residual pixel assignment (per-level).
        # Returns the leftover (unclaimed) mask so we can emit a single
        # synthetic ``residual`` leaf — v13 semantics restored. Named
        # siblings are trusted: no per-sibling judge call.
        leftover = self._v15_assign_sibling_residual_pixels(
            nodes, parent_mask, depth,
        )

        # ── Step D: emit single ``residual`` leaf + judge it ONCE.
        # If the judge marks the leaf residual, keep it as a rejected
        # node (so debug viz still shows it) instead of dropping silently.
        if leftover is not None and leftover.any():
            residual_part = self._make_residual_leaf(leftover, depth, parent_id)
            part_rgba = _apply_mask_rgba(self.img_rgba, residual_part.mask)
            is_res, is_bg, judge_dbg = self._run_residual_judge(
                part_rgba, residual_part.label,
            )
            residual_part.extra["judge"] = judge_dbg
            residual_part.extra["judge_is_residual"] = bool(is_res)

            # Deterministic safety net: when the residual covers a large
            # fraction of the icon (≥ 15%), it is almost certainly a real
            # layer (typically the scene background plate) — override any
            # noisy is_residual=TRUE from the judge. Small residuals
            # (< 15%) still defer to the judge. The override only flips
            # the residual decision; the background judge then decides
            # z-order normally.
            cov_floor = 0.15
            if is_res and residual_part.icon_coverage >= cov_floor:
                print(f"    override residual judge — cov="
                      f"{residual_part.icon_coverage:.3f} ≥ {cov_floor} → keep")
                residual_part.extra["judge_is_residual_override"] = True
                residual_part.extra["judge_is_residual_original"] = True
                is_res = False
                residual_part.extra["judge_is_residual"] = False
                # Fresh background judgement on the kept layer.
                try:
                    from modules.inpainter.judge import judge_background_layer
                    is_bg, bg_dbg = judge_background_layer(
                        self.img_for_vlm, part_rgba, residual_part.label,
                        model=self.cfg.inpaint_model,
                        thinking_level="medium",
                    )
                    if isinstance(judge_dbg, dict):
                        judge_dbg["background"] = bg_dbg
                except Exception as ex:
                    print(f"    [judge:background] override path error "
                          f"{type(ex).__name__}: {ex} → normal z-order")
                    is_bg = False
            if is_res:
                residual_part.rejected = True
                residual_part.extra["drop_reason"] = (
                    f"judge_residual: {judge_dbg.get('reason', '')}"
                )
                print(f"    drop 'residual' (depth {depth}) — "
                      f"judge residual: {judge_dbg.get('reason', '')}")
                nodes.append(residual_part)
            else:
                # Kept residual: stage-2 judge decided whether it's the
                # scene background (paint first, z-order=0). The merge
                # consumes this flag via _collect_effective_leaves.
                residual_part.extra["is_background_layer"] = bool(is_bg)
                if is_bg:
                    bg_reason = (judge_dbg.get("background") or {}).get(
                        "reason", ""
                    )
                    print(f"    keep 'residual' (depth {depth}) as "
                          f"BACKGROUND (z-order=0): {bg_reason}")
                    # Background residual paints FIRST in the layer
                    # stack — also hoist it to the front of the sibling
                    # list so parts_tree.png + segments.json show it as
                    # the bottom-most node (top row of the tree viz).
                    nodes.insert(0, residual_part)
                else:
                    nodes.append(residual_part)

        # ── Step E: recurse on each named child. The residual leaf is a
        # synthetic catch-all by construction and never recurses.
        child_ancestors = (ancestor_labels + [parent_label]
                           if parent_label else list(ancestor_labels))
        all_sibling_labels = [p.label for p in nodes if p.label]
        for part in nodes:
            is_residual_leaf = (
                part.extra.get("mask_source") == "residual_margin_leftover"
            )
            if not is_residual_leaf:
                other_siblings = [lab for lab in all_sibling_labels
                                  if lab != part.label]
                part.children = self._build_v15(
                    parent_mask=part.mask,
                    parent_id=part.id,
                    parent_label=part.label,
                    depth=depth + 1,
                    parent_bbox_raw=part.extra.get("gemini_bbox_raw"),
                    ancestor_labels=child_ancestors,
                    sibling_labels=other_siblings,
                )
            self._register_completed_subtree(part)

        return nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sam_for_child(
        self, scene_pil: Image.Image, scene_rgb_np: np.ndarray,
        label: str, gemini_bbox: list[int],
        parent_mask: np.ndarray | None,
    ) -> dict:
        """SAM3 grounded segmentation for one child. Returns dict:

        ``mask``           — final (post-recover) bool (H, W). Always set.
        ``prob``           — SAM rank-1 prob map, or None on no-detection.
        ``mask_source``    — sam / sam_bad_recover_color / sam_no_detection.
        ``sam_score``      — SAM detection score (0.0 on no-detection).
        ``mask_pre_recover`` — original SAM mask BEFORE color recovery.
                              Equals ``mask`` when recovery didn't trigger
                              (so the caller can always dump pre vs post
                              for review). None on no-detection.
        ``sam_quality``    — dict from ``assess_sam_quality(prob)`` (None
                              on no-detection).
        ``recover_info``   — color-palette recovery dict (None when not
                              triggered).
        """
        clip_mask = (parent_mask if parent_mask is not None
                     else self.content_mask)

        try:
            cands = _sam3_segment_with_prob(
                self._sam_model, self._sam_proc, scene_pil, label,
                device=self.cfg.device, input_box=list(gemini_bbox),
            )
        except Exception as ex:
            print(f"    [sam] '{label}' error {type(ex).__name__}: {ex}")
            cands = []

        if not cands:
            x0, y0, x1, y1 = (int(v) for v in gemini_bbox)
            rect = np.zeros((self.H, self.W), dtype=bool)
            rect[max(0, y0):min(self.H, y1),
                 max(0, x0):min(self.W, x1)] = True
            mask = rect & clip_mask
            return {
                "mask": mask, "prob": None,
                "mask_source": "sam_no_detection",
                "sam_score": 0.0,
                "mask_pre_recover": None,
                "sam_quality": None,
                "recover_info": None,
            }

        top = cands[0]
        sam_mask = top["mask"] & clip_mask
        prob = top["prob"]
        score = float(top["score"])
        sam_quality = assess_sam_quality(prob)

        final_mask = sam_mask
        mask_source = "sam"
        recover_info = None
        if self.cfg.bad_recover_color and sam_quality["quality"] == "bad":
            rec_mask, recover_info = recover_via_color_palette(
                prob, scene_rgb_np, list(gemini_bbox),
                sam_tau=0.5,
                edge_thresh=self.cfg.bad_recover_edge_thresh,
                min_split_area=self.cfg.bad_recover_min_split_area,
            )
            final_mask = rec_mask & clip_mask
            mask_source = "sam_bad_recover_color"

        return {
            "mask": final_mask, "prob": prob,
            "mask_source": mask_source,
            "sam_score": score,
            "mask_pre_recover": sam_mask,
            "sam_quality": sam_quality,
            "recover_info": recover_info,
        }

    def _make_residual_leaf(self, leftover_mask: np.ndarray,
                            depth: int, parent_id: str | None) -> Part:
        """Wrap an unassigned-residual mask into a single
        ``label='residual'`` Part (v13-style ``residual_margin_leftover``).
        Score 0.0, no gemini_bbox; ``mask_source`` set so downstream
        recurse / debug viz can recognise it as the catch-all leaf."""
        self.counter += 1
        pid = f"p{self.counter}"
        cov = float(
            (leftover_mask & self.content_mask).sum()
        ) / max(self.icon_area, 1)
        tau = float(getattr(self.cfg, "residual_margin_threshold", 0.15))
        part = Part(
            id=pid, label="residual",
            bbox_xywh=_bbox_xywh(leftover_mask),
            score=0.0, icon_coverage=cov, mask=leftover_mask,
            layer_image=_apply_mask_rgba(self.img_rgba, leftover_mask),
            depth=depth, parent_id=parent_id,
        )
        part.extra["mask_source"] = "residual_margin_leftover"
        part.extra["residual_margin_threshold"] = tau
        return part

    def _v15_assign_sibling_residual_pixels(
        self, nodes: list[Part], parent_mask: np.ndarray | None, depth: int,
    ) -> np.ndarray | None:
        """Margin-based residual pixel distribution + leftover mask return.

        For pixels uncovered by any sibling, assign the argmax-winner
        sibling when ``top1 − top2 ≥ residual_margin_threshold``; pixels
        below the margin are returned as a leftover bool mask so the
        caller can wrap them into a single ``label='residual'`` leaf and
        judge it once (v13 semantics, restored in v15).

        Returns the leftover ``(H, W)`` bool mask, or ``None`` when there
        is no residual region to process at all (empty input, no
        accepted siblings, or full parent coverage).
        """
        if not nodes:
            return None
        region = (parent_mask & self.content_mask
                  if parent_mask is not None else self.content_mask)
        accepted = [n for n in nodes
                    if n.mask is not None and n.mask.any()]
        if not accepted or not region.any():
            return None

        union = np.zeros_like(region, dtype=bool)
        for n in accepted:
            union |= n.mask
        residual = region & ~union
        n_res = int(residual.sum())
        if n_res == 0:
            return None

        cand_nodes: list[Part] = []
        cand_probs: list[np.ndarray] = []
        for n in accepted:
            prob = n.extra.get("sam_prob")
            if prob is None:
                continue
            cand_nodes.append(n)
            cand_probs.append(prob.astype(np.float32))

        if not cand_probs:
            print(f"    [residual_assign] d={depth} "
                  f"residual={n_res}px → no sam_prob siblings, "
                  f"all leftover")
            return residual

        tau = float(getattr(self.cfg, "residual_margin_threshold", 0.15))
        stack = np.stack(cand_probs, axis=0)
        max_prob = stack.max(axis=0)
        winner = stack.argmax(axis=0)
        if stack.shape[0] >= 2:
            second = np.partition(stack, -2, axis=0)[-2]
            margin_arr = max_prob - second
        else:
            margin_arr = max_prob.copy()
        assign_mask = residual & (margin_arr >= tau)
        n_assign = int(assign_mask.sum())

        if n_assign == 0:
            print(f"    [residual_assign] d={depth} "
                  f"residual={n_res}px → margin τ={tau:.3f} assigned 0, "
                  f"all leftover")
            return residual

        for i, n in enumerate(cand_nodes):
            claim = assign_mask & (winner == i)
            if not claim.any():
                continue
            new_mask = (n.mask | claim) if n.mask is not None else claim
            n.mask = new_mask
            n.bbox_xywh = _bbox_xywh(new_mask)
            n.icon_coverage = float(
                (new_mask & self.content_mask).sum()
            ) / max(self.icon_area, 1)
            n.layer_image = _apply_mask_rgba(self.img_rgba, new_mask)

        leftover = residual & ~assign_mask
        print(f"    [residual_assign] d={depth}  τ={tau:.3f}  "
              f"residual={n_res}px → assigned={n_assign} "
              f"({n_assign / max(n_res, 1):.1%}), "
              f"leftover={int(leftover.sum())}px (residual leaf)")
        return leftover

    def _renumber_background_to_p0(self, tree_parts: list[Part]) -> None:
        """Promote the root-level background residual to ID ``p0`` so
        its layer-stack position (z-order=0, bottom of the stack) is
        also reflected in the ID. The original emission-order ID is
        stashed in ``extra["original_id"]`` for traceability.

        Only the first ROOT-level background residual is renamed.
        ``p0`` is otherwise unused by the counter (which starts at
        ``p1``), so this never collides. Children of the renamed node
        don't exist (the residual is always a leaf), so no
        ``parent_id`` rewrite is needed."""
        for p in tree_parts:
            if (p.extra or {}).get("is_background_layer") and not p.rejected:
                if p.id == "p0":
                    return  # already renumbered
                p.extra["original_id"] = p.id
                p.id = "p0"
                return  # at most one root-level background

    def _cleanup_pre_recover(self, tree_parts: list[Part]) -> None:
        """Drop the cached ``sam_mask_pre_recover`` for nodes that did NOT
        trigger recovery (where pre == post). For recovered nodes we keep
        the bool mask so ``pipeline_controller`` can render the pre-vs-post
        side-by-side in ``parts_tree.png``. The mask is never serialized
        to segments.json — it lives in ``part.extra`` only for the
        downstream viz step."""
        for p in tree_parts:
            src = (p.extra or {}).get("mask_source", "") or ""
            if "bad_recover" not in src:
                p.extra.pop("sam_mask_pre_recover", None)
            if p.children:
                self._cleanup_pre_recover(p.children)

    def _dump_sam_recover_debug(self, part: Part) -> None:
        """Dump pre vs post SAM masks under ``debug/sam_recover/`` for a
        single part. Always writes both PNGs (pre = post when recovery
        didn't trigger, so the side-by-side viewer always has a baseline).
        No-op when ``--no-debug`` is set or pre-recover mask is missing.
        """
        if self.debug_dir is None:
            return
        pre = part.extra.get("sam_mask_pre_recover")
        if pre is None:
            return
        dump_dir = self.debug_dir / "sam_recover"
        dump_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{part.id}_{_safe_name(part.label)}"
        try:
            _apply_mask_rgba(self.img_rgba, pre).save(
                dump_dir / f"{stem}__pre.png"
            )
            _apply_mask_rgba(self.img_rgba, part.mask).save(
                dump_dir / f"{stem}__post.png"
            )
        except Exception as ex:
            print(f"    [sam_recover-dbg] dump failed for {stem}: "
                  f"{type(ex).__name__}: {ex}")

    def _run_residual_judge(
        self, part_rgba: Image.Image, label: str,
    ) -> tuple[bool, bool, dict]:
        """Two-stage gate for the synthetic residual leaf.

        Step 1 — ``judge_residual_only`` decides whether the leaf is
        garbage slivers (drop) or a coherent named sub-shape (keep).

        Step 2 — only when step 1 returns ``is_residual=False``,
        ``judge_background_layer`` decides whether the kept leaf is the
        scene background (z-order=0) or a normal mid-layer part. The
        caller stashes the result in ``part.extra["is_background_layer"]``
        so the downstream merge can hoist background layers to the
        bottom of the painter's stack.

        Both calls run at thinking_level=medium (overrides the global
        ``cfg.inpaint_thinking_level``) — the residual / background
        decision is high-leverage and worth the extra cost.

        Returns ``(is_residual, is_background, debug)``. Errors in
        either stage fail-closed: residual → False (keep),
        background → False (normal z-order).
        """
        from modules.inpainter.judge import (
            judge_residual_only,
            judge_background_layer,
        )
        debug: dict = {}
        thinking = "medium"
        try:
            is_residual, dbg_r = judge_residual_only(
                self.img_for_vlm, part_rgba, label,
                model=self.cfg.inpaint_model,
                thinking_level=thinking,
            )
            debug["residual"] = dbg_r
        except Exception as ex:
            print(f"    [judge:residual] '{label}' error "
                  f"{type(ex).__name__}: {ex} → keep")
            debug["residual_error"] = f"{type(ex).__name__}: {ex}"
            is_residual = False

        # Surface the original residual judge fields at the top level so
        # segments.json + parts_tree viz still find ``reason`` / ``usage``
        # where they used to live (single-judge era).
        rdbg = debug.get("residual") or {}
        for k in ("reason", "usage", "part_visible_px", "scene_color_px",
                  "part_scene_ratio_pct", "raw", "parse_error",
                  "parse_warning"):
            if k in rdbg:
                debug[k] = rdbg[k]

        if is_residual:
            return True, False, debug

        try:
            is_background, dbg_b = judge_background_layer(
                self.img_for_vlm, part_rgba, label,
                model=self.cfg.inpaint_model,
                thinking_level=thinking,
            )
            debug["background"] = dbg_b
        except Exception as ex:
            print(f"    [judge:background] '{label}' error "
                  f"{type(ex).__name__}: {ex} → normal z-order")
            debug["background_error"] = f"{type(ex).__name__}: {ex}"
            is_background = False

        return False, is_background, debug
