"""Main decomposition path.

Originally a bbox-only sanity baseline; evolved over v5-v13 into the
project's single decompose path. Stages, in order ``run()`` executes them:

  1. **Tree build (recursive)** — Gemini ``decompose_one_level`` per
     node; bbox-rect clipped to parent mask is the initial silhouette.
     Catch-all entries (label in ``CATCHALL_LABELS`` and ``coverage_of_
     parent ≥ 0.95``) become ``residual_replaced`` leaves carrying the
     parent − union(named) residual.
  2. **Post-hoc SAM (optional)** — replaces bbox-rects with SAM3 masks
     when ``--bbox-sam`` (basic) or ``--bbox-sam-stability`` (with
     ``assess_sam_quality`` + ``--bad-recover-*``) is set.
  3. **Tree cleanup** — bbox-dup prune → residual subtree drop →
     cross-tree mask dedup → single-child collapse → label dedupe.
  4. **Residual margin assignment** — uncovered pixels with a
     ``top1 − top2`` SAM-prob margin ≥ ``--residual-margin-threshold``
     are awarded to their argmax-winner leaf; the rest collect into
     a single ``residual`` leaf appended at root level. The previous
     argmax-only fill was hijacked by catch-all leaves; margin filters
     those out.

Returns the flat ``parts`` list + ``tree_parts`` for the caller.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .mask_ops import _apply_mask_rgba, _bbox_xywh, _rgba_on_white
from .part import Part
from .sam import _sam3_segment
from .tree_utils import _bbox_from_gemini


class _DecomposeRun:
    """Per-call orchestrator: Gemini tree build → optional post-hoc SAM →
    tree cleanup → optional residual-argmax fill."""

    def __init__(self, decomposer, image: Image.Image,
                 debug_dir: Any | None = None,
                 max_depth: int = 2):
        self.cfg = decomposer
        self.gemini = decomposer._gemini
        self.max_depth = max_depth

        if debug_dir is not None:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = debug_dir

        self.image = image

        self.img_rgba: Image.Image | None = None
        self.img_for_vlm: Image.Image | None = None
        self.W: int = 0
        self.H: int = 0
        self.orig_alpha: np.ndarray | None = None
        self.icon_area: int = 0

        self.counter: int = 0
        self.flat_parts: list[Part] = []
        self.decompose_calls: list[dict] = []
        # Cross-tree awareness: accumulating record of {label, bbox} for
        # every node whose subtree is already fully decomposed. Each
        # recursive ``_build`` snapshots this before its Gemini call so
        # the model sees what OTHER subtrees have already claimed, and
        # registers each node + its descendants once that node's
        # subtree completes (see ``_register_completed_subtree``).
        # Bbox stored here is the raw 0-1000 Gemini bbox, matching the
        # coord frame used by ``focus_bbox`` in the prompt.
        self._tree_so_far: list[dict] = []

    def run(self) -> dict[str, Any]:
        self._prepare_image()

        tree_parts = self._build(
            layer_source_rgba=self.img_rgba,
            parent_mask=None,
            parent_id=None,
            parent_label="",
            depth=0,
        )
        # Gemini-says-atomic at root → no children. Without a fallback the
        # pipeline would emit zero parts and skip merging entirely; treat
        # the whole image as one root-level Part so vtracer at least
        # vectorizes the input as a single layer.
        if not tree_parts:
            tree_parts = [self._build_root_atomic_part()]
        if getattr(self.cfg, "post_hoc_sam", False):
            self._apply_post_hoc_sam(tree_parts)
        if getattr(self.cfg, "post_hoc_segmenter", None) == "sam_stability":
            self._apply_post_hoc_sam_stability(tree_parts)
        # v9: post-hoc clean-up — strip fake decompositions where the
        # child's mask is geometrically the parent (model went deeper
        # without adding info) AND enforce cross-tree exact-label
        # uniqueness (legs/legs → desk-legs/chair-legs).
        tree_parts = self._apply_bbox_dup_prune(tree_parts, parent_mask=None,
                                                parent_label="root")
        # v10: residual_replaced children are catch-alls; recursing into
        # them is meaningless (they have no semantic content). Drop their
        # subtrees so they remain leaves.
        self._drop_residual_subtrees(tree_parts)
        # v10: cross-tree duplicate detection. The model can emit the
        # same physical region as two different parts at different tree
        # positions — 588 has 8 bottles as direct children of root AND
        # the same 8 bottles under ``shelf``. Sibling-mutex doesn't
        # catch these because they're at different levels.
        tree_parts = self._apply_cross_tree_mask_dedup(tree_parts)
        tree_parts = self._apply_single_child_collapse(tree_parts,
                                                       parent_label="root")
        self._apply_label_dedupe(tree_parts)
        self._apply_residual_margin_assign(
            tree_parts,
            parent_mask=None,
            level_depth=0,
            level_parent_id=None,
            level_parent_label="root",
        )
        self._cleanup_sam_prob(tree_parts)
        self._flatten(tree_parts)

        return {
            "tree_parts": tree_parts,
            "parts": self.flat_parts,
            "decompose_calls": self.decompose_calls,
            "img_rgba": self.img_rgba,
            "img_for_vlm": self.img_for_vlm,
            "icon_area": self.icon_area,
        }

    def _prepare_image(self) -> None:
        self.img_rgba = self.image.convert("RGBA")
        self.img_for_vlm = _rgba_on_white(self.img_rgba)
        self.W, self.H = self.img_for_vlm.size
        self.orig_alpha = np.array(self.img_rgba)[..., 3]
        # ``content_mask`` is the real icon shape (non-white, visible).
        # Many input PNGs are RGB-on-white with no alpha, so ``orig_alpha
        # > 0`` would simply be the whole canvas and inflate every coverage
        # / residual fraction with empty whitespace. The flatten-then-
        # threshold approach handles both RGB and RGBA inputs uniformly.
        arr_rgb = np.array(self.img_for_vlm)
        self.content_mask = ~(arr_rgb >= 240).all(axis=2)
        if self.orig_alpha.ndim == 2:
            self.content_mask &= (self.orig_alpha > 0)
        self.icon_area = int(self.content_mask.sum()) or (self.W * self.H)

    def _decompose_level(self, source_image: Image.Image, parent_label: str,
                         depth: int, parent_id: str | None,
                         focus_bbox: list | tuple | None = None,
                         ancestor_labels: list[str] | None = None,
                         sibling_labels: list[str] | None = None,
                         tree_so_far: list[dict] | None = None,
                         ) -> tuple[bool, list[dict]]:
        is_atomic, components, parse_error, raw, usage = (
            self.gemini.decompose_one_level(
                source_image, parent_label, focus_bbox=focus_bbox,
                depth=depth, ancestor_labels=ancestor_labels,
                max_depth=getattr(self, "max_depth", None),
                sibling_labels=sibling_labels,
                tree_so_far=tree_so_far,
            )
        )
        # Post-process model output:
        #   (a) drop any catch-all-labelled entries (H2 — no residual)
        #   (b) strip parent-label tokens from child labels (H1 — child
        #       must not contain parent label or its tokens)
        from modules.gemini_client import (
            _is_catchall_label, _strip_parent_tokens,
        )
        n_raw = len(components)
        cleaned: list[dict] = []
        for c in components:
            lbl = str(c.get("label", "")).strip()
            if not lbl or _is_catchall_label(lbl):
                continue
            # Strip parent's tokens from the child label
            new_lbl = _strip_parent_tokens(lbl, parent_label or "")
            if not new_lbl:
                # Child became empty after strip — drop it
                continue
            cleaned.append({"label": new_lbl, "_bbox": c.get("_bbox")})
        n_dropped = n_raw - len(cleaned)
        # If cleaning left nothing, force atomic
        if not cleaned:
            is_atomic = True
        components = cleaned

        self.decompose_calls.append({
            "depth": depth,
            "parent_id": parent_id,
            "parent_label": parent_label,
            "focus_bbox": list(focus_bbox) if focus_bbox is not None else None,
            "ancestor_labels": list(ancestor_labels or []),
            "tree_so_far": list(tree_so_far or []),
            "is_atomic": bool(is_atomic),
            "n_components": len(components),
            "components": components,
            "parse_error": parse_error,
            "raw_response": raw,
            "usage_metadata": usage,
        })
        ctx = parent_label or "root"
        anc = (" anc=" + "→".join(ancestor_labels)) if ancestor_labels else ""
        drop_note = f" [post-clean dropped {n_dropped}]" if n_dropped else ""
        print(f"    decompose @ depth {depth} '{ctx}'{anc} "
              f"→ atomic={is_atomic}, {len(components)} components"
              f"{drop_note}")
        return is_atomic, components

    def _bbox_to_mask(self, bbox_xyxy: list[int],
                      parent_mask: np.ndarray | None) -> np.ndarray:
        x0, y0, x1, y1 = bbox_xyxy
        mask = np.zeros((self.H, self.W), dtype=bool)
        mask[y0:y1, x0:x1] = True
        if parent_mask is not None:
            mask = mask & parent_mask
        return mask

    # If a sibling's mask covers ≥ this fraction of the parent area, it is
    # the catch-all sibling Gemini emits per Rule 2 (FULL COVERAGE). The
    # rect mask itself overlaps named siblings, so we always replace it
    # with the actual residual = parent − union(named). When the residual
    # is essentially zero (≤ ``CATCHALL_DROP_RESIDUAL_FRAC``) we drop the
    # catch-all entirely (e.g. root-level "background" on a transparent
    # icon — fully redundant); when it is meaningful we keep it as a leaf
    # holding only the residual pixels (e.g. "wall" inside a building
    # whose only named child is "doorway").
    CATCHALL_COVERAGE_THRESHOLD: float = 0.95
    CATCHALL_DROP_RESIDUAL_FRAC: float = 0.01
    # Catch-all detection is label-aware: only entries whose label is
    # one of these words (case-insensitive) AND whose bbox covers
    # ≥CATCHALL_COVERAGE_THRESHOLD of the parent get the catch-all
    # treatment. This prevents the rabbit-case bug where a model-emitted
    # ``head`` with a near-full-canvas bbox got wrongly replaced by the
    # residual and never recursed into its face features.
    CATCHALL_LABELS: frozenset = frozenset({
        "none", "background", "remainder", "residual", "outline",
        "rest", "leftover", "other", "misc",
    })

    def _build(self, layer_source_rgba: Image.Image,
               parent_mask: np.ndarray | None,
               parent_id: str | None,
               parent_label: str,
               depth: int,
               parent_bbox_raw: list | tuple | None = None,
               ancestor_labels: list[str] | None = None,
               sibling_labels: list[str] | None = None) -> list[Part]:
        if depth >= self.max_depth:
            return []

        # ``ancestor_labels`` is the chain root → … → parent EXCLUDING
        # the parent (which is in ``parent_label``). At root both are
        # empty / None. ``sibling_labels`` are this node's siblings
        # (already emitted at the same level by the caller).
        ancestor_labels = list(ancestor_labels or [])
        sibling_labels = list(sibling_labels or [])

        # Always send the FULL original image to Gemini and pass the
        # parent's bbox in text via ``focus_bbox`` — avoids the cropped-
        # white-frame artifact that caused recursion to over-split.
        # Snapshot the cross-tree state BEFORE this call so the prompt
        # sees only nodes already fully resolved by earlier siblings'
        # subtrees — not partial in-flight state.
        tree_so_far_snapshot = list(self._tree_so_far)
        is_atomic, components = self._decompose_level(
            self.img_for_vlm, parent_label, depth, parent_id,
            focus_bbox=parent_bbox_raw,
            ancestor_labels=ancestor_labels,
            sibling_labels=sibling_labels,
            tree_so_far=tree_so_far_snapshot,
        )
        if is_atomic or not components:
            return []

        # Reference area for coverage-based catch-all detection. At root
        # there's no parent_mask yet, so we use the full canvas; deeper
        # levels use the parent's silhouette area.
        parent_area = (int(parent_mask.sum()) if parent_mask is not None
                       else self.W * self.H)

        # Pass 1: build every component as a Part with its bbox-rect mask.
        # We don't decide catch-all behavior here — that's handled in pass 2
        # so we have all sibling masks available for residual computation.
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
            mask = self._bbox_to_mask(gemini_bbox, parent_mask)
            mask_area = int(mask.sum())
            if mask_area == 0:
                print(f"    skip '{label}' (depth {depth}) — empty mask after parent clip")
                continue
            cov_of_parent = mask_area / max(parent_area, 1)
            cov = float((mask & self.content_mask).sum()) / self.icon_area
            self.counter += 1
            pid = f"p{self.counter}"
            layer = _apply_mask_rgba(layer_source_rgba, mask)
            part = Part(
                id=pid, label=label, bbox_xywh=_bbox_xywh(mask),
                score=1.0, icon_coverage=cov, mask=mask,
                layer_image=layer, depth=depth, parent_id=parent_id,
            )
            part.extra["gemini_bbox"] = gemini_bbox
            part.extra["gemini_bbox_raw"] = list(bbox_raw) if bbox_raw else None
            part.extra["coverage_of_parent"] = cov_of_parent
            nodes.append(part)

        # Pass 2: catch-all handling. Replace each catch-all candidate's
        # rect with the residual = parent − union(named siblings); drop it
        # entirely when that residual is negligible.
        # Catch-all detection is LABEL-AWARE: high coverage alone is not
        # enough — the label must also be in CATCHALL_LABELS (none /
        # background / remainder / …). A semantic label like "head" or
        # "body" with a canvas-spanning bbox stays a named sibling and
        # gets recursed into (the model is being lazy with the bbox, not
        # marking it as catch-all).
        def _is_catchall(p) -> bool:
            cov_hi = (p.extra["coverage_of_parent"]
                      >= self.CATCHALL_COVERAGE_THRESHOLD)
            label_match = (p.label or "").strip().lower() in self.CATCHALL_LABELS
            return cov_hi and label_match
        named = [p for p in nodes if not _is_catchall(p)]
        catchall = [p for p in nodes if _is_catchall(p)]
        if named and catchall:
            union = np.zeros_like(named[0].mask, dtype=bool)
            for p in named:
                union |= p.mask
            base = parent_mask if parent_mask is not None else self.content_mask
            # Normalize the residual fraction by the FULL canvas (W×H) so
            # the threshold has the same meaning at every depth — using the
            # parent rect as denominator was an IoU-style ratio that
            # inflated small residuals at deeper levels.
            canvas_area = self.W * self.H
            kept: list[Part] = []
            for cand in catchall:
                # Filter residual to *real content* (non-white visible) so
                # the redundancy check measures meaningful pixels, not
                # whitespace inside the parent rect.
                residual = base & self.content_mask & ~union
                res_area = int(residual.sum())
                res_frac = res_area / canvas_area
                if res_frac < self.CATCHALL_DROP_RESIDUAL_FRAC:
                    print(f"    drop '{cand.label}' (depth {depth}) — catch-all "
                          f"redundant (residual {res_frac:.1%} of canvas)")
                    continue
                cand.mask = residual
                cand.bbox_xywh = _bbox_xywh(residual)
                cand.layer_image = _apply_mask_rgba(layer_source_rgba, residual)
                cand.icon_coverage = float(
                    (residual & self.content_mask).sum()
                ) / self.icon_area
                cand.extra["residual_replaced"] = True
                cand.extra["residual_frac"] = res_frac
                print(f"    keep '{cand.label}' (depth {depth}) — replaced "
                      f"rect with residual ({res_frac:.1%} of canvas)")
                kept.append(cand)
            nodes = named + kept

        # Propagate the tree path so descendants know not to re-emit any
        # ancestor's label. ``parent_label`` is empty at root, so we only
        # append it when truthy.
        child_ancestors = (ancestor_labels + [parent_label]
                           if parent_label else list(ancestor_labels))
        # All emitted-at-this-level labels become "siblings" for each
        # recursive child call — forbidding them as descendants stops
        # the "body → ears + face" duplication where body wrongly
        # re-emits its own siblings as its children.
        all_sibling_labels = [p.label for p in nodes if p.label]
        for part in nodes:
            if part.extra.get("residual_replaced"):
                # Residual is by definition "what's left over" — recursing
                # would just re-trigger the same catch-all problem.
                # Still register so later cross-tree probes see it.
                self._register_completed_subtree(part)
                continue
            other_siblings = [lab for lab in all_sibling_labels
                              if lab != part.label]
            part.children = self._build(
                layer_source_rgba=part.layer_image,
                parent_mask=part.mask,
                parent_id=part.id,
                parent_label=part.label,
                depth=depth + 1,
                parent_bbox_raw=part.extra.get("gemini_bbox_raw"),
                ancestor_labels=child_ancestors,
                sibling_labels=other_siblings,
            )
            # Now that this sibling's entire subtree is resolved, expose
            # it (and every descendant) to subsequent siblings' Gemini
            # calls via the cross-tree snapshot.
            self._register_completed_subtree(part)

        return nodes

    def _register_completed_subtree(self, root: Part) -> None:
        """Append a node + all of its descendants to ``self._tree_so_far``
        so the next Gemini decompose call (a different subtree) sees
        them as already-claimed labels and bboxes. Bbox uses the raw
        0-1000 Gemini coord frame (the same frame ``focus_bbox`` uses
        in the prompt). Nodes without a bbox raw value are skipped."""
        bbox = root.extra.get("gemini_bbox_raw")
        if root.label and bbox and len(bbox) == 4:
            self._tree_so_far.append(
                {"label": root.label, "bbox": list(bbox)}
            )
        for child in root.children or []:
            self._register_completed_subtree(child)




    def _build_root_atomic_part(self) -> Part:
        """Create a single full-image Part when Gemini declares the input
        atomic at depth 0. The silhouette is the content_mask (real icon
        pixels), the bbox is the full canvas, and post-hoc SAM uses an
        empty label (text-only fallback) since Gemini gave no name."""
        self.counter += 1
        pid = f"p{self.counter}"
        mask = self.content_mask if self.content_mask.any() else np.ones(
            (self.H, self.W), dtype=bool
        )
        layer = _apply_mask_rgba(self.img_rgba, mask)
        cov = float(mask.sum()) / max(self.icon_area, 1)
        part = Part(
            id=pid, label="root", bbox_xywh=_bbox_xywh(mask),
            score=1.0, icon_coverage=cov, mask=mask,
            layer_image=layer, depth=0, parent_id=None,
        )
        # Full-canvas bbox in canvas px so post-hoc SAM still gets a usable
        # input_box (the entire image) — SAM can then tighten the silhouette.
        part.extra["gemini_bbox"] = [0, 0, self.W, self.H]
        part.extra["gemini_bbox_raw"] = [0, 0, 1000, 1000]
        part.extra["root_atomic"] = True
        print(f"    root atomic — emitting full-canvas part {pid}")
        return part

    def _flatten(self, parts: list[Part]) -> None:
        for p in parts:
            self.flat_parts.append(p)
            self._flatten(p.children)

    def _apply_post_hoc_sam(self, tree_parts: list[Part]) -> None:
        """v5: After the Gemini-only tree is fully built, run SAM on every
        non-root node using its (label, gemini_bbox) on the *full original
        image*. Replace the bbox-rect mask with the SAM top-1 result.
        """
        sam_model, sam_proc = self.cfg._ensure_loaded()
        if sam_model is None or sam_proc is None:
            print("[post_hoc_sam] SAM not loaded; skipping")
            return

        n_total = 0
        n_sam = 0
        n_no_det = 0

        def _apply_sam_choice(node: Part, score: float,
                              sam_mask: np.ndarray, mask_source: str) -> None:
            node.mask = sam_mask
            node.bbox_xywh = _bbox_xywh(sam_mask)
            node.score = float(score)
            node.icon_coverage = float(
                (sam_mask & self.content_mask).sum()
            ) / self.icon_area
            node.extra["sam_score"] = float(score)
            node.extra["mask_source"] = mask_source

        def visit(node: Part) -> None:
            nonlocal n_total, n_sam, n_no_det
            # Root-atomic synthetic part has no real Gemini label — SAM with
            # text="root" would return junk. Keep the content_mask silhouette.
            if node.extra.get("root_atomic"):
                node.extra["mask_source"] = "root_atomic"
                for ch in node.children:
                    visit(ch)
                return
            n_total += 1
            bbox_canvas = node.extra.get("gemini_bbox")
            try:
                candidates = _sam3_segment(
                    sam_model, sam_proc,
                    self.img_for_vlm, node.label,
                    device=self.cfg.device,
                    score_threshold=0.3,
                    input_box=list(bbox_canvas) if bbox_canvas else None,
                )
            except Exception as ex:
                print(f"    [post_hoc_sam] {node.id} '{node.label}' "
                      f"SAM error {type(ex).__name__}: {ex}")
                candidates = []

            if not candidates:
                node.extra["mask_source"] = "sam_no_detection"
                n_no_det += 1
            else:
                score, _, sam_mask = max(candidates, key=lambda c: c[0])
                _apply_sam_choice(node, score, sam_mask, "sam")
                n_sam += 1

            node.layer_image = _apply_mask_rgba(self.img_rgba, node.mask)

            for ch in node.children:
                visit(ch)

        for top in tree_parts:
            visit(top)

        print(f"[post_hoc_sam] {n_sam}/{n_total} nodes SAM-replaced "
              f"({n_no_det} no-detection)")



    def _apply_post_hoc_sam_stability(self, tree_parts: list[Part]) -> None:
        """Post-hoc SAM with bad-quality classification.

        For each non-root leaf with a gemini_bbox:

          1. SAM3 grounded with (Gemini label, bbox) → rank-1 candidate
             (score, mask, prob).
          2. ``assess_sam_quality(prob)`` (sb_05 / sb_20 thresholds, fit on
             sweep_v11) tags the leaf as ``good`` or ``bad``.
          3. If ``bad`` and ``bad_recover_color`` is on, the mask is
             rewritten via the color-palette / edge-split recovery.

        Failure modes (rare):
          - SAM returns no candidates at all → ``bbox-rect ∩ content_mask``
            fallback (tag ``sam_no_detection``).
          - Leaf has no gemini_bbox (e.g. Phase-1 residual catch-all) →
            keep the existing rect-mask (tag ``sam_no_bbox``).
        """
        from .sam import _sam3_segment_with_prob, assess_sam_quality
        from .sam_recovery import recover_via_color_palette

        sam_model, sam_proc = self.cfg._ensure_loaded()
        if sam_model is None or sam_proc is None:
            print("[post_hoc_sam_stability] SAM not loaded; skipping")
            return

        do_recover_color = bool(getattr(self.cfg, "bad_recover_color", False))
        recover_edge_thr = float(
            getattr(self.cfg, "bad_recover_edge_thresh", 0.30)
        )
        recover_min_area = int(
            getattr(self.cfg, "bad_recover_min_split_area", 30)
        )
        full_rgb_np: np.ndarray | None = None
        if do_recover_color:
            full_rgb_np = np.asarray(
                self.img_for_vlm.convert("RGB"), dtype=np.uint8,
            )

        counters = {
            "total": 0, "good": 0, "bad": 0,
            "no_det": 0, "no_bbox": 0,
            "bad_recover_color": 0,
        }

        def _maybe_recover_bad(node: Part, cand: dict) -> None:
            """If the leaf is tagged ``sam_quality == "bad"`` and
            ``bad_recover_color`` is on, swap ``node.mask`` for the
            color-palette recovered version."""
            if not do_recover_color:
                return
            if node.extra.get("sam_quality") != "bad":
                return
            if full_rgb_np is None:
                return
            bbox = node.extra.get("gemini_bbox")
            if bbox is None or len(bbox) != 4:
                return
            prob = cand["prob"]
            color_mask, info_color = recover_via_color_palette(
                prob, full_rgb_np, list(bbox),
                sam_tau=0.5,
                edge_thresh=recover_edge_thr,
                min_split_area=recover_min_area,
            )
            counters["bad_recover_color"] += 1
            final = color_mask & self.content_mask
            old_source = node.extra.get("mask_source", "sam")
            if node.mask is not None:
                node.extra["sam_mask_pre_recover"] = node.mask.copy()
            _apply_choice(
                node, final, f"{old_source}_bad_recover_color",
                prob=prob,
            )
            node.extra["bad_recover"] = {"color": info_color}

        def _apply_choice(node: Part, mask: np.ndarray, source: str,
                          prob: np.ndarray | None = None) -> None:
            node.mask = mask
            node.bbox_xywh = _bbox_xywh(mask)
            node.icon_coverage = float(
                (mask & self.content_mask).sum()
            ) / self.icon_area
            node.extra["mask_source"] = source
            # Stash the SAM prob map for the residual margin-assign
            # pass that runs after _apply_post_hoc_sam_stability.
            # Peak/color recovery changes the mask but not the prob,
            # so we keep the prob attached on every mask update for
            # consistency. Cleared by _cleanup_sam_prob at end of run().
            if prob is not None:
                node.extra["sam_prob"] = prob

        def _sam_top(label: str, bbox_canvas) -> dict | None:
            try:
                cands = _sam3_segment_with_prob(
                    sam_model, sam_proc,
                    self.img_for_vlm, label,
                    device=self.cfg.device,
                    input_box=list(bbox_canvas),
                )
            except Exception as ex:
                print(f"    [sam] error '{label}' "
                      f"{type(ex).__name__}: {ex}")
                return None
            return cands[0] if cands else None

        def visit(node: Part, parent: Part | None) -> None:
            if node.extra.get("root_atomic"):
                node.extra["mask_source"] = "root_atomic"
                for ch in node.children:
                    visit(ch, node)
                return
            counters["total"] += 1
            bbox_canvas = node.extra.get("gemini_bbox")
            if bbox_canvas is None:
                node.extra["mask_source"] = "sam_no_bbox"
                counters["no_bbox"] += 1
                for ch in node.children:
                    visit(ch, node)
                return

            top = _sam_top(node.label, bbox_canvas)
            if top is None:
                fallback = node.mask & self.content_mask
                _apply_choice(node, fallback, "sam_no_detection")
                counters["no_det"] += 1
            else:
                mask = top["mask"] & self.content_mask
                _apply_choice(node, mask, "sam", prob=top["prob"])
                node.extra["sam_score"] = float(top["score"])
                q = assess_sam_quality(top["prob"])
                node.extra["sam_quality"] = q["quality"]
                node.extra["sam_quality_sb_05"] = q["sb_05"]
                node.extra["sam_quality_sb_20"] = q["sb_20"]
                if q["reason"]:
                    node.extra["sam_quality_reason"] = q["reason"]
                counters[q["quality"]] += 1
                _maybe_recover_bad(node, top)

            node.layer_image = _apply_mask_rgba(self.img_rgba, node.mask)
            for ch in node.children:
                visit(ch, node)

        for top_part in tree_parts:
            visit(top_part, None)

        rec_line = ""
        if do_recover_color:
            rec_line = (
                f"  bad_recover[color={counters['bad_recover_color']}]"
            )
        print(f"[post_hoc_sam_stability] {counters['total']} nodes  "
              f"good={counters['good']}  bad={counters['bad']}  "
              f"no_detection={counters['no_det']}  "
              f"no_bbox={counters['no_bbox']}{rec_line}")

    # ------------------------------------------------------------------
    # v9 post-hoc: bbox-dup prune + cross-tree label dedupe
    # ------------------------------------------------------------------

    BBOX_DUP_IOU_THRESHOLD: float = 0.9

    def _apply_bbox_dup_prune(self, tree_parts: list[Part],
                              parent_mask: np.ndarray | None,
                              parent_label: str) -> list[Part]:
        """Resolve children whose mask is geometrically the parent.

        Two regimes:

        - **Only-child** (``parent → child``): a fake relabel — drop the
          child, parent becomes atomic. Canonical cases:
          960 ``surface → main board → top edge`` and 1702 ``fire →
          outer core`` (the model "decomposed" but the child IS the
          parent).
        - **One of multiple siblings**: the child's bbox spans the
          parent because it's the residual / catch-all sibling
          (e.g. 2607 ``head → face, sunglasses, mouth`` where ``face``
          is most of head minus the other two). Drop would lose the
          partition; instead REPLACE the child's mask with
          ``parent_mask & ~union(other siblings)`` so face stays a
          named child but only claims the skin residual.

        Comparison uses mask IoU (≥ ``BBOX_DUP_IOU_THRESHOLD``). At
        root (``parent_mask is None``) no pruning fires.
        """
        if parent_mask is None or not parent_mask.any():
            for child in tree_parts:
                child.children = self._apply_bbox_dup_prune(
                    child.children or [], child.mask, child.label,
                )
            return tree_parts

        def _iou_with_parent(m: np.ndarray) -> float:
            inter = int((m & parent_mask).sum())
            union = int((m | parent_mask).sum())
            return inter / max(union, 1)

        kept: list[Part] = []
        n_total = len(tree_parts)
        for idx, child in enumerate(tree_parts):
            if child.mask is None or not child.mask.any():
                kept.append(child)
                continue
            iou = _iou_with_parent(child.mask)
            if iou >= self.BBOX_DUP_IOU_THRESHOLD:
                others = [c for j, c in enumerate(tree_parts)
                          if j != idx and c.mask is not None
                          and c.mask.any()]
                if not others:
                    print(f"    drop '{child.label}' (depth {child.depth}) "
                          f"— bbox-dup with parent '{parent_label}' "
                          f"(IoU={iou:.2f}, only child)")
                    continue
                union_others = np.zeros_like(parent_mask, dtype=bool)
                for c in others:
                    union_others |= c.mask
                residual = parent_mask & ~union_others
                if residual.any():
                    child.mask = residual
                    child.bbox_xywh = _bbox_xywh(residual)
                    child.icon_coverage = float(
                        (residual & self.content_mask).sum()
                    ) / self.icon_area
                    child.layer_image = _apply_mask_rgba(
                        self.img_rgba, residual,
                    )
                    child.extra["residual_replaced"] = True
                    print(f"    replace '{child.label}' (depth {child.depth}) "
                          f"with residual of '{parent_label}' "
                          f"(IoU was {iou:.2f}, {n_total - 1} other siblings)")
                else:
                    print(f"    drop '{child.label}' (depth {child.depth}) "
                          f"— bbox-dup with parent '{parent_label}' "
                          f"(IoU={iou:.2f}, residual empty)")
                    continue
            child.children = self._apply_bbox_dup_prune(
                child.children or [], child.mask, child.label,
            )
            kept.append(child)
        return kept

    # ------------------------------------------------------------------
    # v10 post-hoc: cross-tree dedup + single-child collapse
    # ------------------------------------------------------------------

    SINGLE_CHILD_DROP: bool = True

    CROSS_TREE_DEDUP_IOU: float = 0.9

    def _drop_residual_subtrees(self, tree_parts: list[Part]) -> None:
        """Residual-replaced children have no semantic content — they're
        the partition leftover. Recursing into them just rediscovers the
        leftover with smaller bboxes, inflating node count without
        adding info (1702 ``fire → outer core (residual) → top flame +
        side flames``). Strip their subtrees.
        """
        for ch in tree_parts:
            if ch.extra.get("residual_replaced") and ch.children:
                print(f"    drop subtree of '{ch.label}' (depth "
                      f"{ch.depth}) — residual replaced, children "
                      f"have no semantic content ({len(ch.children)} "
                      f"descendants stripped)")
                ch.children = []
            elif ch.children:
                self._drop_residual_subtrees(ch.children)

    def _apply_cross_tree_mask_dedup(self,
                                     tree_parts: list[Part]) -> list[Part]:
        """Drop parts whose mask duplicates an EARLIER-emitted part's
        mask (anywhere in the tree, not just siblings).

        588 ``shelf → 8 bottles + root → same 8 bottles``: each
        root-level bottle has identical bbox to a shelf child. Sibling-
        mutex doesn't see them because they're at different levels of
        the tree. Pre-order DFS keeps the first occurrence (which has
        more context — deeper, has a meaningful parent) and drops
        later ones.

        IoU threshold = ``CROSS_TREE_DEDUP_IOU``. The kept-first rule
        works because DFS visits deeper-context (parent's children)
        before the duplicate emitted at root.
        """
        flat: list[Part] = []

        def collect(parts: list[Part]):
            for p in parts:
                flat.append(p)
                if p.children:
                    collect(p.children)

        collect(tree_parts)

        keep = [True] * len(flat)
        for j in range(len(flat)):
            if not keep[j]:
                continue
            b = flat[j]
            if b.mask is None or not b.mask.any():
                continue
            for i in range(j):
                if not keep[i]:
                    continue
                # Don't dedupe a node against its own ancestor — bbox-dup
                # prune already handled parent/child relationships.
                ancestor = False
                anc_id = b.parent_id
                while anc_id:
                    if anc_id == flat[i].id:
                        ancestor = True
                        break
                    par = next((p for p in flat if p.id == anc_id), None)
                    anc_id = par.parent_id if par else None
                if ancestor:
                    continue
                a = flat[i]
                if a.mask is None or not a.mask.any():
                    continue
                inter = int((a.mask & b.mask).sum())
                union = int((a.mask | b.mask).sum())
                iou = inter / max(union, 1)
                if iou >= self.CROSS_TREE_DEDUP_IOU:
                    print(f"    cross-tree dedup: drop '{b.label}' "
                          f"(depth {b.depth}) — duplicates earlier "
                          f"'{a.label}' (depth {a.depth}, IoU={iou:.2f})")
                    keep[j] = False
                    break

        dropped_ids = {flat[k].id for k in range(len(flat)) if not keep[k]}

        def filter_tree(parts: list[Part]) -> list[Part]:
            out = []
            for p in parts:
                if p.id in dropped_ids:
                    continue
                p.children = filter_tree(p.children or [])
                out.append(p)
            return out

        return filter_tree(tree_parts)

    def _apply_single_child_collapse(self, tree_parts: list[Part],
                                     parent_label: str) -> list[Part]:
        """Drop a parent's only child — a 1-child "decomposition" is a
        relabeling, not a partition. Recurses depth-first.

        Skipped at root (``parent_label == "root"``): a single top-level
        entity is the normal "one-icon image" case and its decomposition
        must be preserved (e.g. 2607's ``root → head → face, mouth,
        sunglass``).
        """
        if not self.SINGLE_CHILD_DROP:
            return tree_parts
        for child in tree_parts:
            child.children = self._apply_single_child_collapse(
                child.children or [], child.label,
            )
        if len(tree_parts) == 1 and parent_label != "root":
            only = tree_parts[0]
            print(f"    collapse '{parent_label}' — single child "
                  f"'{only.label}' carries no partition info; marking atomic")
            return []
        return tree_parts

    def _apply_residual_margin_assign(
        self,
        tree_parts: list[Part],
        parent_mask: np.ndarray | None = None,
        level_depth: int = 0,
        level_parent_id: str | None = None,
        level_parent_label: str = "root",
    ) -> None:
        """Bottom-up recursive residual margin assignment.

        Post-order: each child's subtree is processed first; THEN this
        level's residual = ``parent_mask − union(direct children
        masks)`` is partitioned among the *effective leaves under this
        level only*. For each residual pixel, compute
        ``margin = top1 − top2`` over those leaves' ``sam_prob`` and
        assign pixels with ``margin ≥ cfg.residual_margin_threshold``
        to the argmax-winner leaf. The leftover becomes a new
        ``residual`` leaf appended as a sibling at this level (depth +
        parent_id matched to the level so it nests under the right
        parent).

        Keeps each parent's residual contained within its own subtree
        leaves — no cross-subtree contamination. Bottom-up ordering
        means a leaf that absorbs a child-level residual can then
        compete for its grandparent's residual with its grown mask.
        Leaves without ``sam_prob`` (root_atomic / no_bbox fallbacks)
        are skipped as candidates but still count in the union.
        """
        tau = float(getattr(self.cfg, "residual_margin_threshold", 0.15))

        # Post-order: recurse into each child's subtree first so leaves
        # absorb their immediate parent's residual before competing for
        # higher-up residuals.
        for child in list(tree_parts):
            if child.children:
                self._apply_residual_margin_assign(
                    child.children,
                    parent_mask=child.mask,
                    level_depth=child.depth + 1,
                    level_parent_id=child.id,
                    level_parent_label=child.label or "",
                )

        if not tree_parts:
            return
        region = (parent_mask & self.content_mask
                  if parent_mask is not None else self.content_mask)
        accepted = [c for c in tree_parts
                    if not c.rejected and c.mask is not None]
        if not accepted or not region.any():
            return

        union = np.zeros_like(region, dtype=bool)
        for c in accepted:
            union |= c.mask
        residual = region & ~union
        n_res = int(residual.sum())
        if n_res == 0:
            return

        # Candidates = effective leaves under THIS level's accepted
        # subtrees (subtree-restricted, not the global leaf set).
        leaves = self._collect_effective_leaves(accepted)
        cand_leaves: list[Part] = []
        cand_probs: list[np.ndarray] = []
        for lf in leaves:
            p = lf.extra.get("sam_prob") if lf.extra else None
            if p is None:
                continue
            cand_leaves.append(lf)
            cand_probs.append(p.astype(np.float32))

        n_assign = 0
        if cand_probs:
            stack = np.stack(cand_probs, axis=0)        # (K, H, W)
            max_prob = stack.max(axis=0)
            winner = stack.argmax(axis=0)
            if stack.shape[0] >= 2:
                second = np.partition(stack, -2, axis=0)[-2]
                margin_arr = max_prob - second
            else:
                margin_arr = max_prob.copy()
            assign_mask = residual & (margin_arr >= tau)
            n_assign = int(assign_mask.sum())
            for i, lf in enumerate(cand_leaves):
                claim = assign_mask & (winner == i)
                if not claim.any():
                    continue
                new_mask = (lf.mask | claim) if lf.mask is not None else claim
                lf.mask = new_mask
                lf.bbox_xywh = _bbox_xywh(new_mask)
                lf.icon_coverage = float(
                    (new_mask & self.content_mask).sum()
                ) / self.icon_area
                lf.layer_image = _apply_mask_rgba(self.img_rgba, new_mask)
            leftover = residual & ~assign_mask
        else:
            leftover = residual

        n_left = int(leftover.sum())
        parent_tag = (level_parent_label
                      if level_parent_label else "root") or "root"
        if n_left > 0:
            self.counter += 1
            pid = f"p{self.counter}"
            cov = float((leftover & self.content_mask).sum()
                        ) / max(self.icon_area, 1)
            new_leaf = Part(
                id=pid, label="residual",
                bbox_xywh=_bbox_xywh(leftover),
                score=0.0, icon_coverage=cov, mask=leftover,
                layer_image=_apply_mask_rgba(self.img_rgba, leftover),
                depth=level_depth, parent_id=level_parent_id,
            )
            new_leaf.extra["mask_source"] = "residual_margin_leftover"
            new_leaf.extra["residual_margin_threshold"] = tau
            new_leaf.extra["residual_parent_label"] = parent_tag
            tree_parts.append(new_leaf)
            print(f"    [residual_margin] d={level_depth} "
                  f"parent='{parent_tag}'  τ={tau:.3f}  "
                  f"residual={n_res}px → assigned={n_assign} "
                  f"({n_assign / max(n_res, 1):.1%}), "
                  f"residual leaf {pid}={n_left}px")
        else:
            print(f"    [residual_margin] d={level_depth} "
                  f"parent='{parent_tag}'  τ={tau:.3f}  "
                  f"residual={n_res}px → assigned={n_assign} (100%)")

    def _collect_effective_leaves(self, tree_parts: list[Part]) -> list[Part]:
        """Mirror of ``pipeline_controller._collect_effective_leaves`` —
        accepted parts with no accepted children. Internal-node rejects
        don't cascade; their accepted descendants still count."""
        out: list[Part] = []
        for p in tree_parts:
            if p.rejected:
                if p.children:
                    out.extend(self._collect_effective_leaves(
                        [c for c in p.children if not c.rejected]
                    ))
                continue
            accepted = [c for c in p.children if not c.rejected]
            if not accepted:
                out.append(p)
            else:
                out.extend(self._collect_effective_leaves(accepted))
        return out

    def _cleanup_sam_prob(self, tree_parts: list[Part]) -> None:
        """Drop the cached ``sam_prob`` from every node post-residual-
        argmax. Keeps segments.json clean (these arrays are big and
        not JSON-serialisable) and frees memory before vectorize."""
        for p in tree_parts:
            p.extra.pop("sam_prob", None)
            if p.children:
                self._cleanup_sam_prob(p.children)

    def _apply_label_dedupe(self, tree_parts: list[Part]) -> None:
        """Enforce cross-tree exact-label uniqueness.

        Rule (user-stated D): two parts MAY NOT share the exact same
        label string. ``desk legs`` and ``chair legs`` are fine (they
        differ); the broken case is when the H1 strip collapses both
        children to a bare ``legs``. Fix in-place by prepending the
        immediate parent's label to ALL duplicate occurrences:
        ``legs`` (parent=desk) → ``desk legs``;
        ``legs`` (parent=chair) → ``chair legs``.

        When the parents are themselves identical (rare — same-parent
        siblings would already have been deduped by the recursive
        builder) a numeric suffix is appended instead so we never emit
        two identical labels.
        """
        all_parts: list[Part] = []

        def walk(parts: list[Part]) -> None:
            for p in parts:
                all_parts.append(p)
                walk(p.children or [])

        walk(tree_parts)
        id_to_part = {p.id: p for p in all_parts}

        from collections import defaultdict
        groups: dict[str, list[Part]] = defaultdict(list)
        for p in all_parts:
            groups[(p.label or "").strip().lower()].append(p)

        for label, dups in groups.items():
            if len(dups) <= 1 or not label:
                continue
            parent_labels = []
            for p in dups:
                par = id_to_part.get(p.parent_id) if p.parent_id else None
                parent_labels.append(par.label if par and par.label else "root")
            if len(set(parent_labels)) == len(dups):
                for p, par_lbl in zip(dups, parent_labels):
                    new_lbl = f"{par_lbl} {p.label}".strip()
                    print(f"    rename '{p.label}' → '{new_lbl}' "
                          f"(cross-tree dedup, parent='{par_lbl}')")
                    p.label = new_lbl
                    p.extra["disambiguated"] = True
            else:
                for i, p in enumerate(dups):
                    if i == 0:
                        continue
                    new_lbl = f"{p.label} {i + 1}"
                    print(f"    rename '{p.label}' → '{new_lbl}' "
                          f"(numeric dedup — parents collide)")
                    p.label = new_lbl
                    p.extra["disambiguated"] = True
