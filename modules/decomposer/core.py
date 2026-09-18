"""Public ``Decomposer`` orchestrator.

A thin façade: holds the model identifiers + hyperparameters, lazily loads
SAM3 once, and delegates each ``decompose()`` call to a fresh
:class:`_DecomposeRun`. The heavy lifting lives in ``run.py``;
this file is the stable import target for the rest of the project.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from modules.gemini_client import GeminiClient

from .sam import _get_sam


class Decomposer:
    """Single-level semantic decomposition (VLM labels + SAM 3 masks)."""

    def __init__(
        self,
        gemini_model: str = "gemini-3-flash-preview",
        sam_model: str = "facebook/sam3",
        close_kernel: int = 3,
        smooth_epsilon_px: float = 1.5,
        small_hole_ratio: float = 0.02,
        device: str | None = None,
        post_hoc_sam: bool = False,
        post_hoc_segmenter: str | None = None,
        bad_recover_color: bool = False,
        bad_recover_edge_thresh: float = 0.30,
        bad_recover_min_split_area: int = 30,
        decompose_thinking_budget: int = 0,
        residual_margin_threshold: float = 0.15,
        decompose_mode: str = "v15",
        inpaint_model: str = "gemini-3-flash-preview",
        inpaint_thinking_level: str = "low",
        inpaint_device: str = "cuda:0",
    ):
        self.gemini_model = gemini_model
        self.sam_model = sam_model
        self.close_kernel = close_kernel
        self.smooth_epsilon_px = smooth_epsilon_px
        self.small_hole_ratio = small_hole_ratio
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._dtype = None
        # v5: VLM-first tree, then run SAM post-hoc on every non-root node.
        # Phase 1 builds the tree without SAM, then _apply_post_hoc_sam swaps
        # each node.mask in place.
        self.post_hoc_sam = post_hoc_sam
        # v8 family: post-hoc segmenter that replaces the leaf bbox-rect.
        #   "sam_stability" — SAM3 grounded with (label, bbox); the rank-1
        #                     candidate's mask replaces the rect, and
        #                     ``assess_sam_quality`` (sb_05/sb_20 thresholds
        #                     fit on sweep_v11) tags each leaf good/bad. Bad
        #                     leaves enter the optional ``bad_recover_*``
        #                     path below.
        # Mutually exclusive with ``post_hoc_sam`` (same leaf-mask slot).
        if post_hoc_segmenter not in (None, "sam_stability"):
            raise ValueError(
                f"post_hoc_segmenter must be None or 'sam_stability', "
                f"got {post_hoc_segmenter!r}"
            )
        if post_hoc_segmenter is not None and self.post_hoc_sam:
            raise ValueError(
                f"post_hoc_segmenter={post_hoc_segmenter!r} is mutually "
                f"exclusive with post_hoc_sam=True (both replace the leaf "
                f"mask post-hoc)"
            )
        self.post_hoc_segmenter = post_hoc_segmenter
        # Recovery for leaves flagged ``sam_quality == "bad"`` by
        # ``assess_sam_quality``: exp5 palette + edge-split pipeline.
        self.bad_recover_color = bool(bad_recover_color)
        self.bad_recover_edge_thresh = float(bad_recover_edge_thresh)
        self.bad_recover_min_split_area = int(bad_recover_min_split_area)
        # Reasoning-heavy decompose calls can opt into Gemini-3 thinking.
        self.decompose_thinking_budget = int(decompose_thinking_budget)
        # Residual handler: for each pixel left uncovered by any leaf SAM
        # mask, look at (top1 − top2) sam_prob over all leaves. Pixels
        # with margin ≥ threshold get assigned to the argmax winner;
        # the rest collect into a single "residual" leaf appended at
        # root level so vtracer still fills the gap. Margin (not just
        # max_prob) is used because catch-all leaves otherwise sweep
        # near-zero confidence pixels into one leaf.
        self.residual_margin_threshold = float(residual_margin_threshold)
        # Decompose modes:
        #   v15 (default) — per-node Gemini → SAM → margin-residual →
        #                   residual-only judge → recurse. No inpaint in
        #                   build; cost is leaf-count × judge.
        #   v13 (legacy)  — flat decompose + post-hoc SAM stability +
        #                   bad-quality color recovery. Kept as fallback.
        if decompose_mode not in ("v13", "v15"):
            raise ValueError(
                f"decompose_mode must be 'v13' or 'v15', "
                f"got {decompose_mode!r}"
            )
        self.decompose_mode = decompose_mode
        # Inpaint config — used by v15 residual judge (and reserved for
        # the planned post-tree inpaint pass on accepted leaves).
        self.inpaint_model = inpaint_model
        if inpaint_thinking_level not in ("minimal", "low", "medium", "high"):
            raise ValueError(
                f"inpaint_thinking_level must be one of "
                f"minimal/low/medium/high; got {inpaint_thinking_level!r}"
            )
        self.inpaint_thinking_level = inpaint_thinking_level
        self.inpaint_device = inpaint_device
        # InpaintAgent (FLUX+Gemini) is lazy-loaded only if explicitly
        # requested — keeping ~15 GB FLUX off the GPU for the default
        # v15 build path which only needs the residual judge.
        self._inpaint_agent = None
        self._gemini = GeminiClient(
            model=gemini_model,
            decompose_thinking_budget=self.decompose_thinking_budget,
        )

    def _get_inpaint_agent(self):
        """Lazy InpaintAgent singleton (reserved for post-tree inpaint
        pass; current v15 build does not call this)."""
        if self._inpaint_agent is None:
            from modules.inpainter import get_agent
            self._inpaint_agent = get_agent(device=self.inpaint_device)
        return self._inpaint_agent

    def _ensure_loaded(self):
        # SAM is needed for v13 post-hoc paths (post_hoc_sam,
        # sam_stability) and for v15 (per-node SAM inside the builder).
        needs_sam = (
            self.post_hoc_sam
            or self.post_hoc_segmenter == "sam_stability"
            or self.decompose_mode == "v15"
        )
        if not needs_sam:
            return None, None
        if self._dtype is None:
            import torch
            self._dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        return _get_sam(self.sam_model, self.device, self._dtype)

    def decompose(self, image: Image.Image,
                  debug_dir: Any | None = None,
                  max_depth: int = 2) -> dict[str, Any]:
        if self.decompose_mode == "v15":
            from .run_v15 import _DecomposeRunV15
            return _DecomposeRunV15(self, image, debug_dir,
                                    max_depth=max_depth).run()
        from .run import _DecomposeRun
        return _DecomposeRun(self, image, debug_dir,
                             max_depth=max_depth).run()
