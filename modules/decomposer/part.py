"""``Part`` — tree node container for the semantic decomposition output."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class Part:
    id: str                                     # "p1", "p2", ... (BFS-ish DFS order)
    label: str                                  # semantic label from Gemini
    bbox_xywh: tuple[int, int, int, int]        # canvas-space bbox
    score: float                                # SAM3 detection score
    icon_coverage: float                        # mask area / icon area
    mask: np.ndarray                            # bool (H, W) — visible silhouette
    layer_image: Image.Image                    # RGBA, original-size canvas
    depth: int = 0                              # 0 = top-level part under root scene
    parent_id: str | None = None
    children: list["Part"] = field(default_factory=list)
    rejected: bool = False                      # True → dropped by judge or other post-hoc filter
    extra: dict = field(default_factory=dict)   # gemini_bbox / mask_source / sam_* / etc.
