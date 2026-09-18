"""Tree/parsing helpers for Gemini's hierarchical decomposition output."""
from __future__ import annotations

import re
from typing import Any


def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s).strip("_")
    return (s[:40] or "part").lower()


def _real_children(d: Any) -> list[tuple[str, Any]]:
    """List ``(label, value)`` pairs in a tree dict, skipping ``_``-prefixed
    metadata keys (e.g. ``_bbox``) so the iteration only sees real children."""
    if not isinstance(d, dict):
        return []
    return [(k, v) for k, v in d.items()
            if not (isinstance(k, str) and k.startswith("_"))]


def _bbox_from_gemini(bbox_raw: Any, W: int, H: int) -> list[int] | None:
    """Convert Gemini's ``[ymin, xmin, ymax, xmax]`` in normalized 0-1000
    coordinates to canvas-space ``[x0, y0, x1, y1]`` for SAM3 grounding.
    Returns ``None`` if the input is malformed so the caller falls back to
    text-only segmentation rather than feeding SAM a bogus box."""
    if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in bbox_raw)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= ymin < ymax <= 1000.0 and 0.0 <= xmin < xmax <= 1000.0):
        return None
    x0 = int(round(xmin * W / 1000.0))
    y0 = int(round(ymin * H / 1000.0))
    x1 = int(round(xmax * W / 1000.0))
    y1 = int(round(ymax * H / 1000.0))
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]
