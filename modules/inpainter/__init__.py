"""Inpainter subpackage: Gemini-judge → Gemini-polygon × N → FLUX-Fill × N → Gemini-pick.

Per part, the agent first asks Gemini whether the SAM cutout is
actually occluded by another scene element (``judge``); if yes, Gemini
predicts the part's complete outline as one or more polygons
(``polygon``), each rasterized to a FLUX repaint mask (``mask``), then
FLUX.1-Fill-dev fills in the occluded pixels (``agent``). When more
than one polygon prompt is used (the multi-mask pipeline), a final
Gemini pick-judge (``picker``) chooses the best of the N candidate
inpaints.

The judge gate is what keeps the heavy FLUX path off the table for the
~majority of parts that are already complete (no occlusion) or are
residual scatter (no coherent shape).

Public API::

    from modules.inpainter import get_agent              # lazy singleton (preferred)
    from modules.inpainter import InpaintAgent           # raw class (tests / CLI)
    # InpaintAgent.inpaint(...)        - single-mask path
    # InpaintAgent.inpaint_multi(...)  - multi-mask + pick-judge path

Internals (re-imports / hooks are documented in each submodule):
    agent     - FLUX-backed inpaint orchestrator (single + multi paths)
    judge     - Gemini "needs inpaint?" gate (v2: closed-set occluders)
    polygon   - Gemini polygon prediction (POLYGON_PROMPTS p0..p3)
    mask      - polygon → mask, repaint mask construction
    picker    - Gemini pick-judge over N candidate inpaints
    imaging   - RGBA / RGB conversion helpers
    client    - process-wide lazy singleton for InpaintAgent
    _nothink  - side-effect import that turns off Gemini CoT tokens
    cli       - argparse entry point (`python -m modules.inpainter.cli ...`)
"""
from .agent import InpaintAgent
from .client import get_agent
from .picker import pick_best
from .polygon import POLYGON_PROMPTS

__all__ = [
    "InpaintAgent",
    "get_agent",
    "pick_best",
    "POLYGON_PROMPTS",
]
