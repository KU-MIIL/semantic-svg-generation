"""Lazy singleton wrapper around :class:`modules.inpainter.agent.InpaintAgent`.

Toggle entry point for the pipeline. FLUX.1-Fill-dev is heavy
(~15 GB bf16) so the agent (and thus the model) is only constructed on
the first ``get_agent()`` call. Subsequent calls reuse the same instance.
"""
from __future__ import annotations

import os
from pathlib import Path

_AGENT = None


def _ensure_safe_cache_dirs() -> None:
    """``/tmp`` is noexec on some hosts; redirect compile/inductor caches.

    diffusers transitively triggers torch.compile / inductor on the FLUX
    transformer; the JIT cache mmaps ``.so`` files which fails outright
    when the cache dir is on a noexec mount.

    Root selection, first that works: ``$SVGAGENT_CACHE_ROOT``, then
    ``.cache_runtime/inpaint`` next to the repo (what ``set_env.sh`` pins the
    other JIT caches to). Every var is set with ``setdefault``, so an
    environment that already pinned these — again, ``set_env.sh`` — wins.
    """
    repo_default = Path(__file__).resolve().parents[2] / ".cache_runtime" / "inpaint"
    safe_root = Path(os.environ.get("SVGAGENT_CACHE_ROOT") or repo_default)
    try:
        safe_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return                      # keep whatever the environment already set
    os.environ.setdefault("TMPDIR", str(safe_root))
    os.environ.setdefault("TRITON_CACHE_DIR", str(safe_root / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                          str(safe_root / "inductor"))


def get_agent(device: str = "cuda:0"):
    """Return the singleton :class:`InpaintAgent` (lazy-loaded)."""
    global _AGENT
    if _AGENT is None:
        _ensure_safe_cache_dirs()
        # FLUX itself only loads at the first .inpaint() call (lazy inside
        # InpaintAgent._pipe_).
        from .agent import InpaintAgent
        _AGENT = InpaintAgent(device=device)
    return _AGENT
