"""Import this BEFORE any Gemini call to disable CoT/thinking tokens.

Monkeypatches google.genai Models.generate_content so every call gets
thinking_config = ThinkingConfig(thinking_budget=0, include_thoughts=
False). Saves cost; no behavior change for our polygon use.

    from modules.inpainter import _nothink   # noqa: F401
"""
from __future__ import annotations

from google.genai import models as _m
from google.genai import types as _t

_ORIG = _m.Models.generate_content
_NOTHINK = _t.ThinkingConfig(thinking_budget=0, include_thoughts=False)


def _patched(self, *, model, contents, config=None, **kw):
    if config is None:
        config = _t.GenerateContentConfig(thinking_config=_NOTHINK)
    elif isinstance(config, dict):
        if config.get("thinking_config") is None:
            config = {**config, "thinking_config": _NOTHINK}
    else:
        # Only inject the default when the caller didn't provide one,
        # so Gemini-3.5 ``thinking_level`` callers aren't clobbered.
        if getattr(config, "thinking_config", None) is None:
            try:
                config = config.model_copy(update={"thinking_config": _NOTHINK})
            except Exception:
                try:
                    config.thinking_config = _NOTHINK
                except Exception:
                    pass
    return _ORIG(self, model=model, contents=contents, config=config, **kw)


if getattr(_m.Models.generate_content, "__name__", "") != "_patched":
    _m.Models.generate_content = _patched
