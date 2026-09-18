"""Anthropic Claude client that mimics the slice of ``google.genai`` we use.

Drop-in for the same call sites as :mod:`modules.qwen_client` /
:mod:`modules.openai_client`. Routes any ``claude-*`` / ``Claude-*``
model id through the Anthropic Messages API while presenting the
``client.models.generate_content`` surface the Gemini SDK exposes.

Claude specifics:
    * Temperature defaults to 0.0 (deterministic), matching Gemini.
    * ``max_tokens`` is required — pinned at 4096 for JSON responses
      the pipeline consumes.
    * No native JSON-mode; we rely on the prompt to specify JSON and
      strip surrounding prose with ``_extract_json_object``.
    * ``system`` is not used — every Gemini call site inlines the
      instructions into the user contents, and Anthropic accepts a
      user-only message.

API key via ``ANTHROPIC_API_KEY`` (required).
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image


def is_claude_model(model: str | None) -> bool:
    """Route on model name prefix. ``claude-*`` / ``Claude-*`` → Anthropic."""
    if not model:
        return False
    m = model.strip().lower()
    return m.startswith("claude-") or m.startswith("claude ")


def _encode_image_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    rgb.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _contents_to_blocks(contents: list) -> list[dict]:
    """Flatten a Gemini-style ``contents`` list (strings + PIL images)
    into Anthropic Messages API content blocks. Order is preserved.

    Anthropic blocks: ``{"type":"text","text":...}`` or
    ``{"type":"image","source":{"type":"base64","media_type":"image/png",
    "data":<b64>}}``.
    """
    blocks: list[dict] = []
    for item in contents:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, Image.Image):
            b64 = _encode_image_b64(item)
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })
        else:
            blocks.append({"type": "text", "text": str(item)})
    return blocks


@dataclass
class _UsageMeta:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int = 0
    total_token_count: int = 0


@dataclass
class _Response:
    text: str
    usage_metadata: _UsageMeta


class _Models:
    def __init__(self, client: "ClaudeClient") -> None:
        self._client = client

    def generate_content(self, *, model: str, contents: list,
                          config: Any = None, reasoning: str | None = None,
                          **_kw) -> _Response:
        anth = self._client._anth
        blocks = _contents_to_blocks(contents)
        # ``reasoning`` mirrors the Gemini thinking_level the call site would
        # have applied. When set (inpaint judge/polygon/pick run at "medium",
        # matching Gemini), enable adaptive extended thinking at that effort.
        # Adaptive thinking on Sonnet 4.6 requires temperature to stay at the
        # default, so it is omitted; max_tokens is raised because thinking
        # tokens share the output budget. When unset (decompose path) keep the
        # deterministic, thinking-off configuration.
        # ``reasoning`` (explicit, from inpaint call sites) wins. When absent
        # — the decompose path — fall back to ``VLM_DECOMPOSE_EFFORT`` so a
        # run can sweep the decompose grounding effort without touching call
        # sites; unset keeps the deterministic thinking-off default.
        effort = reasoning or os.environ.get("VLM_DECOMPOSE_EFFORT")
        # Claude's output_config.effort accepts low|medium|high|xhigh|max only,
        # NOT 'minimal' (a Gemini/GPT reasoning level). Map 'minimal' — and
        # anything outside the valid set — to thinking-off, its closest analog.
        if effort not in ("low", "medium", "high", "xhigh", "max"):
            effort = None
        if effort:
            create_kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=16384,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                messages=[{"role": "user", "content": blocks}],
            )
        else:
            # No temperature: the Messages API dropped it in anthropic SDK 1.0
            # (output_config carries only `effort` and `format`), so passing it
            # raises TypeError. The thinking branch above never set it either —
            # adaptive thinking required the default — so both paths now run at
            # the model default, which is what the rebuttal's Claude runs used
            # for every reasoning call.
            create_kwargs = dict(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": blocks}],
            )
        resp = anth.messages.create(**create_kwargs)
        # Anthropic returns a list of content blocks; concatenate the text
        # blocks and skip any thinking blocks the adaptive pass produced.
        parts: list[str] = []
        for b in resp.content or []:
            if getattr(b, "type", None) == "text":
                parts.append(getattr(b, "text", "") or "")
        text = "".join(parts)
        usage = resp.usage
        pt = int(getattr(usage, "input_tokens", 0) or 0)
        # output_tokens already includes any thinking tokens; bill it all at
        # the output rate (the API exposes no separate thinking-token count).
        ct = int(getattr(usage, "output_tokens", 0) or 0)
        return _Response(
            text=text,
            usage_metadata=_UsageMeta(
                prompt_token_count=pt,
                candidates_token_count=ct,
                thoughts_token_count=0,
                total_token_count=pt + ct,
            ),
        )


class ClaudeClient:
    """Drop-in stand-in for ``google.genai.Client`` against the Anthropic
    Messages API. Only ``.models.generate_content`` is implemented — the
    same slice our pipeline needs from the Gemini SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import Anthropic
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._anth = Anthropic(api_key=api_key)
        self.models = _Models(self)


_CLIENT: ClaudeClient | None = None


def get_client() -> ClaudeClient:
    """Process-wide singleton (mirrors :func:`modules.gemini_client._get_client`)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = ClaudeClient()
    return _CLIENT


__all__ = ["ClaudeClient", "is_claude_model", "get_client"]
