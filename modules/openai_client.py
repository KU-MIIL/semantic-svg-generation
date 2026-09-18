"""OpenAI GPT client that mimics the slice of ``google.genai`` we use.

Drop-in for the same call sites as :mod:`modules.qwen_client`. Routes any
``gpt-*`` / ``GPT-*`` model id through the OpenAI Chat Completions API
while presenting the ``client.models.generate_content`` surface the
Gemini SDK exposes.

GPT-5 specifics:
    * Temperature is not configurable — GPT-5 accepts only the API
      default. We therefore omit ``temperature`` on the request rather
      than pinning it to 0.0 (which would 400 the call).
    * ``max_completion_tokens`` replaces the deprecated ``max_tokens``
      knob for GPT-5.
    * ``response_format={"type": "json_object"}`` is honored, matching
      the JSON-shaped prompts every Gemini call site emits.
    * Reasoning tokens (``usage.completion_tokens_details.
      reasoning_tokens``) are surfaced as ``thoughts_token_count`` for
      parity with the Gemini-2.5+ ``thoughts_token_count`` field.

Endpoint override via ``OPENAI_BASE_URL`` (default OpenAI cloud);
API key via ``OPENAI_API_KEY`` (required).
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image


def is_gpt_model(model: str | None) -> bool:
    """Route on model name prefix. ``gpt-*`` / ``GPT-*`` → OpenAI."""
    if not model:
        return False
    m = model.strip().lower()
    return m.startswith("gpt-") or m.startswith("gpt5") or m.startswith("chatgpt")


def _encode_image_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    rgb.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _contents_to_messages(contents: list) -> list[dict]:
    """Flatten a Gemini-style ``contents`` list (strings + PIL images)
    into a single OpenAI ``user`` message with mixed ``text`` and
    ``image_url`` content blocks. Order is preserved."""
    blocks: list[dict] = []
    for item in contents:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, Image.Image):
            b64 = _encode_image_b64(item)
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        else:
            blocks.append({"type": "text", "text": str(item)})
    return [{"role": "user", "content": blocks}]


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
    def __init__(self, client: "OpenAIClient") -> None:
        self._client = client

    def generate_content(self, *, model: str, contents: list,
                          config: Any = None, reasoning: str | None = None,
                          **_kw) -> _Response:
        oai = self._client._oai
        messages = _contents_to_messages(contents)
        # ``reasoning`` mirrors the Gemini thinking_level the call site would
        # have applied: None on the decompose path (→ minimal, matching
        # Gemini's thinking_budget=0), "medium" on the inpaint judge/polygon/
        # pick calls. When reasoning is on, give the output cap extra headroom
        # because reasoning tokens count against ``max_completion_tokens``.
        # ``reasoning`` (explicit, from inpaint call sites) wins. When absent
        # — the decompose path, which calls generate_content without a
        # reasoning hint — fall back to ``VLM_DECOMPOSE_EFFORT`` so a run can
        # sweep the decompose grounding effort without touching call sites;
        # unset keeps the safe "minimal" default (= Gemini decompose off).
        effort = reasoning or os.environ.get("VLM_DECOMPOSE_EFFORT") or "minimal"
        max_out = 32768 if effort != "minimal" else 16384
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_out,
            "response_format": {"type": "json_object"},
            "reasoning_effort": effort,
        }
        resp = oai.chat.completions.create(**create_kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct_total = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", 0) or (pt + ct_total))
        details = getattr(usage, "completion_tokens_details", None)
        rt = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        # OpenAI's ``completion_tokens`` already includes reasoning tokens.
        # Split visible output from reasoning so the cost aggregator (which
        # bills candidates + thoughts at the output rate) doesn't count
        # reasoning twice.
        visible = max(0, ct_total - rt)
        return _Response(
            text=text,
            usage_metadata=_UsageMeta(
                prompt_token_count=pt,
                candidates_token_count=visible,
                thoughts_token_count=rt,
                total_token_count=tt,
            ),
        )


class OpenAIClient:
    """Drop-in stand-in for ``google.genai.Client`` against the OpenAI
    Chat Completions API. Only ``.models.generate_content`` is
    implemented — the same slice our pipeline needs from the Gemini
    SDK."""

    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None) -> None:
        from openai import OpenAI
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        oai_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            oai_kwargs["base_url"] = base_url
        self._oai = OpenAI(**oai_kwargs)
        self.models = _Models(self)


_CLIENT: OpenAIClient | None = None


def get_client() -> OpenAIClient:
    """Process-wide singleton (mirrors :func:`modules.gemini_client._get_client`)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAIClient()
    return _CLIENT


__all__ = ["OpenAIClient", "is_gpt_model", "get_client"]
