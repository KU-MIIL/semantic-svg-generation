"""vLLM-backed Qwen client that mimics the slice of ``google.genai`` we use.

Goal: drop-in for the four Gemini call sites
(``decompose_one_level``, ``judge_needs_inpaint``, ``predict_complete_polygon``,
``pick_best``) without rewriting their call patterns.

The Gemini SDK pattern those sites use is::

    resp = client.models.generate_content(
        model=..., contents=[prompt_str, PIL.Image, ...], config=cfg,
    )
    resp.text
    resp.usage_metadata.prompt_token_count
    resp.usage_metadata.candidates_token_count
    resp.usage_metadata.thoughts_token_count   # Gemini-2.5+; we set to 0
    resp.usage_metadata.total_token_count

We reproduce that surface with :class:`QwenClient`, mapping ``contents``
(list mixing strings and PIL images) to OpenAI Chat Completions
``messages`` with base64-encoded ``image_url`` blocks. The ``config``
argument (Gemini ``GenerateContentConfig``/``ThinkingConfig``) is
discarded — Qwen doesn't expose a comparable knob.

Endpoint override via ``QWEN_BASE_URL`` (default
``http://localhost:8000/v1``); API key via ``QWEN_API_KEY`` (default
``EMPTY``, what vLLM accepts).
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image


def is_qwen_model(model: str | None) -> bool:
    """Route on model name prefix. ``Qwen*`` / ``qwen*`` → vLLM.

    Matches both the HuggingFace id (``Qwen/Qwen3.6-35B-A3B``) and the
    served-name shorthand (``Qwen3.6-35B-A3B``).
    """
    if not model:
        return False
    m = model.strip()
    return m.startswith("Qwen") or m.lower().startswith("qwen")


def _strip_org_prefix(model: str) -> str:
    """``Qwen/Qwen3.6-35B-A3B`` → ``Qwen3.6-35B-A3B``. vLLM's
    ``--served-model-name`` typically drops the org prefix."""
    return model.split("/", 1)[1] if "/" in model else model


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
            # Unknown content type — coerce to str so the call doesn't
            # silently drop it.
            blocks.append({"type": "text", "text": str(item)})
    return [{"role": "user", "content": blocks}]


@dataclass
class _UsageMeta:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    thoughts_token_count: int = 0          # always 0 for Qwen; kept for
                                            # signature parity with Gemini-2.5+
    total_token_count: int = 0


@dataclass
class _Response:
    text: str
    usage_metadata: _UsageMeta


class _Models:
    """``client.models.generate_content`` shim — the only Gemini SDK
    surface our four call sites touch."""

    def __init__(self, client: "QwenClient") -> None:
        self._client = client

    def generate_content(self, *, model: str, contents: list,
                          config: Any = None, **_kw) -> _Response:
        # ``config`` (Gemini GenerateContentConfig / ThinkingConfig) is
        # intentionally ignored. Temperature is fixed at 0 to match the
        # Gemini ``temperature=0.0`` used in modules/gemini_client.py.
        oai = self._client._oai
        messages = _contents_to_messages(contents)
        # Honor served-model-name convention (Qwen3.6-35B-A3B without
        # the ``Qwen/`` org prefix). vLLM rejects the request if the
        # passed name doesn't match ``--served-model-name``.
        served = _strip_org_prefix(model)
        # All four Gemini call sites ask for JSON output in their
        # prompts. Force vLLM's guided-JSON mode so Qwen can't slip
        # natural-language reasoning in front of the JSON. ``extra_body
        # .chat_template_kwargs.enable_thinking=False`` suppresses the
        # Qwen3 hybrid-thinking trace (otherwise emitted before the JSON
        # and breaks our ``_extract_json_object`` parser).
        resp = oai.chat.completions.create(
            model=served,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", 0) or (pt + ct))
        return _Response(
            text=text,
            usage_metadata=_UsageMeta(
                prompt_token_count=pt,
                candidates_token_count=ct,
                thoughts_token_count=0,
                total_token_count=tt,
            ),
        )


class QwenClient:
    """Drop-in stand-in for ``google.genai.Client`` against a vLLM
    OpenAI-compatible endpoint. Only the ``.models.generate_content``
    method is implemented — it's the only Gemini SDK call our pipeline
    uses."""

    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None) -> None:
        from openai import OpenAI
        base_url = (base_url or os.environ.get("QWEN_BASE_URL")
                    or "http://localhost:8000/v1")
        api_key = (api_key or os.environ.get("QWEN_API_KEY") or "EMPTY")
        self._oai = OpenAI(base_url=base_url, api_key=api_key)
        self.models = _Models(self)


_CLIENT: QwenClient | None = None


def get_client() -> QwenClient:
    """Process-wide singleton (mirrors :func:`modules.gemini_client._get_client`)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QwenClient()
    return _CLIENT


__all__ = ["QwenClient", "is_qwen_model", "get_client"]
