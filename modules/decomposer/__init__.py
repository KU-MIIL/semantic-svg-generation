"""Semantic decomposer: Gemini labels + SAM 3 text-prompted masks.

Exposes the public surface (``Decomposer``, ``Part``, ``_safe_name``) for
the pipeline. Implementation is split across submodules:

  - ``part``         : the ``Part`` dataclass
  - ``mask_ops``     : pure image / mask utilities
  - ``tree_utils``   : tree traversal + Gemini-bbox conversion
  - ``sam``          : SAM3 model cache + segmentation primitive
  - ``core``         : ``Decomposer`` thin façade
"""
from __future__ import annotations

import os

# Must be set before CUDA / cuBLAS context is created so deterministic
# algorithms can use a fixed workspace. Done at package import time because
# anything later (after the first `.to("cuda")`) is too late.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .core import Decomposer
from .part import Part
from .tree_utils import _safe_name


__all__ = ["Decomposer", "Part", "_safe_name"]
