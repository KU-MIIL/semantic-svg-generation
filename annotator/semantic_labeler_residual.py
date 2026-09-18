"""Wrapper around semantic_labeler.py — points the labeler at sample_residual
instead of sample_full so the residual set (labeler_3 분량) can be re-labeled
without touching the running 8088 instance.

Run:
    .venv/bin/python semantic_labeler_residual.py
    # serves at http://localhost:8089

Existing tree.json files in sample_residual are loaded as starting points —
the labeler edits them in place.
"""
from __future__ import annotations

from pathlib import Path

import uvicorn

import semantic_labeler as sl

_RES_ROOT = sl._PKG_ROOT / "sample_residual"

sl.SVG_DIRS = [
    _RES_ROOT / "icon" / "svg",
    _RES_ROOT / "illustration" / "svg",
    _RES_ROOT / "emoji" / "svg",
    _RES_ROOT / "sample" / "Icon" / "svg",
    _RES_ROOT / "sample" / "illustration" / "svg",
]
# No assignment file in sample_residual → load_assignments returns {},
# user_files falls back to listing all files (single-user mode).
sl._ASSIGN_PATH = _RES_ROOT / "_assignments.json"

PORT = 8089

if __name__ == "__main__":
    files = sl.list_files()
    print(f"Sample-residual labeler at http://localhost:{PORT}")
    print(f"  source: {_RES_ROOT}")
    print(f"  files:  {len(files)}")
    uvicorn.run(sl.app, host="0.0.0.0", port=PORT, log_level="warning")
