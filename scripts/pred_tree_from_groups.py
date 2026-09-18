#!/usr/bin/env python3
"""Build a tree.json-shaped grouping for a *flat-file* VLM prediction.

The p2 prompt asks the model to emit semantic ``<g>`` groups, so the
prediction already carries a grouping — it just isn't in tree.json form.
This turns the ``<g>`` nesting into the ``{"root": {"children": [...]}}``
schema ``metric.semantic.flatten_tree`` expects, so those methods can be
scored with the same N×M grouping metric as our pipeline.

Path indices follow ``metric.semantic.count_drawables`` document order —
non-rendered containers (``<defs>``, ``<clipPath>``, …) are skipped, so
the indices line up with what ``render_subset_rgb`` will draw. Using a
walker that counts them instead shifts every later index (8/203 GT
samples drift, worst case by 8).

Grouping rules:
  * every ``<g>`` holding >=1 drawable becomes a node; nesting is kept,
    and a parent's ``paths`` is the union over its subtree (mirroring
    ``scripts/semantic_eval_batch.build_pred_tree``);
  * drawables not under any ``<g>`` are collected into one extra
    ``"ungrouped"`` node, so precision still sees every bit of ink;
  * a file with no grouping at all yields a single node holding every
    primitive — the "one group" fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from metric.semantic import (
    _DRAWABLE_TAGS, _NON_RENDERED_CONTAINERS, _strip_ns,
)


def _label_of(el: ET.Element, fallback: str) -> str:
    for key in ("data-label", "id", "aria-label", "class"):
        v = (el.get(key) or "").strip()
        if v:
            return v
    return fallback


def build_tree_from_groups(svg_text: str) -> tuple[dict, dict]:
    """Return ``(tree, info)``.

    ``info`` reports ``n_drawables``, ``n_groups`` (nodes emitted),
    ``n_ungrouped`` and ``fallback_single_group`` so the caller can log
    how each prediction was interpreted.
    """
    root = ET.fromstring(svg_text)
    idx = [0]
    grouped: set[int] = set()

    def walk(parent: ET.Element) -> list[dict]:
        """Emit child nodes for every <g> under ``parent`` that draws."""
        nodes: list[dict] = []
        for child in list(parent):
            tag = _strip_ns(child.tag)
            if tag in _NON_RENDERED_CONTAINERS:
                continue
            if tag in _DRAWABLE_TAGS:
                idx[0] += 1
                continue
            if tag == "g":
                start = idx[0]
                kids = walk(child)
                own = set(range(start, idx[0]))
                if not own:
                    continue                      # <g> with no ink — skip
                grouped.update(own)
                node: dict = {"label": _label_of(child, f"group{len(nodes)}"),
                              "paths": sorted(own)}
                if kids:
                    node["children"] = kids
                nodes.append(node)
            else:
                nodes.extend(walk(child))
        return nodes

    children = walk(root)
    n_draw = idx[0]
    ungrouped = sorted(set(range(n_draw)) - grouped)
    fallback = False

    if not children:
        # No usable grouping anywhere → treat the whole drawing as one group.
        fallback = True
        children = ([{"label": "whole", "paths": list(range(n_draw))}]
                    if n_draw else [])
    elif ungrouped:
        children.append({"label": "ungrouped", "paths": ungrouped})

    info = {"n_drawables": n_draw,
            "n_groups": len(children),
            "n_ungrouped": len(ungrouped),
            "fallback_single_group": fallback}
    return {"root": {"children": children}}, info


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svg", type=Path)
    args = ap.parse_args()
    tree, info = build_tree_from_groups(args.svg.read_text())
    print(json.dumps({"info": info, "tree": tree}, indent=1)[:2000])


if __name__ == "__main__":
    main()
