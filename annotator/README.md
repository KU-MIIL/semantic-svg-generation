# SVGAgent-benchmark — semantic labeler webui (portable bundle)

Hierarchical SVG semantic labeling webui. Fully self-contained: zip this
folder, copy it to any server, and run — no path editing needed.

## Layout
```
SVGAgent-benchmark/
  semantic_labeler.py          # the webui server (run this)
  visualize_tree.py            # imported at runtime for tree preview rendering
  requirements.txt
  sample_full/
    _assignments.json          # per-user workload split (6 labelers)
    icon/svg/*.svg
    illustration/svg/*.svg
  sample_full_shp/             # illustration with >50 paths, split out (not labeled)
```

## Setup on the new server
```bash
pip install -r requirements.txt
# Optional (nicer label fonts; auto-falls back to a built-in font if absent):
#   apt-get install fonts-dejavu-core
python semantic_labeler.py
# open http://localhost:8088   (binds 0.0.0.0 → reachable as http://<server-ip>:8088)
```

## Per-user labeling (access control)
- On open, each labeler types their name in the header box and clicks 확인
  (names autocomplete from `_assignments.json`: labeler_1 / labeler_2 / labeler_3 /
  labeler_4 / labeler_5 / labeler_6).
- Only that person's assigned files are listed, openable, and savable.
  Others' files are not in their index at all (server-enforced), so multiple
  people can use the same server URL concurrently without collisions.
- The 💾 counter shows progress within the user's own assignment only.
- Work is split evenly by total path-count (difficulty), not file-count.

## Notes
- All paths resolve **relative to this folder** (`Path(__file__).parent`),
  including `SVG_DIRS` and `_assignments.json` — nothing is hard-coded to a
  machine. No absolute paths remain except optional font fallbacks.
- Only `*.svg` files whose name is all digits are scanned.
- Saving writes a `<name>.tree.json` sidecar next to each SVG (the labeling
  output). "Fail" writes `{"fail": true, ...}` to that sidecar.
- To re-split the workload, regenerate `sample_full/_assignments.json`
  (`{"users": [...], "assign": {"icon/239": "labeler_1", ...}}`); the app reloads
  it on every request. If the file is missing, it falls back to single-user
  mode (all files visible).
