# Semantic-SVG Benchmark

Code and benchmark for **compositional / semantic SVG generation** — the task of
turning a raster image into an SVG whose structure reflects the image's
*semantics* (objects and their parts), not just its pixels.

- 📄 **Paper:** [arXiv:2609.14657](https://arxiv.org/abs/2609.14657) — *Compositional SVG Generation via VLM-Driven Hierarchical Semantic Parsing* (EMNLP 2026)
- 🌐 **Project page:** https://ku-miil.github.io/semantic-svg-generation
- 🤗 **Benchmark:** [KU-MIIL/Semantic-SVG-Benchmark](https://huggingface.co/datasets/KU-MIIL/Semantic-SVG-Benchmark)

The main contribution of this repo is the **task + benchmark** and a
ready-to-run **evaluation harness**: bring your own image→SVG method, and score
it against the benchmark with one command. A reference pipeline is included
([see below](#reference-pipeline)) but is not required to use the benchmark.

---

## The benchmark

203 evaluation samples (`test` split). Each sample carries:

| file | what |
|------|------|
| `svg/<id>.svg` | ground-truth SVG |
| `svg/<id>.tree.json` | semantic tree — top-level objects, optionally decomposed into parts, each node labelling a set of SVG shape indices |
| `png/<id>.png` | reference render (used as the pixel-fidelity target **and** as the input image your method should vectorize) |

`manifest.json` carries per-sample metadata (`category` ∈ {emoji, icon,
illustration}, `source`, `n_paths`, …). The semantic tree schema is:

```json
{"paths": [0, 1, ...],
 "root": {"children": [
    {"label": "sun", "paths": [0, 1]},
    {"label": "planter", "paths": [2, 3, 4],
     "children": [{"label": "pot", "paths": [2]}, {"label": "plant", "paths": [3, 4]}]}
 ]}}
```
where each integer indexes a drawable shape in document order (`path`, `rect`,
`circle`, `ellipse`, `polygon`, `polyline`, `line`; shapes inside `<defs>`,
`<clipPath>`, `<mask>`, etc. are skipped).

---

## Evaluate your own method

### 1. Fetch the benchmark

```bash
pip install -r requirements.txt
python scripts/prepare_benchmark.py            # → ./benchmark
```

Downloads the SVGs + trees from HuggingFace and renders the reference PNGs. Produces:

```
benchmark/
├── basenames.txt         # the 203 sample ids, one per line
├── manifest.json         # per-sample metadata (category, source, …)
├── svg/<id>.svg          # ground-truth SVG
├── svg/<id>.tree.json    # ground-truth semantic tree
└── png/<id>.png          # reference render / input image
```

### 2. Produce predictions

Run **your** image→SVG method on each `benchmark/png/<id>.png` and save the
output as `<pred_dir>/<id>.svg`, keeping the id as the filename stem:

```
my_preds/
├── emoji_161.svg
├── openclipart__189889.svg
└── ...                    # one .svg per id in benchmark/basenames.txt
```

Missing predictions are reported and skipped; unparseable SVGs are scored
against a blank canvas (a fair penalty) rather than dropped.

### 3. Score

```bash
python evaluate.py --pred-dir my_preds --name MyMethod
```

```
================================================================
  MyMethod   (n=203, missing=0)
================================================================
  pixel      MSE(↓) 0.0412    DINO(↑) 0.8123
  grouping   F1(↑) 0.6540   P(↑) 0.7010   R(↑) 0.6122   [DINO-scored]
  anchor     Recall/DINO(↑) 0.7180   Recall/MSE-dist(↓) 0.0930

  per-category F1/MSE by emoji, icon, illustration  ->  my_preds/_eval/summary.json
  per-sample rows                                   ->  my_preds/_eval/per_sample.csv
```

`summary.json` has the overall + per-category breakdown; `per_sample.csv` has
every sample's scores.

**Options**

| flag | default | purpose |
|------|---------|---------|
| `--pred-dir` | (required) | directory of your `<id>.svg` predictions |
| `--gt-root` | `./benchmark` | benchmark root from `prepare_benchmark.py` |
| `--name` | `method` | label for this run in the report |
| `--metrics` | `all` | subset of `pixel,grouping,anchor` |
| `--no-dino` | off | skip all DINO metrics (no GPU / model download; MSE-side only) |
| `--size` | `256` | render size for the semantic metrics |
| `--out-dir` | `<pred-dir>/_eval` | where to write `summary.json` / `per_sample.csv` |
| `--limit` | — | score only the first N samples (quick check) |

DINO metrics load DINOv2 on the GPU on first use. For a fast, CPU-only sanity
check use `--no-dino`.

---

## Metrics

All three come straight from the paper.

- **Pixel fidelity** — `MSE` (↓) and `DINO` (↑, DINOv2 cosine similarity) between
  the render of your SVG and the reference PNG. Measures *how the image looks*.

- **Semantic grouping** — `Precision` / `Recall` / `F1` (↑). Your SVG's grouping,
  read from its `<g>` nesting, is matched object-by-object against the GT
  semantic tree (N×M union-bbox matching). Measures *whether the SVG is organised
  into the right objects*. A flat SVG (no `<g>`) is treated as a single group.
  Judge = MSE (which groups match), score = MSE **and** DINO (how well they
  match); the headline F1 is DINO-scored (MSE-scored when `--no-dino`).

- **Anchor recall** — grouping-free per-object recall: for each GT object, greedily
  select the predicted shapes that best reconstruct it and score the match
  (`Recall/DINO` ↑, `Recall/MSE-dist` ↓). Needs no `<g>` structure, so it is
  defined for **any** SVG, including flat ones.

If your method emits grouped SVGs (`<g>` per object), all three metrics are
meaningful. If it emits flat SVGs, rely on pixel fidelity + anchor recall.

---

## Reference pipeline

`pipeline_controller.py` is the paper's **training-free** reference method: a VLM
(Gemini) decomposes the image into a hierarchy of labelled parts, SAM 3 segments
each part, vtracer vectorizes the per-part layers, and a compositor stitches them
into one grouped SVG. It is a workflow, not a trained model — included for
reproducibility, not required to use the benchmark.

```bash
export GEMINI_API_KEY=...            # required for the VLM decompose calls
python pipeline_controller.py benchmark/png/ --out-dir runs/mine
python pipeline_controller.py --help
```

Collect its per-image `merged_stacked.svg` into a `<pred_dir>/<id>.svg` layout,
then score with `evaluate.py` as above. Requires a CUDA GPU (SAM 3) and a Gemini
API key; `--amodal-inpaint` additionally loads FLUX.1-Fill-dev (~15 GB).

## Benchmark annotation tool

`annotator/` is the self-contained web UI used to build the benchmark
(hierarchical semantic labelling of SVGs). See `annotator/README.md`.

## License

Apache-2.0. Please cite the paper if you use the benchmark (BibTeX on the
[project page](https://ku-miil.github.io/semantic-svg-generation)).
