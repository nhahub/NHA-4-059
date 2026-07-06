# Clever Hans CLIP Project — project-root

Refactored, modular version of `CH_Detection_Pipeline.ipynb`. See root
`README.md` for the project overview and headline accuracy numbers.

## Structure

```
project-root/
├── src/
│   ├── models/
│   │   ├── clip_model.py       # CLIPModel: load CLIP RN50 + LogisticRegression classifier
│   │   └── ch_mitigation.py    # CHMitigator: find/mask logo-sensitive relu3 filters
│   ├── data/
│   │   └── image_modifier.py   # ImageModifier: logo paste + blur/replace/crop variants
│   ├── xai/
│   │   ├── gradcam.py          # Grad-CAM, targeted at the actual classifier decision function
│   │   └── bilrp.py            # Joint (Hessian-based) pairwise attribution between two images
│   └── dashboard/
│       ├── layout.py           # dbc layout (unchanged, just fixed to a relative import)
│       ├── components.py       # reusable dbc components (unchanged)
│       ├── callbacks.py        # data loading + plotting callbacks (unchanged)
│       └── main.py             # NEW entry point: wires layout + callbacks + runs the app
├── scripts/
│   ├── run_inference.py        # NEW: writes inference_{clean,blur,replace,crop}.csv
│   ├── compute_metrics.py      # fixed: stray ``` fences removed
│   ├── accuracy_delta.py       # unchanged, already solid
│   └── validate_accuracy_delta.py  # unchanged, already solid (Wilson CIs)
├── tests/
│   ├── test_compute_metrics.py # fixed: stray ``` fences removed
│   └── test_image_modifier.py  # NEW: covers logo/blur/replace/crop variants
```

## Latest round of fixes (this session)

1. **FP16 Grad-CAM/BiLRP instability (fixed, `src/models/clip_model.py`)** —
   `clip.load` keeps the model in fp16 on CUDA; backward passes (Grad-CAM,
   and especially BiLRP's double backward) underflow in fp16. Model is now
   forced to fp32 at load time.
2. **Grad-CAM never actually ran end-to-end (fixed, `scripts/generate_heatmaps.py`,
   `src/xai/gradcam.py`)** — added `overlay_heatmap()` (jet colormap
   compositing) and `attention_alignment_score()` (FR-5), and a script that
   runs Grad-CAM over clean/blur/replace/crop, saves PNGs to
   `outputs/heatmaps/{method}/{class}/{stem}.png`, and writes
   `outputs/metrics/energy_ratio_all.csv` + `spurious_flags.csv` — the exact
   paths/schema `src/dashboard/callbacks.py` already expected.
3. **LRP implemented (`src/xai/lrp.py`, `scripts/generate_lrp.py`)** — via
   Zennit, targeting the same classifier-logit decision function Grad-CAM
   does (see `_ClassifierHead` wrapper). Computes Grad-CAM vs LRP spatial
   correlation (FR-7). Known limitation documented in the module docstring:
   CLIP RN50's `AttentionPool2d` isn't a chain of hookable submodules, so
   relevance through it falls back to plain gradient, not a true LRP rule.
4. **Pipeline column-name/path mismatches (fixed)** — `compute_metrics.py`
   expected `true_label`/`predicted_label` and wrote capitalized columns to
   a caller-chosen dir; `run_inference.py` actually writes
   `true_class`/`predicted_class`, and the dashboard reads lowercase columns
   from `outputs/metrics/`. Rewrote `compute_metrics.py` to match both ends
   (also now writes `confusion_matrix.csv` and `summary_metrics.csv`, which
   the dashboard needs and nothing produced before). `accuracy_delta.py` /
   `validate_accuracy_delta.py` default output dirs changed to
   `outputs/metrics` to match.
5. **Dockerfile** — `CMD` was `python -m http.server` (didn't launch the
   app at all). Now runs `gunicorn ... src.dashboard.main:app`.
6. **Data schema (`scripts/build_dataset_registry.py`)** — builds
   `data/metadata/dataset_registry.csv` and
   `data/processed/{train,val,test}.csv` per SAD §6.2, from the
   `data/splits.json` and `data/logo_boxes.json` that already exist.
   `label_id` is a synthetic 0..7 index (sorted class name), not an
   ImageNet synset id — `splits.json`'s classes don't line up 1:1 with
   `TRUCK_CLASSES` (e.g. `beer_truck` has no synset entry there).
7. **Experiment config + logging (FR-8, `src/utils/`)** — `ExperimentConfig`
   (save/load as JSON) and `set_seed()` wired into `generate_heatmaps.py`
   and `generate_lrp.py` via `--config`/`--save-config`/`--seed`.
   `ExperimentLogger` appends one row per run to `outputs/experiment_log.csv`
   matching the SAD's schema (kept under `outputs/`, not the SAD's
   `results/`, to match every other output path in this repo).
8. **Tests added**: `test_image_loader.py` (UT-1/2, needed a minimal
   `src/data/image_loader.py::ImageDataLoader` first — didn't exist),
   `test_clip_model.py` (UT-3/4, adapted to the actual features-based
   `predict()` contract — this codebase doesn't implement the SAD's
   zero-shot `predict(image, prompts)`), `test_gradcam.py` (UT-5/6 — UT-6's
   ViT-B/32 backbone is explicitly skipped/documented as unimplemented,
   not faked), `test_integration_pipeline.py` (IT-1..4), `test_dashboard_uat.py`
   (UAT-1/UAT-5 only — **UAT-2/3/4 and IT-5 are not automated because the
   dashboard has no upload/live-inference UI at all**, see that file's
   docstring for the gap), `test_build_dataset_registry.py`,
   `test_experiment_config.py`. Model/GPU-dependent tests use
   `pytest.importorskip` + a try/except skip around CLIP weight loading, so
   they skip cleanly without torch/network/GPU and actually run in the
   team's Colab/dev environment.

## What changed and why

1. **Grad-CAM targeting bug (fixed, `src/xai/gradcam.py`)**
   The notebook's Grad-CAM backpropped from `features[0, features[0].argmax()]`
   — an arbitrary, unlabeled raw-embedding coordinate, not a class score. It
   had nothing to do with the `LogisticRegression` classifier that actually
   produces the reported truck-class predictions, so a heatmap missing the
   logo proved nothing either way, and before/after-logo comparisons could
   silently target two different embedding dimensions. `GradCAM` now
   backprops from the classifier's own decision-function logit for the
   predicted class (`w_c · features + b_c`, using the trained `coef_`/
   `intercept_`), so the CAM explains the actual prediction.

2. **BiLRP outer-product bug (fixed, `src/xai/bilrp.py`)**
   The notebook computed two independent per-image gradient-saliency maps
   and took their outer product — which draws a strong line between any
   two individually high-gradient patches, regardless of whether the model
   relates them at all. `compute_bilrp()` now computes the true mixed
   second derivative (a mini Hessian-vector product: gradient of image-1's
   patch-gradient-energy with respect to image-2's pixels), which is zero
   unless the similarity score genuinely depends on the two patches
   *jointly*. The old method is kept as `compute_bilrp_naive()`, clearly
   labeled, for side-by-side comparison only.

3. **Markdown code fences (fixed)**
   `scripts/compute_metrics.py`, `tests/test_compute_metrics.py`, and the
   old `src/dashboard/app.py` all had literal ` ```python ` / ` ``` ` lines
   at the top and bottom — syntax errors, not style issues. Stripped.

4. **Dashboard entry point (added, `src/dashboard/main.py`)**
   `layout.py` and `callbacks.py` were already implemented and good, but
   nothing wired them together — `app.py` was an unrelated placeholder
   with hard-coded dummy data and `placehold.co` images. It's been deleted
   and replaced with `main.py`, which does
   `app.layout = serve_layout(); register_callbacks(app); app.run()`.
   `layout.py`'s `from components import ...` was also switched to a
   relative import (`from .components import ...`) so it works as part of
   the `src.dashboard` package. `__init__.py` files were added under
   `src/`, `src/models/`, `src/data/`, `src/xai/`, `src/dashboard/`,
   `scripts/`, and `tests/` for the same reason.

5. **Missing inference pipeline (added, `scripts/run_inference.py`)**
   `accuracy_delta.py` and `compute_metrics.py` both expect
   `outputs/inference/inference_{clean,blur,replace,crop}.csv`, but nothing
   ever produced them — the notebook computed accuracy inline with numpy
   arrays and never serialized per-image predictions. `run_inference.py`
   runs `CLIPModel` + `ImageModifier` over a test set for each variant and
   writes the CSVs those two scripts already know how to read.

## Known caveat carried over from review

`CHMitigator` (Section 4 port) still masks filters in `relu3`, which is a
very early layer in CLIP's ResNet stem (part of the 3-conv stem before the
first residual block), not a deep semantic layer. Masking early filters
can recover accuracy for reasons that aren't necessarily "we removed the
logo-detecting neuron" — it could just be adding noise/regularization that
happens to help. This is documented in `ch_mitigation.py`'s docstring;
treat the "CH Mitigation" accuracy numbers as suggestive rather than
proof of causal logo-neuron removal, and consider ablating a random
same-size set of relu3 filters as a control.

## Requires

`torch`, `git+https://github.com/openai/CLIP.git`, `scikit-learn`,
`pandas`, `matplotlib`, `seaborn`, `dash`, `dash-bootstrap-components`,
`Pillow`, `tqdm` — see `requirements.txt`. The XAI/model modules
(`clip_model.py`, `ch_mitigation.py`, `gradcam.py`, `bilrp.py`,
`run_inference.py`) need a GPU-equipped environment (e.g. Colab T4) to run
at reasonable speed, matching the original notebook's requirements.
