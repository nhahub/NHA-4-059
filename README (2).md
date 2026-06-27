# Revealing Hidden Decision Patterns in Machine Learning Models

CLIP zero-shot classification on ImageNet trucks, explained with Grad-CAM, to
test whether the model relies on genuine vehicle features or spurious cues
like logos and text.

**Team:** 6 people · **Timeline:** 10 days · See `PLAN.md` for the full
day-by-day schedule and per-person checklists.

## Project scope (do not drift from this)

- **Model:** CLIP zero-shot classification (`ViT-B/32` primary, `RN50` fallback)
- **Task:** 8-class ImageNet truck classification
- **XAI:** Grad-CAM (required) + LRP via Zennit on RN50 (optional stretch)
- **Output:** Plotly Dash dashboard, deployed to Vercel
- **NOT in scope:** training new models, SimCLR/BarlowTwins comparison (cut
  unless ahead of schedule on Day 4), reproducing the Kauffmann et al. paper's
  BiLRP similarity analysis — that is a different research question and a
  different repo. See `docs/scope_note.md`.

## Repo layout

```
clever-hans-clip/
├── data/
│   ├── raw/              # downloaded ImageNet truck images, untouched
│   ├── processed/        # cleaned/resized images ready for inference
│   ├── logo_variant/      # synthetic logo-inserted copies of processed/
│   └── annotations/       # logo bounding boxes, has_logo labels, splits.json
├── src/
│   ├── tracking.py         # shared MLflow helper — import this, don't call mlflow directly
│   ├── data_prep/          # Person C (incl. make_splits.py — dev/val/test)
│   ├── clip_inference/      # Person A
│   ├── xai/                 # Person B (Grad-CAM, optional LRP)
│   ├── quant/                # Person D (spurious-feature scoring)
│   └── dashboard/             # Person E (Plotly Dash app)
├── outputs/
│   ├── predictions/         # CLIP predictions + embeddings (.parquet/.npy)
│   ├── heatmaps/             # saved Grad-CAM/LRP heatmaps (.npy or .png)
│   └── figures/              # charts for the report
├── tests/
│   ├── unit/                 # fast, no-model-needed tests of single functions
│   └── integration/          # checks data contracts between modules
├── mlruns/                    # MLflow local tracking store (gitignored)
├── scripts/
│   └── run_pipeline.py        # Person A/F — runs the full pipeline end to end
├── notebooks/                  # exploratory work, not graded deliverables
├── .github/workflows/ci.yml    # runs tests on every push/PR
├── Dockerfile
├── docker-compose.yml           # dashboard + MLflow UI together
├── config.yaml
├── requirements.txt
├── CONTRIBUTING.md              # git/GitHub branch + PR workflow
└── PLAN.md
```

## Validation methodology

Before anyone reports a single accuracy number: run `src/data_prep/make_splits.py`
first. It splits each class into `dev` (build/debug freely), `val` (use to
lock decisions like logo-insertion params or backbone choice — by Day 4, not
later), and `test` (touch exactly once, for the numbers in the final report).
See the docstring in that file for the full rationale — skipping this step
is the easiest way to end up with accuracy numbers that don't replicate.

## Experiment tracking (MLflow)

Every script that produces a number imports `src/tracking.py` and wraps its
work in `start_run(...)`. This logs params (backbone, logo-insertion
settings, config snapshot), metrics (accuracy per class, spurious scores),
and artifacts (prediction files) to a local MLflow store automatically.

```bash
mlflow ui --backend-store-uri ./mlruns
# or: docker compose up mlflow-ui
```

This matters because the backbone choice, logo params, and prompt templates
will all change at least once over 10 days — MLflow is what lets you compare
"accuracy with RN50" vs. "accuracy with ViT-B/32" without manually tracking
spreadsheet rows.

## Testing

```bash
pytest tests/unit -v          # fast, no model/network needed
pytest tests/integration -v   # checks module-to-module data contracts
```

Both run automatically on every push/PR via GitHub Actions
(`.github/workflows/ci.yml`). Write a unit test for any pure-logic function
you add (scoring, bounding-box math, metric calculations) — they take
minutes to write and catch real bugs before the bug bash.

## Docker

```bash
docker compose up              # dashboard on :8050, MLflow UI on :5000
docker build -t clever-hans-clip .
docker run --rm clever-hans-clip pytest tests/unit tests/integration -v
```

Docker is for environment reproducibility across the team's machines and
for the bug-bash / final handoff — keep using your local venv for fast
iteration while actively coding.

## Setup (everyone, Day 1)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

GPU is optional but speeds up ViT-B/32 inference significantly. Google Colab
free tier works for the dataset size here (~2GB).

## Day 1 critical path

Before anyone builds on top of CLIP or Grad-CAM, **Person B must resolve the
backbone decision** (see `src/xai/gradcam_feasibility_spike.py`). ViT-B/32
does not have native spatial conv feature maps, so standard Grad-CAM needs
adaptation (attention-rollout-style) or you fall back to RN50, which works
with standard Grad-CAM out of the box. Don't let two days pass before this is
decided — it blocks Person A and Person D's later work.

## Module entry points

| Module | Owner | Entry point |
|---|---|---|
| Data prep | C | `src/data_prep/download_dataset.py` |
| Logo insertion | C | `src/data_prep/insert_logo.py` |
| CLIP inference | A | `src/clip_inference/run_zero_shot.py` |
| Grad-CAM | B | `src/xai/gradcam.py` |
| Backbone spike | B | `src/xai/gradcam_feasibility_spike.py` |
| Metrics | D | `src/quant/metrics.py` |
| Spurious-feature scoring | D | `src/quant/spurious_score.py` |
| Dashboard | E | `src/dashboard/app.py` |
| Full pipeline | A/F | `scripts/run_pipeline.py` |

## The 8 classes

| Class | Synset ID | Spurious feature risk |
|---|---|---|
| Garbage truck | n03417042 | Municipal markings, distinctive shape — **known real-world logo case** |
| Moving van | n03796401 | Company logos on side panels |
| Pickup truck | n03930630 | Brand emblems, common vehicle type |
| Trailer truck | n04467665 | Text, branding on trailer body |
| Beer truck | n02814533 | Prominent brand logos — **primary synthetic logo test class** |
| Fire engine | n03345487 | Distinctive colour, emergency markings |
| Tow truck | n04461696 | Unique mechanical equipment |
| Minivan | n03770679 | Different body style, fewer logos |

Garbage truck images often already contain real logos — use these directly
for the "original/logo-present" condition. Beer truck and other low-logo
classes are good candidates for the synthetic logo-insertion experiment.
