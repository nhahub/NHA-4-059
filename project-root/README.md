# Clever Hans CLIP Project — project-root

Refactored, modular version of `CH_Detection_Pipeline.ipynb`. See root
`README.md` for the project overview and headline accuracy numbers.

## Structure

```
project-root/
├── src/
│   ├── models/
│   │   ├── clip_model.py              # CLIPModel: load CLIP (RN50 or ViT-B/32) + LogisticRegression classifier
│   │   ├── torchvision_resnet_model.py  # SupervisedResNetModel: ImageNet-supervised ResNet-50 adapter (the CH control model)
│   │   ├── ch_mitigation.py           # CHMitigator: find/mask logo-sensitive relu channels (conv backbones)
│   │   └── vit_head_ablation.py       # ViTHeadAblationMitigator: find/ablate logo-sensitive attention heads (ViT)
│   ├── data/
│   │   └── image_modifier.py          # ImageModifier: logo paste + blur/replace/crop variants
│   ├── xai/
│   │   ├── gradcam.py                 # Grad-CAM, targeted at the actual classifier decision function (conv backbones only)
│   │   ├── bilrp.py                   # Joint (Hessian-based) pairwise attribution between two images (architecture-agnostic)
│   │   ├── lrp.py                     # LRP via Zennit + Grad-CAM/LRP spatial correlation (conv backbones only)
│   │   ├── attention_rollout.py       # AttentionRollout: Grad-CAM analog for ViT (no conv feature maps to hook)
│   │   └── vit_attention_utils.py     # Shared attention-capture + manual multi-head-attention reimplementation
│   ├── utils/
│   │   ├── experiment_config.py       # ExperimentConfig save/load + set_seed
│   │   └── experiment_log.py          # ExperimentLogger -> experiment_log.csv
│   └── dashboard/
│       ├── data.py                    # reads PNGs/CSV from results/, degrades gracefully if a file isn't there yet
│       ├── layout.py                  # Overview tab (comparison table + charts) + one image-gallery tab per model
│       └── main.py                    # entry point
├── results/                           # PNGs + three_way_comparison.csv the dashboard reads (kept in sync by the notebook's push-cell)
└── tests/                             # see each file's docstring
```

## Dashboard

A Dash app that browses the pre-generated results in `results/` — it does
not run inference itself. Run it locally:

```bash
python -m src.dashboard.main
```

Then open http://localhost:8050. It has an Overview tab (the three-way
comparison table + two interactive Plotly charts) and one tab per model
that shows whatever PNGs `results/` currently has for it — a tab renders
an informational message instead of erroring if that model's images
haven't been generated yet. Refreshing the page re-reads `results/`, so
re-running the notebook's push-cell and reloading the page is enough to
pick up new results without restarting the dashboard.

Deploy with `gunicorn src.dashboard.main:app` if you need it running
somewhere persistent.

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
   `intercept_`), so the CAM explains the actual prediction. It also now
   raises a clear error instead of silently producing a meaningless heatmap
   if handed a ViT backbone (no late-stage conv feature map to target).

2. **BiLRP outer-product bug (fixed, `src/xai/bilrp.py`)**
   The notebook computed two independent per-image gradient-saliency maps
   and took their outer product — which draws a strong line between any
   two individually high-gradient patches, regardless of whether the model
   relates them at all. `compute_bilrp()` now computes the true mixed
   second derivative (a mini Hessian-vector product: gradient of image-1's
   patch-gradient-energy with respect to image-2's pixels), which is zero
   unless the similarity score genuinely depends on the two patches
   *jointly*. The old method is kept as `compute_bilrp_naive()`, clearly
   labeled, for side-by-side comparison only. Architecture-agnostic — works
   unmodified on RN50, ViT-B/32, and the supervised ResNet-50.

3. **FP16 Grad-CAM/BiLRP instability (fixed, `src/models/clip_model.py`)**
   `clip.load` keeps the model in fp16 on CUDA; backward passes (Grad-CAM,
   and especially BiLRP's double backward) underflow in fp16. Model is now
   forced to fp32 at load time.

4. **LRP implemented (`src/xai/lrp.py`)** — via Zennit, targeting the same
   classifier-logit decision function Grad-CAM does. Known limitation
   documented in the module docstring: CLIP RN50's `AttentionPool2d` isn't
   a chain of hookable submodules, so relevance through it falls back to
   plain gradient, not a true LRP rule (the supervised ResNet-50 has no
   such gap and gets full LRP coverage). Doesn't apply to ViT at all — no
   conv/linear/relu chain for Zennit to walk.

5. **CLIP ViT-B/32 support added (`src/models/clip_model.py`)** —
   `CLIPModel(model_name=...)`, defaulting to `"RN50"` for backward
   compatibility. Added specifically to test whether the logo shortcut
   generalizes across CLIP backbones (it does — ViT showed an even larger
   accuracy drop than RN50).

6. **Supervised ResNet-50 added (`src/models/torchvision_resnet_model.py`)**
   — a plain ImageNet-supervised (no language supervision) backbone, used
   as the control condition: does a model with no incentive to read pasted
   text as a semantic signal show the same shortcut? It shows a
   meaningfully smaller accuracy drop, supporting the hypothesis that the
   shortcut is a CLIP/vision-language artifact rather than a universal
   deep-learning weakness. Built as an adapter matching `CLIPModel`'s
   shape, so `GradCAM`/`BiLRP`/`LRPExplainer`/`CHMitigator` all work
   against it completely unmodified.

7. **Attention Rollout added (`src/xai/attention_rollout.py`)** — the
   Grad-CAM analog for ViT-B/32, since it has no conv feature maps to hook
   Grad-CAM onto. Standard Abnar & Zuidema rollout: captures each
   transformer block's attention weights (via a monkeypatched forward that
   forces `need_weights=True`), fuses each layer with the residual
   connection, and composes across layers. Class-agnostic by construction
   (explains what the [CLS] token attended to, not what drove a specific
   class's score) — a known, accepted property of the base algorithm, not
   a limitation specific to this implementation.

8. **ViT attention-head ablation added (`src/models/vit_head_ablation.py`,
   `src/xai/vit_attention_utils.py`)** — the CH-mitigation analog for ViT.
   `nn.MultiheadAttention` fuses all heads into one call with no supported
   way to zero an individual head's contribution, so
   `vit_attention_utils.manual_mha_forward` reimplements multi-head
   attention from public, stable attributes (`in_proj_weight`/`in_proj_bias`/
   `out_proj`), verified against real `nn.MultiheadAttention` output by a
   numeric parity test before being trusted for anything.

9. **Experiment config + logging (`src/utils/`)** — `ExperimentConfig`
   (save/load as JSON) and `set_seed()`, `ExperimentLogger` (appends one
   row per run to `experiment_log.csv`). Used directly by the notebook's
   Section 5E.

10. **Tests**: `test_clip_model.py`, `test_gradcam.py`, `test_image_modifier.py`,
    `test_integration_pipeline.py`, `test_experiment_config.py`,
    `test_attention_rollout.py`, `test_vit_attention_utils.py`,
    `test_torchvision_resnet_model.py`, `test_vit_head_ablation.py`.
    Model/GPU-dependent tests use `pytest.importorskip` + a try/except skip
    around CLIP weight loading, so they skip cleanly without torch/network/
    GPU and actually run in the team's Colab/dev environment.

## Data

The dataset is stored as pickled examples on Google Drive, loaded directly
by Section 0 of `CH_Detection_Pipeline.ipynb` — this repo doesn't vendor
the raw images.

- 8 classes: `garbage_truck`, `moving_van`, `pickup`, `trailer_truck`,
  `police_van`, `fire_engine`, `tow_truck`, `minivan` (see
  `src/models/clip_model.py::TRUCK_CLASSES` for the ImageNet class-index
  mapping)
- Train: 10,133 images, Test: 400 images
- Source: HuggingFace `ILSVRC/imagenet-1k`, pre-extracted into
  `truck_samples_train_FULL.pkl` / `truck_samples_test_FULL.pkl`

## Known caveat carried over from review

`CHMitigator`/`ViTHeadAblationMitigator` mask filters/heads identified by
clean-vs-logo activation diff. Masking can recover accuracy for reasons
that aren't necessarily "we removed the logo-detecting unit" — it can also
act as noise/regularization. Treat "CH Mitigation" accuracy numbers as
suggestive rather than proof of causal shortcut removal, and note this was
directly observed in practice: the supervised ResNet-50's own mitigation
attempt did **not** recover accuracy (both clean and logo scores dropped
together), consistent with there being no strong shortcut in that model to
specifically remove in the first place — unlike CLIP RN50 and ViT-B/32,
where mitigation cleanly recovered logo-image accuracy at low-to-no cost
to clean-image accuracy.

## Requires

See `requirements.txt`. The XAI/model modules need a GPU-equipped
environment (e.g. Colab T4) to run at reasonable speed, matching the
original notebook's requirements.
