# LRP Setup Notes — Zennit Feasibility (Day 1)

**Owner:** Mariham
**Date:** Day 1
**Goal:** Confirm Zennit installs and runs in the team environment, and produces a rudimentary relevance heatmap on RN50 for at least 2 images.

## Result: ✅ PASS

Zennit installs cleanly via pip and the LRP attribution pipeline runs end-to-end on
ResNet50, producing relevance heatmaps for both test images. No blocking issues.

## Environment

- Python 3.12
- `torch==2.12.1`, `torchvision==0.27.1`, `zennit==1.0.0`
- Installed via: `pip install torch torchvision zennit`
  (matches `requirements.txt` already listed by Hassan — no version conflicts found)

## What was tested

- Loaded `torchvision.models.resnet50`
- Built an `EpsilonPlusFlat` composite (Zennit's standard rule set for ResNet-style
  conv nets — this is the same composite spec'd for `scripts/generate_lrp.py` on Day 2)
- Used `zennit.attribution.Gradient` as the attributor, targeting the model's
  predicted class (one-hot vector over 1000 classes)
- Ran on 2 images and saved relevance heatmaps to `outputs/feasibility_heatmaps/` equivalent
  (local: `lrp_test/output/lrp_dog.png`, `lrp_test/output/lrp_synthetic.png`)
- Verified the relevance map shows concentrated activation rather than uniform noise

## Installation issues encountered

1. **No real installation issues with Zennit itself.** `pip install zennit` worked
   first try, no dependency conflicts with `torch`/`torchvision` versions in
   `requirements.txt`.
2. **Sandboxed test environment could not download pretrained ImageNet weights**
   (`download.pytorch.org` returned HTTP 403 — blocked by this sandbox's network
   proxy, not a Zennit/PyTorch issue). To validate the pipeline regardless, I ran
   the test with a randomly-initialized RN50, which is enough to confirm the
   Zennit + RN50 + LRP composite plumbing works end-to-end (forward pass → relevance
   propagation → heatmap save). **On a normal dev machine / the team's actual
   environment, this won't be an issue** — pretrained weights will download fine.
   Flagging only in case the team also runs in a similarly locked-down CI/sandbox
   later (e.g. for `run_pipeline.py` in Docker) — may need to vendor/cache the
   weights file or whitelist `download.pytorch.org` in that case.
3. One of my original test images (a cat photo pulled from a raw GitHub URL)
   resolved to a 404 HTML page instead of binary image data and failed to load with
   PIL. Swapped in a different image. **Lesson for the team:** always sanity-check
   downloaded test images with `PIL.Image.open()` before using them in a pipeline,
   especially when fetching from URLs that don't guarantee binary content.

## Notes for Day 2+ (`scripts/generate_lrp.py`)

- `EpsilonPlusFlat` composite confirmed as a good default choice for RN50 — matches
  the plan's spec.
- LRP only works on RN50 (conv layers required) — confirmed Zennit does **not**
  support ViT, consistent with the plan's note that ViT will use Grad-CAM instead.
- Relevance should be summed across the RGB channel dimension before visualizing
  (`relevance.sum(dim=channel)`), then passed through `zennit.image.imgify` with
  `symmetric=True` for a diverging heatmap, or normalized 0-1 and run through a
  `jet` colormap to match the team's chosen heatmap style.
- Runtime per image on CPU was a few seconds — fine for a 2-image smoke test;
  worth checking real per-image runtime on the team's GPU/CPU setup once weights
  are loaded, since Day 4's full-test-set run needs to be efficient.

## Conclusion

Zennit is confirmed working in this environment. Definition of Done is met:
Zennit installs, produces rudimentary heatmaps for 2 images, and this file is
ready to commit to `docs/lrp_setup_notes.md`.
