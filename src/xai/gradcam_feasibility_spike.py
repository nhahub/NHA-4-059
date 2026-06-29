"""
DAY 1 CRITICAL PATH — run this before committing to a backbone.

Standard Grad-CAM assumes spatial convolutional feature maps. CLIP's
ViT-B/32 uses patch tokens + attention, not conv feature maps, so Grad-CAM
needs adaptation (e.g. targeting the last transformer block's output
reshaped to a patch grid, or attention-rollout). RN50 is a standard CNN and
works with captum's LayerGradCam out of the box.

This script tries both on a handful of sample images and reports which
backbone is actually producing usable heatmaps. Make the go/no-go call from
this output by end of Day 1 — it blocks Person A's full inference run and
Person D's quantification work if left undecided.

Owner: Person B (XAI / Grad-CAM Lead)
Day: 1

Run: python src/xai/gradcam_feasibility_spike.py
"""
from pathlib import Path

import clip
import torch
import yaml
from captum.attr import LayerGradCam
from PIL import Image

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def try_rn50(sample_image_path: str, device: str):
    """RN50: standard conv net, should just work with LayerGradCam."""
    print("\n--- Trying RN50 ---")
    model, preprocess = clip.load("RN50", device=device)
    model.eval()

    target_layer = model.visual.layer4
    image = preprocess(Image.open(sample_image_path).convert("RGB")).unsqueeze(0).to(device)

    def forward_fn(x):
        return model.encode_image(x)

    try:
        gradcam = LayerGradCam(forward_fn, target_layer)
        attribution = gradcam.attribute(image, target=0)
        print(f"  SUCCESS: attribution shape {attribution.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def try_vit(sample_image_path: str, device: str):
    """
    ViT-B/32: needs the target layer's output reshaped from
    (batch, n_patches+1, dim) into a spatial grid before LayerGradCam's
    upsampling logic will produce a sensible 2D heatmap. This commonly
    requires a small wrapper module — attempt below, but expect this to need
    real debugging time, which is the whole point of running this spike now.
    """
    print("\n--- Trying ViT-B/32 ---")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    try:
        target_layer = model.visual.transformer.resblocks[-1]
    except AttributeError as e:
        print(f"  FAILED to locate target layer: {e}")
        return False

    image = preprocess(Image.open(sample_image_path).convert("RGB")).unsqueeze(0).to(device)

    def forward_fn(x):
        return model.encode_image(x)

    try:
        gradcam = LayerGradCam(forward_fn, target_layer)
        attribution = gradcam.attribute(image, target=0)
        print(f"  Raw attribution shape: {attribution.shape}")
        print("  NOTE: shape will likely be (batch, n_patches+1, dim) not "
              "(batch, channels, H, W) — this needs a reshape/wrapper to "
              "become a usable 2D heatmap. If this is non-trivial within a "
              "few hours, fall back to RN50 as primary and note ViT as a "
              "stretch goal.")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


if __name__ == "__main__":
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_image = "data/raw/garbage_truck/sample_001.jpg"  # TODO: point at a real downloaded image
    if not Path(sample_image).exists():
        print(f"NOTE: {sample_image} doesn't exist yet — point this at any "
              "real image once Person C has downloaded sample data, even "
              "before the full dataset is ready. The spike only needs a "
              "handful of images.")
    else:
        rn50_ok = try_rn50(sample_image, device)
        vit_ok = try_vit(sample_image, device)

        print("\n=== GO / NO-GO DECISION ===")
        if vit_ok:
            print("ViT-B/32 Grad-CAM is feasible (pending visual sanity check "
                  "of the heatmap). Proceed with ViT-B/32 as primary, RN50 fallback.")
        elif rn50_ok:
            print("ViT-B/32 Grad-CAM is NOT straightforward. Recommend: "
                  "switch RN50 to primary backbone for the whole pipeline, "
                  "keep ViT-B/32 zero-shot numbers as a secondary reported "
                  "metric without heatmaps, or as a stretch goal if time allows.")
            print("ACTION: update config.yaml model.primary_backbone to 'RN50' "
                  "and tell Person A before they run full inference.")
        else:
            print("Neither worked cleanly — escalate to the team immediately, "
                  "this blocks the whole XAI pipeline.")
