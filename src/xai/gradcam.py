"""
Grad-CAM heatmap generation, run across the full dataset once the backbone
decision from gradcam_feasibility_spike.py is made.

Owner: Person B (XAI / Grad-CAM Lead)
Day: 2 (build) -> Day 3 (run on original set) -> Day 4 (run on logo variant)

Run: python src/xai/gradcam.py --dataset original
     python src/xai/gradcam.py --dataset logo_variant

Output: outputs/heatmaps/<dataset>/<class_name>/<image_name>.npy
        (raw heatmap arrays, same H x W as the input image after resize,
        consumed by src/quant/spurious_score.py and the dashboard)
"""
import argparse
from pathlib import Path

import clip
import numpy as np
import torch
import yaml
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
from PIL import Image

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_target_layer(model, backbone: str):
    if backbone.startswith("RN"):
        return model.visual.layer4
    elif backbone.startswith("ViT"):
        # TODO (Person B): confirm this based on feasibility spike results.
        # May need a wrapper module that reshapes (batch, n_patches+1, dim)
        # into (batch, dim, grid_h, grid_w) before LayerGradCam will produce
        # a sensible spatial heatmap.
        return model.visual.transformer.resblocks[-1]
    else:
        raise ValueError(f"Unknown backbone family: {backbone}")


def generate_heatmap(model, target_layer, image_tensor, target_class_idx):
    def forward_fn(x):
        return model.encode_image(x)

    gradcam = LayerGradCam(forward_fn, target_layer)
    attribution = gradcam.attribute(image_tensor, target=target_class_idx)
    upsampled = LayerGradCam.interpolate(attribution, image_tensor.shape[-2:])
    return upsampled.squeeze().detach().cpu().numpy()


def run_gradcam_pass(dataset_split: str, config: dict, backbone: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone or config["model"]["primary_backbone"]
    model, preprocess = clip.load(backbone, device=device)
    model.eval()

    target_layer = get_target_layer(model, backbone)

    data_key = "processed_data" if dataset_split == "original" else "logo_variant_data"
    data_dir = Path(config["paths"][data_key])
    out_dir = Path(config["paths"]["heatmaps_out"]) / dataset_split
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(config["classes"].keys())

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        true_class = class_dir.name
        target_idx = class_names.index(true_class)
        class_out = out_dir / true_class
        class_out.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(class_dir.glob("*.jpg")):
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            heatmap = generate_heatmap(model, target_layer, image, target_idx)
            np.save(class_out / f"{img_path.stem}.npy", heatmap)

        print(f"  {true_class}: heatmaps saved to {class_out}")

    print(f"Grad-CAM pass complete for '{dataset_split}' using {backbone}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original", "logo_variant"], default="original")
    parser.add_argument("--backbone", default=None)
    args = parser.parse_args()

    config = load_config()
    run_gradcam_pass(args.dataset, config, backbone=args.backbone)
