"""
Build the synthetic "logo-inserted" variant of the test set.

This is the controlled experiment at the heart of the project: same images,
with a consistent logo overlay added, so any accuracy drop can be attributed
to the logo rather than to image-to-image variation.

Owner: Person C (Data Engineer)
Day: 2

Run: python src/data_prep/insert_logo.py
Output: data/logo_variant/<class_name>/<image>.jpg  (logo-inserted copies)
        data/annotations/logo_boxes.json             (bounding box per image)
"""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def overlay_logo(image: np.ndarray, logo: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """
    Paste `logo` onto `image` at a consistent relative size/position/opacity,
    as defined in config.yaml -> logo_insertion. Returns the modified image
    and the bounding box {x, y, w, h} in pixel coordinates, for later use by
    the heatmap energy-ratio scorer in src/quant/spurious_score.py.
    """
    img_h, img_w = image.shape[:2]
    logo_w = int(img_w * cfg["relative_size"])
    logo_h = int(logo.shape[0] * (logo_w / logo.shape[1]))
    logo_resized = cv2.resize(logo, (logo_w, logo_h))

    margin = int(0.03 * img_w)
    if cfg["position"] == "lower_left":
        x, y = margin, img_h - logo_h - margin
    elif cfg["position"] == "lower_right":
        x, y = img_w - logo_w - margin, img_h - logo_h - margin
    else:
        raise ValueError(f"Unsupported position: {cfg['position']}")

    out = image.copy()
    roi = out[y:y + logo_h, x:x + logo_w]

    if logo_resized.shape[2] == 4:  # has alpha channel
        alpha = (logo_resized[:, :, 3:4] / 255.0) * cfg["opacity"]
        rgb = logo_resized[:, :, :3]
    else:
        alpha = cfg["opacity"]
        rgb = logo_resized

    blended = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    out[y:y + logo_h, x:x + logo_w] = blended

    bbox = {"x": x, "y": y, "w": logo_w, "h": logo_h}
    return out, bbox


def build_logo_variant_dataset(config):
    cfg = config["logo_insertion"]
    raw_dir = Path(config["paths"]["processed_data"])
    out_dir = Path(config["paths"]["logo_variant_data"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logo_path = Path(cfg["logo_image_path"])
    if not logo_path.exists():
        raise FileNotFoundError(
            f"Synthetic logo image not found at {logo_path}. "
            "Day 2 TODO: source or design a simple text/brand-style logo "
            "PNG with alpha channel and place it here."
        )
    logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)

    all_boxes = {}
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_out = out_dir / class_dir.name
        class_out.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(class_dir.glob("*.jpg")):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  WARNING: could not read {img_path}, skipping")
                continue
            modified, bbox = overlay_logo(image, logo, cfg)
            out_path = class_out / img_path.name
            cv2.imwrite(str(out_path), modified)
            all_boxes[str(out_path)] = bbox

    ann_path = Path(config["paths"]["annotations"]) / "logo_boxes.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump(all_boxes, f, indent=2)

    print(f"Logo-variant dataset built: {len(all_boxes)} images, "
          f"bounding boxes saved to {ann_path}")


if __name__ == "__main__":
    config = load_config()
    build_logo_variant_dataset(config)
