"""
CLIP zero-shot classification on the 8-class truck subset.

Owner: Person A (ML / CLIP Lead)
Day: 1 (smoke test) -> Day 2 (full original set) -> Day 3 (logo variant)

Run: python src/clip_inference/run_zero_shot.py --dataset original
     python src/clip_inference/run_zero_shot.py --dataset logo_variant

Output: outputs/predictions/<dataset>_predictions.parquet
        columns: image_path, true_class, predicted_class, confidence,
                 embedding (stored separately as .npy for size reasons)
"""
import argparse
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tracking import start_run, log_params, log_metrics, log_artifact, log_config_snapshot

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_text_prompts(config, device, model):
    """
    Build the zero-shot classifier weight matrix: one embedding per class,
    averaged across prompt templates (standard CLIP zero-shot recipe).
    """
    class_names = list(config["classes"].keys())
    prompt_names = [config["classes"][c]["prompt_name"] for c in class_names]
    templates = config["model"]["prompt_templates"]

    text_embeddings = []
    with torch.no_grad():
        for prompt_name in prompt_names:
            texts = [t.format(class_name=prompt_name) for t in templates]
            tokens = clip.tokenize(texts).to(device)
            embeds = model.encode_text(tokens)
            embeds /= embeds.norm(dim=-1, keepdim=True)
            text_embeddings.append(embeds.mean(dim=0))

    text_embeddings = torch.stack(text_embeddings)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    return class_names, text_embeddings


def run_inference(dataset_split: str, config: dict, backbone: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone or config["model"]["primary_backbone"]

    with start_run(f"clip_inference_{dataset_split}", tags={"owner": "A", "dataset": dataset_split}):
        log_config_snapshot(config)
        log_params({"backbone": backbone, "device": device, "dataset_split": dataset_split})

        print(f"Loading CLIP {backbone} on {device}...")
        model, preprocess = clip.load(backbone, device=device)
        model.eval()

        class_names, text_embeddings = build_text_prompts(config, device, model)

        data_key = "processed_data" if dataset_split == "original" else "logo_variant_data"
        data_dir = Path(config["paths"][data_key])

        records = []
        embeddings_list = []

        with torch.no_grad():
            for class_dir in sorted(data_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                true_class = class_dir.name
                for img_path in sorted(class_dir.glob("*.jpg")):
                    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    image_embed = model.encode_image(image)
                    image_embed /= image_embed.norm(dim=-1, keepdim=True)

                    similarities = (image_embed @ text_embeddings.T).softmax(dim=-1)
                    confidence, pred_idx = similarities.max(dim=-1)
                    predicted_class = class_names[pred_idx.item()]

                    records.append({
                        "image_path": str(img_path),
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "confidence": confidence.item(),
                        "correct": predicted_class == true_class,
                    })
                    embeddings_list.append(image_embed.cpu().numpy().squeeze())

        df = pd.DataFrame(records)
        out_dir = Path(config["paths"]["predictions_out"])
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_path = out_dir / f"{dataset_split}_predictions.parquet"
        embed_path = out_dir / f"{dataset_split}_embeddings.npy"
        df.to_parquet(pred_path)
        np.save(embed_path, np.stack(embeddings_list))

        acc = df["correct"].mean()
        log_metrics({"overall_accuracy": acc, "n_images": len(df)})
        for class_name, group in df.groupby("true_class"):
            log_metrics({f"accuracy_{class_name}": group["correct"].mean()})

        log_artifact(str(pred_path))

        print(f"[{dataset_split}] n={len(df)}  overall accuracy={acc:.3f}")
        print(df.groupby("true_class")["correct"].mean())
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original", "logo_variant"], default="original")
    parser.add_argument("--backbone", default=None, help="Override config primary_backbone")
    args = parser.parse_args()

    config = load_config()
    run_inference(args.dataset, config, backbone=args.backbone)
