"""
Runs the full pipeline end to end, from raw data to dashboard-ready outputs.
This is what gets run during the bug bash (Day 7) and what the report's
methodology section should describe.

Owner: Person A / Person F (Integration)
Day: 6 (build) -> Day 7 (bug bash target)

Run: python scripts/run_pipeline.py
     python scripts/run_pipeline.py --skip-data-prep   # if data already built
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(description: str, cmd: list[str]):
    print(f"\n{'=' * 60}\n{description}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nFAILED at step: {description}")
        sys.exit(result.returncode)


def main(skip_data_prep: bool, skip_lrp: bool):
    if not skip_data_prep:
        run_step("1/7 Downloading dataset", [sys.executable, "src/data_prep/download_dataset.py"])
        run_step("2/7 Building logo-variant dataset", [sys.executable, "src/data_prep/insert_logo.py"])

    run_step("3/7 CLIP inference — original set", [sys.executable, "src/clip_inference/run_zero_shot.py", "--dataset", "original"])
    run_step("4/7 CLIP inference — logo variant", [sys.executable, "src/clip_inference/run_zero_shot.py", "--dataset", "logo_variant"])

    run_step("5/7 Grad-CAM — original set", [sys.executable, "src/xai/gradcam.py", "--dataset", "original"])
    run_step("5/7 Grad-CAM — logo variant", [sys.executable, "src/xai/gradcam.py", "--dataset", "logo_variant"])

    run_step("6/7 Metrics + accuracy delta", [sys.executable, "src/quant/metrics.py", "--dataset", "both"])
    run_step("6/7 Spurious-feature scoring", [sys.executable, "src/quant/spurious_score.py"])

    print(f"\n{'=' * 60}\nPipeline complete. Launch the dashboard with:\n"
          f"  python src/dashboard/app.py\n{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--skip-lrp", action="store_true", default=True,
                         help="LRP is a Day 5 stretch goal, off by default")
    args = parser.parse_args()
    main(skip_data_prep=args.skip_data_prep, skip_lrp=args.skip_lrp)
