"""
validate_accuracy_delta.py
===========================

Owner: Abdelrahman (scripts/ — Metrics & Analysis)

Purpose
-------
Statistical validation pass on top of accuracy_delta.py's output.

Per-class accuracy figures are proportions estimated from a finite sample
of images. With small samples (e.g. n=20), a single misclassified image
swings "accuracy" by 5 percentage points — so a reported "delta" of, say,
20% can be mostly sampling noise rather than a real effect. This script
attaches a 95% Wilson confidence interval to both the clean and modified
accuracy for every class/method row, flags classes with too few images to
trust, and flags whether the clean-vs-modified difference is likely to be
a real (statistically meaningful) effect or could plausibly be noise.

Why Wilson, not the normal/Wald interval
-----------------------------------------
The standard "p +/- z*sqrt(p(1-p)/n)" interval breaks down badly for small
n and/or accuracy near 0 or 1 (which is common here — some classes hit
100% or 0% accuracy). The Wilson score interval stays well-behaved in
exactly these conditions and is the standard recommendation for small-n
proportion estimation, so it's what we use here.

Inputs
------
    outputs/accuracy_delta.csv                 (from accuracy_delta.py)
    outputs/inference/inference_clean.csv       (to recover n_clean)
    outputs/inference/inference_<method>.csv    (to recover n_modified)

Output
------
    outputs/accuracy_delta_validated.csv
        all original accuracy_delta.csv columns, plus:
        n_clean, n_modified,
        ci_clean_low, ci_clean_high,
        ci_modified_low, ci_modified_high,
        low_n_flag        (True if n_clean < 30 or n_modified < 30)
        ci_overlap        (True if the two 95% CIs overlap)
        delta_significant (True if ci_overlap is False AND low_n_flag is False
                            -- i.e. we have enough data AND the CIs don't
                            overlap, so the delta is unlikely to be noise)

Usage
-----
    python scripts/validate_accuracy_delta.py
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

METHODS = ["blur", "replace", "crop"]
MIN_N = 30          # below this, flag as too small for reliable conclusions
Z_95 = 1.959963985   # z-score for a 95% confidence interval

TRUE_LABEL_ALIASES = ["true_class", "label", "ground_truth", "class", "true_label"]


def _find_column(df: pd.DataFrame, aliases: list[str], required_for: str) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    raise ValueError(
        f"Could not find a column for '{required_for}'. "
        f"Looked for any of {aliases} in columns {list(df.columns)}"
    )


def wilson_interval(p_hat: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion.

    Returns (low, high), both clamped to [0, 1].
    Returns (nan, nan) if n == 0.
    """
    if n == 0 or pd.isna(p_hat):
        return float("nan"), float("nan")

    z2 = z * z
    denom = 1 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z2 / (4 * n * n)))

    low = (center - margin) / denom
    high = (center + margin) / denom
    return max(0.0, low), min(1.0, high)


def load_class_counts(path: Path) -> dict:
    """Count images per ground-truth class in an inference CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    df = pd.read_csv(path)
    true_col = _find_column(df, TRUE_LABEL_ALIASES, "ground-truth label")
    df[true_col] = df[true_col].astype(str).str.strip()
    return df.groupby(true_col).size().to_dict()


def intervals_overlap(low1, high1, low2, high2) -> bool:
    if any(pd.isna(v) for v in (low1, high1, low2, high2)):
        return True  # unknown -> treat conservatively as "can't rule out overlap"
    return low1 <= high2 and low2 <= high1


def main():
    parser = argparse.ArgumentParser(
        description="Attach Wilson 95% CIs and low-n flags to accuracy_delta.csv"
    )
    parser.add_argument("--delta-csv", type=Path, default=Path("outputs/accuracy_delta.csv"))
    parser.add_argument("--inference-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not args.delta_csv.exists():
        print(
            f"ERROR: {args.delta_csv} not found. Run scripts/accuracy_delta.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    delta_df = pd.read_csv(args.delta_csv)

    try:
        n_clean_by_class = load_class_counts(args.inference_dir / "inference_clean.csv")
        n_modified_by_class = {
            method: load_class_counts(args.inference_dir / f"inference_{method}.csv")
            for method in METHODS
        }
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for _, row in delta_df.iterrows():
        cls = row["class"]
        method = row["method"]

        n_clean = n_clean_by_class.get(cls, 0)
        n_modified = n_modified_by_class.get(method, {}).get(cls, 0)

        ci_clean_low, ci_clean_high = wilson_interval(row["accuracy_clean"], n_clean)
        ci_mod_low, ci_mod_high = wilson_interval(row["accuracy_modified"], n_modified)

        low_n_flag = (n_clean < MIN_N) or (n_modified < MIN_N)
        overlap = intervals_overlap(ci_clean_low, ci_clean_high, ci_mod_low, ci_mod_high)
        delta_significant = (not overlap) and (not low_n_flag)

        rows.append(
            {
                **row.to_dict(),
                "n_clean": n_clean,
                "n_modified": n_modified,
                "ci_clean_low": round(ci_clean_low, 4) if not pd.isna(ci_clean_low) else ci_clean_low,
                "ci_clean_high": round(ci_clean_high, 4) if not pd.isna(ci_clean_high) else ci_clean_high,
                "ci_modified_low": round(ci_mod_low, 4) if not pd.isna(ci_mod_low) else ci_mod_low,
                "ci_modified_high": round(ci_mod_high, 4) if not pd.isna(ci_mod_high) else ci_mod_high,
                "low_n_flag": low_n_flag,
                "ci_overlap": overlap,
                "delta_significant": delta_significant,
            }
        )

    validated_df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "accuracy_delta_validated.csv"
    validated_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(validated_df)} rows)")

    # Console summary
    n_low_n = validated_df["low_n_flag"].sum()
    n_sig = validated_df["delta_significant"].sum()
    print(f"\n{n_low_n}/{len(validated_df)} rows flagged low_n_flag (n < {MIN_N})")
    print(f"{n_sig}/{len(validated_df)} rows have a statistically significant delta "
          f"(non-overlapping 95% CIs AND n >= {MIN_N} on both sides)")

    return validated_df


if __name__ == "__main__":
    main()
