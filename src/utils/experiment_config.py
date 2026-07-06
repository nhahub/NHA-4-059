"""
experiment_config.py
======================

FR-8: "All experiments shall be reproducible via fixed random seeds... The
system shall support saving and reloading experiment configurations."

`ExperimentConfig` captures everything a run of generate_heatmaps.py /
generate_lrp.py needs to be re-run identically: the seed, which
classifier/example set was used, which modification methods, and where
outputs went. `save()`/`load()` round-trip it as JSON so a run can be
reproduced or audited later without re-typing the original CLI flags.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    classifier_path: str
    examples_path: str
    methods: List[str] = field(default_factory=lambda: ["clean", "blur", "replace", "crop"])
    output_dir: str = "outputs"
    seed: int = 42
    model_name: str = "CLIP-RN50"
    xai_method: str = "gradcam"
    notes: str = ""

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def set_seed(seed: int) -> None:
    """Fixes random/numpy/torch seeds so a run with the same
    ExperimentConfig is reproducible (FR-8, NFR-4)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_config(args, config_path: Optional[str]) -> ExperimentConfig:
    """Builds an ExperimentConfig from parsed CLI args, or loads one from
    `config_path` if given (CLI flags in that case are ignored in favour of
    the saved config, so a run can be reproduced exactly)."""
    if config_path:
        return ExperimentConfig.load(config_path)
    return ExperimentConfig(
        classifier_path=str(args.classifier_path),
        examples_path=str(args.examples_path),
        methods=list(args.methods),
        output_dir=str(args.output_dir),
        seed=args.seed,
    )
