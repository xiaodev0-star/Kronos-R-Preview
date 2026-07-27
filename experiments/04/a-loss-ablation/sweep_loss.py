"""Exp 04-A refresh: epoch-wise Focal(gamma=4) versus Cross-Entropy.

Both arms inherit Exp 02's 64x192 @ 9+7 tokenizer and Exp 03's selected GPT
architecture.  The only arm-level variables are loss type and focal gamma.
Every epoch is evaluated on four validation windows; the holdout stays sealed.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
EXP04_DIR = SCRIPT_PATH.parents[1]
if str(EXP04_DIR) not in sys.path:
    sys.path.insert(0, str(EXP04_DIR))

from ablation_common import run_ablation


ARMS = [
    {
        "name": "focal",
        "description": "Focal loss with gamma=4",
        "train": {"loss": "focal", "gamma": 4.0},
    },
    {
        "name": "ce",
        "description": "Standard cross-entropy",
        "train": {"loss": "ce", "gamma": 0.0},
    },
]

FIXED_RECIPE = {
    "optimizer": "adamw",
    "lr": 3e-4,
    "dropout": 0.10,
    "weight_decay": 0.01,
    "fine_weight": 0.30,
    "heteroscedastic": True,
    "het_weight": 0.10,
    "label_smoothing": 0.0,
    "entropy_alpha": 0.0,
    "warmup_ratio": 0.05,
}


def main() -> int:
    return run_ablation(
        wrapper_path=SCRIPT_PATH,
        experiment_key="exp04a",
        experiment_label="Exp 04-A loss ablation",
        arm_definitions=ARMS,
        fixed_recipe=FIXED_RECIPE,
        default_output_root=SCRIPT_PATH.parent / "run_seed42",
        selected_arm="ce",
        selection_rationale=(
            "CE is the preregistered downstream working arm: the historical "
            "controlled comparison selected it decisively, while this refresh "
            "rechecks that conclusion under the new tokenizer, Exp 03-selected "
            "architecture, constant accumulation, and mature-window evaluation. The "
            "published envelopes and health count remain the authoritative result."
        ),
        default_epochs=30,
    )


if __name__ == "__main__":
    raise SystemExit(main())
