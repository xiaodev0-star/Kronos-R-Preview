"""Exp 04-B refresh: epoch-wise AdamW versus Muon+AdamW.

The experiment requires the refreshed Exp 04-A selection to remain CE.  Both
optimizer arms inherit the Exp 02 tokenizer and Exp 03 architecture, and differ
only in optimizer family plus Muon's optimizer-specific learning rate.
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
        "name": "adamw",
        "description": "AdamW(lr=3e-4)",
        "train": {"optimizer": "adamw"},
    },
    {
        "name": "muon",
        "description": "Muon(lr=0.02) for 2D weights plus AdamW",
        "train": {"optimizer": "muon", "lr_muon": 0.02},
    },
]

FIXED_RECIPE = {
    "loss": "ce",
    "gamma": 0.0,
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

EXP04A_SELECTION = (
    SCRIPT_PATH.parents[1] / "a-loss-ablation" / "run_seed42" / "selection.json"
)


def main() -> int:
    return run_ablation(
        wrapper_path=SCRIPT_PATH,
        experiment_key="exp04b",
        experiment_label="Exp 04-B optimizer ablation",
        arm_definitions=ARMS,
        fixed_recipe=FIXED_RECIPE,
        default_output_root=SCRIPT_PATH.parent / "run_seed42",
        selected_arm="adamw",
        selection_rationale=(
            "AdamW is the preregistered downstream working arm: the historical "
            "controlled comparison found substantially healthier predictions "
            "than Muon(lr=0.02). This refresh tests the same optimizer contrast "
            "under the new upstream dependencies and epoch-wise protocol."
        ),
        required_selections={"exp04a_loss": (EXP04A_SELECTION, "ce")},
        default_epochs=30,
    )


if __name__ == "__main__":
    raise SystemExit(main())
