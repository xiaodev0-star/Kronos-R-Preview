"""Exp 04-A: epoch-wise AdamW versus Muon+AdamW.

Cross-entropy is fixed (decided without a separate loss ablation).  Both
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

from ablation_common import default_study_roots, run_ablation


DEFAULT_WEIGHTS_ROOT, DEFAULT_RESULTS_ROOT = default_study_roots(
    "04a-optimizer-ablation", seed=42
)


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


def main() -> int:
    return run_ablation(
        wrapper_path=SCRIPT_PATH,
        experiment_key="exp04a",
        experiment_label="Exp 04-A optimizer ablation",
        arm_definitions=ARMS,
        fixed_recipe=FIXED_RECIPE,
        default_weights_root=DEFAULT_WEIGHTS_ROOT,
        default_results_root=DEFAULT_RESULTS_ROOT,
        selected_arm="muon",
        selection_rationale=(
            "Pending Muon arm execution. Selection is decided by the "
            "token-balance ranking (coarse codebook balance, support F1, "
            "token JSD, effective token alignment, collapse, unique tokens) "
            "under downstream guardrails; DA/RankIC/MAPE/AmpRatio only gate, "
            "never rank. AdamW arm reuses the Exp 03 depth6 trajectory "
            "(50 epochs, same recipe). Muon arm must run 50 epochs for a "
            "like-for-like paired comparison; the pre-registered muon "
            "preference stands only if it leads on token-balance and clears "
            "guardrails against the AdamW reference."
        ),
        default_epochs=50,
    )


if __name__ == "__main__":
    raise SystemExit(main())
