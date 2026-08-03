"""Record the reviewed Exp 04-B selection as selection.json (human-reviewed, token-quality-only rule).

Run locally after reviewing EXPERIMENT_REPORT.md:
    python experiments/04/b-hpo/record_selection.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Kronos-R-Preview\server_runs\results\04b-hpo\seed42")
TID = "trial_4c721141ab"

lb = json.loads((ROOT / "leaderboard.json").read_text(encoding="utf-8"))
s = pd.read_csv(ROOT / "combined_trial_summary.csv").set_index("tid")
row = {r["tid"]: r for r in lb["rows"]}
t, b = s.loc[TID], s.loc["baseline"]
lr = row[TID]

recipe = {"dropout": 0.1, "entropy_alpha": 0.0, "fine_weight": 0.3, "gamma": 0.0,
          "het_weight": 0.1, "heteroscedastic": True, "label_smoothing": 0.0,
          "loss": "ce", "lr": 0.0003, "lr_muon": 0.005, "optimizer": "muon",
          "warmup_ratio": 0.05, "weight_decay": 0.01}

selection = {
    "experiment": "Exp 04-B HPO",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "selected": {
        "arm": TID,
        "description": "Muon lr_muon=0.005 (all other hyperparameters identical to the Exp 04-A baseline recipe)",
        "recipe": json.dumps(recipe),
        "n_epochs": 50,
        "healthy_epochs": 0,
        "late_healthy_epochs": 0,
        "mature_window_epochs": [1, 50],
        "window_policy": "all_epochs",
        "maturity_onset_epoch": lr["maturity_onset_epoch"],
        "representative_epoch": lr["representative_epoch"],
        "model_path": lr["model_path"],
        "minimum_val_loss": lr["best_val_loss"],
        "best_da": lr["best_da"],
        "best_da_epoch": lr["best_da_epoch"],
        "mature_median_codebook_balance": float(t["coarse_balance"]),
        "mature_p10_codebook_balance": float(t["coarse_balance_p10"]),
        "mature_median_token_support_f1": float(t["coarse_support_f1"]),
        "mature_median_token_jsd": float(t["coarse_jsd"]),
        "mature_median_effective_token_alignment": float(t["coarse_eff_align"]),
        "mature_median_collapse_rate": float(t["coarse_collapse"]),
        "mature_median_unique_tokens": float(t["coarse_unique"]),
        "mature_median_target_support_recall": lr["median_daily_target_support_recall"],
        "mature_median_fine_codebook_balance": float(t["fine_balance"]),
        "mature_median_fine_token_support_f1": float(t["fine_support_f1"]),
        "mature_median_fine_n_unique_tokens": float(t["fine_unique"]),
        "mature_median_joint_codebook_balance": float(t["joint_balance"]),
        "mature_median_joint_token_support_f1": float(t["joint_support_f1"]),
        "mature_median_joint_n_unique_tokens": float(t["joint_unique"]),
        "coarse_info_bits_h_minus_ce": float(t["coarse_mi_bits"]),
        "fine_info_bits_h_minus_ce": float(t["fine_mi_bits"]),
        "downstream_observed_not_used": {
            "avg_da_per_date": float(t["da"]),
            "avg_daily_rank_ic": float(t["rank_ic"]),
            "avg_mape": float(t["mape"]),
            "avg_ampratio": float(t["ampratio"]),
            "delta_vs_baseline": {
                "avg_da_per_date": float(t["da"] - b["da"]),
                "avg_daily_rank_ic": float(t["rank_ic"] - b["rank_ic"]),
                "avg_mape": float(t["mape"] - b["mape"]),
                "ampratio_log_error": float(t["ampratio_log_error"] - b["ampratio_log_error"]),
            },
        },
        "primary_rank": 1,
    },
    "runner_up": {
        "arm": "trial_8ceb4d575b",
        "note": "lr_muon=0.01; adjacent point on the lr_muon slope (robust region). Best val CE "
                "(coarse H-CE 1.10 bits/token). Reserved as the Continue-PreTrain control arm.",
    },
    "selection_rule": (
        "Token quality only. Primary ranking by target-relative coarse codebook balance (median, p10), "
        "token support F1, token JSD, effective token alignment, collapse rate, and unique tokens over the "
        "full 1..50 epoch trajectory as a single window. Downstream DA/RankIC/MAPE/AmpRatio are deliberately "
        "excluded from selection (recorded for reference only): downstream capability is deferred to the "
        "Continue-PreTrain -> SFT -> PostTrain stages. Val CE (H(target)-CE bits) is kept as the likelihood "
        "sanity check so diversity cannot be bought with noise."
    ),
    "rationale": (
        "trial_4c721141ab (lr_muon=0.005) ranks first on every primary token-quality metric: coarse balance "
        "0.513 vs baseline 0.211 (+143%), p10 0.383 vs 0.162, support F1 0.634 vs 0.161, token JSD 0.366 vs "
        "0.580, median collapse 0.338 vs 0.567, median unique tokens 47 vs 8 (target 97). The OFAT sweep "
        "shows lr_muon is the dominant factor and is monotone over 0.005..0.04; the top-2 trials are "
        "adjacent points on that slope, so the optimum is a robust region, not a lucky draw. Likelihood "
        "check passes: coarse H-CE 1.00 bits/token (baseline 0.27), fine 3.72 bits (baseline 3.46). Raw "
        "token accuracy is explicitly not used: collapse inflates it (baseline 0.106 > selected 0.094 while "
        "covering only 8/97 target tokens). Downstream metrics, though excluded from selection, also "
        "improve (RankIC -0.033 -> +0.039, MAPE 4.29 -> 3.30, AmpRatio 1.80 -> 1.20), so nothing is "
        "sacrificed. lr_muon=0.005 sits at the swept lower edge; probing 0.0025 and the dropout interaction "
        "is folded into the Continue-PreTrain stage plan rather than blocking this selection."
    ),
    "upstream_eligible": True,
    "human_review_recorded": True,
    "holdout_used": False,
    "n_completed_trials": lb["n_completed"],
    "n_planned_trials": 20,
    "missing_trials": [
        "trial_5834942bd7 (warmup_ratio=0.02)",
        "trial_7def749250 (fine_weight=0.5)",
        "trial_78be72d5dd (het_weight=0.2)",
    ],
    "study_manifest": str((ROOT / "study_manifest.json").resolve()),
    "leaderboard_snapshot_utc": lb["updated_at_utc"],
}

out = ROOT / "selection.json"
out.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
print("written", out)
