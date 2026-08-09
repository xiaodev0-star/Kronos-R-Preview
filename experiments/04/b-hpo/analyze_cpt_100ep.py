"""CPT 100-epoch full-run curve diagnosis.

Combines three sources:
  1. Local CPT training history (checkpoints/history_default.json) — loss/LR/step curves.
  2. Local CPT 400-window token-quality trajectory (server_runs/results/04b-cpt/.../epoch_*.json).
  3. Exp 04-B reference 4c72 trajectory (same recipe, 50-epoch HPO arm) for alignment.

Outputs a diagnostic table + plot set under a per-run output directory.
No holdout positions are used (offsets 0-399 only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)

LN2 = math.log(2.0)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def target_entropy_bits(level: str, aggregate: dict[str, Any]) -> float:
    """Target marginal entropy for the requested level (coarse/fine/joint)."""
    key = "median_daily_target_token_entropy_bits"
    if level != "coarse":
        key = f"median_daily_{level}_{'target_token_entropy_bits'}"
    value = aggregate.get(key)
    return float(value) if value is not None else float("nan")


def parse_epoch_files(directory: Path) -> list[dict[str, Any]]:
    """Load epoch_*.json payloads (excludes epoch_summary.json / manifest.json)."""
    results: list[dict[str, Any]] = []
    pattern = re.compile(r"^epoch_(\d{3})\.json$")
    if not directory.exists():
        return results
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        payload = load_json(path)
        if payload.get("status") != "completed":
            continue
        results.append(payload)
    results.sort(key=lambda item: item["epoch"])
    return results


def make_trajectory_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        epoch = payload["epoch"]
        training = payload["training"]
        agg = payload["aggregate"]
        rows.append(
            {
                "epoch": epoch,
                "global_step": training.get("global_step"),
                "val_loss": training.get("val_loss"),
                "val_fine_loss": training.get("val_fine_loss"),
                "coarse_balance": agg.get("median_daily_codebook_balance_score"),
                "p10_balance": agg.get("p10_daily_codebook_balance_score"),
                "joint_balance": agg.get("median_daily_joint_codebook_balance_score"),
                "jsd": agg.get("median_daily_token_jsd"),
                "support_f1": agg.get("median_daily_token_support_f1"),
                "unique": agg.get("median_daily_unique_tokens"),
                "target_unique": agg.get("median_daily_target_n_unique_tokens"),
                "collapse": agg.get("median_daily_collapse_rate"),
                "p90_collapse": agg.get("p90_daily_collapse_rate"),
                "pred_entropy": agg.get("median_daily_pred_token_entropy_bits"),
                "target_entropy": target_entropy_bits("coarse", agg),
                "da": agg.get("avg_da_per_date"),
                "rank_ic": agg.get("avg_daily_rank_ic"),
                "mape": agg.get("avg_mape"),
                "ampratio": agg.get("avg_ampratio"),
            }
        )
    return rows


def h_ce_bits(coarse_entropy_bits: float, val_coarse_loss: float) -> float:
    """H(target) - CE, in bits: how much target marginal entropy the model captures."""
    if not np.isfinite(coarse_entropy_bits) or val_coarse_loss is None:
        return float("nan")
    return float(coarse_entropy_bits - val_coarse_loss / LN2)


def print_rows(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'ep':>4} {'step':>7} {'valL':>6} {'bal':>5} {'p10':>5} "
        f"{'jsd':>5} {'suppF1':>6} {'uniq':>4} {'coll%':>6} {'H-CE':>5} "
        f"{'DA%':>5} {'IC':>5} {'MAPE':>6} {'amp':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        hce = h_ce_bits(r["target_entropy"], r["val_loss"])
        print(
            f"{r['epoch']:>4} {r['global_step'] or 0:>7} "
            f"{r['val_loss']:>6.3f} "
            f"{(r['coarse_balance'] or 0):>5.3f} "
            f"{(r['p10_balance'] or 0):>5.3f} "
            f"{(r['jsd'] or 0):>5.3f} "
            f"{(r['support_f1'] or 0):>6.3f} "
            f"{(r['unique'] or 0):>4.0f} "
            f"{(r['collapse'] or 0) * 100:>6.1f} "
            f"{hce:>5.3f} "
            f"{(r['da'] or 0) * 100:>5.1f} "
            f"{(r['rank_ic'] or 0):>5.3f} "
            f"{(r['mape'] or 0):>6.3f} "
            f"{(r['ampratio'] or 0):>5.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("checkpoints/history_default.json"),
    )
    parser.add_argument(
        "--trajectory_dir",
        type=Path,
        default=Path(
            "server_runs/results/04b-cpt/seed42/trials/local_cpt/epoch_trajectory"
        ),
    )
    parser.add_argument(
        "--ref_trajectory_dir",
        type=Path,
        default=Path(
            "server_runs/results/04b-hpo/seed42/trials/"
            "trial_4c721141ab/epoch_trajectory"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "server_runs/results/04b-cpt/seed42/trials/local_cpt/diagnostics"
        ),
    )
    args = parser.parse_args()

    history = load_json(args.history)
    local_rows = make_trajectory_rows(
        parse_epoch_files(args.trajectory_dir)
    )
    ref_rows = make_trajectory_rows(
        parse_epoch_files(args.ref_trajectory_dir)
    )

    print("=" * 78)
    print("CPT 100-epoch full run — token-quality trajectory (400-window)")
    print("=" * 78)
    print_rows(local_rows)

    if ref_rows:
        print()
        print("=" * 78)
        print("Reference: Exp 04-B 4c72 (same recipe, 50-epoch HPO arm)")
        print("=" * 78)
        print_rows(ref_rows)

    # ---- Key diagnostic numbers -----------------------------------------
    print()
    print("-" * 78)
    print("Key diagnostics")
    print("-" * 78)
    if local_rows:
        epochs = [r["epoch"] for r in local_rows]
        balances = [r["coarse_balance"] for r in local_rows]
        val_losses = [r["val_loss"] for r in local_rows]
        collapses = [r["collapse"] for r in local_rows]
        best_bal_idx = int(np.nanargmax(balances))
        min_val_idx = int(np.nanargmin(val_losses))
        max_hce_idx = int(
            np.nanargmax(
                [
                    h_ce_bits(r["target_entropy"], r["val_loss"])
                    for r in local_rows
                ]
            )
        )
        print(f"Epoch count (sampled): {len(local_rows)}")
        print(
            f"Best coarse balance: {balances[best_bal_idx]:.3f} @ ep"
            f"{epochs[best_bal_idx]}"
        )
        print(
            f"Min val_loss: {val_losses[min_val_idx]:.3f} @ ep"
            f"{epochs[min_val_idx]}"
        )
        print(f"Best H-CE @ ep{epochs[max_hce_idx]}")
        # red line relative to 04-B anchor
        anchor = 0.513
        last = local_rows[-1]
        print(
            f"Final (ep{last['epoch']}): balance={last['coarse_balance']:.3f} "
            f"vs 04-B anchor {anchor} "
            f"({'OK' if last['coarse_balance'] >= anchor else 'BELOW ANCHOR'})"
        )
        print(
            f"JSD final={last['jsd']:.3f} vs 04-B {0.366} "
            f"({'OK' if last['jsd'] <= 0.366 else 'ABOVE (degraded)'})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save a merged CSV for downstream analysis
    import csv

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "cpt_trajectory.csv", local_rows)
    if ref_rows:
        write_csv(args.output_dir / "ref_4c72_trajectory.csv", ref_rows)

    # ---- Plots -----------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if local_rows:
            ep = np.asarray([r["epoch"] for r in local_rows], dtype=float)
            fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

            ax = axes[0]
            ax.plot(
                ep,
                [r["coarse_balance"] for r in local_rows],
                "-o",
                label="coarse balance",
                color="#1f77b4",
            )
            ax.plot(
                ep,
                [r["p10_balance"] for r in local_rows],
                "-s",
                label="p10 balance",
                color="#ff7f0e",
            )
            ax.axhline(0.513, color="#d62728", linestyle="--", label="04-B anchor")
            ax.axhline(0.383, color="#d62728", linestyle=":", alpha=0.7)
            ax.set_ylabel("codebook balance")
            ax.legend()
            ax.grid(alpha=0.2)

            ax = axes[1]
            ax.plot(
                ep,
                [r["jsd"] for r in local_rows],
                "-o",
                label="JSD",
                color="#2ca02c",
            )
            ax.plot(
                ep,
                [r["support_f1"] for r in local_rows],
                "-s",
                label="support F1",
                color="#9467bd",
            )
            ax.axhline(0.366, color="#d62728", linestyle="--", label="04-B JSD anchor")
            ax.set_ylabel("distribution quality")
            ax.legend()
            ax.grid(alpha=0.2)

            ax = axes[2]
            ax.plot(
                ep,
                [r["collapse"] * 100 for r in local_rows],
                "-o",
                label="median collapse %",
                color="#d62728",
            )
            ax.plot(
                ep,
                [r["p90_collapse"] * 100 for r in local_rows],
                "-s",
                label="p90 collapse %",
                color="#ff9896",
            )
            ax.set_ylabel("collapse %")
            ax.set_xlabel("epoch")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.suptitle("CPT 100-epoch: token quality (400-window protocol)")
            fig.tight_layout()
            fig.savefig(args.output_dir / "cpt_token_quality.png", dpi=150)
            plt.close(fig)

            fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            ax = axes[0]
            ax.plot(
                ep,
                [r["val_loss"] for r in local_rows],
                "-o",
                label="val loss",
                color="0.25",
            )
            ax.set_ylabel("validation loss")
            ax.legend()
            ax.grid(alpha=0.2)
            ax = axes[1]
            ax.plot(
                ep,
                [
                    h_ce_bits(r["target_entropy"], r["val_loss"])
                    for r in local_rows
                ],
                "-o",
                label="H(target)-CE bits",
                color="#8c564b",
            )
            ax.set_xlabel("epoch")
            ax.set_ylabel("captured info (bits)")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.suptitle("CPT 100-epoch: loss vs captured information")
            fig.tight_layout()
            fig.savefig(args.output_dir / "cpt_info_bits.png", dpi=150)
            plt.close(fig)

        if ref_rows:
            ep_ref = np.asarray([r["epoch"] for r in ref_rows], dtype=float)
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(
                ep_ref,
                [r["coarse_balance"] for r in ref_rows],
                "-o",
                label="4c72 ref (50-ep)",
                color="#1f77b4",
            )
            if local_rows:
                ax.plot(
                    ep,
                    [r["coarse_balance"] for r in local_rows],
                    "-s",
                    label="CPT local (100-ep)",
                    color="#ff7f0e",
                )
            ax.axhline(0.513, color="#d62728", linestyle="--", label="04-B anchor")
            ax.set_xlabel("epoch")
            ax.set_ylabel("coarse balance")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.suptitle("Coarse balance: CPT 100-epoch vs 04-B 4c72 reference")
            fig.tight_layout()
            fig.savefig(args.output_dir / "cpt_vs_ref_balance.png", dpi=150)
            plt.close(fig)

        print()
        print(f"Diagnostics saved to {args.output_dir}")
    except ImportError:
        print("matplotlib not available; plots skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
