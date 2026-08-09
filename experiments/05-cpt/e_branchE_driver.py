"""Branch E (DPO PostTrain): per-seed orchestration + red-line gating + acted check.

Pipeline per (beta, seed):
  Phase0  Measure pi_ref under the 400-window protocol once -> red-line and
          bootstrap reference (baseline.json).
  Phase1  Build preference pairs with the FROZEN pi_ref (e_build_pairs.py).
  Phase2  Train pi with DPO (e_train_dpo.py); snapshots every 0.5 epoch.
          Red-line gate every snapshot via evaluate_epoch_trajectory; roll back
          to the last good snapshot and mark the seed FAILED on any breach.
          Pick the best passing snapshot by avg_daily_rank_ic.
  Acted   Paired bootstrap (rank_ic, da) + T3 mean-sampling on the best
          snapshot.  Acceptance per beta (3 seeds): all seeds pass red lines,
          >= 2/3 seeds have a positive acted delta, 0 seeds significantly
          negative.

Run (two-phase, single GPU):
    python experiments/05-cpt/e_branchE_driver.py \\
        --ref-ckpt checkpoints/branchA_dm030_8ceb_ep5.pt \\
        --betas 0.1,0.5 --seeds 43,44,45 \\
        --tokenizer checkpoints/tokenizer_v2_ohlc.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "experiments" / "05-cpt"
EVAL_SCRIPT = ROOT / "experiments" / "04" / "b-hpo" / "evaluate_epoch_trajectory.py"
PAIRS_SCRIPT = SCRIPT_DIR / "e_build_pairs.py"
TRAIN_SCRIPT = SCRIPT_DIR / "e_train_dpo.py"
T3_SCRIPT = SCRIPT_DIR / "t3_sampling_self_consistency.py"

REF_OVERRIDE = (
    ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials"
    / "local_cpt" / "override.json"
)
CPT_CACHE = (
    ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials"
    / "local_cpt" / "cache"
)

# Red-line tolerances relative to the pi_ref measured baseline (ToDo §1.2).
RED_BALANCE_DELTA = 0.02
RED_JSD_DELTA = 0.02
RED_JSD_ABS_FLOOR = 0.366
RED_COLLAPSE_DELTA = 0.05
RED_UNIQUE_DELTA = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-ckpt", type=Path, required=True,
                        help="pi_ref checkpoint (Branch A output).")
    parser.add_argument("--tokenizer", type=Path,
                        default=ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt")
    parser.add_argument("--betas", type=str, default="0.1,0.5",
                        help="Comma-separated beta values to sweep.")
    parser.add_argument("--seeds", type=str, default="43,44,45",
                        help="Comma-separated seeds; each seed is used for BOTH "
                             "pair building (--seed) and DPO training "
                             "(--controlled-loader-seed / --seed).")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--prepared-cache-dir", type=Path, default=CPT_CACHE)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--accum", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=6144)
    parser.add_argument("--batch-cap", type=int, default=64)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="Debug: limit stocks in pair building and DPO training.")
    parser.add_argument("--no-phase0", action="store_true",
                        help="Skip the pi_ref baseline measurement (reuse existing).")
    parser.add_argument("--force-pairs", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-redline", action="store_true",
                        help="Debug: skip the per-snapshot 400-window red-line eval.")
    parser.add_argument("--skip-t3", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--pair-seed-offset", type=int, default=0,
                        help="pair_seed = seed + this offset (debug override).")
    return parser.parse_args()


def _cmd(python, *parts):
    return [python, *[str(p) for p in parts]]


def write_model_checkpoints(trial_dir: Path, tag: str, save_path: Path,
                            entries: list[dict]) -> Path:
    """Write model_checkpoints.json with integer-epoch entries (eval protocol)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "tag": tag,
        "save_path": str(save_path.resolve()),
        "updated_epoch": max((e["epoch"] for e in entries), default=0),
        "checkpoints": entries,
    }
    index_path = trial_dir / "model_checkpoints.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index_path


def run_evaluate_epochs(trial_dir: Path, epochs_csv: str, tokenizer: Path,
                        output_dir: Path, prepared_cache: Path) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd = _cmd(
        sys.executable, EVAL_SCRIPT,
        "--trial_dir", trial_dir,
        "--tokenizer", tokenizer,
        "--output_dir", output_dir,
        "--epochs", epochs_csv,
        "--offsets", "0-399",
        "--n_days", "1",
        "--batch_size", "4",
        "--seed", "42",
        "--prepared_cache_dir", prepared_cache,
        "--no_reference_check",
    )
    print(f"  eval cmd: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"evaluate_epoch_trajectory failed for {epochs_csv}")


def read_epoch_json(output_dir: Path, epoch: int) -> dict:
    path = output_dir / f"epoch_{epoch:03d}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_redline_metrics(payload: dict) -> dict:
    agg = payload["aggregate"]
    return {
        "balance": float(agg["median_daily_codebook_balance_score"]),
        "jsd": float(agg["median_daily_token_jsd"]),
        "collapse": float(agg["median_daily_collapse_rate"]),
        "unique": float(agg["median_daily_unique_tokens"]),
        "da": float(agg["avg_da_per_date"]),
        "ic": float(agg["avg_daily_rank_ic"]),
    }


def check_redlines(cand: dict, ref: dict) -> tuple[bool, dict]:
    breaches = {}
    if cand["balance"] < ref["balance"] - RED_BALANCE_DELTA:
        breaches["balance"] = (
            cand["balance"], ref["balance"] - RED_BALANCE_DELTA)
    jsd_floor = min(ref["jsd"] + RED_JSD_DELTA, RED_JSD_ABS_FLOOR)
    if cand["jsd"] > jsd_floor:
        breaches["jsd"] = (cand["jsd"], jsd_floor)
    if cand["collapse"] > ref["collapse"] + RED_COLLAPSE_DELTA:
        breaches["collapse"] = (
            cand["collapse"], ref["collapse"] + RED_COLLAPSE_DELTA)
    if cand["unique"] < ref["unique"] - RED_UNIQUE_DELTA:
        breaches["unique"] = (
            cand["unique"], ref["unique"] - RED_UNIQUE_DELTA)
    return not breaches, breaches


def paired_bootstrap(cand_payload: dict, ref_payload: dict, key: str,
                     n_boot: int, seed: int) -> dict:
    """Block bootstrap over per-date deltas (mirrors bootstrap_compare.py)."""
    cand = load_per_date_direct(cand_payload, key)
    ref = load_per_date_direct(ref_payload, key)
    common = sorted(set(cand) & set(ref))
    if not common:
        return {"n_dates": 0, "significant": False, "ci": [0.0, 0.0]}
    c = np.asarray([cand[d] for d in common])
    r = np.asarray([ref[d] for d in common])
    delta = c - r
    mean_delta = float(delta.mean())
    rng = np.random.RandomState(seed)
    n = len(common)
    means = np.asarray(
        [delta[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    )
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {
        "n_dates": n,
        "mean_delta": mean_delta,
        "ci": [lo, hi],
        "significant_positive": lo > 0,
        "significant_negative": hi < 0,
        "significant": not (lo <= 0 <= hi),
    }


def load_per_date_direct(payload: dict, key: str) -> dict[str, float]:
    out = {}
    for offset, window in payload.get("windows", {}).items():
        for date, metrics in window.get("per_date", {}).items():
            if key in metrics and metrics[key] is not None:
                out[date] = float(metrics[key])
    return out


def phase0_baseline(args: argparse.Namespace, out_root: Path,
                    tokenizer: Path) -> dict:
    ref_trial = out_root / "ref_baseline"
    baseline_metrics_path = out_root / "baseline.json"
    baseline_epoch_json = ref_trial / "epoch_trajectory" / "epoch_001.json"
    if baseline_epoch_json.is_file():
        print(f"  [phase0] reusing baseline {baseline_epoch_json}", flush=True)
        payload = json.loads(baseline_epoch_json.read_text(encoding="utf-8"))
    else:
        ref_trial.mkdir(parents=True, exist_ok=True)
        shutil.copy(REF_OVERRIDE, ref_trial / "override.json")
        entries = [{
            "epoch": 1,
            "path": str(args.ref_ckpt.resolve()),
            "size_bytes": args.ref_ckpt.stat().st_size,
            "train_loss": 3.0,
            "val_loss": 3.6,
            "learning_rate": 0.0,
            "learning_rate_adam": 0.0,
            "optimizer_steps_this_epoch": 0,
            "global_step": 0,
            "best_so_far": False,
        }]
        write_model_checkpoints(ref_trial, "ref_baseline", args.ref_ckpt, entries)
        run_evaluate_epochs(
            ref_trial, "1", tokenizer,
            ref_trial / "epoch_trajectory", args.prepared_cache_dir,
        )
        payload = read_epoch_json(ref_trial / "epoch_trajectory", 1)
    metrics = read_redline_metrics(payload)
    baseline = {
        "ref_ckpt": str(args.ref_ckpt.resolve()),
        "ref_ckpt_sha256": file_sha256(args.ref_ckpt),
        "epoch_json": str(baseline_epoch_json.resolve()),
        "redline_ref": metrics,
        "raw_aggregate": payload.get("aggregate", {}),
    }
    with baseline_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"  [phase0] baseline balance={metrics['balance']:.3f} "
          f"jsd={metrics['jsd']:.3f} collapse={metrics['collapse']:.3f} "
          f"unique={metrics['unique']} da={metrics['da']:.4f} "
          f"ic={metrics['ic']:.4f}", flush=True)
    return baseline


def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_seed(args: argparse.Namespace, beta: float, seed: int,
             baseline: dict, tokenizer: Path, out_root: Path) -> dict:
    bdir = out_root / f"b{beta:g}"
    seed_dir = bdir / f"s{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if not (seed_dir / "override.json").is_file():
        shutil.copy(REF_OVERRIDE, seed_dir / "override.json")

    pairs_path = bdir / f"pairs_s{seed}.npz"
    selection = {"beta": beta, "seed": seed, "status": "pending"}
    selection_path = seed_dir / "selection.json"

    # ---- Phase1: build pairs -------------------------------------------------
    if not pairs_path.is_file() or args.force_pairs:
        pair_seed = seed + args.pair_seed_offset
        cmd = _cmd(
            sys.executable, PAIRS_SCRIPT,
            "--ref-ckpt", args.ref_ckpt,
            "--tokenizer", tokenizer,
            "--k", 8,
            "--temperature", 1.0,
            "--alpha", 0.5,
            "--min-reward-gap", 0.15,
            "--min-lookback", 20,
            "--seed", pair_seed,
            "--out", pairs_path,
        )
        if args.max_stocks > 0:
            cmd += ["--max-stocks", args.max_stocks]
        print(f"  [phase1] building pairs (seed {pair_seed}): {' '.join(cmd)}",
              flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            selection.update({"status": "failed", "stage": "phase1_pairs",
                              "reason": "e_build_pairs non-zero exit"})
            write_json(selection_path, selection)
            return selection
    else:
        print(f"  [phase1] reusing pairs {pairs_path}", flush=True)

    # ---- Phase2: DPO training ------------------------------------------------
    metrics_dir = seed_dir
    save_path = out_root / f"branchE_b{beta:g}_s{seed}.pt"
    force_train = args.force_train or args.force_pairs
    if not (seed_dir / "model_checkpoints.json").is_file() or force_train:
        cmd = _cmd(
            sys.executable, TRAIN_SCRIPT,
            "--init-ckpt", args.ref_ckpt,
            "--tokenizer", tokenizer,
            "--pairs", pairs_path,
            "--beta", beta,
            "--lr", args.lr,
            "--epochs", args.epochs,
            "--accum", args.accum,
            "--batch-tokens", args.batch_tokens,
            "--batch-cap", args.batch_cap,
            "--warmup-ratio", 0.05,
            "--controlled-loader-seed", seed,
            "--ce-weight", args.ce_weight,
            "--val-holdout-ratio", 0.05,
            "--early-stop-patience", 2,
            "--snapshot-every", 0.5,
            "--seed", seed,
            "--tag", f"branchE_b{beta:g}_s{seed}",
            "--save-path", save_path,
            "--metrics-dir", metrics_dir,
        )
        if args.max_stocks > 0:
            cmd += ["--max-stocks", args.max_stocks]
        print(f"  [phase2] training DPO: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            selection.update({"status": "failed", "stage": "phase2_train",
                              "reason": "e_train_dpo non-zero exit"})
            write_json(selection_path, selection)
            return selection
    else:
        print(f"  [phase2] reusing trained run {seed_dir}", flush=True)

    index = json.loads((seed_dir / "model_checkpoints.json").read_text(encoding="utf-8"))
    snapshots = index["checkpoints"]
    if not snapshots:
        selection.update({"status": "failed", "stage": "phase2_train",
                          "reason": "no snapshots produced"})
        write_json(selection_path, selection)
        return selection
    print(f"  [phase2] {len(snapshots)} snapshots", flush=True)

    # ---- Red-line gating over every snapshot ----------------------------------
    eval_out = seed_dir / "epoch_trajectory"
    passing = []
    rolled_back = None
    for snap in sorted(snapshots, key=lambda e: int(e["epoch"])):
        ep = int(snap["epoch"])
        epoch_json = eval_out / f"epoch_{ep:03d}.json"
        if not epoch_json.is_file() and not args.skip_redline:
            run_evaluate_epochs(seed_dir, str(ep), tokenizer, eval_out,
                                args.prepared_cache_dir)
        if args.skip_redline:
            # No gating in debug mode: treat every snapshot as passing.  If the
            # snapshot already has a formal eval JSON use it for the acted step;
            # otherwise carry a None payload so bootstrap/T3 are skipped.
            metrics = {"balance": float("nan"), "jsd": float("nan"),
                       "collapse": float("nan"), "unique": float("nan"),
                       "da": float("nan"), "ic": float("nan")}
            payload = None
            if epoch_json.is_file():
                payload = read_epoch_json(eval_out, ep)
                metrics = read_redline_metrics(payload)
            passing.append((ep, snap, epoch_json, metrics, payload))
            continue
        payload = read_epoch_json(eval_out, ep)
        metrics = read_redline_metrics(payload)
        ok, breaches = check_redlines(metrics, baseline["redline_ref"])
        print(f"    snapshot ep{ep:02d}: balance={metrics['balance']:.3f} "
              f"jsd={metrics['jsd']:.3f} collapse={metrics['collapse']:.3f} "
              f"unique={metrics['unique']} ok={ok}", flush=True)
        if not ok:
            rolled_back = {"epoch": ep, "breaches": breaches}
            print(f"    RED-LINE BREACH: {breaches} -> rolling back to last good",
                  flush=True)
            break
        passing.append((ep, snap, epoch_json, metrics, payload))

    if rolled_back is not None:
        selection.update({
            "status": "failed", "stage": "redline",
            "reason": "snapshot breached a token-quality red line",
            "breach": rolled_back,
            "n_snapshots": len(snapshots),
            "n_passing": len(passing),
        })
        write_json(selection_path, selection)
        return selection

    if not passing:
        selection.update({"status": "failed", "stage": "redline",
                          "reason": "no passing snapshots"})
        write_json(selection_path, selection)
        return selection

    # ---- Acted: best passing snapshot (by avg_daily_rank_ic) ------------------
    best_ep, best_snap, best_json, best_metrics, best_payload = max(
        passing, key=lambda item: item[3]["ic"]
        if np.isfinite(item[3]["ic"]) else -1.0
    )
    ref_json = Path(baseline["epoch_json"])
    ref_payload = json.loads(ref_json.read_text(encoding="utf-8"))
    if best_payload is None and best_json is not None and best_json.is_file():
        best_payload = read_epoch_json(best_json.parent, best_ep)

    boot_ic = {"n_dates": 0, "mean_delta": 0.0, "ci": [0.0, 0.0],
               "significant": False}
    boot_da = dict(boot_ic)
    if best_payload is not None:
        boot_ic = paired_bootstrap(best_payload, ref_payload, "rank_ic",
                                   args.bootstrap, 42)
        boot_da = paired_bootstrap(best_payload, ref_payload, "da",
                                   args.bootstrap, 42)

    # ---- Secondary acted evidence: T3 mean-sampling ---------------------------
    t3_result = None
    if not args.skip_t3:
        t3_out = seed_dir / "t3_self_consistency"
        cmd = _cmd(
            sys.executable, T3_SCRIPT,
            "--ckpt", best_snap["path"],
            "--tokenizer", tokenizer,
            "--k", 8,
            "--temperature", 1.0,
            "--output_dir", t3_out,
            "--prepared_cache_dir", args.prepared_cache_dir,
        )
        print(f"  [acted] T3: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode == 0:
            summary_path = t3_out / "summary.json"
            if summary_path.is_file():
                t3_result = json.loads(summary_path.read_text(encoding="utf-8"))

    # Direction consistency is aggregated across seeds afterwards.
    selection.update({
        "status": "passed",
        "best_snapshot": {
            "epoch": best_ep,
            "path": best_snap["path"],
            "fractional_epoch": best_snap.get("fractional_epoch"),
        },
        "redline": {
            "balance": best_metrics["balance"],
            "jsd": best_metrics["jsd"],
            "collapse": best_metrics["collapse"],
            "unique": best_metrics["unique"],
            "da": best_metrics["da"],
            "ic": best_metrics["ic"],
        },
        "boot_rank_ic": boot_ic,
        "boot_da": boot_da,
        "acted_positive": (
            boot_ic["significant_positive"] or boot_da["significant_positive"]
        ),
        "t3_mean_ic": t3_result.get("mean_ic") if t3_result else None,
        "t3_mean_da": t3_result.get("mean_da") if t3_result else None,
        "n_passing_snapshots": len(passing),
        "pairs_sha256": file_sha256(pairs_path),
        "ref_ckpt_sha256": baseline["ref_ckpt_sha256"],
        "ref_redline_ref": baseline["redline_ref"],
    })
    write_json(selection_path, selection)
    print(f"  [acted] seed {seed}: boot_ic={boot_ic['mean_delta']:+.4f} "
          f"CI={boot_ic['ci']} boot_da={boot_da['mean_delta']:+.4f} "
          f"CI={boot_da['ci']}", flush=True)
    return selection


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    tokenizer = args.tokenizer.resolve()
    if not args.ref_ckpt.is_file():
        raise FileNotFoundError(args.ref_ckpt)
    out_root = args.out_root or (
        ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials"
        / "branchE"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    betas = [float(v) for v in args.betas.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]

    if not args.no_phase0:
        baseline = phase0_baseline(args, out_root, tokenizer)
    else:
        baseline_path = out_root / "baseline.json"
        if not baseline_path.is_file():
            raise RuntimeError("--no-phase0 but no baseline.json exists")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    all_selections: dict[str, dict] = {}
    for beta in betas:
        beta_selections = []
        for seed in seeds:
            print(f"\n==== beta={beta} seed={seed} ====", flush=True)
            sel = run_seed(args, beta, seed, baseline, tokenizer, out_root)
            beta_selections.append(sel)
        all_selections[f"b{beta:g}"] = beta_selections

        # Per-beta 3-seed direction consistency: a seed counts as positive only
        # when its paired-bootstrap CI excludes zero in the improving direction
        # (rank_ic or da); a significantly negative CI on either key fails it.
        passed = [s for s in beta_selections if s["status"] == "passed"]
        n_pos = sum(
            1 for s in passed
            if s["boot_rank_ic"]["significant_positive"]
            or s["boot_da"]["significant_positive"]
        )
        n_sig_neg = sum(
            1 for s in passed
            if s["boot_rank_ic"]["significant_negative"]
            or s["boot_da"]["significant_negative"]
        )
        n_seeds = len(beta_selections)
        all_pass_red = len(passed) == n_seeds
        direction_ok = n_pos >= max(2, int(np.ceil(2 * n_seeds / 3))) \
            and n_sig_neg == 0
        verdict = "PASS" if (all_pass_red and direction_ok and passed) else "FAIL"
        print(f"  [beta {beta}] passed_seeds={len(passed)}/{n_seeds} "
              f"positive={n_pos} sig_negative={n_sig_neg} -> {verdict}",
              flush=True)
        all_selections[f"b{beta:g}_verdict"] = {
            "n_seeds": n_seeds,
            "n_passed": len(passed),
            "n_positive": n_pos,
            "n_significant_negative": n_sig_neg,
            "all_pass_redline": all_pass_red,
            "direction_consistent": direction_ok,
            "verdict": verdict,
        }

    overall_path = out_root / "branchE_selection.json"
    write_json(overall_path, all_selections)
    print(f"\nSaved overall selection -> {overall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
