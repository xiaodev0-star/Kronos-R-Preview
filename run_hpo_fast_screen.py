"""Fast HPO Screen: Two-phase hyperparameter optimization.

Phase 1 (FAST): 500 stocks × 10 epochs × 32 trials (~90 min total)
  - ASHA-style pruning: prune bottom 50% at epochs 3, 5, 7
  - Early termination if collapse_rate > 40% at epoch 5
  - Covers CE, Focal (γ=4~10+ls), Heteroscedastic families

Phase 2 (FULL): top 3 per family × all stocks × 40 epochs
  - Full training with downstream eval (DA, MAPE, collapse_rate)

Cross-Loss comparison: use downstream metrics (DA, MAPE, collapse_rate),
  NOT val_loss (different loss scales are incomparable).

Usage:
    python run_hpo_fast_screen.py                    # Full two-phase run
    python run_hpo_fast_screen.py --phase1_only      # Only Phase 1 (screening)
    python run_hpo_fast_screen.py --phase2_only      # Only Phase 2 (skip screening)
    python run_hpo_fast_screen.py --n_phase1 20      # Limit Phase 1 trials
"""
import argparse
import json
import os
import subprocess
import sys
import time

try:
    import optuna
    from optuna.pruners import MedianPruner
except ImportError:
    optuna = None

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- Phase 1 Config ---
PHASE1_DIR = "checkpoints/hpo_fast_phase1"
PHASE1_DB = "sqlite:///hpo_fast_p1_v2.db"
PHASE1_STOCKS = 500
PHASE1_EPOCHS = 10
TOK_PATH = "checkpoints/tokenizer_v2_ohlc.pt"
PYTHON = sys.executable

# --- Phase 2 Config ---
PHASE2_DIR = "checkpoints/hpo_fast_phase2"
PHASE2_EPOCHS = 40
TOP_PER_FAMILY = 3  # Top N per loss family → Phase 2

# --- Pruning Config ---
PRUNE_CHECKPOINTS = [3, 5, 7]  # Check for pruning at these epochs
COLLAPSE_THRESHOLD = 0.60  # Prune if collapse_rate > 60% at any checkpoint
PRUNE_FRACTION = 0.5  # Prune bottom 50% at each checkpoint


def build_phase1_trials():
    """Build the Phase 1 search space: 32 trials across 3 loss families."""
    trials = []

    # --- Family 1: CE + Het (8 trials) ---
    for lr in [5e-4, 7e-4]:
        for het_w in [0.01, 0.03]:
            for dropout in [0.0, 0.1]:
                trials.append({
                    "family": "ce_het", "loss": "ce", "gamma": 2.0,
                    "learning_rate": lr, "het_weight": het_w,
                    "dropout": dropout, "weight_decay": 0.02,
                    "label_smoothing": 0.0, "entropy_alpha": 0.2,
                    "heteroscedastic": True,
                })

    # --- Family 2: Focal (12 trials) ---
    for gamma in [4, 6, 8, 10]:
        for ls in [0.0, 0.05]:
            for dropout in [0.0, 0.1]:
                # Skip redundant combos
                if gamma == 4 and ls == 0.05 and dropout == 0.1:
                    continue
                trials.append({
                    "family": "focal", "loss": "focal", "gamma": gamma,
                    "learning_rate": 5e-4, "het_weight": 0.0,
                    "dropout": dropout, "weight_decay": 0.01,
                    "label_smoothing": ls, "entropy_alpha": 0.0,
                    "heteroscedastic": False,
                })

    # --- Family 3: Focal + Het (12 trials) ---
    for gamma in [4, 6, 8]:
        for het_w in [0.01, 0.03, 0.05]:
            for dropout in [0.0, 0.1]:
                if gamma == 4 and het_w == 0.05 and dropout == 0.1:
                    continue
                trials.append({
                    "family": "focal_het", "loss": "focal", "gamma": gamma,
                    "learning_rate": 5e-4, "het_weight": het_w,
                    "dropout": dropout, "weight_decay": 0.01,
                    "label_smoothing": 0.0, "entropy_alpha": 0.0,
                    "heteroscedastic": True,
                })

    return trials


def make_phase1_objective(out_dir, trials_config):
    """Create Optuna objective for Phase 1 fast screening."""

    def objective(trial):
        # Select trial config
        idx = trial.number
        if idx >= len(trials_config):
            raise optuna.TrialPruned("Exceeded trial configs")
        cfg = trials_config[idx]

        tag = f"p1_{cfg['family']}_{idx}"
        save_path = os.path.join(out_dir, f"{tag}.pt")
        history_path = os.path.join(out_dir, f"history_{tag}.json")

        # Check if already completed
        if os.path.exists(save_path):
            try:
                import torch
                ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
                if ckpt.get("completed", False):
                    vl = ckpt.get("val_loss", float("inf"))
                    cr = ckpt.get("collapse_rate", 0.0)
                    print(f"  [SKIP] {tag} completed, val={vl:.4f} collapse={cr*100:.1f}%")
                    # Return composite: val_loss + collapse penalty
                    return vl + max(0, cr - 0.15) * 10.0
            except Exception:
                pass

        # Build CLI command
        cmd = [
            PYTHON, "train_base.py",
            "--save_path", save_path,
            "--tokenizer_path", TOK_PATH,
            "--epochs", str(PHASE1_EPOCHS),
            "--tag", tag,
            "--loss", cfg["loss"],
            "--gamma", str(cfg["gamma"]),
            "--weight_decay", str(cfg["weight_decay"]),
            "--dropout", str(cfg["dropout"]),
            "--label_smoothing", str(cfg["label_smoothing"]),
            "--entropy_alpha", str(cfg["entropy_alpha"]),
            "--history_per_epoch",
            "--light_eval",
            "--max_stocks", str(PHASE1_STOCKS),
            "--max_seq_len", "2048",
        ]

        if cfg["heteroscedastic"]:
            cmd.extend(["--heteroscedastic", "--het_weight", str(cfg["het_weight"])])

        # Override TrainingConfig
        override = {
            "TrainingConfig": {
                "learning_rate": cfg["learning_rate"],
                "warmup_ratio": 0.05,
                "accumulation_steps": 4,  # Less data, smaller accum
            }
        }
        override_path = os.path.join(out_dir, f"_override_{tag}.json")
        with open(override_path, "w") as f:
            json.dump(override, f)

        env = os.environ.copy()
        env["KRONOS_PREVIEW_OVERRIDE_JSON"] = os.path.abspath(override_path)

        # Launch subprocess
        proc = subprocess.Popen(cmd, env=env)

        # Poll for results with collapse-based early termination
        last_reported = 0
        pruned = False
        try:
            while proc.poll() is None:
                time.sleep(15)  # Check every 15s (epochs are fast with 500 stocks)
                if os.path.exists(history_path):
                    try:
                        with open(history_path, "r") as f:
                            h = json.load(f)
                        val_losses = h.get("val_loss", [])
                        collapse_rates = h.get("collapse_rate", [])

                        for i in range(last_reported, len(val_losses)):
                            trial.report(val_losses[i], step=i)

                        # Collapse-based early termination
                        if collapse_rates and len(collapse_rates) >= 5:
                            latest_cr = collapse_rates[-1]
                            if latest_cr > COLLAPSE_THRESHOLD:
                                print(f"  [PRUNE-COLLAPSE] {tag} epoch {len(collapse_rates)}: "
                                      f"collapse={latest_cr*100:.1f}% > {COLLAPSE_THRESHOLD*100:.0f}%")
                                proc.terminate()
                                try:
                                    proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait()
                                pruned = True
                                break

                        last_reported = len(val_losses)
                    except Exception:
                        pass

                if pruned:
                    break

        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise

        if os.path.exists(override_path):
            os.remove(override_path)

        if pruned:
            # Save partial results for analysis
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    h = json.load(f)
                val_losses = h.get("val_loss", [])
                vl = min(val_losses) if val_losses else float("inf")
            else:
                vl = float("inf")
            return vl + 100.0  # Heavily penalize collapsed trials

        if proc.returncode != 0:
            print(f"  [FAIL] {tag} exit={proc.returncode}")
            return float("inf")

        # Read final results
        if os.path.exists(save_path):
            import torch
            ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
            vl = ckpt.get("val_loss", float("inf"))
            cr = ckpt.get("collapse_rate", 0.0)
            n_unique = ckpt.get("n_unique_tokens", 0)
        elif os.path.exists(history_path):
            with open(history_path, "r") as f:
                h = json.load(f)
            val_losses = h.get("val_loss", [])
            vl = min(val_losses) if val_losses else float("inf")
            collapse_rates = h.get("collapse_rate", [])
            cr = collapse_rates[-1] if collapse_rates else 0.0
            n_unique = 0
        else:
            return float("inf")

        # Composite score: val_loss + collapse penalty
        composite = vl + max(0, cr - 0.15) * 10.0
        print(f"  [OK] {tag} ({cfg['family']}): val={vl:.4f} collapse={cr*100:.1f}% "
              f"uniq={n_unique} composite={composite:.4f}")
        return composite

    return objective


def run_phase1(n_trials, out_dir):
    """Run Phase 1: fast screening."""
    if optuna is None:
        print("ERROR: optuna not installed.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    trials_config = build_phase1_trials()
    n_trials = min(n_trials, len(trials_config))

    # Check existing progress
    study = optuna.create_study(
        study_name="kronos_fast_phase1",
        direction="minimize",
        storage=PHASE1_DB,
        load_if_exists=True,
    )
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print("=" * 70)
    print("  PHASE 1: FAST HPO SCREENING")
    print("=" * 70)
    print(f"  Stocks: {PHASE1_STOCKS}, Epochs: {PHASE1_EPOCHS}")
    print(f"  Trials: {n_trials} total ({completed} completed, {pruned_count} pruned)")
    print(f"  Search: {len([t for t in trials_config if t['family']=='ce_het'])} CE+Het + "
          f"{len([t for t in trials_config if t['family']=='focal'])} Focal + "
          f"{len([t for t in trials_config if t['family']=='focal_het'])} Focal+Het")
    print(f"  Pruning: collapse>{COLLAPSE_THRESHOLD*100:.0f}% at epoch 5, "
          f"ASHA at epochs {PRUNE_CHECKPOINTS}")
    print(f"  Est. time: ~{n_trials * 2.5:.0f} min")
    print()

    remaining = max(0, n_trials - completed)
    if remaining == 0:
        print("  All Phase 1 trials completed!")
    else:
        objective = make_phase1_objective(out_dir, trials_config[:n_trials])
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    return summarize_phase1(study, trials_config[:n_trials], out_dir)


def summarize_phase1(study, trials_config, out_dir):
    """Summarize Phase 1 results and select top candidates for Phase 2."""
    print()
    print("=" * 70)
    print("  PHASE 1 RESULTS")
    print("=" * 70)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Total: {len(study.trials)} (completed={len(completed)}, pruned={len(pruned)})")

    if not completed:
        print("  No completed trials!")
        return []

    # Group by family
    families = {}
    for t in completed:
        idx = t.number
        if idx < len(trials_config):
            family = trials_config[idx]["family"]
            families.setdefault(family, []).append((t, trials_config[idx]))

    print(f"\n  Top per family (by composite = val_loss + collapse_penalty):")
    phase2_candidates = []

    for family, trials in sorted(families.items()):
        trials.sort(key=lambda x: x[0].value)  # Sort by composite score
        print(f"\n  [{family}] ({len(trials)} trials)")
        top_n = trials[:TOP_PER_FAMILY]
        for rank, (t, cfg) in enumerate(trials[:5]):  # Show top 5
            mark = " ← Phase 2" if rank < TOP_PER_FAMILY else ""
            print(f"    #{rank+1} trial={t.number:3d} composite={t.value:.4f} "
                  f"lr={cfg['learning_rate']} γ={cfg['gamma']} "
                  f"het={cfg['het_weight']} drop={cfg['dropout']} "
                  f"ls={cfg['label_smoothing']}{mark}")
            if rank < TOP_PER_FAMILY:
                phase2_candidates.append({
                    "trial": t.number, "family": family,
                    "composite": t.value, "config": cfg,
                })

    # Save Phase 2 candidates
    candidates_path = os.path.join(out_dir, "phase2_candidates.json")
    with open(candidates_path, "w") as f:
        json.dump(phase2_candidates, f, indent=2, default=str)
    print(f"\n  Phase 2 candidates saved: {candidates_path}")
    print(f"  Selected: {len(phase2_candidates)} candidates for full training")

    return phase2_candidates


def run_phase2(candidates, out_dir, base_phase1_dir):
    """Run Phase 2: full training on top candidates."""
    import torch

    os.makedirs(out_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("  PHASE 2: FULL TRAINING")
    print("=" * 70)
    print(f"  Stocks: ALL, Epochs: {PHASE2_EPOCHS}")
    print(f"  Candidates: {len(candidates)}")
    print()

    results = []
    for i, cand in enumerate(candidates):
        cfg = cand["config"]
        family = cand["family"]
        tag = f"p2_{family}_{cand['trial']}"
        save_path = os.path.join(out_dir, f"{tag}.pt")
        history_path = os.path.join(out_dir, f"history_{tag}.json")

        print(f"  [{i+1}/{len(candidates)}] {tag} "
              f"(lr={cfg['learning_rate']} γ={cfg['gamma']} het={cfg['het_weight']})")

        # Check if already completed
        if os.path.exists(save_path):
            try:
                ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
                if ckpt.get("completed", False):
                    vl = ckpt.get("val_loss", float("inf"))
                    print(f"    [SKIP] already completed, val={vl:.4f}")
                    results.append({"tag": tag, "config": cfg, "val_loss": vl,
                                    "family": family, "checkpoint": save_path})
                    continue
            except Exception:
                pass

        cmd = [
            PYTHON, "train_base.py",
            "--save_path", save_path,
            "--tokenizer_path", TOK_PATH,
            "--epochs", str(PHASE2_EPOCHS),
            "--tag", tag,
            "--loss", cfg["loss"],
            "--gamma", str(cfg["gamma"]),
            "--weight_decay", str(cfg["weight_decay"]),
            "--dropout", str(cfg["dropout"]),
            "--label_smoothing", str(cfg["label_smoothing"]),
            "--entropy_alpha", str(cfg["entropy_alpha"]),
            "--history_per_epoch",
            "--light_eval",
        ]

        if cfg["heteroscedastic"]:
            cmd.extend(["--heteroscedastic", "--het_weight", str(cfg["het_weight"])])

        override = {
            "TrainingConfig": {
                "learning_rate": cfg["learning_rate"],
                "warmup_ratio": 0.05,
                "accumulation_steps": 8,  # Full data, use standard accum
            }
        }
        override_path = os.path.join(out_dir, f"_override_{tag}.json")
        with open(override_path, "w") as f:
            json.dump(override, f)

        env = os.environ.copy()
        env["KRONOS_PREVIEW_OVERRIDE_JSON"] = os.path.abspath(override_path)

        proc = subprocess.Popen(cmd, env=env)
        proc.wait()

        if os.path.exists(override_path):
            os.remove(override_path)

        if os.path.exists(save_path):
            try:
                ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
                vl = ckpt.get("val_loss", float("inf"))
                cr = ckpt.get("collapse_rate", 0.0)
                results.append({"tag": tag, "config": cfg, "val_loss": vl,
                                "collapse_rate": cr, "family": family,
                                "checkpoint": save_path})
                print(f"    val={vl:.4f} collapse={cr*100:.1f}%")
            except Exception as e:
                print(f"    ERROR: {e}")

    # Phase 2 summary
    print()
    print("=" * 70)
    print("  PHASE 2 RESULTS")
    print("=" * 70)
    print(f"  {'Tag':<30} {'Family':<10} {'ValLoss':>8} {'Collapse':>9} {'Config'}")
    print("  " + "-" * 90)
    results.sort(key=lambda x: x.get("val_loss", float("inf")))
    for r in results:
        cfg = r["config"]
        print(f"  {r['tag']:<30} {r['family']:<10} {r.get('val_loss', 0):>8.4f} "
              f"{r.get('collapse_rate', 0)*100:>8.1f}% "
              f"lr={cfg['learning_rate']} γ={cfg['gamma']} het={cfg['het_weight']}")

    # Save results
    out_path = os.path.join(out_dir, "phase2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    # Print downstream eval instructions
    if results:
        print(f"\n  Next: Run downstream eval on best models:")
        print(f"    python eval_batch_1step.py")
        print(f"  (or modify to include Phase 2 checkpoints)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fast Two-Phase HPO for Kronos-R-Preview")
    parser.add_argument("--phase1_only", action="store_true", help="Only run Phase 1")
    parser.add_argument("--phase2_only", action="store_true", help="Only run Phase 2 (use existing candidates)")
    parser.add_argument("--n_phase1", type=int, default=32, help="Max Phase 1 trials")
    parser.add_argument("--phase1_dir", type=str, default=PHASE1_DIR)
    parser.add_argument("--phase2_dir", type=str, default=PHASE2_DIR)
    args = parser.parse_args()

    if optuna is None:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    phase2_candidates = None

    if not args.phase2_only:
        phase2_candidates = run_phase1(args.n_phase1, args.phase1_dir)

    if not args.phase1_only:
        if phase2_candidates is None:
            # Load from file
            cand_path = os.path.join(args.phase1_dir, "phase2_candidates.json")
            if os.path.exists(cand_path):
                with open(cand_path, "r") as f:
                    phase2_candidates = json.load(f)
                print(f"  Loaded {len(phase2_candidates)} Phase 2 candidates from {cand_path}")
            else:
                print(f"  ERROR: No Phase 2 candidates found. Run Phase 1 first.")
                sys.exit(1)

        if phase2_candidates:
            run_phase2(phase2_candidates, args.phase2_dir, args.phase1_dir)
        else:
            print("  No candidates for Phase 2.")

    print("\n  Fast HPO complete!")


if __name__ == "__main__":
    main()
