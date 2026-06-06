"""HPO v4: Refined heteroscedastic HPO based on v3 results.

V3 findings (top 7 all converged to same config):
  - loss=CE (NOT focal), lr=5e-4, dropout=0.1, wd=0.01
  - het_weight=0.01-0.04 (low is better), entropy_alpha=0.2, ls=0.0
  - Val loss STILL DECREASING at epoch 30 → increase to 40 epochs

Refined search space:
  - Narrow around best config, explore edges (higher lr, lower het, lower dropout)
  - Fix loss=CE, ls=0.0, entropy_alpha=0.2 (clear winners)
  - Search: lr, dropout, wd, het_weight, plus architectural variants

Usage:
    python run_hpo_v4_het_refined.py
    python run_hpo_v4_het_refined.py --reset
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

OUT_DIR = "checkpoints/hpo_v4_het_refined"
TOK_PATH = "checkpoints/tokenizer_v2_ohlc.pt"
STUDY_DB = "sqlite:///hpo_v4_het_refined.db"
PYTHON = sys.executable
EPOCHS = 40  # v3 best was at epoch 30 (still decreasing)


def make_objective(out_dir, tok_path):

    def objective(trial):
        import torch

        # ---- Refined search space (narrower, CE-focused) ----
        learning_rate = trial.suggest_categorical("learning_rate", [3e-4, 5e-4, 7e-4, 1e-3])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1, 0.15])
        weight_decay = trial.suggest_categorical("weight_decay", [0.005, 0.01, 0.02, 0.05])
        het_weight = trial.suggest_float("het_weight", 0.005, 0.05, log=True)
        entropy_alpha = trial.suggest_categorical("entropy_alpha", [0.1, 0.2, 0.3])
        warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.03, 0.05, 0.10])
        accum_steps = trial.suggest_categorical("accum_steps", [4, 8, 16])

        tag = f"v4_t{trial.number}"
        save_path = os.path.join(out_dir, f"{tag}.pt")
        history_path = os.path.join(out_dir, f"history_{tag}.json")

        # ---- Check if already completed ----
        if os.path.exists(save_path):
            try:
                ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
                if ckpt.get("completed", False):
                    vl = ckpt.get("val_loss", float("inf"))
                    print(f"  [SKIP] {tag} already completed, val_loss={vl:.4f}")
                    return vl
            except Exception:
                pass

        # ---- Build CLI command (fixed: CE, ls=0) ----
        cmd = [
            PYTHON, "train_base.py",
            "--save_path", save_path,
            "--tokenizer_path", tok_path,
            "--epochs", str(EPOCHS),
            "--tag", tag,
            "--heteroscedastic",
            "--het_weight", str(het_weight),
            "--history_per_epoch",
            "--loss", "ce",
            "--weight_decay", str(weight_decay),
            "--dropout", str(dropout),
            "--label_smoothing", "0.0",
            "--entropy_alpha", str(entropy_alpha),
        ]

        # ---- Override multiple TrainingConfig params via env var ----
        override = {
            "TrainingConfig": {
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "accumulation_steps": accum_steps,
            }
        }
        override_path = os.path.join(out_dir, f"_override_{tag}.json")
        with open(override_path, "w") as f:
            json.dump(override, f)

        env = os.environ.copy()
        env["KRONOS_PREVIEW_OVERRIDE_JSON"] = os.path.abspath(override_path)

        # ---- Launch subprocess ----
        proc = subprocess.Popen(cmd, env=env)

        # ---- Poll for per-epoch results ----
        last_reported = 0
        try:
            while proc.poll() is None:
                time.sleep(30)
                if os.path.exists(history_path):
                    try:
                        with open(history_path, "r") as f:
                            h = json.load(f)
                        val_losses = h.get("val_loss", [])
                        for i in range(last_reported, len(val_losses)):
                            trial.report(val_losses[i], step=i)
                            if trial.should_prune():
                                print(f"  [PRUNE] {tag} at epoch {i+1}, val={val_losses[i]:.4f}")
                                proc.terminate()
                                try:
                                    proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait()
                                raise optuna.TrialPruned(f"Pruned at epoch {i+1}")
                        last_reported = len(val_losses)
                    except optuna.TrialPruned:
                        raise
                    except Exception:
                        pass

        except optuna.TrialPruned:
            if os.path.exists(override_path):
                os.remove(override_path)
            raise

        if os.path.exists(override_path):
            os.remove(override_path)

        if proc.returncode != 0:
            print(f"  [FAIL] {tag} exited with code {proc.returncode}")
            return float("inf")

        # ---- Return best val_loss ----
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                h = json.load(f)
            val_losses = h.get("val_loss", [])
            for i in range(last_reported, len(val_losses)):
                trial.report(val_losses[i], step=i)
            best = min(val_losses) if val_losses else float("inf")
        else:
            if os.path.exists(save_path):
                ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
                best = ckpt.get("val_loss", float("inf"))
            else:
                best = float("inf")

        print(f"  [OK] {tag}: best_val_loss={best:.4f}")
        return best

    return objective


def main():
    parser = argparse.ArgumentParser(description="HPO v4: Refined heteroscedastic search")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--tok_path", type=str, default=TOK_PATH)
    args = parser.parse_args()

    if optuna is None:
        print("ERROR: optuna not installed.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.reset and os.path.exists("hpo_v4_het_refined.db"):
        os.remove("hpo_v4_het_refined.db")
        print("Study DB cleared.")

    study = optuna.create_study(
        study_name="kronos_v4_het_refined",
        direction="minimize",
        storage=STUDY_DB,
        load_if_exists=True,
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,  # 10 warmup epochs (more than v3's 8)
            interval_steps=1,
        ),
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Study: {completed} completed, target={args.n_trials} trials, {EPOCHS} epochs each")
    print(f"Search: lr=[3e-4,5e-4,7e-4,1e-3], dropout=[0,0.05,0.1,0.15], "
          f"wd=[0.005,0.01,0.02,0.05], het=[0.005-0.05], accum=[4,8,16]")
    print(f"Fixed: loss=CE, ls=0, entropy_alpha∈[0.1,0.2,0.3], warmup∈[0.03,0.05,0.10]")
    print()

    objective = make_objective(args.out_dir, args.tok_path)

    remaining = max(0, args.n_trials - completed)
    if remaining == 0:
        print(f"All {args.n_trials} trials completed.")
    else:
        print(f"Running {remaining} more trials...")
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  HPO v4 REFINED SUMMARY")
    print("=" * 70)

    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Total: {len(study.trials)} (completed={len(completed_trials)}, pruned={len(pruned_trials)})")

    if completed_trials:
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        print(f"\n  Top 5:")
        for i, t in enumerate(sorted_trials[:5]):
            p = t.params
            print(f"    #{i+1} trial={t.number:3d} val={t.value:.4f}")
            for k, v in sorted(p.items()):
                print(f"        {k}: {v}")

        best = study.best_trial
        print(f"\n  Best: trial={best.number}, val_loss={best.value:.4f}")
        best_path = os.path.join(args.out_dir, "hpo_v4_best.json")
        with open(best_path, "w") as f:
            json.dump({"trial": best.number, "val_loss": best.value,
                        "params": best.params}, f, indent=2)
        print(f"  Saved to {best_path}")


if __name__ == "__main__":
    main()
