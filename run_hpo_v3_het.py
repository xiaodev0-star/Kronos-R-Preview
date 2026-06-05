"""HPO v3: Heteroscedastic BaseModel — Optuna + MedianPruner (30 epochs).

搜索空间：loss/gamma/lr/dropout/wd/het_weight/entropy_alpha/label_smoothing
每个 trial 训练 30 epoch，前 8 个 epoch 不剪枝，之后按 median pruner 剪枝。

Usage:
    python run_hpo_v3_het.py              # start / resume
    python run_hpo_v3_het.py --n_trials 20
    python run_hpo_v3_het.py --reset      # clear study, start fresh
"""
import argparse
import json
import os
import subprocess
import sys
import time

# Optuna import — required at module level for TrialPruned in objective()
try:
    import optuna
    from optuna.pruners import MedianPruner
except ImportError:
    optuna = None
    MedianPruner = None
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "checkpoints/hpo_v3_het"
TOK_PATH = "checkpoints/tokenizer_v2_ohlc.pt"
STUDY_DB = "sqlite:///hpo_het.db"
PYTHON = sys.executable
EPOCHS = 30


def make_objective(out_dir, tok_path):
    """Return an Optuna objective function with closures over shared config."""

    def objective(trial):
        import torch  # lazy to avoid CUDA init before needed

        # ---- Suggest hyperparameters ----
        loss_type = trial.suggest_categorical("loss", ["ce", "focal"])
        if loss_type == "focal":
            gamma = trial.suggest_float("gamma", 2.0, 10.0, step=1.0)
        else:
            gamma = 2.0  # unused placeholder

        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1, step=0.05)
        weight_decay = trial.suggest_categorical("weight_decay", [0.001, 0.01, 0.1])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
        learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 5e-4])
        het_weight = trial.suggest_float("het_weight", 0.01, 1.0, log=True)
        entropy_alpha = trial.suggest_float("entropy_alpha", 0.0, 0.3, step=0.1)

        tag = f"het_t{trial.number}"
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

        # ---- Build CLI command ----
        cmd = [
            PYTHON, "train_base.py",
            "--save_path", save_path,
            "--tokenizer_path", tok_path,
            "--epochs", str(EPOCHS),
            "--tag", tag,
            "--heteroscedastic",
            "--het_weight", str(het_weight),
            "--history_per_epoch",
            "--loss", loss_type,
            "--weight_decay", str(weight_decay),
            "--dropout", str(dropout),
            "--label_smoothing", str(label_smoothing),
            "--entropy_alpha", str(entropy_alpha),
        ]
        if loss_type == "focal":
            cmd += ["--gamma", str(gamma)]

        # ---- LR override via env var ----
        override = {"TrainingConfig": {"learning_rate": learning_rate}}
        override_path = os.path.join(out_dir, f"_override_{tag}.json")
        with open(override_path, "w") as f:
            json.dump(override, f)

        env = os.environ.copy()
        env["KRONOS_PREVIEW_OVERRIDE_JSON"] = os.path.abspath(override_path)

        # ---- Launch subprocess (tqdm progress bar visible in terminal) ----
        proc = subprocess.Popen(cmd, env=env)

        # ---- Poll for per-epoch results and prune if needed ----
        last_reported = 0
        try:
            while proc.poll() is None:
                time.sleep(30)
                # Check for pruning
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
                        pass  # JSON read error during write, retry next poll
        except optuna.TrialPruned:
            # Cleanup override file
            if os.path.exists(override_path):
                os.remove(override_path)
            raise

        # ---- Process finished ----
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
            # Report all remaining epochs
            for i in range(last_reported, len(val_losses)):
                trial.report(val_losses[i], step=i)
            best = min(val_losses) if val_losses else float("inf")
        else:
            # Fallback: read from checkpoint
            if os.path.exists(save_path):
                ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
                best = ckpt.get("val_loss", float("inf"))
            else:
                best = float("inf")

        print(f"  [OK] {tag}: best_val_loss={best:.4f}")
        return best

    return objective

def main():
    parser = argparse.ArgumentParser(description="HPO v3: Heteroscedastic BaseModel")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--reset", action="store_true", help="Delete study DB and start fresh")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--tok_path", type=str, default=TOK_PATH)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if optuna is None:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    # Reset study if requested
    if args.reset and os.path.exists("hpo_het.db"):
        os.remove("hpo_het.db")
        print("Study DB cleared.")

    # Create or load study
    study = optuna.create_study(
        study_name="kronos_het_hpo",
        direction="minimize",
        storage=STUDY_DB,
        load_if_exists=True,
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=8,
            interval_steps=1,
        ),
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"Study loaded: {completed} completed, {pruned} pruned, target={args.n_trials} trials")
    print(f"Epochs per trial: {EPOCHS}")
    print(f"Pruner: MedianPruner(startup=5, warmup=8)")
    print(f"Search space: loss/gamma/lr/dropout/wd/het_weight/entropy_alpha/label_smoothing")
    print()

    objective = make_objective(args.out_dir, args.tok_path)

    remaining = max(0, args.n_trials - completed)
    if remaining == 0:
        print(f"All {args.n_trials} trials already completed.")
    else:
        print(f"Running {remaining} more trials...")
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  HPO v3 SUMMARY")
    print("=" * 70)

    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Total trials: {len(study.trials)} "
          f"(completed={len(completed_trials)}, pruned={len(pruned_trials)})")

    if completed_trials:
        # Top 5
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        print(f"\n  Top 5 (by val_loss):")
        for i, t in enumerate(sorted_trials[:5]):
            print(f"    #{i+1} trial={t.number:3d} val_loss={t.value:.4f}")
            for k, v in sorted(t.params.items()):
                print(f"        {k}: {v}")

        # Best
        best = study.best_trial
        print(f"\n  Best: trial={best.number}, val_loss={best.value:.4f}")
        print(f"  Params: {json.dumps(best.params, indent=2)}")

        # Save best params
        best_path = os.path.join(args.out_dir, "hpo_het_best.json")
        with open(best_path, "w") as f:
            json.dump({
                "trial": best.number,
                "val_loss": best.value,
                "params": best.params,
                "n_trials": len(study.trials),
                "n_completed": len(completed_trials),
                "n_pruned": len(pruned_trials),
            }, f, indent=2)
        print(f"\n  Best params saved to {best_path}")

    # Per-trial summary CSV
    csv_path = os.path.join(args.out_dir, "hpo_het_trials.csv")
    with open(csv_path, "w") as f:
        f.write("trial,val_loss,state,loss,gamma,lr,dropout,wd,het_weight,entropy_alpha,label_smoothing\n")
        for t in study.trials:
            p = t.params
            val = f"{t.value:.4f}" if t.value is not None else "inf"
            state = t.state.name
            f.write(f"{t.number},{val},{state},"
                    f"{p.get('loss','')},{p.get('gamma','')},"
                    f"{p.get('learning_rate','')},{p.get('dropout','')},"
                    f"{p.get('weight_decay','')},{p.get('het_weight','')},"
                    f"{p.get('entropy_alpha','')},{p.get('label_smoothing','')}\n")
    print(f"  Trial summary saved to {csv_path}")


if __name__ == "__main__":
    main()
