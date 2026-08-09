"""T2 (redo): cross-trajectory model soup over the two Exp 04-B HPO top trials.

Original ToDo T2 soups HPO top trials 4c721141ab (lr_muon=0.005) +
8ceb4d575b (lr_muon=0.01) — two different optimization trajectories. The 4c72
leg is the locked CPT 100-epoch run (checkpoints/exp04b_best_ep*.pt); the 8ceb
leg is a fresh 100-epoch CPT run of the same recipe with lr_muon=0.01
(checkpoints/exp04b_8ceb_ep*.pt, trained by temp_train_8ceb.py).

Per-run weights are selected by BEST TOKEN QUALITY (max median daily coarse
codebook balance over the evaluated trajectory; JSD/collapse/unique as
tie-breakers). Barrier check (ToDo): the soup IS the line-segment midpoint of
the two weights — quick token-quality read on a subset of windows must not
collapse before the full 400-window eval. Acceptance (ToDo T2): balance/JSD
not worse than the best single trial -> adopt as the new CPT start.

Stages (run sequentially):
    python experiments/05-cpt/t2_cross_config_soup.py --stage prepare_traj
    python experiments/05-cpt/t2_cross_config_soup.py --stage select
    python experiments/05-cpt/t2_cross_config_soup.py --stage soup
    python experiments/05-cpt/t2_cross_config_soup.py --stage barrier
    python experiments/05-cpt/t2_cross_config_soup.py --stage eval
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
os.chdir(ROOT)

TRIALS = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials"
EVAL_SCRIPT = ROOT / "experiments" / "04" / "b-hpo" / "evaluate_epoch_trajectory.py"
BOOTSTRAP = ROOT / "experiments" / "05-cpt" / "bootstrap_compare.py"

RUN_A = "local_cpt"        # 4c72 leg (trained): exp04b_best_ep*.pt
RUN_B = "8ceb_cpt"         # 8ceb leg (trained): exp04b_8ceb_ep*.pt
CKPT_A_PREFIX = "exp04b_best"
CKPT_B_PREFIX = "exp04b_8ceb"
TOKENIZER = ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt"
CACHE_A = TRIALS / RUN_A / "cache"
TRAJ_EPOCHS = [1] + list(range(5, 101, 5))   # 21 sampled points (same as local_cpt)


def run(cmd: list[str]) -> int:
    print("CMD:", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


# ---------------------------------------------------------------- stage 1
def stage_prepare_traj(args: argparse.Namespace) -> int:
    """Copy the training-written index into the 8ceb trial dir, then evaluate
    the 8ceb trajectory under the 400-window protocol (shared prepared cache)."""
    src = ROOT / "checkpoints" / f"{CKPT_B_PREFIX}_checkpoints.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} — train 8ceb first (temp_train_8ceb.py)")
    trial = TRIALS / RUN_B
    shutil.copy(src, trial / "model_checkpoints.json")
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--trial_dir", str(trial),
        "--tokenizer", str(TOKENIZER),
        "--output_dir", str(trial / "epoch_trajectory"),
        "--epochs", ",".join(str(e) for e in TRAJ_EPOCHS),
        "--offsets", "0-399",
        "--n_days", "1",
        "--batch_size", "4",
        "--seed", "42",
        "--prepared_cache_dir", str(CACHE_A),
        "--no_reference_check",
    ]
    return run(cmd)


# ---------------------------------------------------------------- stage 2
def load_aggregate(trial: Path, epoch: int) -> dict | None:
    p = trial / "epoch_trajectory" / f"epoch_{epoch:03d}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("aggregate", {})


def stage_select(args: argparse.Namespace) -> int:
    """Per-run best-token-quality epoch (max median coarse balance; tie-break
    by lower JSD, then lower collapse)."""
    best = {}
    for name, trial, prefix in ((RUN_A, TRIALS / RUN_A, CKPT_A_PREFIX),
                                (RUN_B, TRIALS / RUN_B, CKPT_B_PREFIX)):
        candidates = []
        for ep in TRAJ_EPOCHS:
            agg = load_aggregate(trial, ep)
            if agg is None:
                continue
            candidates.append({
                "epoch": ep,
                "balance": agg["median_daily_codebook_balance_score"],
                "p10": agg["p10_daily_codebook_balance_score"],
                "jsd": agg["median_daily_token_jsd"],
                "collapse": agg["median_daily_collapse_rate"],
                "unique": agg["median_daily_unique_tokens"],
                "support_f1": agg["median_daily_token_support_f1"],
            })
        if not candidates:
            raise RuntimeError(f"No trajectory evals for {name}")
        pick = max(candidates, key=lambda c: (c["balance"], -c["jsd"], -c["collapse"]))
        best[name] = pick
        print(f"\n[{name}] best token-quality epoch = ep{pick['epoch']}:")
        for k in ("balance", "p10", "jsd", "collapse", "unique", "support_f1"):
            print(f"    {k}: {pick[k]:.4f}" if isinstance(pick[k], float) else f"    {k}: {pick[k]}")
    best_a, best_b = best[RUN_A], best[RUN_B]
    ref = best_a if best_a["balance"] >= best_b["balance"] else best_b
    print(f"\nBest single trial: {ref['epoch']} ({ref['balance']:.4f}) — "
          f"{'4c72' if ref is best_a else '8ceb'} leg")
    (TRIALS / "t2_selection.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items()}
                    for k, v in best.items()}, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- stage 3
def ckpt_path(prefix: str, epoch: int) -> Path:
    return ROOT / "checkpoints" / f"{prefix}_ep{epoch}.pt"


def stage_soup(args: argparse.Namespace) -> int:
    """Average the two selected weights -> soup checkpoint (the midpoint)."""
    sel = json.loads((TRIALS / "t2_selection.json").read_text(encoding="utf-8"))
    pa, pb = ckpt_path(CKPT_A_PREFIX, sel[RUN_A]["epoch"]), ckpt_path(CKPT_B_PREFIX, sel[RUN_B]["epoch"])
    ca, cb = torch.load(pa, map_location="cpu", weights_only=False), torch.load(pb, map_location="cpu", weights_only=False)
    if ca["config"] != cb["config"]:
        raise RuntimeError("Config mismatch between the two legs — cannot soup")
    state = {k: (v.float().clone() + cb["model_state_dict"][k].float()) / 2
             for k, v in ca["model_state_dict"].items()}
    out = ROOT / "checkpoints" / f"t2_soup_{CKPT_A_PREFIX}ep{sel[RUN_A]['epoch']}_{CKPT_B_PREFIX}ep{sel[RUN_B]['epoch']}.pt"
    torch.save({
        "model_state_dict": state,
        "config": ca["config"],
        "tag": f"soup_{RUN_A}ep{sel[RUN_A]['epoch']}_{RUN_B}ep{sel[RUN_B]['epoch']}",
        "soup_type": "cross_trajectory_average",
        "soup_weights": {RUN_A: sel[RUN_A]["epoch"], RUN_B: sel[RUN_B]["epoch"]},
        "barrier_check": "stage barrier (subset windows) + full 400-window eval",
    }, out)
    print(f"Saved soup -> {out}", flush=True)
    return 0


# ---------------------------------------------------------------- stage 4
def eval_soup_full(trial: Path) -> int:
    """Full 400-window eval of the soup (writes t2_soup/epoch_trajectory/epoch_001.json)."""
    sel = json.loads((TRIALS / "t2_selection.json").read_text(encoding="utf-8"))
    soup = ROOT / "checkpoints" / f"t2_soup_{CKPT_A_PREFIX}ep{sel[RUN_A]['epoch']}_{CKPT_B_PREFIX}ep{sel[RUN_B]['epoch']}.pt"
    if not soup.exists():
        raise FileNotFoundError(f"{soup} — run --stage soup first")
    trial.mkdir(parents=True, exist_ok=True)
    shutil.copy(TRIALS / RUN_A / "override.json", trial / "override.json")
    payload = torch.load(soup, map_location="cpu", weights_only=False)
    entries = [{
        "epoch": 1, "path": str(soup.resolve()),
        "size_bytes": soup.stat().st_size,
        "train_loss": 3.0, "val_loss": 3.6,
        "learning_rate": 0.0, "learning_rate_adam": 0.0,
        "optimizer_steps_this_epoch": 0, "global_step": 0, "best_so_far": False,
    }]
    (trial / "model_checkpoints.json").write_text(json.dumps({
        "tag": payload["tag"], "save_path": str(soup.resolve()),
        "updated_epoch": 1, "checkpoints": entries}, indent=2), encoding="utf-8")
    p = trial / "epoch_trajectory" / "epoch_001.json"
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            print("Soup full eval already present (completed) — reusing")
            return 0
        print("Soup eval present but not completed — re-running")
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--trial_dir", str(trial),
        "--tokenizer", str(TOKENIZER),
        "--output_dir", str(trial / "epoch_trajectory"),
        "--epochs", "1",
        "--offsets", "0-399",
        "--n_days", "1",
        "--batch_size", "4",
        "--seed", "42",
        "--prepared_cache_dir", str(CACHE_A),
        "--no_reference_check",
    ]
    return run(cmd)


def stage_barrier(args: argparse.Namespace) -> int:
    """Full 400-window eval of the midpoint (soup) and compare token quality
    against BOTH endpoints on the same protocol. A barrier (ridge) shows as
    balance collapse / JSD spike vs the endpoints. Literal ToDo barrier check
    uses val loss, but raw val_loss is disabled as a health metric (ToDo §1.3)
    — token quality is the health metric here. Single-checkpoint full eval is
    ~65 s, so the subset-window shortcut is unnecessary."""
    sel = json.loads((TRIALS / "t2_selection.json").read_text(encoding="utf-8"))
    trial = TRIALS / "t2_soup"
    rc = eval_soup_full(trial)
    if rc:
        return rc
    rows = [("soup(midpoint)", load_aggregate(trial, 1))]
    for name, run_name in ((f"{CKPT_A_PREFIX}", RUN_A), (f"{CKPT_B_PREFIX}", RUN_B)):
        rows.append((f"{name} ep{sel[run_name]['epoch']}",
                     load_aggregate(TRIALS / run_name, sel[run_name]["epoch"])))
    print(f"\nBarrier check — full 400-window protocol:")
    print(f"{'model':<28}{'balance':>10}{'JSD':>10}{'collapse':>12}{'unique':>8}")
    for name, agg in rows:
        print(f"{name:<28}{agg['median_daily_codebook_balance_score']:>10.4f}"
              f"{agg['median_daily_token_jsd']:>10.4f}"
              f"{agg['median_daily_collapse_rate']:>12.4f}"
              f"{agg['median_daily_unique_tokens']:>8.1f}")
    bal = [r[1]["median_daily_codebook_balance_score"] for r in rows]
    jsd = [r[1]["median_daily_token_jsd"] for r in rows]
    s_bal, s_jsd = bal[0], jsd[0]
    barrier = s_bal < min(bal[1:]) - 0.02 or s_jsd > max(jsd[1:]) + 0.02
    print(f"\nBarrier detected: {barrier}")
    if barrier:
        print(f"  soup balance {s_bal:.4f} vs min endpoint {min(bal[1:]) - 0.02:.4f}, "
              f"soup JSD {s_jsd:.4f} vs max endpoint {max(jsd[1:]) + 0.02:.4f}")
    else:
        print("  midpoint within endpoints — soup premise holds")
    return 0


# ---------------------------------------------------------------- stage 4b
def stage_interp(args: argparse.Namespace) -> int:
    """Evaluate an interpolation point w = (1-a)*w4c72 + a*w8ceb along the
    line segment between the two selected weights. Maps the barrier shape
    (ToDo T2: barrier exists -> drop the soup)."""
    sel = json.loads((TRIALS / "t2_selection.json").read_text(encoding="utf-8"))
    a = args.alpha
    pa = ckpt_path(CKPT_A_PREFIX, sel[RUN_A]["epoch"])
    pb = ckpt_path(CKPT_B_PREFIX, sel[RUN_B]["epoch"])
    ca, cb = torch.load(pa, map_location="cpu", weights_only=False), torch.load(pb, map_location="cpu", weights_only=False)
    state = {k: (1 - a) * v.float() + a * cb["model_state_dict"][k].float()
             for k, v in ca["model_state_dict"].items()}
    out = ROOT / "checkpoints" / f"t2_interp_a{a:g}.pt"
    torch.save({"model_state_dict": state, "config": ca["config"],
                "tag": f"interp_alpha_{a:g}", "soup_type": "linear_interpolation",
                "soup_weights": {RUN_A: sel[RUN_A]["epoch"], RUN_B: sel[RUN_B]["epoch"],
                                 "alpha_towards_B": a}}, out)
    print(f"Saved interp alpha={a} -> {out}", flush=True)
    trial = TRIALS / f"t2_interp_a{a:g}"
    trial.mkdir(parents=True, exist_ok=True)
    shutil.copy(TRIALS / RUN_A / "override.json", trial / "override.json")
    (trial / "model_checkpoints.json").write_text(json.dumps({
        "tag": f"interp_alpha_{a:g}", "save_path": str(out.resolve()),
        "updated_epoch": 1, "checkpoints": [{
            "epoch": 1, "path": str(out.resolve()),
            "size_bytes": out.stat().st_size,
            "train_loss": 3.0, "val_loss": 3.6,
            "learning_rate": 0.0, "learning_rate_adam": 0.0,
            "optimizer_steps_this_epoch": 0, "global_step": 0, "best_so_far": False,
        }]}, indent=2), encoding="utf-8")
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--trial_dir", str(trial),
        "--tokenizer", str(TOKENIZER),
        "--output_dir", str(trial / "epoch_trajectory"),
        "--epochs", "1",
        "--offsets", "0-399",
        "--n_days", "1",
        "--batch_size", "4",
        "--seed", "42",
        "--prepared_cache_dir", str(CACHE_A),
        "--no_reference_check",
    ]
    rc = run(cmd)
    if rc:
        return rc
    agg = load_aggregate(trial, 1)
    print(f"\nalpha={a:g} (towards 8ceb): balance={agg['median_daily_codebook_balance_score']:.4f} "
          f"JSD={agg['median_daily_token_jsd']:.4f} collapse={agg['median_daily_collapse_rate']:.4f} "
          f"unique={agg['median_daily_unique_tokens']:.0f}")
    return 0


# ---------------------------------------------------------------- stage 5
def stage_eval(args: argparse.Namespace) -> int:
    """Paired bootstrap of the soup vs the best single trial (ToDo T2
    acceptance: balance/JSD not worse than the best single -> adopt)."""
    sel = json.loads((TRIALS / "t2_selection.json").read_text(encoding="utf-8"))
    trial = TRIALS / "t2_soup"
    rc = eval_soup_full(trial)
    if rc:
        return rc
    # reference = best single trial
    ref_run = RUN_A if sel[RUN_A]["balance"] >= sel[RUN_B]["balance"] else RUN_B
    ref_ep = sel[ref_run]["epoch"]
    print(f"\nReference (best single): {ref_run} ep{ref_ep}")
    print(f"\n=== balance bootstrap ===")
    rc = run([sys.executable, BOOTSTRAP,
              "--candidate", str(trial / "epoch_trajectory" / "epoch_001.json"),
              "--reference", str(TRIALS / ref_run / "epoch_trajectory" / f"epoch_{ref_ep:03d}.json"),
              "--key", "codebook_balance_score"])
    print(f"\n=== JSD ===")
    rc |= run([sys.executable, BOOTSTRAP,
               "--candidate", str(trial / "epoch_trajectory" / "epoch_001.json"),
               "--reference", str(TRIALS / ref_run / "epoch_trajectory" / f"epoch_{ref_ep:03d}.json"),
               "--key", "token_jsd"])
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["prepare_traj", "select", "soup", "barrier", "interp", "eval"])
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Interpolation weight towards the 8ceb weight (interp stage)")
    args = parser.parse_args()
    if args.stage == "prepare_traj":
        return stage_prepare_traj(args)
    if args.stage == "select":
        return stage_select(args)
    if args.stage == "soup":
        return stage_soup(args)
    if args.stage == "barrier":
        return stage_barrier(args)
    if args.stage == "interp":
        return stage_interp(args)
    return stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
