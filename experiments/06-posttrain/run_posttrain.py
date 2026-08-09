"""PostTrain stage orchestrator (PT-00 .. PT-90).

Usage:
    python run_posttrain.py preflight
    python run_posttrain.py tests
    python run_posttrain.py hidden_cache
    python run_posttrain.py pt01 --hidden <path>
    python run_posttrain.py train_cache
    python run_posttrain.py calibrate
    python run_posttrain.py probes
    python run_posttrain.py sets
    python run_posttrain.py proposal --decoder J3_posterior_median

Each stage validates the reviewed selection and refuses offsets >= 400.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from posttrain_common import load_reviewed_selection, require_offsets  # noqa: E402
from posttrain_common import resolve_roots, append_trial  # noqa: E402


def _run(cmd, *args):
    full = [sys.executable, str(HERE / cmd), *args]
    print(f"[run_posttrain] {' '.join(full)}")
    return subprocess.run(full, cwd=str(ROOT))


def preflight():
    sel = load_reviewed_selection()
    require_offsets(range(0, 400))
    print("[preflight] reviewed selection OK; offsets 0..399 allowed; holdout sealed")


def stage_hidden_cache():
    return _run("cache_hidden.py", "--require_cuda", "--device", "cuda")


def stage_pt01(hidden):
    return _run("evaluate_posttrain.py", "--hidden", hidden, "--chunk", "512")


def stage_train_cache():
    return _run("build_training_cache.py", "--device", "cuda")


def stage_calibrate():
    return _run("calibrate_posttrain.py", "--device", "cuda")


def stage_probes():
    return _run("train_heads.py")


def stage_proposal(decoder):
    return _run("record_selection.py", "--decoder", decoder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preflight", "tests", "hidden_cache", "pt01",
                                      "train_cache", "calibrate", "probes", "sets",
                                      "confirm", "proposal"])
    ap.add_argument("--hidden", default="server_runs/weights/06-posttrain/seed42/hidden_cache.npz")
    ap.add_argument("--decoder", default="J3_posterior_median")
    ap.add_argument("--arm", default="P6_mlp_rank_spearman")
    ap.add_argument("--seeds", default="43,44")
    args = ap.parse_args()

    if args.stage == "preflight":
        preflight()
    elif args.stage == "tests":
        rc = subprocess.run([sys.executable, str(HERE / "tests" / "test_contracts.py")])
        return rc.returncode
    elif args.stage == "hidden_cache":
        return stage_hidden_cache().returncode
    elif args.stage == "pt01":
        return stage_pt01(args.hidden).returncode
    elif args.stage == "train_cache":
        return stage_train_cache().returncode
    elif args.stage == "calibrate":
        return stage_calibrate().returncode
    elif args.stage == "probes":
        return _run("train_heads.py", "--stage", "pt03").returncode
    elif args.stage == "sets":
        return _run("train_heads.py", "--stage", "pt04").returncode
    elif args.stage == "confirm":
        return _run("confirm_seed.py", "--arm", args.arm, "--seeds", args.seeds).returncode
    elif args.stage == "proposal":
        return stage_proposal(args.decoder).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
