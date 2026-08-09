"""PT-90: generate a PostTrain proposal selection for human review.

Writes a proposal ``selection.json`` with ``human_review_recorded=false``,
``upstream_eligible=false``, ``ready_for_holdout=false``.  Training scripts never
flip these booleans; only a human record-selection step may (PostTrain-ToDo §20).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from posttrain_common import (  # noqa: E402
    POSTTRAIN_SELECTION_PATH, CPT_SELECTION_PATH, utc_now, dict_sha256,
    load_json, write_json,
)
from experiment_io import file_sha256  # noqa: E402


def build_proposal(pipeline: dict, claim: dict, protocol: dict,
                   out_path: Path = POSTTRAIN_SELECTION_PATH) -> Path:
    parent = load_json(CPT_SELECTION_PATH)
    ckpt = ROOT / parent["upstream"]["checkpoint"]
    parent_sel = {
        "path_under_results_root": "04b-cpt/seed42/trials/selection.json",
        "sha256": file_sha256(CPT_SELECTION_PATH),
        "checkpoint_id": "branchA_dm030_8ceb_ep1",
        "checkpoint_sha256": parent["upstream"]["checkpoint_sha256"],
        "tokenizer_sha256": parent["upstream"]["tokenizer_sha256"],
    }
    proposal = {
        "schema_version": "posttrain-selection-v1",
        "experiment": "Exp 06 PostTrain",
        "stage": "proposal",
        "created_at_utc": utc_now(),
        "parent_selection": parent_sel,
        "selected_pipeline": pipeline,
        "claim": claim,
        "protocol": {
            "validation_offsets": {"start": 0, "stop_exclusive": 400, "step": 1},
            "holdout_offset": 400,
            "holdout_days": 80,
            "holdout_used": False,
            "recipe_locked": False,
            "comparison_source_fingerprint": protocol.get("comparison_source_fingerprint"),
            "arm_source_hashes": protocol.get("arm_source_hashes", {}),
        },
        "statistics": {},
        "token_guardrail": {},
        "development_summary": {},
        "confirmation_summary": {},
        "full400_summary_after_confirmation": {},
        "upstream_cpt_provisional_single_seed": True,
        "seed_evidence": {
            "applicability": "not_applicable" if claim.get("training") is False
                              else "optimizer_replication",
            "development_seed": 42,
            "confirmation_seeds": [43, 44],
            "posttrain_three_seed_consistent": False,
            "seed_population_inference": False,
            "backbone_seed_robust": False,
        },
        # These three are ONLY flipped true by a human record-selection step.
        "human_review_recorded": False,
        "upstream_eligible": False,
        "ready_for_holdout": False,
    }
    write_json(out_path, proposal)
    print(f"[record_selection] proposal written (review pending): {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Write PostTrain proposal selection")
    ap.add_argument("--decoder", default="J3_posterior_median")
    ap.add_argument("--primary_metric", default="avg_daily_rank_ic")
    ap.add_argument("--minimum_effect", type=float, default=0.01)
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pipeline = {
        "decoder": {"arm": args.decoder, "temperature": {"Tc": 1.0, "Tf": 1.0}},
        "calibration": {},
        "heads": [],
        "ensemble": {},
        "abstention": {},
        "artifact_sha256": [],
        "training_seeds": [],
    }
    claim = {
        "primary_metric": args.primary_metric,
        "minimum_effect": args.minimum_effect,
        "reference": "same_upstream_exact_joint_decoder",
        "training": args.training,
    }
    out = Path(args.out) if args.out else POSTTRAIN_SELECTION_PATH
    build_proposal(pipeline, claim, {}, out_path=out)


if __name__ == "__main__":
    main()
