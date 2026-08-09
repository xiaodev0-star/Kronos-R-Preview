"""compare_t1_r1.py — T1 (scoring-aligned fine-tune) vs mlm_v1 R1 comparison.

Reads r_series_r1.json (mlm_v1) and the T1-suffixed rerun, prints a side-by-side
E_BERT table + bootstrap deltas.  The T1 acceptance (plan §4 T1): C-a not
regressed (gate already checked) + E_BERT[r] RankIC improves over mlm_v1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

RROOT = ROOT / "server_runs" / "results" / "07-bert-critic" / "seed42"
FIELDS = ["E_BERT_mean", "E_BERT_median", "P_BERT_up_raw", "P_BERT_up_naive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", default=str(RROOT / "r_series_r1_t1_w512.json"))
    args = ap.parse_args()
    v1 = json.load(open(RROOT / "r_series_r1.json", encoding="utf-8"))
    t1 = json.load(open(args.t1, encoding="utf-8"))
    print("=" * 88)
    print("R1: mlm_v1 vs T1 (W=512) — full 0..399")
    print("=" * 88)
    print(f"  {'field':20s} {'mlm_v1_ic':>10s} {'T1_ic':>10s} {'delta':>8s} "
          f"{'dev(mlm/T1)':>20s} {'cfm(mlm/T1)':>20s}")
    for f in FIELDS:
        a = v1["full"].get(f, {}); b = t1["full"].get(f, {})
        if not a or not b:
            print(f"  {f:20s} (missing)")
            continue
        ia, ib = a.get("avg_daily_rank_ic"), b.get("avg_daily_rank_ic")
        da = v1["dev_0_299"].get(f, {}).get("avg_daily_rank_ic")
        db = t1["dev_0_299"].get(f, {}).get("avg_daily_rank_ic")
        ca = v1["confirm_300_399"].get(f, {}).get("avg_daily_rank_ic")
        cb = t1["confirm_300_399"].get(f, {}).get("avg_daily_rank_ic")
        print(f"  {f:20s} {ia:10.4f} {ib:10.4f} {ib-ia:+8.4f} "
              f"({da:.4f}/{db:.4f}) {('('+f'{ca:.4f}'+'/'+f'{cb:.4f}'+')'):>20s}")
    print("\n  references (mlm_v1): J3=0.0398 J2=0.0365 J4=0.0393 P6=0.0535")
    print("  T1 bootstrap vs J3 (point / robust):")
    for f in FIELDS:
        bv = t1["bootstrap_vs"]["vs_J3"].get(f)
        if bv and bv.get("point") is not None:
            print(f"    {f:20s} {bv['point']:+.4f}  robust={bv['block_robust']}")


if __name__ == "__main__":
    main()
