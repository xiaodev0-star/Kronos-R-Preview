"""analyze_r_series.py — consolidated R-series decision table (plan §3/§5).

Reads the six r_series_*.json artifacts and prints a one-page decision table:
per-arm full/dev/confirm RankIC, bootstrap vs the plan's references (J3/P6/J4),
plus the §5 decision-tree verdict.  Pure CPU, no caches loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
sys.path.insert(0, str(SEVEN))

RROOT = ROOT / "server_runs" / "results" / "07-bert-critic" / "seed42"

REFS = {"J3": 0.0398, "J2": 0.0365, "J4": 0.0393, "P6": 0.0535}

ARM_LABELS = {
    "r1": ["E_BERT_mean", "E_BERT_median", "P_BERT_up_raw", "P_BERT_up_naive"],
    "r3_bert": ["critic_pick_center", "critic_weighted_e"],
    "r5": ["P_up_avg_0.5", "P_up_fit_w1.0"],
}


def _arm_rows(d, labels):
    out = []
    for lbl in labels:
        m = d.get("full", {}).get(lbl)
        if not m:
            continue
        dev = d.get("dev_0_299", {}).get(lbl, {}).get("avg_daily_rank_ic")
        cfm = d.get("confirm_300_399", {}).get(lbl, {}).get("avg_daily_rank_ic")
        bv = {}
        for refname, refdata in (d.get("bootstrap_vs") or {}).items():
            b = (refdata or {}).get(lbl)
            if b and b.get("point") is not None:
                bv[refname] = (b["point"], b["block_robust"])
        out.append({"arm": lbl, "ic": m.get("avg_daily_rank_ic"),
                    "da": m.get("avg_da_per_date"), "dev": dev, "cfm": cfm, "bv": bv})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    tables = {}
    for key, labels in ARM_LABELS.items():
        p = RROOT / f"r_series_{key}.json"
        if p.exists():
            tables[key] = _arm_rows(json.load(open(p, encoding="utf-8")), labels)
    # r2 / r3_poe have dynamic labels
    for key in ("r2", "r3_poe"):
        p = RROOT / f"r_series_{key}.json"
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            if key == "r2":
                lam = d.get("calib_chosen_lambda")
                tables[key] = _arm_rows(d, [f"PoE_e_lambda_{lam:.1f}"] if lam is not None else [])
                if lam is not None:
                    tables[key].insert(0, {"arm": f"calib_chosen_λ={lam}", "ic": None,
                                           "dev": None, "cfm": None,
                                           "bv": {"__note__": d.get("calib_chosen_lambda_ic")}})
            else:
                tables[key] = _arm_rows(d, [k for k in d.get("full", {}) if k != "J3_median" and k != "P6"])

    print("=" * 90)
    print("R-series consolidated decision table (full 0..399, dense 3634)")
    print("=" * 90)
    for key, rows in tables.items():
        print(f"\n--- {key} ---")
        print(f"  {'arm':28s} {'full_ic':>9s} {'dev':>9s} {'cfm':>9s}  bootstrap point (robust)")
        for r in rows:
            parts = []
            for rn, val in r["bv"].items():
                if isinstance(val, tuple):
                    pt, rb = val
                    parts.append(f"{rn}:{pt:+.4f}{'*' if rb else ''}")
                else:
                    parts.append(f"{rn}:{val}")
            bv = "  ".join(parts)
            print(f"  {r['arm']:28s} {r['ic'] if r['ic'] is not None else '':>9} "
                  f"{str(round(r['dev'],4)) if r['dev'] is not None else '':>9} "
                  f"{str(round(r['cfm'],4)) if r['cfm'] is not None else '':>9}  {bv}")
    print("\nReferences: J2=0.0365  J3=0.0398  J4=0.0393  P6=0.0535")
    print("(* = block-robust bootstrap; __note__ carries calib diagnostics)")


if __name__ == "__main__":
    main()
