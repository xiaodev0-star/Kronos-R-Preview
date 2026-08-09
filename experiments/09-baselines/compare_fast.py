"""compare_fast.py — fast comparison (vectorized coverage curves).

Same output contract as compare.py but with O(N log N) vectorized coverage
curves (reuses the within-date rank grouping), fast enough on a degraded
machine.  Loads every saved prediction + the per-model metrics JSON, and
writes results/comparison_v2.json + comparison_v2.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import common
from common import (load_eval_rows, full_metrics, daily_rank_ic, daily_da,
                    daily_mape, load_predictions, RESULTS)

EXP08_MAG_MAPE = 0.0229
EXP08_MAG_DA = 0.5181


def _split_points(dates):
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(sd)]]).astype(np.int64)
    uniq = [str(sd[int(bounds[i])]) for i in range(len(bounds) - 1)]
    return order, bounds, uniq


def coverage_curve(score, rows, coverages=(1.0, 0.8, 0.6, 0.4, 0.2)):
    """Per-date top-fraction by |score|; acted daily RankIC (vectorized)."""
    conf = np.abs(np.asarray(score, dtype=np.float64))
    dates = np.asarray(rows["date_key"])
    true = np.asarray(rows["true_logret"], dtype=np.float64)
    valid = np.isfinite(conf) & np.isfinite(true) & np.asarray(rows["quality"]).astype(bool)
    dense = rows["dense_threshold"]
    order, bounds, uniq = _split_points(dates)
    conf_o = conf[order]
    out = {}
    for cv in coverages:
        acted = np.zeros(len(conf), dtype=bool)
        for i in range(len(bounds) - 1):
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            m = np.where((conf_o[lo:hi] > -np.inf) & (conf_o[lo:hi] < np.inf))[0]  # all
            keep = max(1, int(round(cv * (hi - lo))))
            idx = np.argsort(-conf_o[lo:hi])[:keep]
            acted[order[lo + idx]] = True
        sub = {"date_key": dates[acted], "true_logret": true[acted],
               "quality": np.ones(acted.sum(), dtype=bool),
               "dense_threshold": max(5, int(round(cv * dense)))}
        ic = daily_rank_ic(score[acted], sub)
        out[str(cv)] = float(np.mean(list(ic.values()))) if ic else None
    return out


def main():
    names = ["xgboost", "xgboost_rank", "xgboost_rank2", "mlp", "transformer",
             "transformer_rank", "tcn", "random_walk"]
    rows = load_eval_rows()
    models = {"reference_Exp08": load_predictions("reference_Exp08")}
    for n in names:
        models[n] = load_predictions(n)

    table = {}
    for name, d in models.items():
        if d is None:
            print(f"[cmp] WARNING: no predictions for '{name}'")
            continue
        score = d["score"]
        assert len(score) == rows["n_rows"], f"{name}: len mismatch"
        m = full_metrics(score, rows)
        entry = {"full": m, "coverage_rankic": coverage_curve(score, rows),
                 "meta": d["meta"] if isinstance(d["meta"], dict) else {}}
        if name == "reference_Exp08":
            entry["full"]["avg_mape"] = EXP08_MAG_MAPE
            entry["full"]["avg_da_per_date"] = EXP08_MAG_DA
            entry["note"] = ("Exp08 fused rank score; MAPE/DA from its "
                             "isotonic-calibrated magnitude channel (f27)")
        table[name] = entry
        print(f"[cmp] {name:16s} RankIC={m['avg_daily_rank_ic']:.4f} "
              f"ic@20%={entry['coverage_rankic'].get('0.2')}")

    lines = ["# 09-Baselines vs Exp08 — 对比结果 (v2)", "",
             "| 模型 | RankIC (full) | DA | MAPE | RankIC@80% | RankIC@20% |",
             "|---|---|---|---|---|---|"]
    order = ["reference_Exp08"] + names
    for name in order:
        if name not in table:
            lines.append(f"| {name} | — | — | — | — | — |")
            continue
        e = table[name]["full"]
        cov = table[name]["coverage_rankic"]
        lines.append(
            f"| {name} | {e['avg_daily_rank_ic']:.4f} | {e['avg_da_per_date']:.4f} "
            f"| {e['avg_mape']:.4f} | {cov.get('0.8','—')} | {cov.get('0.2','—')} |")
    md = "\n".join(lines)

    common.save_json(RESULTS / "comparison_v2.json", table)
    (RESULTS / "comparison_v2.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[cmp] -> {RESULTS/'comparison_v2.json'}, {RESULTS/'comparison_v2.md'}")


if __name__ == "__main__":
    main()
