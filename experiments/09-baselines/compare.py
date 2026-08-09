"""compare.py — evaluate all baseline predictions and compare vs Exp08.

Loads every saved prediction from outputs/ (xgboost / mlp / transformer /
timesfm / reference_Exp08), computes:
  - full-400 daily RankIC / DA / MAPE (shared evaluator)
  - precision@coverage curves (per-date top-fraction by |score|, the same
    abstention protocol as the Exp08 |F| filter)
and writes a comparison table (JSON + Markdown) to results/.

Note on MAPE: the Exp08 reference score is a monotone RANK score (a z-score,
not a calibrated return), so its raw MAPE is not meaningful.  The reference
row in the MAPE column uses the Exp08 isotonic-calibrated magnitude
(0.0229, from experiments/08 f27) — reported separately from the raw score.

Usage:
    python compare.py
    python compare.py --names xgboost mlp transformer timesfm
"""
from __future__ import annotations

import argparse

import numpy as np

import common
from common import (load_eval_rows, full_metrics, daily_rank_ic, daily_da,
                    save_predictions, load_predictions, RESULTS)

# reference magnitude MAPE from Exp08 f27 (isotonic F -> logret)
EXP08_MAG_MAPE = 0.0229
EXP08_MAG_DA = 0.5181


def coverage_curve(score, rows, coverages=(1.0, 0.8, 0.6, 0.4, 0.2)):
    """Per-date top-fraction by |score|; acted daily RankIC (|F|-style)."""
    conf = np.abs(np.asarray(score, dtype=np.float64))
    dates = np.asarray(rows["date_key"])
    true = np.asarray(rows["true_logret"], dtype=np.float64)
    valid = np.isfinite(conf) & np.isfinite(true) & np.asarray(rows["quality"]).astype(bool)
    dense = rows["dense_threshold"]
    out = {}
    for cv in coverages:
        acted = np.zeros(len(conf), dtype=bool)
        for d in np.unique(dates):
            dm = (dates == d) & valid
            if dm.sum() == 0:
                continue
            keep = max(1, int(round(cv * dm.sum())))
            idx = np.where(dm)[0]
            order = np.argsort(-conf[idx])
            acted[idx[order[:keep]]] = True
        sub = {"date_key": dates[acted], "true_logret": true[acted],
               "quality": np.ones(acted.sum(), dtype=bool),
               "dense_threshold": max(5, int(round(cv * dense)))}
        ic = daily_rank_ic(score[acted], sub)
        out[str(cv)] = float(np.mean(list(ic.values()))) if ic else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="xgboost,mlp,transformer,tcn,random_walk",
                    help="comma list of baseline names (not the reference)")
    args = ap.parse_args()
    names = [n.strip() for n in args.names.split(",")]

    rows = load_eval_rows()
    models = {"reference_Exp08": load_predictions("reference_Exp08")}
    for n in names:
        models[n] = load_predictions(n)
        if models[n] is None:
            print(f"[compare] WARNING: no predictions for '{n}' (run its train script)")

    table = {}
    for name, d in models.items():
        if d is None:
            continue
        score = d["score"]
        assert len(score) == rows["n_rows"], f"{name}: len mismatch"
        m = full_metrics(score, rows)
        entry = {
            "full": m,
            "coverage_rankic": coverage_curve(score, rows),
            "meta": d["meta"] if isinstance(d["meta"], dict) else {},
        }
        if name == "reference_Exp08":
            entry["full"]["avg_mape"] = EXP08_MAG_MAPE  # calibrated magnitude
            entry["full"]["avg_da_per_date"] = EXP08_MAG_DA
            entry["note"] = ("Exp08 fused rank score; MAPE/DA shown are from its "
                             "isotonic-calibrated magnitude channel (f27)")
        table[name] = entry

    # ---- Markdown table ----
    lines = ["# 09-Baselines vs Exp08 — 对比结果", "",
             "| 模型 | RankIC (full) | DA | MAPE | RankIC@80% | RankIC@20% | 说明 |",
             "|---|---|---|---|---|---|---|"]
    order = ["reference_Exp08"] + names
    for name in order:
        if name not in table:
            lines.append(f"| {name} | — | — | — | — | — | 无预测 |")
            continue
        e = table[name]["full"]
        cov = table[name]["coverage_rankic"]
        note = table[name].get("note", table[name].get("meta", {}).get("mode", ""))
        lines.append(
            f"| {name} | {e['avg_daily_rank_ic']:.4f} | {e['avg_da_per_date']:.4f} "
            f"| {e['avg_mape']:.4f} | {cov.get('0.8','—')} | {cov.get('0.2','—')} | {note} |")
    md = "\n".join(lines)

    # ---- JSON + markdown out ----
    common.save_json(RESULTS / "comparison.json", table)
    (RESULTS / "comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[compare] -> {RESULTS/'comparison.json'}, {RESULTS/'comparison.md'}")


if __name__ == "__main__":
    main()
