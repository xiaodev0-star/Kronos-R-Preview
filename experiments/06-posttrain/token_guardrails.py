"""Token-health guardrail reference (PostTrain-ToDo §14.4).

Computes the coarse/fine/joint token metrics (codebook balance, collapse, JSD,
support F1, unique) per dense date from a verified decode (the J0 greedy path
matching ``predict_selected_ids``), then the median / p10 across the 400-day
window.  These are the ``ep1`` reference values that weight-changing methods
must satisfy (absolute floor + relative non-inferiority).

The metric formulas replicate ``eval_helpers.token_distribution_metrics``
(which is nested inside ``compute_windowed_metrics`` and not importable).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from posttrain_common import resolve_roots, write_json  # noqa: E402


def token_distribution_metrics(pred_tokens, true_tokens):
    """Identical math to eval_helpers.token_distribution_metrics."""
    pred_tokens = np.asarray(pred_tokens, dtype=np.int64)
    true_tokens = np.asarray(true_tokens, dtype=np.int64)
    paired = true_tokens >= 0
    pred_tokens = pred_tokens[paired]
    true_tokens = true_tokens[paired]
    if not len(true_tokens):
        return {}
    pred_labels, pred_counts = np.unique(pred_tokens, return_counts=True)
    true_labels, true_counts = np.unique(true_tokens, return_counts=True)
    labels = np.union1d(pred_labels, true_labels)
    pc = np.zeros(len(labels), dtype=np.float64)
    tc = np.zeros(len(labels), dtype=np.float64)
    pc[np.searchsorted(labels, pred_labels)] = pred_counts
    tc[np.searchsorted(labels, true_labels)] = true_counts
    pp = pc / pc.sum()
    tp = tc / tc.sum()
    mid = 0.5 * (pp + tp)

    def ent(p):
        nz = p > 0
        return float(-np.sum(p[nz] * np.log2(p[nz])))

    def kl(l, r):
        nz = l > 0
        return float(np.sum(l[nz] * np.log2(l[nz] / r[nz])))

    jsd = 0.5 * kl(pp, mid) + 0.5 * kl(tp, mid)
    pe = ent(pp)
    peff = float(2.0 ** pe)
    ps = set(pred_labels.tolist())
    ts = set(true_labels.tolist())
    ov = len(ps & ts)
    prec = ov / len(ps)
    rec = ov / len(ts)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    pcol = float(pc.max() / pc.sum())

    def sr(l, r):
        return 0.0 if l <= 0 or r <= 0 else float(min(l / r, r / l))

    bal = float((f1 * max(0.0, 1.0 - jsd) * sr(peff, 2.0 ** ent(tp))
                 * sr(pcol, float(tc.max() / tc.sum()))) ** 0.25)
    return {
        "balance": bal, "jsd": jsd, "collapse": pcol, "support_f1": f1,
        "unique": int(len(ps)), "token_accuracy": float(np.mean(pred_tokens == true_tokens)),
    }


def per_date_token_metrics(records, level_key_c, level_key_f, dense_threshold):
    """Per-date coarse/fine/joint token metrics from per-row records.

    ``records`` is a dict of arrays with date_key, pred_{level}_id, true_{level}_id.
    Returns dict of {date: metrics}.
    """
    dates = np.asarray(records["date_key"])
    uniq, inv = np.unique(dates, return_inverse=True)
    out = {}
    for i, d in enumerate(uniq):
        m = inv == i
        if m.sum() < dense_threshold:
            continue
        day = {k: records[k][m] for k in records}
        c = token_distribution_metrics(day["pred_coarse_id"], day["true_coarse_id"])
        f = token_distribution_metrics(day["pred_fine_id"], day["true_fine_id"])
        pred_joint = day["pred_coarse_id"] * 128 + day["pred_fine_id"]
        true_joint = day["true_coarse_id"] * 128 + day["true_fine_id"]
        j = token_distribution_metrics(pred_joint, true_joint)
        out[str(d)] = {
            "coarse": c, "fine": f, "joint": j, "n": int(m.sum()),
        }
    return out


def median_p10(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return None
    return {"median": float(np.median(arr)), "p10": float(np.percentile(arr, 10))}


def compute_ep1_reference(records, dense_threshold, out_path=None):
    """Compute the ep1 token guardrail reference from J0 records."""
    per_date = per_date_token_metrics(
        {
            "date_key": records["date_key"],
            "pred_coarse_id": records["greedy_c"],
            "pred_fine_id": records["greedy_f"],
            "true_coarse_id": records["true_coarse_id"],
            "true_fine_id": records["true_fine_id"],
        },
        None, None, dense_threshold,
    )
    ref = {"schema_version": "ep1-token-guardrail-v1", "source": "pt01 J0 greedy (verified)"}
    for level in ("coarse", "fine", "joint"):
        for metric in ("balance", "jsd", "collapse", "support_f1", "unique"):
            vals = [d[level][metric] for d in per_date.values()]
            agg = median_p10(vals)
            ref[f"{level}_{metric}"] = agg
    # H(target)-CE uses the dataset token summary (target marginals)
    import json as _json
    from pathlib import Path
    ts = _json.loads(Path("checkpoints/dataset_token_summary.json").read_text())
    ref["dataset_token_summary"] = {
        "coarse_entropy_bits": ts["splits"]["train"]["coarse"]["entropy_bits"],
        "fine_entropy_bits": ts["splits"]["train"]["fine"]["entropy_bits"],
    }
    ref["n_dense_dates"] = len(per_date)
    if out_path:
        write_json(out_path, ref)
    return ref


if __name__ == "__main__":
    rec = np.load(sys.argv[1], allow_pickle=True)
    dense_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 3634
    roots = resolve_roots()
    ref = compute_ep1_reference(rec, dense_threshold,
                                out_path=roots.results_root / "ep1_token_guardrail.json")
    print(json.dumps(ref, indent=2, ensure_ascii=False))
