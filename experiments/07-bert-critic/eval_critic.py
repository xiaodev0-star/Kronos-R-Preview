"""eval_critic.py — plan §10: C-a/C-b controls + B0-B5 arm evaluation.

Pipeline gate (§10.2, contract T8): C-a (true-token sorting control) and C-b
(shuffled-history control) must run and write their artifact BEFORE any B1-B5
evaluation summary is emitted.

Arms (§10.1):
    B0  no critic (P6 rank score primary; J3 posterior median fallback)  — reference
    B1  log p_BERT alone ranking (diagnostic, not a candidate)
    B2  F-INT interpolation (zero-training fusion main arm)
    B3  F-STK stacking / B4 F-SEL (trained combiner, produced by fuse_scores.py)
    B5  B3/B4 + ELECTRA features (conditional on Stage 2a)

Statistics: daily RankIC + DA + MAE, paired circular moving-block bootstrap
(L=5/10/20) vs B0; dev=0..299, confirm=300..399.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_io import file_sha256  # noqa: E402
from config import DataConfig  # noqa: E402

from bert_data import VOCAB_BASE  # noqa: E402
from critic_common import resolve_roots, upstream_paths, append_trial  # noqa: E402
from compare_posttrain import paired_bootstrap_ci  # noqa: E402
from evaluate_posttrain import arm_metrics  # noqa: E402
from score_bert import load_bert, build_index, score_rows, NumpyRowTable  # noqa: E402
from build_gpt_candidates import coarse_q_for_hidden, load_model_cpu  # noqa: E402


def require_controls(controls_path) -> None:
    """Refuse to produce B1-B5 summaries without the C-a/C-b artifact."""
    if not Path(controls_path).exists():
        raise RuntimeError(
            "C-a/C-b controls must run and write their artifact before B1-B5 "
            f"(missing {controls_path})")


# ============================================================================
# C-a / C-b controls
# ============================================================================

def _sample_fit_rows(training_cache, n_target=50_000):
    """Sample n_target (uid, date) rows from the fit-region cache by stride.

    Returns (NumpyRowTable, cache_npz, idx, true_coarse_ids).
    """
    from score_bert import NumpyRowTable
    c = np.load(training_cache, allow_pickle=True)
    n = len(c["stock_uid"])
    stride = max(1, n // n_target)
    idx = np.arange(0, n, stride)
    table = NumpyRowTable(c["stock_uid"][idx], c["date_key"][idx])
    true_coarse = c["true_coarse_id"][idx].astype(np.int64)
    return table, c, idx, true_coarse


@torch.no_grad()
def _gpt_q_for_rows(model, hidden_cache, idx):
    """q(c) at T_c=1.4 for the sampled training-cache rows."""
    h = torch.from_numpy(hidden_cache["hidden"][idx]).float()
    h = h.to(next(model.parameters()).device)
    q, sm = coarse_q_for_hidden(model, h, t_c=1.4,
                                vocab_base=model._vocab_l1)
    return q.cpu().numpy(), sm.cpu().numpy()


def _sample_distractors(q_row, true_id, n_dist=7, seed=0):
    """Sample n_dist distractors from q (excluding the true token)."""
    rng = np.random.RandomState(seed)
    probs = q_row.copy()
    probs[true_id] = 0.0
    probs /= probs.sum()
    candidates = np.arange(VOCAB_BASE, dtype=np.int64)
    d = rng.choice(candidates, size=n_dist, p=probs, replace=False)
    return d


def _true_rank(logp_row, true_id, distractor_ids):
    """Rank of the true token among {true} ∪ distractors (1 = best)."""
    scores = np.concatenate([[logp_row[true_id]], logp_row[distractor_ids]])
    # lower rank value = higher score (rank 1 = highest score)
    order = np.argsort(-scores)
    return int(np.where(order == 0)[0][0]) + 1


def run_controls(*, bert_path, index, rows, gpt_model, hidden_cache,
                 true_coarse, cache_idx, n_shuffle=10_000, window=512,
                 batch_size=32, device="cuda"):
    """Run C-a (true-token sorting) and C-b (shuffled-history) controls.

    ``rows`` is a NumpyRowTable sampled from the fit region; ``true_coarse`` and
    ``cache_idx`` give the aligned true coarse id and the training-cache row
    index per sampled row (used to pull hidden states for GPT q(c)).  Returns
    the controls dict.
    """
    from score_bert import NumpyRowTable
    model, cfg, _ = load_bert(bert_path, torch.device(device))
    n = len(rows)
    # ---- C-a: score the true next-day slot ----
    scores_a = score_rows(index, rows, model, window=window,
                          batch_size=batch_size, device=device)
    logp = scores_a["logp_bert_full"]                       # [N, 128]
    q, _ = _gpt_q_for_rows(gpt_model, hidden_cache, cache_idx)

    ranks = np.empty(n, dtype=np.float64)
    hits = np.empty(n, dtype=bool)
    auc_vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = _sample_distractors(q[i], int(true_coarse[i]), n_dist=7, seed=i)
        rank = _true_rank(logp[i], int(true_coarse[i]), d)
        ranks[i] = rank
        hits[i] = (rank == 1)
        auc_vals[i] = np.mean(logp[i, int(true_coarse[i])] > logp[i, d])
    c_a = {
        "n_rows": n,
        "avg_true_rank": float(ranks.mean()),
        "uniform_rank_expectation": 4.5,           # (8+1)/2
        "top1_hit_rate": float(hits.mean()),
        "uniform_top1_expectation": 1.0 / 8.0,
        "auc_true_vs_distractors": float(auc_vals.mean()),
        "pass": bool(ranks.mean() < 4.5 - 0.5) and bool(hits.mean() > 1.0 / 8.0 + 0.02),
    }

    # ---- C-b: shuffle history order -> discrimination should collapse ----
    m = min(n_shuffle, n)
    shuf_rows = NumpyRowTable(rows._u[:m], rows._d[:m])
    shuf_scores = score_rows(index, shuf_rows, model, window=window,
                             batch_size=batch_size, device=device,
                             shuffle_history=True)
    logp_sh = shuf_scores["logp_bert_full"]
    q_sh, _ = _gpt_q_for_rows(gpt_model, hidden_cache, cache_idx[:m])
    ranks_sh = np.empty(m, dtype=np.float64)
    hits_sh = np.empty(m, dtype=bool)
    for i in range(m):
        d = _sample_distractors(q_sh[i], int(true_coarse[i]), n_dist=7, seed=i)
        ranks_sh[i] = _true_rank(logp_sh[i], int(true_coarse[i]), d)
        hits_sh[i] = (ranks_sh[i] == 1)
    c_b = {
        "n_rows": m,
        "shuffled_avg_true_rank": float(ranks_sh.mean()),
        "shuffled_top1_hit_rate": float(hits_sh.mean()),
        "collapsed_toward_uniform": bool(ranks_sh.mean() >= 4.5 - 0.3),
    }
    return {"c_a": c_a, "c_b": c_b, "config": {"window": window,
                                               "tc_candidates": 1.4,
                                               "n_distractors": 7}}


# ============================================================================
# B0-B5 arm evaluation
# ============================================================================

def load_p6_scores(head_path, hidden):
    """Evaluate the 06 P6 MLP rank head on hidden [N, dim] -> [N] scores."""
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    sd = ck["head_state"]
    # MlpRankHead(dim=256, hidden=64, dropout=0.0) -> Linear(0)/SiLU(1)/Identity(2)/Linear(3)
    net = torch.nn.Sequential(
        torch.nn.Linear(256, 64), torch.nn.SiLU(), torch.nn.Identity(),
        torch.nn.Linear(64, 1))
    net.load_state_dict({k.replace("net.", ""): v for k, v in sd.items()})
    net.eval()
    with torch.no_grad():
        s = net(torch.from_numpy(hidden).float()).squeeze(-1).numpy()
    return s


def _dev_confirm(offsets):
    """Split offsets into dev (0..299) and confirm (300..399)."""
    dev = [o for o in offsets if o <= 299]
    conf = [o for o in offsets if 300 <= o < 400]
    return dev, conf


def bootstrap_vs_reference(cand_rec, ref_rec, score_field, dense_min=3634,
                           block_lengths=(5, 10, 20), n_replicates=10_000):
    """Paired moving-block bootstrap of candidate vs reference daily RankIC.

    Both recs must share the same (date, stock_uid) universe and finite mask.
    ``cand_rec[score_field]`` is the candidate's per-row rank score;
    ``ref_rec`` uses its own ``p6_score`` as the reference.  Returns the
    compare_posttrain result dict (point + L=5/10/20 CIs).
    """
    from compare_posttrain import paired_bootstrap_ci

    def to_rows(rec, field):
        rows = []
        valid = np.isfinite(rec[field]) & np.isfinite(rec["true_logret"]) & rec["quality"]
        for i in np.where(valid)[0]:
            rows.append({"date_key": str(rec["date_key"][i]),
                         "stock_uid": str(rec["stock_uid"][i]),
                         "rank_score": float(rec[field][i]),
                         "true_logret": float(rec["true_logret"][i])})
        return rows

    cand_rows = to_rows(cand_rec, score_field)
    ref_rows = to_rows(ref_rec, "p6_score")
    return paired_bootstrap_ci(cand_rows, ref_rows, "rank_ic",
                               dense_min=dense_min,
                               block_lengths=block_lengths,
                               n_replicates=n_replicates)


def evaluate_arms(*, candidates, scores_eval, p6_path, dense_threshold,
                  out_json=None, score_fields=None, fused_scores_npz=None):
    """Build B0/B1/B2 arm records and compute metrics + paired bootstrap.

    ``score_fields``: dict arm_name -> dict(score_field, source) where source
    is one of ``p6``, ``post_median``, ``bert_max``, ``fint`` (needs fused_scores).
    """
    cand = np.load(candidates, allow_pickle=True)
    sc = np.load(scores_eval, allow_pickle=True) if scores_eval else None
    n = len(cand["stock_uid"])

    rec = {
        "date_key": cand["date_key"], "stock_uid": cand["stock_uid"],
        "true_logret": cand["true_logret"].astype(np.float64),
        "quality": cand["quality"].astype(bool),
        "post_median": cand["post_median"].astype(np.float64),
        "p_up": cand["p_up"].astype(np.float64),
        "post_std": cand["post_std"].astype(np.float64),
        "topk_logq": cand["topk_logq"], "topk_ids": cand["topk_ids"],
    }
    if sc is not None:
        # row alignment of scores_eval vs candidates (same region/order)
        if len(sc["stock_uid"]) != n:
            raise RuntimeError(
                f"scores_eval rows {len(sc['stock_uid'])} != candidates rows {n}")
        rec["logp_bert_topk"] = sc["logp_bert_topk"]
        rec["logp_bert_full"] = sc["logp_bert_full"]
        rec["bert_margin"] = sc["bert_margin"]

    # P6 (B0 primary)
    p6 = load_p6_scores(p6_path, np.load(ROOT / "server_runs" / "weights" /
                         "06-posttrain" / "seed42" / "hidden_cache.npz",
                         allow_pickle=True)["hidden"])
    if len(p6) != n:
        raise RuntimeError(f"P6 scores {len(p6)} != candidates rows {n}")
    rec["p6_score"] = p6

    # B1: max over candidates of log p_BERT
    rec["bert_score"] = np.nanmax(rec["logp_bert_topk"], axis=1) if sc is not None else None
    # B2 (F-INT) fused scores injected via score_fields from fuse_scores.py
    if fused_scores_npz is not None and Path(fused_scores_npz).exists():
        fs = np.load(fused_scores_npz, allow_pickle=True)
        if len(fs["stock_uid"]) != n:
            raise RuntimeError("fused F-INT scores misaligned with candidates")
        rec["fint_score"] = fs["fint_score"].astype(np.float64)
    arms = {}
    if "B0_p6" in (score_fields or {}):
        arms["B0_p6"] = arm_metrics(rec, "p6_score", dense_threshold)
    if "B0_j3" in (score_fields or {}):
        arms["B0_j3"] = arm_metrics(rec, "post_median", dense_threshold)
    if "B1_bert" in (score_fields or {}):
        arms["B1_bert"] = arm_metrics(rec, "bert_score", dense_threshold)
    for name, fld in (score_fields or {}).items():
        if name in arms or fld == "p6_score" or fld == "post_median" or fld == "bert_score":
            continue
        if fld in rec:
            arms[name] = arm_metrics(rec, fld, dense_threshold)

    result = {"schema": "critic-eval-v1", "n_rows": n,
              "dense_threshold": dense_threshold, "arms": arms,
              "offsets_scope": "0_399", "holdout_used": False}
    if out_json:
        from critic_common import write_json
        write_json(out_json, result)
    return result, rec


def main():
    ap = argparse.ArgumentParser(description="BERT critic controls + arm evaluation")
    ap.add_argument("--mode", choices=["controls", "arms", "all"], default="all")
    ap.add_argument("--n_controls", type=int, default=50_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bert", default=None,
                    help="BERT checkpoint for controls (default mlm_v1; use a "
                         "fine-tuned checkpoint to gate T1/T2 via require_ca_not_regressed)")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    controls_path = roots.results_root / "controls.json"
    training_cache = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "training_cache.npz"
    candidates = roots.weights_root / "candidates_eval_K8.npz"
    scores_eval = roots.weights_root / "scores_eval_K8_w512_stride1.npz"
    p6_path = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "head_P6_mlp_rank_spearman.pt"
    bert_path = Path(args.bert) if args.bert else ROOT / "checkpoints" / "bert_critic_mlm_v1.pt"

    if args.mode in ("controls", "all"):
        if not bert_path.exists():
            raise RuntimeError(f"Stage-1 BERT checkpoint missing: {bert_path}")
        ckpt, tok_path = upstream_paths()
        index_cache = roots.weights_root / "bert_input_index.pkl"
        index = build_index(tok_path, device="cpu", cache_path=index_cache)
        gpt_model, _ = load_model_cpu(ckpt, tok_path)
        rows, cache, cache_idx, true_coarse = _sample_fit_rows(
            training_cache, args.n_controls)
        # filter to stocks present in the index (uid axis)
        keep = np.array([rows.stock_uid(i) in index for i in range(len(rows))])
        rows = NumpyRowTable(rows._u[keep], rows._d[keep])
        cache_idx = cache_idx[keep]
        true_coarse = true_coarse[keep]
        device = args.device if torch.cuda.is_available() else "cpu"
        gpt_model = gpt_model.to(device)
        controls = run_controls(bert_path=bert_path, index=index, rows=rows,
                                gpt_model=gpt_model, hidden_cache=cache,
                                true_coarse=true_coarse, cache_idx=cache_idx,
                                device=device)
        from critic_common import write_json
        write_json(controls_path, controls)
        append_trial({"event": "controls", "n": len(rows), "status": "ok",
                      "c_a_pass": controls["c_a"]["pass"]})
        print(json.dumps(controls, indent=2, ensure_ascii=False))

    if args.mode in ("arms", "all"):
        require_controls(controls_path)
        if not scores_eval.exists():
            raise RuntimeError(f"eval BERT scores missing: {scores_eval}")
        cand = np.load(candidates, allow_pickle=True)
        # F-INT fused scores (zero-training main arm, B2) — lambda fit on calib
        from fuse_scores import run_fint
        fused_npz = roots.weights_root / "fint_scores_eval.npz"
        calib_cand = roots.weights_root / "candidates_calib_K8.npz"
        calib_sc = roots.weights_root / "scores_calib_K8_w512_stride1.npz"
        run_fint(candidates=candidates, scores_eval=scores_eval,
                 out_json=roots.results_root / "fint_result.json",
                 out_scores_npz=fused_npz,
                 calib_candidates=calib_cand if calib_cand.exists() else None,
                 calib_scores=calib_sc if calib_sc.exists() else None)
        result, rec = evaluate_arms(
            candidates=candidates, scores_eval=scores_eval, p6_path=p6_path,
            dense_threshold=int(cand["dense_threshold"][0]),
            out_json=roots.results_root / "arms_b012.json",
            score_fields={"B0_p6": "p6_score", "B0_j3": "post_median",
                          "B1_bert": "bert_score", "B2_fint": "fint_score"},
            fused_scores_npz=fused_npz)
        # paired moving-block bootstrap of each candidate arm vs B0 (P6)
        from critic_common import write_json
        comparisons = {}
        for name, field in (("B2_fint", "fint_score"), ("B1_bert", "bert_score")):
            if field in rec:
                comparisons[name] = bootstrap_vs_reference(
                    rec, rec, field, dense_min=int(cand["dense_threshold"][0]))
        result["bootstrap_vs_B0_p6"] = comparisons
        write_json(roots.results_root / "arms_b012.json", result)
        print(json.dumps({"arms": {k: v for k, v in result["arms"].items()},
                          "bootstrap_vs_B0_p6": comparisons},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
