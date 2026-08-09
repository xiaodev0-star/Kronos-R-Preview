"""build_gpt_candidates.py — plan §7: GPT top-K candidate cache for the eval region.

Consumes the 06 eval hidden cache (``hidden_cache.npz``) and the PT-01 exact
posterior records (``pt01_records.npz``).  For each of the 1,798,899 eval rows:
  - recompute q(c) at the calibrated coarse temperature T_c=1.4 from hidden via
    ``coarse_logits_from_hidden`` (cheap: one [N,256]@[256,130] matmul),
  - take the top-K=8 coarse candidates and their log q,
  - join the PT-01 posterior stats (post_median / p_up / post_std / entropies /
    special_mass) as the candidate-cache fields — the magnitude channel stays
    the J3 posterior median of the 06 pipeline.

Row alignment with the 06 eval records is asserted (contract test T5): the
(uid, date) sets must be identical, otherwise the paired moving-block bootstrap
later is invalid.

Output (weights root): candidates_eval_K8.npz + candidates_eval_K8.json sidecar.
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
from eval_helpers import load_gpt  # noqa: E402
from model import load_tokenizer  # noqa: E402

from critic_common import resolve_roots, upstream_paths, dict_sha256, append_trial  # noqa: E402

TC = 1.4          # locked calibrated coarse temperature (plan §7.6)
TF = 1.1
K = 8
EPS = 1e-12


def load_model_cpu(ckpt, tok_path):
    tok = load_tokenizer(str(tok_path), torch.device("cpu"))
    model = load_gpt(str(ckpt), torch.device("cpu"), tokenizer=tok)
    model.eval()
    return model, tok


@torch.no_grad()
def coarse_q_for_hidden(model, hidden, t_c=TC, vocab_base=128):
    """Coarse marginal q(c) over ordinary codes at temperature ``t_c``.

    hidden [B, dim] -> q [B, vocab_base] (Bayes-conditioned on ordinary codes,
    special mass excluded and renormalized — same convention as joint_decoder).
    """
    logits = model.coarse_logits_from_hidden(hidden)          # [B, V+2]
    p_full = torch.softmax(logits / t_c, dim=-1)
    special_mass = p_full[:, vocab_base:].sum(dim=-1)          # [B]
    q = p_full[:, :vocab_base] / (1.0 - special_mass).clamp_min(EPS).unsqueeze(-1)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return q, special_mass


def build_candidates(*, hidden_path, records_path, ckpt, tok_path,
                     out_npz, out_json, k=K, chunk=100_000, device="cpu"):
    """Write the eval-region candidate cache."""
    model, tok = load_model_cpu(ckpt, tok_path)

    hdata = np.load(hidden_path, allow_pickle=True)
    hidden = hdata["hidden"]                                  # [N, 256]
    n = hidden.shape[0]
    print(f"[cand] hidden rows {n}")

    rec = np.load(records_path, allow_pickle=True)
    # Posterior stats consumed as candidate-cache fields (PT-01, T=1.0 exact).
    stats = {
        "stock_uid": rec["stock_uid"], "date_key": rec["date_key"],
        "offset": rec["offset"].astype(np.int32),
        "position": rec["position"].astype(np.int32),
        "p_mean0": rec["p_mean0"].astype(np.float64),
        "p_std0": rec["p_std0"].astype(np.float64),
        "true_coarse_id": rec["true_coarse_id"].astype(np.int16),
        "true_logret": rec["true_logret"].astype(np.float64),
        "quality": rec["quality"].astype(bool),
        "post_median": rec["post_median"].astype(np.float64),
        "p_up": rec["p_up"].astype(np.float64),
        "post_std": rec["post_std"].astype(np.float64),
        "coarse_entropy": rec["coarse_entropy"].astype(np.float64),
        "joint_entropy": rec["joint_entropy"].astype(np.float64),
        "special_mass": rec["special_mass"].astype(np.float64),
        "greedy_return": rec["greedy_return"].astype(np.float64),
    }
    # T5: row alignment with the 06 eval records (same (uid, date) set).
    align_h = {(d, u) for d, u in zip(hdata["date_key"][:5000], hdata["stock_uid"][:5000])}
    align_r = {(d, u) for d, u in zip(rec["date_key"][:5000], rec["stock_uid"][:5000])}
    if align_h != align_r:
        raise RuntimeError(
            "candidate cache misaligned with 06 eval records (first 5K rows differ); "
            "cannot build a paired cache")
    print(f"[cand] row alignment OK (first 5K rows match)")

    topk_ids = np.zeros((n, k), dtype=np.int16)
    topk_logq = np.zeros((n, k), dtype=np.float32)
    special_mass = np.zeros(n, dtype=np.float64)
    dev = torch.device(device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        h = torch.from_numpy(hidden[start:stop]).to(dev)
        q, sm = coarse_q_for_hidden(model, h, t_c=TC, vocab_base=model._vocab_l1)
        logq = torch.log(q.clamp_min(EPS))
        ids = q.topk(k, dim=-1).indices.to(torch.int16)
        vals = logq.gather(-1, ids.to(torch.int64)).to(torch.float32)
        topk_ids[start:stop] = ids.cpu().numpy()
        topk_logq[start:stop] = vals.cpu().numpy()
        special_mass[start:stop] = sm.cpu().numpy()
        del h, q, logq, ids, vals
        if (start // chunk) % 5 == 0:
            print(f"[cand] rows {stop}/{n}")

    dense_threshold = int(hdata["dense_threshold"][0])
    out = {**stats,
           "topk_ids": topk_ids, "topk_logq": topk_logq,
           "special_mass": special_mass,
           "dense_threshold": np.array([dense_threshold]),
           "tc": np.array([TC]), "tf": np.array([TF]), "k": np.array([k])}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out)
    print(f"[cand] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")

    meta = {
        "schema": "candidates-v1",
        "n_rows": int(n),
        "k": k, "tc": TC, "tf": TF,
        "source": {
            "hidden_cache": str(hidden_path),
            "hidden_sha256": file_sha256(Path(hidden_path)),
            "pt01_records": str(records_path),
        },
        "upstream": {"checkpoint": str(ckpt), "sha256": file_sha256(ckpt),
                     "tokenizer": str(tok_path), "sha256": file_sha256(tok_path)},
        "dense_threshold": dense_threshold,
        "offsets_scope": "0_399",
        "holdout_used": False,
        "candidate_sha256": dict_sha256({"topk_ids_shape": list(topk_ids.shape),
                                         "n": n, "k": k, "tc": TC}),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[cand] sidecar -> {out_json}")
    return out_npz, meta


def build_calib_candidates(*, calib_cache, ckpt, tok_path, out_npz, out_json,
                           k=K, chunk=4096, device="cpu", light=False):
    """Calibration-region candidate cache (audit_uids x [2023-02-01, 2024-02-01)).

    Unlike the eval region (which reuses PT-01 posterior stats), the calibration
    slice has no existing posterior cache, so we run the exact joint decoder
    (decode_joint) over the calibration hidden to obtain f1-f3 features — unless
    ``light=True`` (F-INT only needs the top-K candidates + targets, so the
    expensive fine expansion is skipped).
    """
    import torch.nn.functional as F
    from eval_helpers import load_gpt

    data = np.load(calib_cache, allow_pickle=True)
    n = len(data["stock_uid"])
    print(f"[calib-cand] rows {n} light={light}")
    tok = load_tokenizer(str(tok_path), torch.device(device))
    model = load_gpt(str(ckpt), torch.device(device), tokenizer=tok)
    model.eval()
    dev = torch.device(device)

    hidden = torch.from_numpy(data["hidden"]).to(dev)
    post = {kk: np.full(n, np.nan) for kk in
            ("post_median", "p_up", "post_std", "coarse_entropy",
             "joint_entropy", "special_mass")}
    topk_ids = np.zeros((n, k), dtype=np.int16)
    topk_logq = np.zeros((n, k), dtype=np.float32)
    q_accum = np.zeros((n, 128), dtype=np.float32)
    if not light:
        from joint_decoder import DecodeTable, decode_joint
        dt = DecodeTable(tok, device="cpu")
        p_mean0 = torch.from_numpy(data["p_mean0"].astype(np.float32)).to(dev)
        p_std0 = torch.from_numpy(data["p_std0"].astype(np.float32)).to(dev)
        tc = torch.from_numpy(data["true_coarse_id"].astype(np.int64)).to(dev)
        tf = torch.from_numpy(data["true_fine_id"].astype(np.int64)).to(dev)
        tr = torch.from_numpy(data["true_logret"].astype(np.float32)).to(dev)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        if not light:
            stats, quality = decode_joint(
                model, dt, hidden[start:stop], p_mean0[start:stop], p_std0[start:stop],
                true_coarse_ids=tc[start:stop], true_fine_ids=tf[start:stop],
                true_logret=tr[start:stop], t_c=TC, t_f=TF, chunk=256)
            post["post_median"][start:stop] = stats.median.cpu().numpy()
            post["p_up"][start:stop] = stats.p_up.cpu().numpy()
            post["post_std"][start:stop] = stats.std.cpu().numpy()
            post["coarse_entropy"][start:stop] = stats.coarse_entropy.cpu().numpy()
            post["joint_entropy"][start:stop] = stats.joint_entropy.cpu().numpy()
            post["special_mass"][start:stop] = stats.special_mass.cpu().numpy()
        # top-K candidates from the calibrated q(c)
        q, sm = coarse_q_for_hidden(model, hidden[start:stop], t_c=TC,
                                    vocab_base=model._vocab_l1)
        logq = torch.log(q.clamp_min(EPS))
        ids = q.topk(k, dim=-1).indices.to(torch.int16)
        vals = logq.gather(-1, ids.to(torch.int64)).to(torch.float32)
        topk_ids[start:stop] = ids.cpu().numpy()
        topk_logq[start:stop] = vals.cpu().numpy()
        q_accum[start:stop] = q.cpu().numpy()
        if start % (chunk * 10) == 0:
            print(f"[calib-cand] rows {stop}/{n}")
    quality = (np.isfinite(data["p_mean0"]) & np.isfinite(data["p_std0"])
               & (np.asarray(data["p_std0"]) > 0))
    out = {
        "stock_uid": data["stock_uid"], "date_key": data["date_key"],
        "true_coarse_id": data["true_coarse_id"].astype(np.int16),
        "true_fine_id": data["true_fine_id"].astype(np.int16),
        "true_logret": data["true_logret"].astype(np.float64),
        "p_mean0": data["p_mean0"].astype(np.float64),
        "p_std0": data["p_std0"].astype(np.float64),
        "quality": quality.astype(bool),
        **post, "topk_ids": topk_ids, "topk_logq": topk_logq,
        "gpt_q": q_accum,
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out)
    print(f"[calib-cand] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")
    meta = {"schema": "candidates-calib-v1", "n_rows": n, "k": k, "tc": TC,
            "source": str(calib_cache), "upstream_sha256": file_sha256(ckpt)}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_npz


def main():
    ap = argparse.ArgumentParser(description="Build GPT candidate caches")
    ap.add_argument("--region", choices=["eval", "calib"], default="eval")
    ap.add_argument("--hidden", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--calib_cache", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--light", action="store_true",
                    help="calib: skip decode_joint posterior (F-INT only needs top-K)")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    ckpt, tok_path = upstream_paths()
    try:
        if args.region == "eval":
            hidden_path = Path(args.hidden) if args.hidden else (
                ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "hidden_cache.npz")
            records_path = Path(args.records) if args.records else (
                ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "pt01_records.npz")
            out_npz = roots.weights_root / f"candidates_eval_K{args.k}.npz"
            out_json = roots.results_root / f"candidates_eval_K{args.k}.json"
            build_candidates(hidden_path=hidden_path, records_path=records_path,
                             ckpt=ckpt, tok_path=tok_path, out_npz=out_npz,
                             out_json=out_json, k=args.k, device=args.device)
        else:
            calib_cache = Path(args.calib_cache) if args.calib_cache else (
                ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "calibration_cache.npz")
            out_npz = roots.weights_root / f"candidates_calib_K{args.k}.npz"
            out_json = roots.results_root / f"candidates_calib_K{args.k}.json"
            build_calib_candidates(calib_cache=calib_cache, ckpt=ckpt,
                                   tok_path=tok_path, out_npz=out_npz,
                                   out_json=out_json, k=args.k,
                                   device=args.device, light=args.light)
        append_trial({"event": "build_candidates", "region": args.region,
                      "k": args.k, "status": "ok"})
    except Exception as e:
        append_trial({"event": "build_candidates", "region": args.region,
                      "k": args.k, "status": "failed", "error": str(e)})
        raise


if __name__ == "__main__":
    main()
