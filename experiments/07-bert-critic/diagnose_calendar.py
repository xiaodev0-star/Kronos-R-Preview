"""C-b follow-up: is the BERT critic's true-token discrimination driven by the
target-date calendar (a potential side channel) or by history content?

Controls ran with C-b (shuffled history) NOT collapsing.  This diagnostic
scrambles the MASK slot's target calendar to a random date; if the
discrimination collapses, the score leaned on the calendar; if it survives,
the signal comes from history content (legitimate context dependence).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from critic_common import resolve_roots, upstream_paths  # noqa: E402
from bert_data import build_bert_input  # noqa: E402
from score_bert import load_bert, build_index, score_rows, NumpyRowTable  # noqa: E402
from build_gpt_candidates import load_model_cpu, coarse_q_for_hidden  # noqa: E402
from eval_critic import _sample_fit_rows, _gpt_q_for_rows, _sample_distractors, _true_rank  # noqa: E402


@torch.no_grad()
def score_scrambled_cal(index, rows, model, cache_idx, gpt_model, training_cache,
                        true_coarse, n_rows, window=512, device="cuda"):
    """Score rows with the MASK slot's target calendar scrambled (random date)."""
    model.eval()
    dev = torch.device(device)
    logp = np.full((n_rows, 128), np.nan, dtype=np.float32)
    for i in range(n_rows):
        p = index.by_uid[rows.stock_uid(i)]
        pos = index.position_for(rows.stock_uid(i), rows.date_key(i))
        if pos is None:
            continue
        start = max(0, pos + 1 - window)
        rng = np.random.RandomState(i)
        hist_ids = p["inp_ids"][start:pos + 1]
        hist_time = p["time_ids"][start:pos + 1]
        hist_va = p["va"][start:pos + 1]
        target_time = torch.tensor([rng.randint(1, 28), rng.randint(1, 12),
                                    rng.randint(0, 60)], dtype=torch.long)
        ids, tids, va, _ = build_bert_input(hist_ids, hist_time, hist_va,
                                            target_time)
        L = ids.shape[0]
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda"),
                                dtype=torch.bfloat16):
            logits = model(ids.unsqueeze(0).to(dev), tids.unsqueeze(0).to(dev),
                           torch.arange(L, device=dev).unsqueeze(0),
                           va_values=va.unsqueeze(0).to(dev))
        logp[i] = torch.log_softmax(logits[0, -1].float(), dim=-1).cpu().numpy()
    q, _ = _gpt_q_for_rows(gpt_model, training_cache, cache_idx)
    ranks = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        d = _sample_distractors(q[i], int(true_coarse[i]), n_dist=7, seed=i)
        ranks[i] = _true_rank(logp[i], int(true_coarse[i]), d)
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    roots = resolve_roots(seed=42)
    ckpt, tok_path = upstream_paths()
    index = build_index(tok_path, device="cpu",
                        cache_path=roots.weights_root / "bert_input_index.pkl")
    model, cfg, _ = load_bert(str(ROOT / "checkpoints" / "bert_critic_mlm_v1.pt"),
                              torch.device(args.device))
    gpt, _ = load_model_cpu(ckpt, tok_path)
    gpt = gpt.to(args.device)

    training_cache = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42" / "training_cache.npz"
    rows, cache, cache_idx, true_coarse = _sample_fit_rows(training_cache, args.n)
    keep = np.array([rows.stock_uid(i) in index for i in range(len(rows))])
    rows = rows.select(keep)
    cache_idx = cache_idx[keep]
    true_coarse = true_coarse[keep]

    # baseline C-a on the same rows
    t0 = time.time()
    out = score_rows(index, rows, model, window=512, batch_size=32,
                     device=args.device)
    logp = out["logp_bert_full"]
    q, _ = _gpt_q_for_rows(gpt, cache, cache_idx)
    ranks_base = np.empty(len(rows), dtype=np.float64)
    for i in range(len(rows)):
        d = _sample_distractors(q[i], int(true_coarse[i]), n_dist=7, seed=i)
        ranks_base[i] = _true_rank(logp[i], int(true_coarse[i]), d)
    print(f"BASELINE C-a (n={len(rows)}): avg_rank={ranks_base.mean():.3f} "
          f"top1={np.mean(ranks_base==1):.3f} ({time.time()-t0:.0f}s)")

    ranks_sc = score_scrambled_cal(index, rows, model, cache_idx, gpt, cache,
                                   true_coarse, len(rows), device=args.device)
    print(f"SCRAMBLED-CAL: avg_rank={ranks_sc.mean():.3f} "
          f"top1={np.mean(ranks_sc==1):.3f} "
          f"collapsed={ranks_sc.mean() >= 4.2}")


if __name__ == "__main__":
    main()
