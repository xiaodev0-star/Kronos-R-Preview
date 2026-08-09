"""build_gpt_full_q.py — plan §3 R2 prep: GPT full-128 q(c) for the eval region.

The eval candidate cache stores only GPT's top-8 (``topk_logq``); R2's
product-of-experts fusion needs q over the FULL 128-code support.
``coarse_q_for_hidden`` (build_gpt_candidates.py) recomputes it cheaply — one
[256,130] matmul per hidden row.  This is the only GPU step the R series needs
(plan §5: "第二步（数十分钟 GPU）：重算 GPT 全 128 q → R2").

Output:  ``gpt_q_eval_full128.npz``  [N,128] float32 (probability, ordinary-code
renormalized, same convention as ``candidates_calib_K8.npz["gpt_q"]``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from critic_common import resolve_roots, upstream_paths, append_trial  # noqa: E402
from build_gpt_candidates import coarse_q_for_hidden, load_model_cpu  # noqa: E402

TC = 1.4  # locked calibrated coarse temperature (same as candidates)

CHUNK = 8192


def main():
    ap = argparse.ArgumentParser(description="Recompute GPT full-128 q for eval")
    ap.add_argument("--hidden", default=str(ROOT / "server_runs" / "weights" /
                                            "06-posttrain" / "seed42" / "hidden_cache.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    args = ap.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA unavailable")
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    roots = resolve_roots(seed=42)
    out_npz = roots.weights_root / "gpt_q_eval_full128.npz"
    if out_npz.exists():
        print(f"[gptq] exists, skipping: {out_npz}")
        return

    ckpt, tok_path = upstream_paths()
    model, tok = load_model_cpu(ckpt, tok_path)
    model = model.to(dev).eval()
    vocab_base = int(model._vocab_l1)
    print(f"[gptq] model on {dev}, vocab_base={vocab_base}, TC={TC}")

    hdata = np.load(args.hidden, allow_pickle=True)
    hidden = hdata["hidden"]                                    # [N, 256]
    n = hidden.shape[0]
    print(f"[gptq] hidden rows {n}")

    q = np.zeros((n, vocab_base), dtype=np.float32)
    start = 0
    while start < n:
        stop = min(start + args.chunk, n)
        h = torch.from_numpy(hidden[start:stop]).to(dev)
        qc, _ = coarse_q_for_hidden(model, h, t_c=TC, vocab_base=vocab_base)
        q[start:stop] = qc.cpu().numpy()
        start = stop
        if stop % (args.chunk * 20) == 0 or stop == n:
            print(f"[gptq] rows {stop}/{n}")

    # sanity: row sums must be 1 (ordinary-code renormalized posterior)
    sums = q.sum(axis=1)
    bad = np.sum(np.abs(sums - 1.0) > 1e-3)
    print(f"[gptq] row-sum sanity: max_dev={np.max(np.abs(sums-1.0)):.2e} bad={bad}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, stock_uid=hdata["stock_uid"], date_key=hdata["date_key"],
             q=q, tc=np.array([TC]))
    print(f"[gptq] wrote {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB)")
    append_trial({"event": "build_gpt_full_q", "n_rows": n, "status": "ok"})


if __name__ == "__main__":
    main()
