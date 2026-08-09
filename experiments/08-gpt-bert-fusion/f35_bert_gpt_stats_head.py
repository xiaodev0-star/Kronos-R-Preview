"""f35_bert_gpt_stats_head.py — augment the BERT-hidden rank head with GPT posterior stats.

The Exp08 rank head uses BERT hidden only.  This arm feeds the head
  [BERT hidden (256) ∥ GPT posterior stats (post_median, p_up, post_std)]
so the head can use GPT's predicted magnitude/direction alongside BERT's
bidirectional representation — the literal "GPT predict + BERT score" at the
feature level.

Steps:
  1. compute fit-region GPT posterior stats via decode_joint (cache to
     server_runs/weights/07-bert-critic/seed42/gpt_post_fit.npz)
  2. train a soft-Spearman rank head on [BERT fit hidden ∥ GPT post stats]
  3. evaluate on eval 0..399 vs the plain BERT-head ensemble and the fused F.

Usage:
    python f35_bert_gpt_stats_head.py --seed 140
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_heads import fold_split, final_fit_split, train_rank_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from improve_common import (  # noqa: E402
    weights_root, results_root, metrics_table, build_rec, cand_path,
    write_json_ledger,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from eval_helpers import load_gpt  # noqa: E402
from critic_common import upstream_paths  # noqa: E402
from joint_decoder import DecodeTable, decode_joint  # noqa: E402


def compute_fit_post_stats(wr, device):
    """GPT posterior stats for fit rows via decode_joint (cached)."""
    out = wr / "gpt_post_fit.npz"
    if out.exists():
        d = np.load(out, allow_pickle=True)
        print(f"[f35] fit post stats cached ({len(d['post_median'])} rows)")
        return {k: d[k] for k in d.files}
    ckpt, tok_path = upstream_paths()
    from model import load_tokenizer
    tok = load_tokenizer(str(tok_path), torch.device(device))
    model = load_gpt(str(ckpt), torch.device(device), tokenizer=tok)
    model.eval()
    dt = DecodeTable(tok, device="cpu")
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    H = tc["hidden"]
    pm, ps = tc["p_mean0"].astype(np.float64), tc["p_std0"].astype(np.float64)
    n = len(H)
    post = {k: np.full(n, np.nan) for k in ("post_median", "p_up", "post_std")}
    dev = torch.device(device)
    for s in range(0, n, 4096):
        stop = min(s + 4096, n)
        stats, quality = decode_joint(
            model, dt, torch.from_numpy(H[s:stop]).to(dev), pm[s:stop], ps[s:stop],
            t_c=1.4, t_f=1.1, chunk=512)
        ok = quality.cpu().numpy().astype(bool)
        for k, arr in (("post_median", stats.median), ("p_up", stats.p_up),
                       ("post_std", stats.std)):
            v = arr.cpu().numpy()
            post[k][s:stop] = np.where(ok, v, np.nan)
        if stop % 200000 == 0 or stop == n:
            print(f"[f35] fit post {stop}/{n}", flush=True)
    np.savez(out, **{k: np.asarray(v) for k, v in post.items()})
    print(f"[f35] fit post stats -> {out}")
    return post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=140)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--apply_only", action="store_true",
                    help="load the saved head and only run the eval apply")
    args = ap.parse_args()
    wr = weights_root()
    rr = results_root()

    if args.apply_only:
        _apply(args, wr, rr)
        return

    post = compute_fit_post_stats(wr, args.device)

    # ---- fit rows: BERT hidden ∥ GPT post stats ----
    b = np.load(wr / "bert_hidden_fit_w512_t2.npz", allow_pickle=True)
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    tc = np.load(sw / "training_cache.npz", allow_pickle=True)
    stats = np.stack([post["post_median"], post["p_up"], post["post_std"]], axis=1)
    rows = {k: tc[k] for k in ("stock_uid", "date_key", "true_logret")}
    rows["hidden"] = np.concatenate([b["hidden"], stats.astype(np.float32)], axis=1)
    print(f"[f35] aug fit rows={len(rows['stock_uid'])} dim={rows['hidden'].shape[1]}")

    # R2 sanity
    fit_r2, val_r2 = fold_split(rows, "R2")
    h0, _ = train_rank_per_date(MlpRankHead(dim=rows["hidden"].shape[1], hidden=64,
                                            dropout=0.1, loss="soft_spearman"),
                                rows, fit_r2, val_r2, "soft_spearman",
                                lr=3e-4, epochs=4, seed=args.seed)
    # final fit
    fit_idx = final_fit_split(rows)
    head = MlpRankHead(dim=rows["hidden"].shape[1], hidden=64, dropout=0.1,
                       loss="soft_spearman")
    _, hist = train_rank_per_date(head, rows, fit_idx, fit_idx, "soft_spearman",
                                  lr=3e-4, epochs=16, seed=args.seed)
    torch.save({"head_state": head.state_dict(), "seed": args.seed,
                "recipe": {"lr": 3e-4, "epochs": 16, "dropout": 0.1}},
               wr / f"head_BERT_gptstats_rank_seed{args.seed}.pt")
    print(f"[f35] trained head")

    # ---- eval ----
    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    Xe = np.concatenate([de["hidden"],
                         np.stack([cand["post_median"].astype(np.float32),
                                   cand["p_up"].astype(np.float32),
                                   cand["post_std"].astype(np.float32)], axis=1)],
                        axis=1)
    head.eval()
    with torch.no_grad():
        rec["aug_head"] = head(torch.from_numpy(Xe.astype(np.float32))).numpy().astype(np.float64)
    # reference: plain BERT-ens + P6 fusion (Exp08)
    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in range(45, 51):
        ck = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(ck["head_state"]); h.eval()
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    rec["ens"] = ens
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)

    res = {"schema": "f35-bert-gptstats-head-v1", "seed": args.seed,
           "dense_threshold": dense}
    res["full"] = metrics_table(rec, {"AUG": "aug_head", "BERT_ens": "ens",
                                      "F_BERT6_P6": "F", "J3": "post_median"}, dense)
    print(f"[f35] full AUG RankIC={res['full']['AUG']['avg_daily_rank_ic']:.4f} "
          f"(BERT_ens {res['full']['BERT_ens']['avg_daily_rank_ic']:.4f}, "
          f"F {res['full']['F_BERT6_P6']['avg_daily_rank_ic']:.4f})")
    out = rr / f"f35_bert_gptstats_head_seed{args.seed}.json"
    write_json_ledger(out, res, "f35_bert_gptstats_head", seed=args.seed)
    print(f"[f35] -> {out}")


def _apply(args, wr, rr):
    """Load the saved augmented head and run the eval apply only."""
    head_path = wr / f"head_BERT_gptstats_rank_seed{args.seed}.pt"
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    head = MlpRankHead(dim=259, hidden=64, dropout=0.1, loss="soft_spearman")
    head.load_state_dict(ck["head_state"])
    head.eval()
    print(f"[f35-apply] head <- {head_path}")

    cand = np.load(cand_path("eval"), allow_pickle=True)
    rec = build_rec(cand)
    n = len(rec["stock_uid"])
    dates = np.asarray(rec["date_key"])
    dense = int(cand["dense_threshold"][0])
    de = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    Xe = np.concatenate([de["hidden"],
                         np.stack([cand["post_median"].astype(np.float32),
                                   cand["p_up"].astype(np.float32),
                                   cand["post_std"].astype(np.float32)], axis=1)],
                        axis=1)
    with torch.no_grad():
        rec["aug_head"] = head(torch.from_numpy(Xe.astype(np.float32))).numpy().astype(np.float64)

    H = torch.from_numpy(de["hidden"].astype(np.float32))
    ranks = []
    for s in range(45, 51):
        c2 = torch.load(str(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt"),
                        map_location="cpu", weights_only=False)
        h = MlpRankHead(dim=256, hidden=64, dropout=0.1, loss="soft_spearman")
        h.load_state_dict(c2["head_state"]); h.eval()
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).numpy().astype(np.float64)))
    ens = np.mean(ranks, axis=0)
    rec["ens"] = ens
    p6 = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    rec["F"] = 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)

    res = {"schema": "f35-bert-gptstats-head-apply-v1", "seed": args.seed,
           "dense_threshold": dense}
    res["full"] = metrics_table(rec, {"AUG": "aug_head", "BERT_ens": "ens",
                                      "F_BERT6_P6": "F", "J3": "post_median"}, dense)
    print(f"[f35-apply] full AUG RankIC={res['full']['AUG']['avg_daily_rank_ic']:.4f} "
          f"(BERT_ens {res['full']['BERT_ens']['avg_daily_rank_ic']:.4f}, "
          f"F {res['full']['F_BERT6_P6']['avg_daily_rank_ic']:.4f})")
    out = rr / f"f35_bert_gptstats_head_seed{args.seed}.json"
    write_json_ledger(out, res, "f35_bert_gptstats_head_apply", seed=args.seed)
    print(f"[f35-apply] -> {out}")


if __name__ == "__main__":
    main()
