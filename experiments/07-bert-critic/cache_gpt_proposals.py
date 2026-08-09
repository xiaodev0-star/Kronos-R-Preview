"""cache_gpt_proposals.py — plan §4 T2 prep: frozen-GPT per-position proposals.

T2 denoises the critic by training on sequences where ~15% of history tokens
are replaced with samples from the frozen GPT at that position.  This script
caches the frozen GPT's coarse logits (ordinary codes only, [N,128]) for every
packed training sequence, so T2's per-epoch replacement sampling reuses one
GPU forward instead of re-running GPT each epoch.

Negatives come from TWO GPT snapshots (the plan §6.2 anti-mirroring rule):
the ep1 parent and the exp04b_8ceb_ep100 parent.  ``--which`` selects one.

Output: ``gpt_proposals_{which}_{n_seqs}seqs.pt`` — a dict
  {"proposals": [n_seqs, max_len, 128] float16, "lengths": [n_seqs] int64,
   "ckpt": str, "sha256": str, "temp_note": "raw logits (no temperature)"}.
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

from config import DataConfig, ModelConfig, TrainingConfig  # noqa: E402
from data_processor import load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2  # noqa: E402
from model import load_tokenizer  # noqa: E402
from model.kronos_preview import KronosPreview  # noqa: E402
from experiment_io import file_sha256  # noqa: E402
from critic_common import append_trial  # noqa: E402

SKELETON = dict(dim=256, depth=6, heads=4, num_kv_heads=1,
                ffn_multiplier=4, dropout=0.1)

# two GPT parents for the anti-mirroring negative mix (plan §6.2):
#   ep1  = the reviewed upstream (branchA_dm030_8ceb_ep1, an ep1-parent checkpoint)
#   ep100 = the Exp04B CPT parent (exp04b_8ceb_ep100)
EP1_CKPT = ROOT / "checkpoints" / "branchA_dm030_8ceb_ep1.pt"
EP100_CKPT = ROOT / "checkpoints" / "exp04b_8ceb_ep100.pt"


def _resolve_checkpoint(which):
    if which == "ep100":
        ckpt = EP100_CKPT
    elif which == "ep1":
        ckpt = EP1_CKPT
    else:
        raise ValueError(which)
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint missing: {ckpt}")
    return ckpt


def main():
    ap = argparse.ArgumentParser(description="Cache GPT proposals for T2")
    ap.add_argument("--which", choices=["ep1", "ep100"], default="ep100")
    ap.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    ap.add_argument("--max_stocks", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA unavailable")
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for k, v in SKELETON.items():
        setattr(ModelConfig, k, v)
    if args.max_stocks > 0:
        DataConfig.max_stocks = args.max_stocks

    tokenizer = load_tokenizer(args.tokenizer_path, dev)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size

    ckpt_path = _resolve_checkpoint(args.which)
    out_path = ROOT / "checkpoints" / f"gpt_proposals_{args.which}.pt"
    if out_path.exists():
        print(f"[props] exists, skipping: {out_path}")
        return

    model = KronosPreview().to(dev)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    if set(sd.keys()) != set(model.state_dict().keys()):
        # allow exp04b checkpoint layout (may carry extras/renames)
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd)
    model.eval()
    print(f"[props] GPT <- {ckpt_path} (sha {file_sha256(ckpt_path)[:10]})")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, _, _ = split_stocks(stocks)
    cache_tag = os.path.basename(args.tokenizer_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir, f"token_cache_{cache_tag}")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train", cache_dir=cache_dir,
                                max_seq_len=args.max_seq_len)
    loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=False)
    print(f"[props] train seqs: {len(train_seqs)}")

    V = ModelConfig.vocab_size
    proposals = []
    lengths = []
    n_done = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_ids, _, _, time_id, pos_id, _, va_val, _, _, _ = batch
            input_ids = input_ids.to(dev)
            time_id = time_id.to(dev)
            pos_id = pos_id.to(dev)
            va_val = va_val.to(dev)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                time_id = time_id.unsqueeze(0)
                pos_id = pos_id.unsqueeze(0)
                va_val = va_val.unsqueeze(0)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda"),
                                    dtype=torch.bfloat16):
                logits = model(input_ids, time_id, pos_id, va_values=va_val,
                               compute_reg_loss=False, fine_targets=None)[0]
            prop = logits[0, :, :V].float().half().cpu().numpy()   # [N, 128] fp16
            proposals.append(prop)
            lengths.append(prop.shape[0])
            n_done += 1
            if n_done % 200 == 0 or n_done == len(train_seqs):
                print(f"[props] {n_done}/{len(train_seqs)}", flush=True)

    max_len = max(lengths)
    arr = np.zeros((len(train_seqs), max_len, V), dtype=np.float16)
    for i, (p, L) in enumerate(zip(proposals, lengths)):
        arr[i, :L] = p
    torch.save({"proposals": torch.from_numpy(arr),
                "lengths": torch.as_tensor(lengths),
                "ckpt": str(ckpt_path), "sha256": file_sha256(ckpt_path),
                "vocab_base": V, "n_seqs": len(train_seqs),
                "max_len": max_len}, out_path)
    print(f"[props] wrote {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")
    append_trial({"event": "cache_gpt_proposals", "which": args.which,
                  "n_seqs": len(train_seqs), "status": "ok"})


if __name__ == "__main__":
    import os
    main()
