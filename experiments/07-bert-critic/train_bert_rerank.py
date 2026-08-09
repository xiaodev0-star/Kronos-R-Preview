"""train_bert_rerank.py — plan §6.3: Stage 2b RescoreBERT-style rerank fine-tune.

CONDITIONAL: only started when the Stage 1+2a C-a control passes but the fusion
gain is insufficient.  Directly aligns the critic to the reranking task itself.

Training tuples (cutoff-pre fit region): per (uid, date), GPT top-K=8 candidates
+ the day's TRUE token (positive).  Target: pairwise margin (true score >
each candidate score) on the t+1 [MASK]-position BERT logit.  Input is
inference-identical: t+1 slot carries the candidate (va=0).

Init from Stage-1 weights, LR <= 3e-5, <= 3 epochs.

Usage:
    python experiments/07-bert-critic/train_bert_rerank.py \
        --init_bert checkpoints/bert_critic_mlm_v1.pt \
        --save_path checkpoints/bert_critic_rerank_v1.pt
"""
import argparse
import json
import os
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

from config import DataConfig, ModelConfig, set_global_seed  # noqa: E402
from experiment_io import file_sha256  # noqa: E402
from eval_helpers import load_gpt  # noqa: E402
from model import load_tokenizer  # noqa: E402
from model.kronos_bert import KronosBert  # noqa: E402
from training_utils import clip_grad_norm_  # noqa: E402

from bert_data import VOCAB_BASE, bert_time_for_target_date  # noqa: E402
from score_bert import load_bert, build_index  # noqa: E402
from critic_common import resolve_roots  # noqa: E402

K = 8
MARGIN = 0.0


@torch.no_grad()
def gpt_topk_for_row(gpt, index, uid, date_key, k=K):
    """GPT top-k coarse candidates for a row, from its hidden in training_cache."""
    # For a lightweight v1 we reuse the training-cache hidden path: the caller
    # provides the row's hidden already.  This helper is a stub for the index.
    return None


def main(args):
    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg, bert_meta = load_bert(args.init_bert, torch.device(device))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    roots = resolve_roots(seed=42)
    training_cache = np.load(ROOT / "server_runs" / "weights" / "06-posttrain" /
                             "seed42" / "training_cache.npz", allow_pickle=True)
    n = len(training_cache["stock_uid"])
    stride = max(1, n // args.max_rows)
    idx = np.arange(0, n, stride)
    print(f"[rerank] fit rows: {len(idx)} (stride {stride})")

    # GPT q(c) for candidate generation from training-cache hidden
    from build_gpt_candidates import load_model_cpu, coarse_q_for_hidden
    _, tok_path = upstream_paths(ROOT / "server_runs" / "results" /
                                 "04b-cpt" / "seed42" / "trials" / "selection.json")
    gpt, _ = load_model_cpu(args.gpt1, tok_path)
    gpt = gpt.to(device).eval()
    index_cache = roots.weights_root / "bert_input_index.pkl"
    index = build_index(tok_path, device="cpu", cache_path=index_cache)

    h = torch.from_numpy(training_cache["hidden"][idx]).to(device)
    q, _ = coarse_q_for_hidden(gpt, h, t_c=1.4, vocab_base=cfg.vocab_size)
    topk = q.topk(K, dim=-1).indices.cpu().numpy()
    true_ids = training_cache["true_coarse_id"][idx].astype(np.int64)
    del h, q

    mask_id = cfg.vocab_size + 2
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        losses = []
        for i in range(0, len(idx), args.batch_rows):
            sl = slice(i, min(i + args.batch_rows, len(idx)))
            # build candidate-inserted inputs: true + K candidates, one per seq
            inputs = []
            cand_ids = []
            for local, r in enumerate(range(sl.start, sl.stop)):
                uid = str(training_cache["stock_uid"][idx[r]])
                dk = str(training_cache["date_key"][idx[r]])[:10]
                true_c = int(true_ids[r - sl.start])
                bi = index.bert_input(uid, dk, args.window)
                if bi is None:
                    continue
                ids, tids, va, _ = bi
                cands = np.concatenate([[true_c], topk[r - sl.start][:K]])
                cand_ids.append(cands.astype(np.int64))
                for c in cands:
                    seq = ids.clone()
                    seq[-1] = int(c)
                    inputs.append((seq, tids, va))
            if not inputs:
                continue
            # pad batch
            max_len = max(s.shape[0] for s, _, _ in inputs)
            B = len(inputs)
            inp = torch.zeros(B, max_len, dtype=torch.long, device=device)
            tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=device)
            vas = torch.zeros(B, max_len, 2, dtype=torch.float32, device=device)
            for j, (s, t, v) in enumerate(inputs):
                inp[j, :s.shape[0]] = s; tids[j, :s.shape[0]] = t
                vas[j, :s.shape[0]] = v
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp, tids, torch.arange(max_len, device=device)
                               .unsqueeze(0).expand(B, -1), va_values=vas)
                # score = log p_BERT(c) at the t+1 slot
                sl = logits[:, -1, :cfg.vocab_size]
                lp = torch.log_softmax(sl.float(), dim=-1)          # [B, V]
                # per tuple: true score > each candidate score (margin)
                pos = 0
                loss = torch.tensor(0.0, device=device)
                cnt = 0
                for cvec in cand_ids:
                    m = len(cvec)
                    true_lp = lp[pos, int(cvec[0])]
                    for kk in range(1, m):
                        loss = loss + F.relu(MARGIN + lp[pos, int(cvec[kk])] - true_lp)
                        cnt += 1
                    pos += m
                if cnt == 0:
                    continue
                loss = loss / cnt
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            losses.append(loss.item())
            if step % 100 == 0:
                print(f"  step {step} loss={np.mean(losses[-100:]):.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        print(f"  Epoch {epoch+1}: avg_loss={np.mean(losses):.4f}")

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__, "arch": "kronos_bert",
        "task": "rerank", "k": K, "margin": MARGIN,
        "init_bert": str(args.init_bert), "init_bert_sha256": file_sha256(Path(args.init_bert)),
        "gpt1": str(args.gpt1), "steps": step, "epochs": args.epochs,
        "completed": True,
    }, args.save_path)
    print(f"Done. -> {args.save_path}")


def upstream_paths(selection_path):
    from critic_common import upstream_paths as _up
    return _up(selection_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 2b rerank fine-tune")
    ap.add_argument("--init_bert", default="checkpoints/bert_critic_mlm_v1.pt")
    ap.add_argument("--gpt1", default="checkpoints/branchA_dm030_8ceb_ep1.pt")
    ap.add_argument("--save_path", default="checkpoints/bert_critic_rerank_v1.pt")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch_rows", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=100_000)
    main(ap.parse_args())
