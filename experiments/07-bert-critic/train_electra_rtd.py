"""train_electra_rtd.py — plan §6.2: Stage 2a ELECTRA RTD discriminator.

Init: Stage-1 KronosBert backbone weights -> KronosElectraDiscriminator (same
backbone, binary "original vs replaced" head).

Negatives (F3): the frozen upstream GPT (``branchA_dm030_8ceb_ep1.pt``) samples
one replacement per selected position from q(c|h) at replace_prob=0.15.  To
prevent the discriminator from memorising a single snapshot's error pattern, the
negative source is mixed between TWO GPT snapshots (ep1 + the CPT parent
``exp04b_8ceb_ep100.pt``), chosen per batch by a fixed seed.  Uniform-random
replacement is kept as the control arm (``--neg_mode uniform``).

Loss: discriminator_loss (per-position BCE over all non-special positions).

The [MASK]-row va=0 consistency rule from Stage 1 carries over: replaced
positions get va_values forced to 0, matching the critic's scoring-time input
shape (§8.1).

LR 3e-5, <=3 epochs, val replaced/original acc early stop.  Acceptance:
replaced-acc > 50% AND original-acc does not collapse (both reported).

Usage:
    python experiments/07-bert-critic/train_electra_rtd.py \
        --init_bert checkpoints/bert_critic_mlm_v1.pt \
        --save_path checkpoints/electra_critic_rtd_v1.pt
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (ROOT, SEVEN, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import DataConfig, ModelConfig, TrainingConfig, set_global_seed
from data_processor import load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2
from experiment_io import file_sha256
from eval_helpers import load_gpt
from model import load_tokenizer
from model.kronos_electra import KronosElectraDiscriminator, make_replaced_batch
from model.kronos_electra import discriminator_loss, discriminator_accuracy
from training_utils import clip_grad_norm_

SNAPSHOT2 = ROOT / "checkpoints" / "exp04b_8ceb_ep100.pt"   # CPT parent (归因 reference)
NEG_TEMP = 1.4                                               # matches candidate generation §7.6


def load_electra_from_bert(bert_path, device):
    """KronosElectraDiscriminator initialized from the Stage-1 KronosBert."""
    ck = torch.load(str(bert_path), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    import types
    model_cfg = types.SimpleNamespace(
        vocab_size=int(cfg["vocab_size"]), vocab_fine=int(cfg.get("vocab_fine", 128)),
        dim=int(cfg["dim"]), depth=int(cfg["depth"]), heads=int(cfg["heads"]),
        num_kv_heads=int(cfg.get("num_kv_heads", 1)),
        ffn_multiplier=int(cfg.get("ffn_multiplier", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        va_hidden_dim=int(cfg.get("va_hidden_dim", 64)),
        rope_base=float(cfg.get("rope_base", 10000.0)),
    )
    model = KronosElectraDiscriminator(model_cfg).to(device)
    sd = ck["model_state_dict"]
    # shared backbone keys only (skip head_coarse -> disc_head)
    shared = {k: v for k, v in sd.items() if not k.startswith("head_coarse")}
    missing, unexpected = model.load_state_dict(shared, strict=False)
    assert not unexpected, f"unexpected keys from BERT: {unexpected}"
    return model, model_cfg, ck


@torch.no_grad()
def gpt_proposal_for_sequence(gpt, input_ids, time_ids, position_ids, va_values,
                              vocab_base, temp=NEG_TEMP):
    """Coarse logits (ordinary codes only) aligned to input positions.

    GPT logits at position p predict input_ids[p+1]; the ELECTRA replacement for
    input position j (>=1) is therefore sampled from GPT logits at position j-1.
    proposal[j] = gpt_logits[j-1, :vocab] for j>=1, zeros at BOS.
    """
    gpt.eval()
    # 1D input -> gpt() returns coarse_logits [N, V+2] (no batch dim)
    with torch.amp.autocast("cuda", enabled=(position_ids.device.type == "cuda"),
                            dtype=torch.bfloat16):
        coarse_logits, _ = gpt(input_ids, time_ids, position_ids,
                               va_values=va_values)
    logits = coarse_logits[:, :vocab_base].float()              # [N, V]
    prop = torch.zeros_like(logits)
    prop[1:] = logits[:-1]                                      # shift by one
    return prop


def build_replaced_batch(gpt, input_ids, time_ids, position_ids, va_values,
                         vocab_base, replace_prob, temp, generator, neg_mode):
    if neg_mode == "uniform":
        replaced_ids, labels = make_replaced_batch(
            input_ids, vocab_base, replace_prob=replace_prob,
            generator=generator)
    else:  # gpt
        prop = gpt_proposal_for_sequence(gpt, input_ids, time_ids, position_ids,
                                         va_values, vocab_base, temp=temp)
        replaced_ids, labels = make_replaced_batch(
            input_ids, vocab_base, replace_prob=replace_prob,
            generator=generator, proposal=prop, temp=temp)
    # replaced rows get va=0 (leakage-safe scoring shape, consistent with Stage 1)
    va_repl = va_values.clone()
    if (labels == 0).any():
        va_repl = va_repl.masked_fill((labels == 0).unsqueeze(-1), 0.0)
    return replaced_ids, labels, va_repl


def main(args):
    set_global_seed(TrainingConfig.random_seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, tag={args.tag}, neg_mode={args.neg_mode}")

    tokenizer = load_tokenizer(args.tokenizer_path, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    vocab_base = ModelConfig.vocab_size
    print(f"Tokenizer: vocab_coarse={vocab_base}")

    # ---- snapshots ----
    ckpt1 = Path(args.gpt1)
    ckpt2 = Path(args.gpt2) if args.gpt2 else SNAPSHOT2
    gpt1 = load_gpt(str(ckpt1), device, tokenizer=tokenizer).eval()
    if args.neg_mode == "gpt" and ckpt2 != ckpt1:
        gpt2 = load_gpt(str(ckpt2), device, tokenizer=tokenizer).eval()
    else:
        gpt2 = None
    print(f"  GPT1 (snapshot1): {ckpt1.name}")
    if gpt2 is not None:
        print(f"  GPT2 (snapshot2): {ckpt2.name}")

    # ---- data ----
    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    cache_dir = os.path.join(TrainingConfig.save_dir,
                             f"token_cache_{os.path.basename(args.tokenizer_path).replace('.pt','')}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir,
                                 f"token_cache_{os.path.basename(args.tokenizer_path).replace('.pt','')}")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train", cache_dir=cache_dir,
                                max_seq_len=args.max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train", cache_dir=cache_dir,
                              max_seq_len=args.max_seq_len)
    train_loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=True)
    val_loader = make_dataloader_v2(val_seqs, batch_size=1, shuffle=False)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    # ---- model ----
    model, cfg, bert_meta = load_electra_from_bert(args.init_bert, device)
    print(f"Discriminator params: {sum(p.numel() for p in model.parameters()):,}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    start_epoch = 0
    best_val = None
    best_state = None
    if args.resume and os.path.exists(args.save_path + ".ckpt"):
        try:
            ck = torch.load(args.save_path + ".ckpt", map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            start_epoch = ck["epoch"] + 1
            best_val = ck.get("best_val")
            print(f"  Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"  Resume failed ({e}); fresh start")

    history = {"train_loss": [], "val_replaced_acc": [], "val_original_acc": [],
               "val_loss": []}
    t0 = time.time()
    rng_g = torch.Generator(device=device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses, acc_repl, acc_orig = [], [], []
        pbar = tqdm(train_loader, desc=f"[{args.tag}] Epoch {epoch+1}/{args.epochs}")
        for bi, batch in enumerate(pbar):
            input_ids, _, _, time_id, pos_id, _, va_val, _, _, _ = batch
            input_ids = input_ids.to(device, non_blocking=True)
            time_id = time_id.to(device, non_blocking=True)
            pos_id = pos_id.to(device, non_blocking=True)
            va_val = va_val.to(device, non_blocking=True)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                time_id = time_id.unsqueeze(0)
                pos_id = pos_id.unsqueeze(0)
                va_val = va_val.unsqueeze(0)

            rng_g.manual_seed(epoch * 100_000 + bi)
            # dual-snapshot mix: assign this batch's negatives to one snapshot
            gpt = gpt1 if (gpt2 is None or bi % 2 == 0) else gpt2
            replaced_ids, labels, va_repl = build_replaced_batch(
                gpt, input_ids[0], time_id[0], pos_id[0], va_val[0],
                vocab_base, args.replace_prob, NEG_TEMP, rng_g, args.neg_mode)
            replaced_ids = replaced_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)
            va_repl = va_repl.unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(replaced_ids, time_id, pos_id, va_values=va_repl,
                               return_logits=True)
                loss = discriminator_loss(logits, labels, ignore_special=True,
                                          vocab_base=vocab_base,
                                          input_ids=replaced_ids)
                if not torch.isfinite(loss):
                    continue
                acc_o, acc_r, acc_orig_acc = discriminator_accuracy(
                    logits, labels, vocab_base=vocab_base, input_ids=replaced_ids)
            loss.backward()
            clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            acc_repl.append(acc_r)
            acc_orig.append(acc_orig_acc)
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "orig": f"{np.mean(acc_orig[-50:]):.3f}",
                              "repl": f"{np.mean(acc_repl[-50:]):.3f}"})

        # validation
        model.eval()
        vloss, vrepl, vorig = [], [], []
        with torch.inference_mode():
            for batch in val_loader:
                input_ids, _, _, time_id, pos_id, _, va_val, _, _, _ = batch
                input_ids = input_ids.to(device)
                time_id = time_id.to(device)
                pos_id = pos_id.to(device)
                va_val = va_val.to(device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    time_id = time_id.unsqueeze(0)
                    pos_id = pos_id.unsqueeze(0)
                    va_val = va_val.unsqueeze(0)
                rng_g.manual_seed(900_000 + epoch * 1000 + len(vloss))
                replaced_ids, labels, va_repl = build_replaced_batch(
                    gpt1, input_ids[0], time_id[0], pos_id[0], va_val[0],
                    vocab_base, args.replace_prob, NEG_TEMP, rng_g, args.neg_mode)
                replaced_ids = replaced_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)
                va_repl = va_repl.unsqueeze(0)
                logits = model(replaced_ids, time_id, pos_id, va_values=va_repl,
                               return_logits=True)
                vloss.append(discriminator_loss(
                    logits, labels, ignore_special=True, vocab_base=vocab_base,
                    input_ids=replaced_ids).item())
                _, vr, vo = discriminator_accuracy(logits, labels,
                                                   vocab_base=vocab_base,
                                                   input_ids=replaced_ids)
                vrepl.append(vr); vorig.append(vo)
        v_loss = float(np.mean(vloss))
        v_repl = float(np.mean(vrepl))
        v_orig = float(np.mean(vorig))
        history["train_loss"].append(float(np.mean(losses)))
        history["val_replaced_acc"].append(v_repl)
        history["val_original_acc"].append(v_orig)
        history["val_loss"].append(v_loss)
        print(f"  Epoch {epoch+1}: train={np.mean(losses):.4f} "
              f"val_loss={v_loss:.4f} replaced_acc={v_repl:.3f} "
              f"original_acc={v_orig:.3f} elapsed={time.time()-t0:.0f}s", flush=True)

        # early-stop: keep the best combined val metric (replaced acc, orig not collapsing)
        score = v_repl - max(0.0, 1.0 - v_orig) * 0.5
        if best_val is None or score > best_val:
            best_val = score
            best_state = model.state_dict()
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "tag": args.tag,
            "best_val": best_val, "history": history,
            "config": cfg.__dict__, "neg_mode": args.neg_mode,
        }, args.save_path + ".ckpt")

    if best_state is not None:
        torch.save({
            "model_state_dict": best_state,
            "config": cfg.__dict__,
            "neg_mode": args.neg_mode, "neg_temp": NEG_TEMP,
            "replace_prob": args.replace_prob,
            "init_bert": str(args.init_bert),
            "init_bert_sha256": file_sha256(Path(args.init_bert)),
            "gpt1": str(ckpt1), "gpt1_sha256": file_sha256(ckpt1),
            "gpt2": str(ckpt2) if gpt2 is not None else None,
            "gpt2_sha256": file_sha256(ckpt2) if gpt2 is not None else None,
            "val_replaced_acc": v_repl, "val_original_acc": v_orig,
            "completed": True, "arch": "kronos_electra",
            "epochs_run": epoch + 1, "history": history,
        }, args.save_path)
    with open(os.path.join(os.path.dirname(args.save_path) or ".",
                           f"electra_{args.tag}_history.json"), "w") as f:
        json.dump({"history": history, "best_val": best_val,
                   "neg_mode": args.neg_mode, "replace_prob": args.replace_prob,
                   "val_replaced_acc": v_repl, "val_original_acc": v_orig}, f, indent=2)
    print(f"\nDone. val replaced_acc={v_repl:.3f}, original_acc={v_orig:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2a: ELECTRA RTD")
    parser.add_argument("--init_bert", type=str, default="checkpoints/bert_critic_mlm_v1.pt")
    parser.add_argument("--gpt1", type=str, default="checkpoints/branchA_dm030_8ceb_ep1.pt")
    parser.add_argument("--gpt2", type=str, default=None, help="default exp04b_8ceb_ep100.pt")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--save_path", type=str, default="checkpoints/electra_critic_rtd_v1.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--replace_prob", type=float, default=0.15)
    parser.add_argument("--neg_mode", choices=["gpt", "uniform"], default="gpt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tag", type=str, default="v1")
    parser.add_argument("--max_stocks", type=int, default=0)
    main(parser.parse_args())
