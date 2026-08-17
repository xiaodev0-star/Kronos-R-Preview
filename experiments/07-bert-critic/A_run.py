"""07-A: BERT training pipeline.

The public entry point runs the complete A pipeline when called without
arguments: base MLM pre-training, scoring-aligned fine-tuning, GPT proposal
cache preparation, and proposal-augmented fine-tuning.  Individual steps are
kept only as debugging hooks for development.
"""
from __future__ import annotations

import argparse
import hashlib
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

# ============================================================================
# Path bootstrap
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
for _p in (SEVEN, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ============================================================================
# Repo-module imports
# ============================================================================
from config import DataConfig, ModelConfig, TrainingConfig, set_global_seed  # noqa: E402
from data_processor import load_stocks, split_stocks, pack_stocks_v2, make_dataloader_v2  # noqa: E402
from experiment_io import file_sha256  # noqa: E402
from model import load_tokenizer  # noqa: E402
from model.kronos_bert import KronosBert, make_mlm_batch  # noqa: E402
from model.kronos_preview import KronosPreview  # noqa: E402
from training_utils import clip_grad_norm_  # noqa: E402

# ============================================================================
# common.py imports
# ============================================================================
from common import (  # noqa: E402
    resolve_roots, write_json, append_trial,
    make_scoring_batch, stage_weights, stage_results, weights_artifact,
)

# ============================================================================
# Shared constants
# ============================================================================

SKELETON = dict(dim=256, depth=6, heads=4, num_kv_heads=1,
                ffn_multiplier=4, dropout=0.1)
NEG_TEMP = 1.4

# Two GPT parents for the anti-mirroring negative mix (plan S6.2):
#   ep1   = branchA_dm030_8ceb_ep1 (an ep1-parent checkpoint)
#   ep100 = exp04b_8ceb_ep100 (the Exp04B CPT parent)
EP1_CKPT = ROOT / "checkpoints" / "branchA_dm030_8ceb_ep1.pt"
EP100_CKPT = ROOT / "checkpoints" / "exp04b_8ceb_ep100.pt"


def _weights(name):
    """Standardized weight path."""
    return stage_weights("A", seed=42) / name


def _results(name):
    """Standardized result path."""
    return stage_results("A", seed=42) / name


# ############################################################################
#  Base MLM pre-training (from train_bert_mlm.py)
# ############################################################################

def _data_fingerprint(cache_dir, tokenizer_path, mlm_prob, epochs, max_seq_len):
    """Short fingerprint of the Stage-1 data + masking configuration."""
    h = hashlib.sha256()
    for item in (str(cache_dir), str(tokenizer_path), str(mlm_prob),
                 str(epochs), str(max_seq_len), "80_10_10", "mask_va_zero"):
        h.update(item.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def stage_a1():
    """Train the base BERT MLM model."""
    parser = argparse.ArgumentParser(description="Base BERT MLM pretraining")
    parser.add_argument("--save_path", type=str,
                        default=str(_weights("BERT.pt")))
    parser.add_argument("--tokenizer_path", type=str,
                        default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max_stocks", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--corrupt_fracs", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--no_mask_va_zero", action="store_true")
    parser.add_argument("--tag", type=str, default="BERT")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=False)
    args = parser.parse_args()

    set_global_seed(TrainingConfig.random_seed,
                    deterministic=getattr(args, "deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    ckpt_path = save_path + ".ckpt"

    if args.max_stocks > 0:
        DataConfig.max_stocks = args.max_stocks
    for k, v in SKELETON.items():
        setattr(ModelConfig, k, v)

    print(f"Device: {device}, tag={args.tag}")
    print(f"  save={save_path}, ep={args.epochs}, mlm_prob={args.mlm_prob}")
    print(f"  model: dim={ModelConfig.dim} depth={ModelConfig.depth} "
          f"heads={ModelConfig.heads} num_kv_heads={ModelConfig.num_kv_heads} "
          f"ffn_mult={ModelConfig.ffn_multiplier} dropout={ModelConfig.dropout} "
          f"(GPT-identical skeleton, v1)")

    tokenizer = load_tokenizer(args.tokenizer_path, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    print(f"Tokenizer loaded. vocab_coarse={ModelConfig.vocab_size}, "
          f"vocab_fine={ModelConfig.vocab_fine}")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    cache_tag = os.path.basename(args.tokenizer_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir,
                             f"token_cache_{cache_tag}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir,
                                 f"token_cache_{cache_tag}")
    print(f"Encoding v2 (cache: {cache_dir}) ...")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train",
                                cache_dir=cache_dir, max_seq_len=args.max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train",
                              cache_dir=cache_dir, max_seq_len=args.max_seq_len)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    train_loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=True)
    val_loader = make_dataloader_v2(val_seqs, batch_size=1, shuffle=False)

    model = KronosBert().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing: enabled")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_updates = len(train_loader) * args.epochs
    warmup = max(1, int(total_updates * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16

    # ---- resume ----
    start_epoch = 0
    best_val = float("inf")
    global_step = 0
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", float("inf"))
            global_step = ckpt.get("global_step", 0)
            print(f"  Resumed from epoch {start_epoch}, best_val={best_val:.4f}, "
                  f"step={global_step}")
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume (incompatible): {e}; starting fresh.")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "lr": [], "mlm_acc": []}
    t0 = time.time()

    vocab_base = ModelConfig.vocab_size
    mask_id = ModelConfig.vocab_size + 2
    corrupt_fracs = tuple(float(x) for x in args.corrupt_fracs.split(","))
    zero_mask_va = not args.no_mask_va_zero

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses, accs = [], []
        optimizer.zero_grad(set_to_none=True)

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

            B, N = input_ids.shape
            mlm_ids_list, mlm_labels_list = [], []
            for b in range(B):
                mlm_ids_b, mlm_labels_b = make_mlm_batch(
                    input_ids[b], vocab_base, mask_id, mlm_prob=args.mlm_prob,
                    corrupt_fracs=corrupt_fracs)
                mlm_ids_list.append(mlm_ids_b)
                mlm_labels_list.append(mlm_labels_b)
            mlm_ids = torch.stack(mlm_ids_list, dim=0)
            mlm_labels = torch.stack(mlm_labels_list, dim=0)

            va_mlm = va_val.clone()
            if zero_mask_va:
                va_mlm = va_mlm.masked_fill(
                    (mlm_ids == mask_id).unsqueeze(-1), 0.0)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(mlm_ids, time_id, pos_id, va_values=va_mlm)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       mlm_labels.view(-1), ignore_index=-100)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    valid = (mlm_labels != -100)
                    if valid.any():
                        accs.append((pred[valid] == mlm_labels[valid])
                                    .float().mean().item())

            if not torch.isfinite(loss):
                print(f"  [skip] non-finite loss at step {global_step}; continuing")
                continue

            loss.backward()
            clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            losses.append(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{np.mean(accs[-50:]) if accs else 0:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        avg_train = sum(losses) / max(len(losses), 1)
        avg_train_acc = sum(accs) / max(len(accs), 1) if accs else 0.0

        # Validation
        model.eval()
        vlosses, vaccs = [], []
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
                B, N = input_ids.shape
                mlm_ids_list, mlm_labels_list = [], []
                for b in range(B):
                    mlm_ids_b, mlm_labels_b = make_mlm_batch(
                        input_ids[b], vocab_base, mask_id,
                        mlm_prob=args.mlm_prob, corrupt_fracs=corrupt_fracs)
                    mlm_ids_list.append(mlm_ids_b)
                    mlm_labels_list.append(mlm_labels_b)
                mlm_ids = torch.stack(mlm_ids_list, dim=0)
                mlm_labels = torch.stack(mlm_labels_list, dim=0)
                va_mlm = va_val.clone()
                if zero_mask_va:
                    va_mlm = va_mlm.masked_fill(
                        (mlm_ids == mask_id).unsqueeze(-1), 0.0)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(mlm_ids, time_id, pos_id, va_values=va_mlm)
                    vloss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                            mlm_labels.view(-1), ignore_index=-100)
                    pred = logits.argmax(dim=-1)
                    valid = (mlm_labels != -100)
                    if valid.any():
                        vaccs.append((pred[valid] == mlm_labels[valid])
                                     .float().mean().item())
                vlosses.append(vloss.item())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        avg_val_acc = sum(vaccs) / max(len(vaccs), 1) if vaccs else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["mlm_acc"].append(avg_val_acc)
        history["lr"].append(cur_lr)

        save_tag = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads,
                           "num_kv_heads": ModelConfig.num_kv_heads,
                           "ffn_multiplier": ModelConfig.ffn_multiplier,
                           "dropout": ModelConfig.dropout,
                           "vocab_size": ModelConfig.vocab_size,
                           "vocab_fine": ModelConfig.vocab_fine,
                           "mask_id": mask_id,
                           "va_hidden_dim": ModelConfig.va_hidden_dim,
                           "rope_base": ModelConfig.rope_base,
                           "arch": "kronos_bert"},
                "val_loss": best_val, "mlm_acc": avg_val_acc, "epoch": epoch,
                "completed": epoch == args.epochs - 1, "tag": args.tag,
                "mlm_prob": args.mlm_prob,
                "corrupt_fracs": list(corrupt_fracs),
                "mask_va_zero": zero_mask_va,
            }, save_path)
            save_tag = "  -> Saved best"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "best_val": best_val, "global_step": global_step,
            "tag": args.tag,
        }, ckpt_path)

        epochs_done = epoch - start_epoch + 1
        epochs_left = args.epochs - epoch - 1
        eta = (elapsed / epochs_done) * epochs_left if epochs_done > 0 else 0
        print(f"  [{epochs_done}/{args.epochs - start_epoch}] Epoch {epoch+1}: "
              f"train={avg_train:.4f} val={avg_val:.4f} best={best_val:.4f} "
              f"mlm_acc={avg_val_acc:.3f} lr={cur_lr:.2e} elapsed={elapsed:.0f}s "
              f"ETA={eta:.0f}s step={global_step}{save_tag}", flush=True)

    # Mark completed
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    # ---- sidecar fingerprint + history ----
    meta = {
        "tag": args.tag, "arch": "kronos_bert",
        "data": {
            "cache_dir": cache_dir,
            "tokenizer_path": str(args.tokenizer_path),
            "tokenizer_sha256": file_sha256(Path(args.tokenizer_path)),
            "cutoff_date": DataConfig.cutoff_date,
            "max_stocks": int(DataConfig.max_stocks),
            "max_seq_len": int(args.max_seq_len),
            "mode": "train",
            "n_train_seqs": len(train_seqs),
            "n_val_seqs": len(val_seqs),
            "fingerprint": _data_fingerprint(
                cache_dir, args.tokenizer_path, args.mlm_prob,
                args.epochs, args.max_seq_len),
        },
        "model": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                  "heads": ModelConfig.heads,
                  "num_kv_heads": ModelConfig.num_kv_heads,
                  "ffn_multiplier": ModelConfig.ffn_multiplier,
                  "dropout": ModelConfig.dropout,
                  "vocab_size": ModelConfig.vocab_size,
                  "vocab_fine": ModelConfig.vocab_fine,
                  "n_params": int(n_params)},
        "train": {"mlm_prob": args.mlm_prob,
                  "corrupt_fracs": list(corrupt_fracs),
                  "mask_va_zero": zero_mask_va, "lr": args.lr,
                  "weight_decay": args.weight_decay, "warmup_ratio": 0.05,
                  "scheduler": "cosine", "grad_clip": 1.0, "bf16": True},
        "result": {"best_val": best_val, "final_val": avg_val,
                   "final_mlm_acc": avg_val_acc,
                   "random_baseline_acc": 1.0 / ModelConfig.vocab_size},
        "history": history,
    }
    meta_path = os.path.join(os.path.dirname(save_path) or ".",
                             f"bert_mlm_{args.tag}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. best val_loss: {best_val:.4f}, mlm_acc: {avg_val_acc:.3f} "
          f"(random baseline {1.0/ModelConfig.vocab_size:.4f})")


# ############################################################################
#  GPT proposal cache (from cache_gpt_proposals.py)
# ############################################################################

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


def stage_cache():
    """Cache frozen-GPT per-position proposals for proposal-augmented training."""
    ap = argparse.ArgumentParser(description="Cache GPT proposals for proposal-augmented training")
    ap.add_argument("--which", choices=["ep1", "ep100"], default="ep100")
    ap.add_argument("--tokenizer_path", type=str,
                    default="checkpoints/tokenizer_v2_ohlc.pt")
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
    out_path = (weights_artifact("gpt-proposals") if args.which == "ep100"
                else stage_weights("A") / "gpt-proposals-ep1.pt")
    if out_path.exists():
        print(f"[props] exists, skipping: {out_path}")
        return

    model = KronosPreview().to(dev)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    if set(sd.keys()) != set(model.state_dict().keys()):
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd)
    model.eval()
    print(f"[props] GPT <- {ckpt_path} (sha {file_sha256(ckpt_path)[:10]})")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, _, _ = split_stocks(stocks)
    cache_tag = os.path.basename(args.tokenizer_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir,
                             f"token_cache_{cache_tag}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir,
                                 f"token_cache_{cache_tag}")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train",
                                cache_dir=cache_dir, max_seq_len=args.max_seq_len)
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


# ############################################################################
#  Scoring-aligned MLM fine-tune
# ############################################################################

def stage_a2():
    """Fine-tune BERT with the scoring-aligned masking recipe."""
    parser = argparse.ArgumentParser(description="Scoring-aligned BERT MLM fine-tune")
    parser.add_argument("--init_from", type=str,
                        default=str(_weights("BERT.pt")))
    parser.add_argument("--save_path", type=str,
                        default=str(_weights("BERT-FT.pt")))
    parser.add_argument("--tokenizer_path", type=str,
                        default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_stocks", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--corrupt_fracs", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--final_pos_frac", type=float, default=0.5)
    parser.add_argument("--va_zero_frac", type=float, default=0.5)
    parser.add_argument("--recency_window", type=int, default=64)
    parser.add_argument("--tag", type=str, default="BERT-FT")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=False)
    args = parser.parse_args()

    set_global_seed(TrainingConfig.random_seed,
                    deterministic=getattr(args, "deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    ckpt_path = save_path + ".ckpt"

    if args.max_stocks > 0:
        DataConfig.max_stocks = args.max_stocks
    for k, v in SKELETON.items():
        setattr(ModelConfig, k, v)

    print(f"Device: {device}, tag={args.tag}")
    print(f"  init={args.init_from}, save={save_path}, ep={args.epochs}, "
          f"lr={args.lr}, window={args.window}")

    tokenizer = load_tokenizer(args.tokenizer_path, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    print(f"Tokenizer. vocab_coarse={ModelConfig.vocab_size}, "
          f"vocab_fine={ModelConfig.vocab_fine}")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}")

    cache_tag = os.path.basename(args.tokenizer_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir,
                             f"token_cache_{cache_tag}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir,
                                 f"token_cache_{cache_tag}")
    print(f"Encoding v2 (cache: {cache_dir}) ...")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train",
                                cache_dir=cache_dir, max_seq_len=args.max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train",
                              cache_dir=cache_dir, max_seq_len=args.max_seq_len)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

    train_loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=True)
    val_loader = make_dataloader_v2(val_seqs, batch_size=1, shuffle=False)

    model = KronosBert().to(device)
    if args.init_from and Path(args.init_from).exists():
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init["model_state_dict"])
        print(f"  init weights <- {args.init_from} "
              f"(val_loss={init.get('val_loss'):.4f}, "
              f"mlm_acc={init.get('mlm_acc'):.3f})")
    else:
        print(f"  WARNING: init_from {args.init_from} missing; training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_updates = len(train_loader) * args.epochs
    warmup = max(1, int(total_updates * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16

    start_epoch = 0
    best_val = float("inf")
    global_step = 0
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", float("inf"))
            global_step = ckpt.get("global_step", 0)
            print(f"  Resumed from epoch {start_epoch}")
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume: {e}; fresh")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "lr": [], "mlm_acc": []}
    t0 = time.time()

    vocab_base = ModelConfig.vocab_size
    mask_id = ModelConfig.vocab_size + 2
    corrupt_fracs = tuple(float(x) for x in args.corrupt_fracs.split(","))

    def run_one_batch(input_ids, time_id, pos_id, va_val):
        B, N = input_ids.shape
        out_ids, out_time, out_va, out_lab, out_pos = [], [], [], [], []
        for b in range(B):
            mid, mt, mv, ml, mp, _off = make_scoring_batch(
                input_ids[b], time_id[b], va_val[b], vocab_base, mask_id,
                window=args.window, mlm_prob=args.mlm_prob,
                corrupt_fracs=corrupt_fracs,
                final_pos_frac=args.final_pos_frac,
                va_zero_frac=args.va_zero_frac,
                recency_window=args.recency_window,
                generator=(torch.Generator().manual_seed(global_step + b)
                           if args.deterministic else None))
            out_ids.append(mid); out_time.append(mt); out_va.append(mv)
            out_lab.append(ml); out_pos.append(mp)
        Lmax = max(x.shape[0] for x in out_ids)
        pad_ids = torch.zeros(B, Lmax, dtype=torch.long, device=input_ids.device)
        pad_time = torch.zeros(B, Lmax, 3, dtype=torch.long, device=input_ids.device)
        pad_va = torch.zeros(B, Lmax, 2, dtype=torch.float32, device=input_ids.device)
        pad_lab = torch.full((B, Lmax), -100, dtype=torch.long, device=input_ids.device)
        pad_pos = torch.zeros(B, Lmax, dtype=torch.long, device=input_ids.device)
        for b in range(B):
            L = out_ids[b].shape[0]
            pad_ids[b, :L] = out_ids[b]
            pad_time[b, :L] = out_time[b]
            pad_va[b, :L] = out_va[b]
            pad_lab[b, :L] = out_lab[b]
            pad_pos[b, :L] = out_pos[b]
        return pad_ids, pad_time, pad_va, pad_lab, pad_pos

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses, accs = [], []
        optimizer.zero_grad(set_to_none=True)
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

            mlm_ids, mlm_time, mlm_va, mlm_labels, mlm_pos = run_one_batch(
                input_ids, time_id, pos_id, va_val)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(mlm_ids, mlm_time, mlm_pos, va_values=mlm_va)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       mlm_labels.view(-1), ignore_index=-100)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    valid = mlm_labels != -100
                    if valid.any():
                        accs.append((pred[valid] == mlm_labels[valid])
                                    .float().mean().item())

            if not torch.isfinite(loss):
                print(f"  [skip] non-finite loss at step {global_step}")
                continue
            loss.backward()
            clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            losses.append(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{np.mean(accs[-50:]) if accs else 0:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        avg_train = sum(losses) / max(len(losses), 1)
        avg_train_acc = sum(accs) / max(len(accs), 1) if accs else 0.0

        model.eval()
        vlosses, vaccs = [], []
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
                mlm_ids, mlm_time, mlm_va, mlm_labels, mlm_pos = run_one_batch(
                    input_ids, time_id, pos_id, va_val)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(mlm_ids, mlm_time, mlm_pos, va_values=mlm_va)
                    vloss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                            mlm_labels.view(-1), ignore_index=-100)
                    pred = logits.argmax(dim=-1)
                    valid = mlm_labels != -100
                    if valid.any():
                        vaccs.append((pred[valid] == mlm_labels[valid])
                                     .float().mean().item())
                vlosses.append(vloss.item())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        avg_val_acc = sum(vaccs) / max(len(vaccs), 1) if vaccs else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["mlm_acc"].append(avg_val_acc)
        history["lr"].append(cur_lr)

        save_tag = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads,
                           "num_kv_heads": ModelConfig.num_kv_heads,
                           "ffn_multiplier": ModelConfig.ffn_multiplier,
                           "dropout": ModelConfig.dropout,
                           "vocab_size": ModelConfig.vocab_size,
                           "vocab_fine": ModelConfig.vocab_fine,
                           "mask_id": mask_id,
                           "va_hidden_dim": ModelConfig.va_hidden_dim,
                           "rope_base": ModelConfig.rope_base,
                           "arch": "kronos_bert"},
                "val_loss": best_val, "mlm_acc": avg_val_acc, "epoch": epoch,
                "completed": epoch == args.epochs - 1, "tag": args.tag,
                "mlm_prob": args.mlm_prob,
                "corrupt_fracs": list(corrupt_fracs),
                "training_recipe": {"name": "scoring-aligned",
                                     "window": args.window,
                                     "final_pos_frac": args.final_pos_frac,
                                     "va_zero_frac": args.va_zero_frac,
                                     "recency_window": args.recency_window},
                "init_from": str(args.init_from),
            }, save_path)
            save_tag = "  -> Saved best"

        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "best_val": best_val,
                    "global_step": global_step, "tag": args.tag}, ckpt_path)
        print(f"  [{epoch+1}/{args.epochs}] train={avg_train:.4f} "
              f"val={avg_val:.4f} best={best_val:.4f} "
              f"mlm_acc={avg_val_acc:.3f} lr={cur_lr:.2e} "
              f"elapsed={elapsed:.0f}s{save_tag}", flush=True)

    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["completed"] = True
        torch.save(ckpt, save_path)

    meta = {"tag": args.tag, "arch": "kronos_bert",
            "training_recipe": {"name": "scoring-aligned",
                                 "window": args.window,
                                 "final_pos_frac": args.final_pos_frac,
                                 "va_zero_frac": args.va_zero_frac,
                                 "recency_window": args.recency_window},
            "init_from": str(args.init_from),
            "data": {"cutoff_date": DataConfig.cutoff_date,
                     "n_train_seqs": len(train_seqs),
                     "n_val_seqs": len(val_seqs)},
            "train": {"mlm_prob": args.mlm_prob,
                      "corrupt_fracs": list(corrupt_fracs),
                      "lr": args.lr, "weight_decay": args.weight_decay,
                      "bf16": True},
            "result": {"best_val": best_val,
                       "final_mlm_acc": avg_val_acc,
                       "random_baseline_acc": 1.0 / ModelConfig.vocab_size},
            "history": history,
    }
    meta_path = os.path.join(os.path.dirname(save_path) or ".",
                             f"bert_mlm_{args.tag}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. best val_loss: {best_val:.4f}, mlm_acc: {avg_val_acc:.3f}")


# ############################################################################
#  Proposal-augmented MLM fine-tune
# ############################################################################

def _make_proposal_augmented_batch(real_ids, time_id, va, vocab_base, mask_id,
                   proposal, window, mlm_prob, corrupt_fracs, final_pos_frac,
                   va_zero_frac, recency_window, gpt_replace_prob=0.15,
                   temp=NEG_TEMP, generator=None):
    """Scoring-aligned masking plus GPT proposal replacement.

    ``proposal`` is [N, V] logits for the FULL sequence (cached).
    ``offset`` from make_scoring_batch aligns the truncated window to the proposal
    slice.  GPT-replaced rows get va=0 (the rollout shape: a model-generated
    token has unknown future volume/amount).
    Returns (mid, mt, mv, ml, mp).
    """
    mid, mt, mv, ml, mp, off = make_scoring_batch(
        real_ids, time_id, va, vocab_base, mask_id, window, mlm_prob,
        corrupt_fracs, final_pos_frac, va_zero_frac, recency_window, generator)
    L = mid.shape[0]
    if proposal is not None:
        noise = (ml == -100) & (mid >= 0) & (mid < vocab_base) & (mid != mask_id)
        if noise.any():
            prop_pos = np.arange(off, off + L) - 1
            prop_pos[0] = off  # BOS has no proposal -> reuse
            prop_slice = np.asarray(proposal[prop_pos], dtype=np.float32)
            gen_r = torch.rand(L, device=mid.device, generator=generator)
            do_replace = noise & (gen_r < gpt_replace_prob)
            if do_replace.any():
                idx = torch.nonzero(do_replace, as_tuple=False).squeeze(-1)
                prop = torch.from_numpy(
                    prop_slice[idx.cpu().numpy()]).to(mid.device)
                p = torch.softmax(prop / temp, dim=-1)
                sample = torch.multinomial(p, 1).squeeze(-1)
                mid[idx] = sample
                mv[idx] = 0.0  # rollout shape: model token -> va unknown
    return mid, mt, mv, ml, mp


def stage_a3():
    """Fine-tune BERT with GPT proposal-augmented denoising."""
    parser = argparse.ArgumentParser(description="Proposal-augmented BERT MLM fine-tune")
    parser.add_argument("--init_from", type=str,
                        default=str(_weights("BERT-FT.pt")))
    parser.add_argument("--proposals1", type=str,
                    default=str(weights_artifact("gpt-proposals")))
    parser.add_argument("--proposals2", type=str, default=None)
    parser.add_argument("--save_path", type=str,
                        default=str(_weights("BERT-PPS.pt")))
    parser.add_argument("--tokenizer_path", type=str,
                        default="checkpoints/tokenizer_v2_ohlc.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_stocks", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--corrupt_fracs", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--final_pos_frac", type=float, default=0.5)
    parser.add_argument("--va_zero_frac", type=float, default=0.5)
    parser.add_argument("--recency_window", type=int, default=64)
    parser.add_argument("--gpt_replace_prob", type=float, default=0.15)
    parser.add_argument("--tag", type=str, default="BERT-PPS")
    parser.add_argument("--deterministic", action="store_true", default=False)
    args = parser.parse_args()

    set_global_seed(TrainingConfig.random_seed,
                    deterministic=getattr(args, "deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    ckpt_path = save_path + ".ckpt"

    if args.max_stocks > 0:
        DataConfig.max_stocks = args.max_stocks
    for k, v in SKELETON.items():
        setattr(ModelConfig, k, v)

    print(f"Device: {device}, tag={args.tag}")
    print(f"  init={args.init_from}, save={save_path}, ep={args.epochs}, lr={args.lr}")

    # ---- proposal banks ----
    def _load_proposals(path):
        if not Path(path).exists():
            raise RuntimeError(f"proposal cache missing: {path} "
                               "(run stage_a3 first with stage_a3)")
        c = torch.load(path, map_location="cpu", weights_only=False)
        return {"proposals": c["proposals"].numpy(),
                "lengths": c["lengths"].numpy(),
                "ckpt": c["ckpt"]}

    prop1 = _load_proposals(args.proposals1)
    prop2 = _load_proposals(args.proposals2) if args.proposals2 else None
    print(f"  proposals1: {prop1['ckpt']}")
    if prop2:
        print(f"  proposals2: {prop2['ckpt']} (anti-mirroring mix)")

    tokenizer = load_tokenizer(args.tokenizer_path, device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, _ = split_stocks(stocks)
    cache_tag = os.path.basename(args.tokenizer_path).replace(".pt", "")
    cache_dir = os.path.join(TrainingConfig.save_dir,
                             f"token_cache_{cache_tag}_het_vol")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(TrainingConfig.save_dir,
                                 f"token_cache_{cache_tag}")
    train_seqs = pack_stocks_v2(train_s, tokenizer, mode="train",
                                cache_dir=cache_dir, max_seq_len=args.max_seq_len)
    val_seqs = pack_stocks_v2(val_s, tokenizer, mode="train",
                              cache_dir=cache_dir, max_seq_len=args.max_seq_len)
    train_loader = make_dataloader_v2(train_seqs, batch_size=1, shuffle=True)
    val_loader = make_dataloader_v2(val_seqs, batch_size=1, shuffle=False)
    print(f"Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")
    if prop1["lengths"].shape[0] != len(train_seqs):
        print(f"  WARNING: proposal bank ({prop1['lengths'].shape[0]}) != "
              f"train seqs ({len(train_seqs)}) -- assuming same order.")

    model = KronosBert().to(device)
    if args.init_from and Path(args.init_from).exists():
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init["model_state_dict"])
        print(f"  init weights <- {args.init_from} "
              f"(val_loss={init.get('val_loss'):.4f})")
    else:
        print(f"  WARNING: init_from {args.init_from} missing; from scratch")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_updates = len(train_loader) * args.epochs
    warmup = max(1, int(total_updates * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16

    start_epoch = 0
    best_val = float("inf")
    global_step = 0
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            start_epoch = ck["epoch"] + 1
            best_val = ck.get("best_val", float("inf"))
            global_step = ck.get("global_step", 0)
        except (RuntimeError, KeyError) as e:
            print(f"  Cannot resume: {e}; fresh")
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "mlm_acc": []}
    t0 = time.time()
    vocab_base = ModelConfig.vocab_size
    mask_id = ModelConfig.vocab_size + 2
    corrupt_fracs = tuple(float(x) for x in args.corrupt_fracs.split(","))

    def run_one_batch(batch, seq_i):
        input_ids, _, _, time_id, pos_id, _, va_val, _, _, _ = batch
        input_ids = input_ids.to(device); time_id = time_id.to(device)
        pos_id = pos_id.to(device); va_val = va_val.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0); time_id = time_id.unsqueeze(0)
            pos_id = pos_id.unsqueeze(0); va_val = va_val.unsqueeze(0)
        B, N = input_ids.shape
        out_ids, out_time, out_va, out_lab, out_pos = [], [], [], [], []
        for b in range(B):
            prop = prop1["proposals"][seq_i + b]
            if prop2 is not None and (seq_i + b) % 2 == 1:
                prop = prop2["proposals"][seq_i + b]
            mid, mt, mv, ml, mp = _make_proposal_augmented_batch(
                input_ids[b], time_id[b], va_val[b], vocab_base, mask_id, prop,
                window=args.window, mlm_prob=args.mlm_prob,
                corrupt_fracs=corrupt_fracs,
                final_pos_frac=args.final_pos_frac,
                va_zero_frac=args.va_zero_frac,
                recency_window=args.recency_window,
                gpt_replace_prob=args.gpt_replace_prob, temp=NEG_TEMP,
                generator=(torch.Generator().manual_seed(global_step + b)
                           if args.deterministic else None))
            out_ids.append(mid); out_time.append(mt); out_va.append(mv)
            out_lab.append(ml); out_pos.append(mp)
        Lmax = max(x.shape[0] for x in out_ids)
        pad_ids = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        pad_time = torch.zeros(B, Lmax, 3, dtype=torch.long, device=device)
        pad_va = torch.zeros(B, Lmax, 2, dtype=torch.float32, device=device)
        pad_lab = torch.full((B, Lmax), -100, dtype=torch.long, device=device)
        pad_pos = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        for b in range(B):
            L = out_ids[b].shape[0]
            pad_ids[b, :L] = out_ids[b]; pad_time[b, :L] = out_time[b]
            pad_va[b, :L] = out_va[b]; pad_lab[b, :L] = out_lab[b]
            pad_pos[b, :L] = out_pos[b]
        return pad_ids, pad_time, pad_va, pad_lab, pad_pos

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses, accs = [], []
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"[{args.tag}] Epoch {epoch+1}/{args.epochs}")
        for bi, batch in enumerate(pbar):
            mid, mt, mv, ml, mp = run_one_batch(batch, bi)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(mid, mt, mp, va_values=mv)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       ml.view(-1), ignore_index=-100)
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    valid = ml != -100
                    if valid.any():
                        accs.append((pred[valid] == ml[valid])
                                    .float().mean().item())
            if not torch.isfinite(loss):
                print(f"  [skip] non-finite at step {global_step}")
                continue
            loss.backward()
            clip_grad_norm_(trainable, 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            scheduler.step(); global_step += 1
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "acc": f"{np.mean(accs[-50:]) if accs else 0:.3f}"})

        avg_train = float(np.mean(losses))
        model.eval()
        vlosses, vaccs = [], []
        with torch.inference_mode():
            for bi, batch in enumerate(val_loader):
                mid, mt, mv, ml, mp = run_one_batch(batch, bi)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(mid, mt, mp, va_values=mv)
                    vloss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                            ml.view(-1), ignore_index=-100)
                    pred = logits.argmax(dim=-1)
                    valid = ml != -100
                    if valid.any():
                        vaccs.append((pred[valid] == ml[valid])
                                     .float().mean().item())
                vlosses.append(vloss.item())
        avg_val = float(np.mean(vlosses))
        avg_val_acc = float(np.mean(vaccs)) if vaccs else 0.0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["mlm_acc"].append(avg_val_acc)
        save_tag = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                           "heads": ModelConfig.heads,
                           "num_kv_heads": ModelConfig.num_kv_heads,
                           "ffn_multiplier": ModelConfig.ffn_multiplier,
                           "dropout": ModelConfig.dropout,
                           "vocab_size": ModelConfig.vocab_size,
                           "vocab_fine": ModelConfig.vocab_fine,
                           "mask_id": mask_id,
                           "va_hidden_dim": ModelConfig.va_hidden_dim,
                           "rope_base": ModelConfig.rope_base,
                           "arch": "kronos_bert"},
                "val_loss": best_val, "mlm_acc": avg_val_acc, "epoch": epoch,
                "completed": epoch == args.epochs - 1, "tag": args.tag,
                "training_recipe": {"name": "proposal-augmented",
                                     "gpt_replace_prob": args.gpt_replace_prob,
                                     "temp": NEG_TEMP, "window": args.window,
                                     "final_pos_frac": args.final_pos_frac,
                                     "va_zero_frac": args.va_zero_frac,
                                     "recency_window": args.recency_window},
                "proposals1": str(args.proposals1),
                "proposals2": str(args.proposals2),
                "init_from": str(args.init_from),
            }, save_path)
            save_tag = "  -> Saved best"
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "best_val": best_val,
                    "global_step": global_step, "tag": args.tag}, ckpt_path)
        print(f"  [{epoch+1}/{args.epochs}] train={avg_train:.4f} "
              f"val={avg_val:.4f} best={best_val:.4f} "
              f"mlm_acc={avg_val_acc:.3f} "
              f"elapsed={time.time()-t0:.0f}s{save_tag}", flush=True)

    if os.path.exists(save_path):
        ck = torch.load(save_path, map_location="cpu", weights_only=False)
        ck["completed"] = True
        torch.save(ck, save_path)
    meta = {"tag": args.tag, "arch": "kronos_bert",
            "training_recipe": {"name": "proposal-augmented",
                                 "gpt_replace_prob": args.gpt_replace_prob,
                                 "temp": NEG_TEMP},
            "init_from": str(args.init_from),
            "proposals1": str(args.proposals1),
            "proposals2": str(args.proposals2), "history": history,
            "result": {"best_val": best_val, "final_mlm_acc": avg_val_acc}}
    meta_path = os.path.join(os.path.dirname(save_path) or ".",
                             f"bert_mlm_{args.tag}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. best val_loss: {best_val:.4f}, mlm_acc: {avg_val_acc:.3f}")


# ############################################################################
#  CLI dispatch
# ############################################################################

DEBUG_STAGES = {
    "base-mlm": ("A: base MLM pre-training", stage_a1),
    "scoring-aligned": ("A: scoring-aligned fine-tuning", stage_a2),
    "proposal-cache": ("A: GPT proposal cache", stage_cache),
    "proposal-augmented": ("A: proposal-augmented fine-tuning", stage_a3),
}


def _run_stage(label, fn, stage_args=()):
    print(f"\n{'='*60}\n  {label}\n{'='*60}\n")
    sys.argv = [sys.argv[0], *stage_args]
    fn()


def _run_all():
    """Run the complete A pipeline with no user-managed sub-steps."""
    for name in ("base-mlm", "scoring-aligned", "proposal-cache",
                 "proposal-augmented"):
        label, fn = DEBUG_STAGES[name]
        _run_stage(label, fn)


def main():
    if len(sys.argv) == 1:
        _run_all()
        return

    requested = sys.argv[1]
    if requested in {"-h", "--help"}:
        print("A_run.py 无参数时会依次生成 BERT.pt、BERT-FT.pt、BERT-PPS.pt，")
        print("并自动准备 GPT proposal cache。")
        print("仅调试单个环节时可传入语义名称：base-mlm、scoring-aligned、")
        print("proposal-cache、proposal-augmented。")
        return

    if requested in {"all", "A-all"}:
        _run_all()
        return

    if requested in {"--stage", "-s"}:
        if len(sys.argv) < 3:
            print("缺少调试阶段名称；A_run.py 默认运行完整 A 流程。")
            sys.exit(2)
        requested = sys.argv[2]
        stage_args = sys.argv[3:]
    else:
        stage_args = sys.argv[2:]

    if requested not in DEBUG_STAGES:
        print("A_run.py 默认运行完整 A 流程；未知的调试阶段：", requested)
        print("可用名称：base-mlm、scoring-aligned、proposal-cache、proposal-augmented")
        sys.exit(2)

    label, fn = DEBUG_STAGES[requested]
    _run_stage(label, fn, stage_args)


if __name__ == "__main__":
    main()
