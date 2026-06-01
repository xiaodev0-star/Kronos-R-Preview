"""
Kronos-R-Preview Follow-up HPO (4-hour budget)
Focus:
  1. Full training verification (15-20 epochs)
  2. Focal γ sensitivity: {2.5, 3.5, 4.0}
  3. Reasoning module variants (16 tokens, 2 layers)
  4. Ensemble evaluation

Supports checkpoint/resume via STATE_FILE.
"""
import json, os, sys, time, math, copy, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from config import DataConfig, ModelConfig, TrainingConfig, NormConfig
from data_processor import load_stocks, split_stocks, pack_stocks, make_dataloader, rolling_normalize
from model.tokenizer import HierarchicalQuantizer
from model.tokenizer_config import build_tokenizer_kwargs
from model.kronos_preview import KronosPreview
from reproducibility import set_global_seed

# ─── Constants ───────────────────────────────────────────────────────────────
STATE_FILE = "hpo_followup_state.json"
RESULTS_FILE = "hpo_followup_results.json"
TOTAL_BUDGET_S = 4 * 3600  # 4 hours
EVAL_N = 100  # eval stocks
GEN_STEPS = 10
AMP_DTYPE = torch.bfloat16

# ─── Loaders ─────────────────────────────────────────────────────────────────
def load_tokenizer(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = HierarchicalQuantizer(**build_tokenizer_kwargs(ckpt.get("config", {})))
    tok.load_state_dict(ckpt["model_state_dict"])
    tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad = False
    return tok

def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = KronosPreview().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ─── Custom Losses ───────────────────────────────────────────────────────────
def focal_loss(logits, targets, gamma=2.0, ignore_index=-100):
    mask = (targets != ignore_index).float()
    n_valid = mask.sum().clamp(min=1)
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1.0)
    focal_w = (1 - pt) ** gamma
    return (focal_w * ce * mask).sum() / n_valid


def train_model(tokenizer, train_seqs, val_seqs, device, epochs, loss_type="ce",
                loss_kwargs=None, overrides=None, label=""):
    """Train with custom loss."""
    if overrides is None:
        overrides = {}
    saved = {}
    for cls_name, params in overrides.items():
        cfg = ModelConfig if cls_name == "ModelConfig" else TrainingConfig
        for k, v in params.items():
            saved[(cls_name, k)] = getattr(cfg, k)
            setattr(cfg, k, v)

    train_loader = make_dataloader(train_seqs, batch_size=TrainingConfig.batch_size, shuffle=True)
    val_loader = make_dataloader(val_seqs, batch_size=TrainingConfig.batch_size, shuffle=False)
    model = KronosPreview().to(device)
    if TrainingConfig.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=TrainingConfig.learning_rate,
                                  weight_decay=TrainingConfig.weight_decay)
    total_updates = len(train_loader) * epochs
    warmup = max(1, int(total_updates * TrainingConfig.warmup_ratio))
    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val, best_state, best_epoch = float("inf"), None, 0
    history = {"train": [], "val": []}
    loss_kw = loss_kwargs or {}

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        optimizer.zero_grad()
        for step, (inp, tgt, tid, pos, mask) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=True).unsqueeze(0)
            tgt = tgt.to(device, non_blocking=True).unsqueeze(0)
            tid = tid.to(device, non_blocking=True).unsqueeze(0)
            pos = pos.to(device, non_blocking=True).unsqueeze(0)
            mask = mask.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                if loss_type == "ce":
                    _, _, loss = model(inp, tid, pos, mask, tgt)
                elif loss_type == "focal":
                    logits_coarse, _, _ = model(inp, tid, pos, mask)
                    shift_logits = logits_coarse[:, :-1, :].contiguous()
                    shift_targets = tgt.contiguous()
                    if (shift_targets == -100).all(): continue
                    loss = focal_loss(shift_logits.view(-1, shift_logits.size(-1)),
                                      shift_targets.view(-1), **loss_kw)
                else:
                    _, _, loss = model(inp, tid, pos, mask, tgt)

            if loss is None: continue
            (loss / TrainingConfig.accumulation_steps).backward()
            if (step + 1) % TrainingConfig.accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, TrainingConfig.grad_clip)
                optimizer.step(); optimizer.zero_grad(); scheduler.step()
            losses.append(loss.item())

        avg_train = sum(losses) / max(len(losses), 1)
        model.eval()
        vlosses = []
        with torch.no_grad():
            for inp, tgt, tid, pos, mask in val_loader:
                inp = inp.to(device).unsqueeze(0); tgt = tgt.to(device).unsqueeze(0)
                tid = tid.to(device).unsqueeze(0); pos = pos.to(device).unsqueeze(0)
                mask = mask.to(device)
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    _, _, loss = model(inp, tid, pos, mask, tgt)
                if loss is not None: vlosses.append(loss.item())
        avg_val = sum(vlosses) / max(len(vlosses), 1)
        elapsed = time.time() - t0
        history["train"].append(avg_train); history["val"].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val; best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1}/{epochs}: train={avg_train:.4f} val={avg_val:.4f} "
                  f"best={best_val:.4f}(ep{best_epoch+1}) {elapsed:.0f}s")

    if best_state: model.load_state_dict(best_state)
    for (cls_name, k), v in saved.items():
        cfg = ModelConfig if cls_name == "ModelConfig" else TrainingConfig
        setattr(cfg, k, v)
    return model, best_val, best_epoch, history


# ─── Reasoning Model ─────────────────────────────────────────────────────────
class CausalReasoningBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False), nn.SiLU(),
            nn.Linear(dim * 2, dim, bias=False))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, memory):
        h = self.norm1(x)
        h, _ = self.cross_attn(h, memory, memory)
        x = x + self.gate.tanh() * h
        x = x + self.ffn(self.norm2(x))
        return x


class KronosPreviewWithReasoning(nn.Module):
    def __init__(self, base_state, n_reason_tokens=8, n_reason_layers=1,
                 dropout=0.0, learnable_temp=False):
        super().__init__()
        cfg = ModelConfig
        from model.kronos_preview import TransformerBlock, RotaryEmbedding, RMSNorm
        self.token_emb = nn.Embedding(cfg.vocab_size + 2, cfg.dim)
        self.time_emb_day = nn.Embedding(32, cfg.dim)
        self.time_emb_month = nn.Embedding(13, cfg.dim)
        self.time_emb_year = nn.Embedding(100, cfg.dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.num_kv_heads, cfg.ffn_multiplier, cfg.dropout)
            for _ in range(cfg.depth)])
        self.norm = RMSNorm(cfg.dim)
        self.head_coarse = nn.Linear(cfg.dim, cfg.vocab_size + 2, bias=True)
        self.head_fine = nn.Linear(cfg.dim, cfg.vocab_size + 2, bias=True)
        self.rotary = RotaryEmbedding(cfg.dim // cfg.heads, base=cfg.rope_base)
        self.reason_tokens = nn.Parameter(torch.randn(1, n_reason_tokens, cfg.dim) * 0.02)
        self.reason_blocks = nn.ModuleList([
            CausalReasoningBlock(cfg.dim, heads=cfg.heads, dropout=dropout)
            for _ in range(n_reason_layers)])
        self._gradient_checkpointing = False
        self.learnable_temp = learnable_temp
        if learnable_temp:
            self.logit_temp = nn.Parameter(torch.zeros(1))  # log(temperature)

        missing, unexpected = self.load_state_dict(base_state, strict=False)
        print(f"  Loaded base model. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    def enable_gradient_checkpointing(self): self._gradient_checkpointing = True

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None, targets=None):
        no_batch = input_ids.dim() == 1
        if no_batch:
            input_ids = input_ids.unsqueeze(0); time_ids = time_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
            attn_mask = attn_mask.unsqueeze(0) if attn_mask is not None else None
            if targets is not None: targets = targets.unsqueeze(0)

        x = self.token_emb(input_ids)
        x = x + self.time_emb_day(time_ids[..., 0])
        x = x + self.time_emb_month(time_ids[..., 1])
        x = x + self.time_emb_year(time_ids[..., 2])
        sin, cos = self.rotary(position_ids)

        for block in self.blocks:
            x = block(x, sin, cos, attn_mask)

        B = x.size(0)
        memory = self.reason_tokens.expand(B, -1, -1)
        for rblock in self.reason_blocks:
            x = rblock(x, memory)

        x = self.norm(x)
        logits_coarse = self.head_coarse(x)
        logits_fine = self.head_fine(x)

        if self.learnable_temp:
            temp = torch.exp(self.logit_temp).clamp(0.1, 10.0)
            logits_coarse = logits_coarse / temp
            logits_fine = logits_fine / temp

        loss = None
        if targets is not None:
            shift_logits = logits_coarse[:, :-1, :].contiguous()
            shift_targets = targets.contiguous()
            if (shift_targets != -100).any():
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                       shift_targets.view(-1), ignore_index=-100)
        if no_batch:
            logits_coarse = logits_coarse.squeeze(0); logits_fine = logits_fine.squeeze(0)
        return logits_coarse, logits_fine, loss


def train_reasoning_model(base_ckpt_path, tokenizer, train_seqs, val_seqs, device, epochs,
                          n_reason_tokens=8, n_reason_layers=1, dropout=0.0,
                          learnable_temp=False, freeze_base=True, overrides=None, label=""):
    if overrides is None: overrides = {}
    saved = {}
    for cls_name, params in overrides.items():
        cfg = ModelConfig if cls_name == "ModelConfig" else TrainingConfig
        for k, v in params.items():
            saved[(cls_name, k)] = getattr(cfg, k); setattr(cfg, k, v)

    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    model = KronosPreviewWithReasoning(
        base_state=base_ckpt["model_state_dict"],
        n_reason_tokens=n_reason_tokens, n_reason_layers=n_reason_layers,
        dropout=dropout, learnable_temp=learnable_temp).to(device)

    if freeze_base:
        for name, param in model.named_parameters():
            if "reason" not in name and "logit_temp" not in name:
                param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}")

    train_loader = make_dataloader(train_seqs, batch_size=TrainingConfig.batch_size, shuffle=True)
    val_loader = make_dataloader(val_seqs, batch_size=TrainingConfig.batch_size, shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=TrainingConfig.learning_rate,
                                  weight_decay=TrainingConfig.weight_decay)
    total_updates = len(train_loader) * epochs
    warmup = max(1, int(total_updates * TrainingConfig.warmup_ratio))
    def lr_lambda(step):
        if step < warmup: return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val, best_state, best_epoch = float("inf"), None, 0
    history = {"train": [], "val": []}

    t0 = time.time()
    for epoch in range(epochs):
        model.train(); losses = []; optimizer.zero_grad()
        for step, (inp, tgt, tid, pos, mask) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=True).unsqueeze(0)
            tgt = tgt.to(device, non_blocking=True).unsqueeze(0)
            tid = tid.to(device, non_blocking=True).unsqueeze(0)
            pos = pos.to(device, non_blocking=True).unsqueeze(0)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                _, _, loss = model(inp, tid, pos, mask, tgt)
            if loss is None: continue
            (loss / TrainingConfig.accumulation_steps).backward()
            if (step + 1) % TrainingConfig.accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, TrainingConfig.grad_clip)
                optimizer.step(); optimizer.zero_grad(); scheduler.step()
            losses.append(loss.item())
        avg_train = sum(losses) / max(len(losses), 1)
        model.eval()
        vlosses = []
        with torch.no_grad():
            for inp, tgt, tid, pos, mask in val_loader:
                inp = inp.to(device).unsqueeze(0); tgt = tgt.to(device).unsqueeze(0)
                tid = tid.to(device).unsqueeze(0); pos = pos.to(device).unsqueeze(0)
                mask = mask.to(device)
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    _, _, loss = model(inp, tid, pos, mask, tgt)
                if loss is not None: vlosses.append(loss.item())
        avg_val = sum(vlosses) / max(len(vlosses), 1)
        elapsed = time.time() - t0
        history["train"].append(avg_train); history["val"].append(avg_val)
        if avg_val < best_val: best_val = avg_val; best_epoch = epoch
        if best_state is None or avg_val < best_val:
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1}/{epochs}: train={avg_train:.4f} val={avg_val:.4f} "
                  f"best={best_val:.4f}(ep{best_epoch+1}) {elapsed:.0f}s")
    if best_state: model.load_state_dict(best_state)
    for (cls_name, k), v in saved.items():
        cfg = ModelConfig if cls_name == "ModelConfig" else TrainingConfig
        setattr(cfg, k, v)
    return model, best_val, best_epoch, history


# ─── Evaluation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_all(model, tokenizer, test_stocks, device, label="", max_stocks=EVAL_N):
    """1-step accuracy + collapse/mape/da. Returns dict of metrics."""
    vocab = tokenizer.bsq_coarse.vocab_size
    bos_id = vocab
    # 1-step
    stocks = test_stocks[:max_stocks]
    total_correct, total_tokens = 0, 0
    for stock in stocks:
        feat = stock["features_raw"]
        day, month, year = stock["day"], stock["month"], stock["year"]
        cutoff = np.datetime64(pd.Timestamp(DataConfig.cutoff_date))
        ci = int(np.searchsorted(stock["dates_dt"], cutoff, side="left"))
        if ci >= len(feat) - 5: continue
        feat_t = feat[ci:]
        if len(feat_t) < NormConfig.min_lookback + 5: continue
        normed = rolling_normalize(feat_t)
        idx_c, _ = tokenizer.encode(torch.from_numpy(normed).float().unsqueeze(0).to(device))
        token_ids = idx_c[0].cpu().numpy()
        ids = [bos_id] + token_ids.tolist()
        d_l = [day[ci]] + day[ci:].tolist(); m_l = [month[ci]] + month[ci:].tolist()
        y_l = [year[ci]] + year[ci:].tolist()
        S = len(ids)
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        tids = torch.stack([torch.tensor([d_l[:-1]], dtype=torch.long),
                            torch.tensor([m_l[:-1]], dtype=torch.long),
                            torch.tensor([y_l[:-1]], dtype=torch.long)], dim=-1).to(device)
        pos = torch.arange(S - 1, device=device).unsqueeze(0)
        cav_mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool, device=device))
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            lc, _, _ = model(inp, tids, pos, cav_mask)
        preds = lc.argmax(dim=-1)
        total_correct += (preds == tgt).float().sum().item()
        total_tokens += tgt.shape[1]
    step1 = total_correct / max(total_tokens, 1)

    # Collapse + MAPE + DA
    rng = np.random.RandomState(42)
    eval_stocks = rng.choice(test_stocks, min(max_stocks, len(test_stocks)), replace=False)
    all_mape, all_da, pred_counts = [], [], {}
    all_pred_amp, all_acc_amp = [], []

    for stock in eval_stocks:
        feat = stock["features_raw"]
        day, month, year = stock["day"], stock["month"], stock["year"]
        cutoff = np.datetime64(pd.Timestamp(DataConfig.cutoff_date))
        ci = int(np.searchsorted(stock["dates_dt"], cutoff, side="left"))
        if ci >= len(feat) - 15: continue
        feat_t, day_t, month_t, year_t = feat[ci:], day[ci:], month[ci:], year[ci:]
        if len(feat_t) < NormConfig.min_lookback + 15: continue
        normed = rolling_normalize(feat_t)
        idx_c, _ = tokenizer.encode(torch.from_numpy(normed).float().unsqueeze(0).to(device))
        token_ids = idx_c[0].cpu().numpy()
        split_point = min(len(token_ids) - GEN_STEPS, 128)
        if split_point < 10: continue
        gt_tokens = token_ids[split_point:split_point + GEN_STEPS].tolist()
        ids = [bos_id] + token_ids[:split_point].tolist()
        d_ctx = [day_t[0]] + day_t[:split_point].tolist()
        m_ctx = [month_t[0]] + month_t[:split_point].tolist()
        y_ctx = [year_t[0]] + year_t[:split_point].tolist()
        generated = []
        for step in range(GEN_STEPS):
            S = len(ids)
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            tids = torch.stack([torch.tensor([d_ctx], dtype=torch.long),
                                torch.tensor([m_ctx], dtype=torch.long),
                                torch.tensor([y_ctx], dtype=torch.long)], dim=-1).to(device)
            pos = torch.arange(S, device=device).unsqueeze(0)
            causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                lc, _, _ = model(inp, tids, pos, causal)
            nxt = lc[0, -1].argmax().item()
            generated.append(nxt)
            pred_counts[nxt] = pred_counts.get(nxt, 0) + 1
            ids.append(nxt)
            ni = split_point + step
            d_ctx.append(day_t[ni] if ni < len(day_t) else d_ctx[-1])
            m_ctx.append(m_ctx[-1]); y_ctx.append(y_ctx[-1])
        pt = torch.tensor([generated], dtype=torch.long, device=device)
        gtt = torch.tensor([gt_tokens], dtype=torch.long, device=device)
        pf = tokenizer.decode_all(pt.unsqueeze(-1).expand(-1, -1, 2).contiguous())[0].cpu().numpy()
        gf = tokenizer.decode_all(gtt.unsqueeze(-1).expand(-1, -1, 2).contiguous())[0].cpu().numpy()
        eps = 1e-6
        all_mape.append(np.mean(np.abs(pf - gf) / (np.abs(gf) + eps)) * 100)
        if len(pf) > 1: all_da.append(np.mean(np.sign(pf[:, 0]) == np.sign(gf[:, 0])))
        all_pred_amp.append(np.mean(np.abs(pf[:, 0])))
        all_acc_amp.append(np.mean(np.abs(gf[:, 0])))

    total_preds = sum(pred_counts.values())
    top_ratio = max(pred_counts.values()) / max(total_preds, 1) if pred_counts else 0
    return {
        "step1_acc": step1,
        "mape": float(np.mean(all_mape)) if all_mape else None,
        "da": float(np.mean(all_da)) if all_da else None,
        "collapse": float(np.mean(all_pred_amp) - np.mean(all_acc_amp)) if all_pred_amp else None,
        "pred_amp": float(np.mean(all_pred_amp)) if all_pred_amp else None,
        "acc_amp": float(np.mean(all_acc_amp)) if all_acc_amp else None,
        "unique_pred_tokens": len(pred_counts),
        "top_token_ratio": top_ratio,
        "zero_collapse": top_ratio > 0.5,
        "n_stocks": len(all_mape),
    }


# ─── Ensemble ────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_ensemble(models, weights, tokenizer, test_stocks, device, max_stocks=EVAL_N):
    """Ensemble of multiple models by weighted averaging logits."""
    vocab = tokenizer.bsq_coarse.vocab_size; bos_id = vocab
    rng = np.random.RandomState(42)
    eval_stocks = rng.choice(test_stocks, min(max_stocks, len(test_stocks)), replace=False)
    all_mape, all_da, pred_counts = [], [], {}
    all_pred_amp, all_acc_amp = [], []

    for stock in eval_stocks:
        feat = stock["features_raw"]
        day, month, year = stock["day"], stock["month"], stock["year"]
        cutoff = np.datetime64(pd.Timestamp(DataConfig.cutoff_date))
        ci = int(np.searchsorted(stock["dates_dt"], cutoff, side="left"))
        if ci >= len(feat) - 15: continue
        feat_t, day_t, month_t, year_t = feat[ci:], day[ci:], month[ci:], year[ci:]
        if len(feat_t) < NormConfig.min_lookback + 15: continue
        normed = rolling_normalize(feat_t)
        idx_c, _ = tokenizer.encode(torch.from_numpy(normed).float().unsqueeze(0).to(device))
        token_ids = idx_c[0].cpu().numpy()
        split_point = min(len(token_ids) - GEN_STEPS, 128)
        if split_point < 10: continue
        gt_tokens = token_ids[split_point:split_point + GEN_STEPS].tolist()
        ids = [bos_id] + token_ids[:split_point].tolist()
        d_ctx = [day_t[0]] + day_t[:split_point].tolist()
        m_ctx = [month_t[0]] + month_t[:split_point].tolist()
        y_ctx = [year_t[0]] + year_t[:split_point].tolist()
        generated = []
        for step in range(GEN_STEPS):
            S = len(ids)
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            tids = torch.stack([torch.tensor([d_ctx], dtype=torch.long),
                                torch.tensor([m_ctx], dtype=torch.long),
                                torch.tensor([y_ctx], dtype=torch.long)], dim=-1).to(device)
            pos = torch.arange(S, device=device).unsqueeze(0)
            causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
            # Weighted average of logits
            ensemble_logits = None
            for model, w in zip(models, weights):
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    lc, _, _ = model(inp, tids, pos, causal)
                if ensemble_logits is None:
                    ensemble_logits = w * F.softmax(lc, dim=-1)
                else:
                    ensemble_logits += w * F.softmax(lc, dim=-1)
            nxt = ensemble_logits[0, -1].argmax().item()
            generated.append(nxt)
            pred_counts[nxt] = pred_counts.get(nxt, 0) + 1
            ids.append(nxt)
            ni = split_point + step
            d_ctx.append(day_t[ni] if ni < len(day_t) else d_ctx[-1])
            m_ctx.append(m_ctx[-1]); y_ctx.append(y_ctx[-1])
        pt = torch.tensor([generated], dtype=torch.long, device=device)
        gtt = torch.tensor([gt_tokens], dtype=torch.long, device=device)
        pf = tokenizer.decode_all(pt.unsqueeze(-1).expand(-1, -1, 2).contiguous())[0].cpu().numpy()
        gf = tokenizer.decode_all(gtt.unsqueeze(-1).expand(-1, -1, 2).contiguous())[0].cpu().numpy()
        eps = 1e-6
        all_mape.append(np.mean(np.abs(pf - gf) / (np.abs(gf) + eps)) * 100)
        if len(pf) > 1: all_da.append(np.mean(np.sign(pf[:, 0]) == np.sign(gf[:, 0])))
        all_pred_amp.append(np.mean(np.abs(pf[:, 0])))
        all_acc_amp.append(np.mean(np.abs(gf[:, 0])))

    total_preds = sum(pred_counts.values())
    top_ratio = max(pred_counts.values()) / max(total_preds, 1) if pred_counts else 0
    return {
        "mape": float(np.mean(all_mape)) if all_mape else None,
        "da": float(np.mean(all_da)) if all_da else None,
        "collapse": float(np.mean(all_pred_amp) - np.mean(all_acc_amp)) if all_pred_amp else None,
        "pred_amp": float(np.mean(all_pred_amp)) if all_pred_amp else None,
        "acc_amp": float(np.mean(all_acc_amp)) if all_acc_amp else None,
        "unique_pred_tokens": len(pred_counts),
        "top_token_ratio": top_ratio,
        "zero_collapse": top_ratio > 0.5,
    }


# ─── Experiment Definitions ──────────────────────────────────────────────────
# Ordered by priority
EXPERIMENTS = [
    # Phase 1: Full training verification (15 epochs) — HIGHEST PRIORITY
    {"name": "fu_reason_frozen_15ep", "desc": "Verify best MAPE: frozen reasoning, 15 epochs",
     "type": "reasoning", "base_ckpt": "checkpoints/base_model.pt",
     "epochs": 15, "n_reason_tokens": 8, "n_reason_layers": 1,
     "freeze_base": True, "ptype": 1},
    {"name": "fu_focal_g3_15ep", "desc": "Verify best calibration: focal gamma=3.0, 15 epochs",
     "type": "train", "loss_type": "focal", "loss_kwargs": {"gamma": 3.0},
     "epochs": 15, "ptype": 1},

    # Phase 2: Focal gamma sensitivity — γ=3.5 (closest to best γ=3.0)
    {"name": "fu_focal_g3p5_10ep", "desc": "Focal gamma=3.5, 10 epochs",
     "type": "train", "loss_type": "focal", "loss_kwargs": {"gamma": 3.5},
     "epochs": 10, "ptype": 2},
    {"name": "fu_focal_g2p5_10ep", "desc": "Focal gamma=2.5, 10 epochs",
     "type": "train", "loss_type": "focal", "loss_kwargs": {"gamma": 2.5},
     "epochs": 10, "ptype": 2},

    # Phase 3: Reasoning module variant — 16 tokens (lower priority)
    {"name": "fu_reason_16tok_10ep", "desc": "Reasoning: 16 tokens, frozen base",
     "type": "reasoning", "base_ckpt": "checkpoints/base_model.pt",
     "epochs": 10, "n_reason_tokens": 16, "n_reason_layers": 1,
     "freeze_base": True, "ptype": 3},
]


def run_train_exp(exp, tokenizer, train_seqs, val_seqs, test_stocks, device):
    """Run a standard training experiment."""
    name = exp["name"]
    print(f"\n{'='*65}")
    print(f"EXPERIMENT: {name} — {exp['desc']}")
    print(f"  Loss: {exp.get('loss_type','ce')}, Epochs: {exp['epochs']}, "
          f"Loss kwargs: {exp.get('loss_kwargs',{})}")
    print(f"{'='*65}")

    t0 = time.time()
    model, best_val, best_epoch, history = train_model(
        tokenizer, train_seqs, val_seqs, device, epochs=exp["epochs"],
        loss_type=exp.get("loss_type", "ce"),
        loss_kwargs=exp.get("loss_kwargs"),
        label=name)
    train_time = time.time() - t0

    save_path = f"checkpoints/{name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                   "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads},
        "val_loss": best_val, "epoch": best_epoch, "experiment": name,
    }, save_path)

    print(f"  Evaluating...")
    metrics = eval_all(model, tokenizer, test_stocks, device, label=name)

    result = {
        "name": name, "desc": exp["desc"], "type": exp["type"],
        "loss_type": exp.get("loss_type", "ce"),
        "loss_kwargs": exp.get("loss_kwargs", {}),
        "epochs": exp["epochs"],
        "best_val_loss": best_val, "best_epoch": best_epoch + 1,
        "train_time_s": train_time,
        **metrics,
    }

    c = metrics.get("collapse", 0) or 0
    ar = (metrics.get("pred_amp", 1) or 1) / max(metrics.get("acc_amp", 1) or 1, 1e-6)
    print(f"  RESULTS: val={best_val:.4f}(ep{best_epoch+1}) | MAPE={metrics['mape']:.1f}% "
          f"DA={metrics['da']:.3f} | Collapse={c:.4f} | AR={ar:.2f}x | "
          f"Tok={metrics['unique_pred_tokens']} | {train_time:.0f}s")
    return result


def run_reasoning_exp(exp, tokenizer, train_seqs, val_seqs, test_stocks, device):
    """Run a reasoning module experiment."""
    name = exp["name"]
    print(f"\n{'='*65}")
    print(f"REASONING EXPERIMENT: {name} — {exp['desc']}")
    print(f"  Tokens: {exp.get('n_reason_tokens',8)}, "
          f"Layers: {exp.get('n_reason_layers',1)}, "
          f"Freeze base: {exp.get('freeze_base',True)}")
    print(f"{'='*65}")

    t0 = time.time()
    model, best_val, best_epoch, history = train_reasoning_model(
        exp["base_ckpt"], tokenizer, train_seqs, val_seqs, device,
        epochs=exp["epochs"],
        n_reason_tokens=exp.get("n_reason_tokens", 8),
        n_reason_layers=exp.get("n_reason_layers", 1),
        learnable_temp=exp.get("learnable_temp", False),
        freeze_base=exp.get("freeze_base", True),
        label=name)
    train_time = time.time() - t0

    save_path = f"checkpoints/{name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "experiment": name}, save_path)

    print(f"  Evaluating...")
    metrics = eval_all(model, tokenizer, test_stocks, device, label=name)

    result = {
        "name": name, "desc": exp["desc"], "type": exp["type"],
        "n_reason_tokens": exp.get("n_reason_tokens", 8),
        "n_reason_layers": exp.get("n_reason_layers", 1),
        "freeze_base": exp.get("freeze_base", True),
        "epochs": exp["epochs"],
        "best_val_loss": best_val, "best_epoch": best_epoch + 1,
        "train_time_s": train_time,
        **metrics,
    }

    c = metrics.get("collapse", 0) or 0
    ar = (metrics.get("pred_amp", 1) or 1) / max(metrics.get("acc_amp", 1) or 1, 1e-6)
    print(f"  RESULTS: val={best_val:.4f}(ep{best_epoch+1}) | MAPE={metrics['mape']:.1f}% "
          f"DA={metrics['da']:.3f} | Collapse={c:.4f} | AR={ar:.2f}x | "
          f"Tok={metrics['unique_pred_tokens']} | {train_time:.0f}s")
    return result


def run_ensemble_exp(all_results, tokenizer, test_stocks, device):
    """Run ensemble evaluation combining best configurations."""
    print(f"\n{'='*65}")
    print(f"ENSEMBLE EVALUATION")
    print(f"{'='*65}")

    # Find the two models to ensemble
    models_to_load = {"w4_reason_frozen": None, "w2_focal_g3": None}
    for name in models_to_load:
        ckpt_path = f"checkpoints/{name}.pt"
        if os.path.exists(ckpt_path):
            print(f"  Loading {name}...")
            if "reason" in name:
                base_ckpt = torch.load("checkpoints/base_model.pt", map_location="cpu", weights_only=False)
                m = KronosPreviewWithReasoning(
                    base_state=base_ckpt["model_state_dict"],
                    n_reason_tokens=8, n_reason_layers=1).to(device)
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                m.load_state_dict(ckpt["model_state_dict"], strict=False)
                m.eval()
            else:
                m = load_model(ckpt_path, device)
            models_to_load[name] = m

    if not all(models_to_load.values()):
        print("  Missing checkpoints, skipping ensemble")
        return None

    # Test weight combinations
    weight_combos = [
        ([0.5, 0.5], "equal"),
        ([0.6, 0.4], "reason_bias"),
        ([0.4, 0.6], "focal_bias"),
    ]

    ensemble_results = []
    models_list = [models_to_load["w4_reason_frozen"], models_to_load["w2_focal_g3"]]

    for weights, wname in weight_combos:
        print(f"  Ensemble: {wname} weights={weights}")
        metrics = eval_ensemble(models_list, weights, tokenizer, test_stocks, device)
        c = metrics.get("collapse", 0) or 0
        ar = (metrics.get("pred_amp", 1) or 1) / max(metrics.get("acc_amp", 1) or 1, 1e-6)
        print(f"    MAPE={metrics['mape']:.1f}% DA={metrics['da']:.3f} "
              f"Collapse={c:.4f} AR={ar:.2f}x Tok={metrics['unique_pred_tokens']}")
        ensemble_results.append({
            "name": f"ensemble_{wname}",
            "desc": f"Ensemble w4_reason_frozen + w2_focal_g3, weights={weights}",
            "type": "ensemble", "weights": weights,
            "mape": metrics["mape"], "da": metrics["da"],
            "collapse": c,
            "pred_amplitude": metrics["pred_amp"],
            "acc_amplitude": metrics["acc_amp"],
            "unique_pred_tokens": metrics["unique_pred_tokens"],
            "top_token_ratio": metrics["top_token_ratio"],
            "zero_collapse": metrics["zero_collapse"],
        })

    return ensemble_results


# ─── State management ────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "results": [], "start_time": time.time()}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Budget: {TOTAL_BUDGET_S/3600:.1f} hours")

    # Load state for resume
    state = load_state()
    completed = set(state["completed"])
    all_results = state["results"]
    if "start_time" not in state:
        state["start_time"] = time.time()

    print(f"Resuming from state: {len(completed)} experiments already done")
    if completed:
        print(f"  Completed: {', '.join(sorted(completed))}")

    tokenizer = load_tokenizer(TrainingConfig.tokenizer_path, device)
    print("Tokenizer loaded.")

    stocks = load_stocks(max_stocks=DataConfig.max_stocks)
    train_s, val_s, test_stocks = split_stocks(stocks)
    print(f"Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_stocks)}")

    train_seqs = pack_stocks(train_s, tokenizer, mode="train",
                             cache_dir=TrainingConfig.token_cache_dir)
    val_seqs = pack_stocks(val_s, tokenizer, mode="train",
                           cache_dir=TrainingConfig.token_cache_dir)
    print(f"Sequences: train={len(train_seqs)}, val={len(val_seqs)}")

    # Run experiments in priority order
    for exp in EXPERIMENTS:
        if exp["name"] in completed:
            continue

        elapsed = time.time() - state["start_time"]
        remaining = TOTAL_BUDGET_S - elapsed
        if remaining < 600:  # less than 10 min
            print(f"\n[!] Budget low ({remaining/60:.0f}min remaining). Stopping.")
            break

        exp_type = exp["type"]

        # Estimate time (from previous runs: ~42 min for 10ep, ~59 min for 15ep, ~33 min for frozen reason 10ep)
        if exp_type == "reasoning":
            est = exp["epochs"] * 2.7 + 8  # ~2.7 min/epoch frozen
        else:
            est = exp["epochs"] * 3.4 + 8  # ~3.4 min/epoch

        if remaining < est * 60:
            print(f"\n[!] Not enough time for {exp['name']} "
                  f"(need {est:.0f}min, have {remaining/60:.0f}min). Skipping.")
            continue

        print(f"\n[{elapsed/3600:.2f}h elapsed, {remaining/3600:.2f}h remaining]")

        result = None
        if exp_type == "train":
            result = run_train_exp(exp, tokenizer, train_seqs, val_seqs, test_stocks, device)
        elif exp_type == "reasoning":
            result = run_reasoning_exp(exp, tokenizer, train_seqs, val_seqs, test_stocks, device)

        if result:
            all_results.append(result)
            completed.add(exp["name"])
            state["completed"] = list(completed)
            state["results"] = all_results
            save_state(state)
            # Also save results standalone
            with open(RESULTS_FILE, "w") as f:
                json.dump({"total_time_h": (time.time() - state["start_time"]) / 3600,
                           "results": all_results}, f, indent=2)

    # ── Ensemble ────────────────────────────────────────────────────────
    elapsed = time.time() - state["start_time"]
    remaining = TOTAL_BUDGET_S - elapsed
    if "ensemble" not in str(state.get("completed", [])) and remaining > 300:
        print(f"\n[{elapsed/3600:.2f}h] Running ensemble evaluation...")
        ensemble_results = run_ensemble_exp(all_results, tokenizer, test_stocks, device)
        if ensemble_results:
            all_results.extend(ensemble_results)
            state["results"] = all_results
            state["completed"] = list(completed) + ["ensemble"]
            save_state(state)
            with open(RESULTS_FILE, "w") as f:
                json.dump({"total_time_h": (time.time() - state["start_time"]) / 3600,
                           "results": all_results}, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - state["start_time"]
    print(f"\n{'='*75}")
    print(f"FOLLOW-UP HPO COMPLETE — Total time: {total_time/3600:.2f} hours")
    print(f"Experiments: {len(all_results)}")
    print(f"{'='*75}")

    # Merge with previous results for comparison
    all_valid = [r for r in all_results if r.get("mape") is not None and not r.get("zero_collapse")]
    print(f"\n{'Experiment':<35} MAPE     DA     Collapse  AmpRatio  Tokens  Type")
    print("-" * 90)
    for r in all_results:
        c = r.get("collapse", 0) or 0
        ar = (r.get("pred_amplitude", 1) or 1) / max(r.get("acc_amplitude", 1) or 1, 1e-6)
        print(f'{r["name"]:<35} {r["mape"]:>7.1f}% {r.get("da",0):>5.3f}  {c:>+8.4f}  {ar:>7.2f}x  '
              f'{r.get("unique_pred_tokens",0):>5}  {r.get("type","?")}')

    # Comparison with baseline
    print(f"\n=== WITH BASELINE CONTEXT ===")
    print(f"Baseline (w1_baseline_10ep):         MAPE=564.9%  DA=0.625  Collapse=-0.1011  AR=0.70x")
    print(f"Previous best (w4_reason_frozen):    MAPE=516.4%  DA=0.626  Collapse=-0.0947  AR=0.72x")
    print(f"Previous best cal (w2_focal_g3):     MAPE=530.4%  DA=0.580  Collapse=-0.0352  AR=0.90x")

    total_time_h = time.time() - state["start_time"]
    with open(RESULTS_FILE, "w") as f:
        json.dump({"total_time_h": total_time_h / 3600, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"State saved to {STATE_FILE} (can resume later)")


if __name__ == "__main__":
    main()
