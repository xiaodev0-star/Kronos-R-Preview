"""
14-Hour HPO for Kronos-R-Preview
Focus: Anti-zero-collapse loss functions + traditional hyperparameter sweep
Collapse metric: Pred_x - Acc_x (predicted vs actual price movement amplitude)

Waves:
  1. Traditional HPO: lr, wd, dropout, warmup, accum (8 experiments, 15ep each)
  2. Loss experiments: label smoothing, focal, entropy-reg, combined (6 experiments, 15ep)
  3. Fine-tune: best configs × 20 epochs (3 experiments)
  4. Reasoning module: lightweight causal reasoning blocks (2-3 experiments)
"""
import json, os, sys, time, math, copy, warnings
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
RESULTS_FILE = "hpo_14h_results.json"
TOTAL_BUDGET_S = 14 * 3600  # 14 hours in seconds
EVAL_N_1STEP = 100           # stocks for 1-step accuracy eval (reduced for speed)
EVAL_N_COLLAPSE = 100        # stocks for collapse metric eval (reduced for speed)
GEN_STEPS = 10               # AR steps for collapse metric
CONTEXT_LEN_EVAL = 128       # context length for AR evaluation
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


# ─── Custom Loss Functions ───────────────────────────────────────────────────
def focal_loss(logits, targets, gamma=2.0, label_smoothing=0.0, ignore_index=-100):
    """Focal loss: downweight easy examples, focus on hard tokens."""
    n_cls = logits.size(-1)
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1.0)
    mask = (targets != ignore_index).float()
    focal_weight = (1 - pt) ** gamma
    loss = (focal_weight * ce * mask).sum() / mask.sum().clamp(min=1)
    if label_smoothing > 0:
        smooth_loss = -F.log_softmax(logits, dim=-1).sum(dim=-1) / n_cls
        loss = (1 - label_smoothing) * loss + label_smoothing * (smooth_loss * mask).sum() / mask.sum().clamp(min=1)
    return loss


def entropy_reg_loss(logits, targets, alpha=0.3, label_smoothing=0.05, ignore_index=-100):
    """Cross-entropy + entropy maximization + label smoothing.
    Directly penalizes low-entropy (collapsed) predictions."""
    n_cls = logits.size(-1)
    mask = (targets != ignore_index).float()
    n_valid = mask.sum().clamp(min=1)

    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing)
    ce_loss = (ce * mask).sum() / n_valid

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = math.log(n_cls)
    entropy_loss = (entropy * mask).sum() / n_valid
    normalized_entropy = entropy_loss / max_entropy

    return ce_loss - alpha * normalized_entropy


def variance_weighted_loss(logits, targets, beta=0.3, ignore_index=-100):
    """Weight loss higher when target distribution has high variance (real market moves).
    Penalizes model for being equally uncertain on volatile and calm days."""
    mask = (targets != ignore_index).float()
    n_valid = mask.sum().clamp(min=1)
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)

    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        pred_entropy = -(probs * probs.clamp(1e-10).log()).sum(dim=-1)
        max_ent = math.log(logits.size(-1))
        weight = 1.0 + beta * (pred_entropy / max_ent)
        weight = weight.clamp(0.5, 3.0)

    loss = (weight * ce * mask).sum() / n_valid
    return loss


def combined_anti_collapse(logits, targets, gamma=1.5, alpha=0.2, label_smoothing=0.03, ignore_index=-100):
    """Combined: focal + entropy regularization + label smoothing."""
    n_cls = logits.size(-1)
    mask = (targets != ignore_index).float()
    n_valid = mask.sum().clamp(min=1)

    # Focal CE
    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1.0)
    focal_w = (1 - pt) ** gamma
    focal_loss = (focal_w * ce * mask).sum() / n_valid

    # Label smoothing CE
    if label_smoothing > 0:
        smooth_loss = -F.log_softmax(logits, dim=-1).sum(dim=-1) / n_cls
        focal_loss = (1 - label_smoothing) * focal_loss + label_smoothing * (smooth_loss * mask).sum() / n_valid

    # Entropy regularization
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_ent = math.log(n_cls)
    entropy_term = (entropy * mask).sum() / n_valid

    return focal_loss - alpha * (entropy_term / max_ent)


def sharpness_penalty_loss(logits, targets, label_smoothing=0.05, sharpness_penalty=0.15, ignore_index=-100):
    """Cross-entropy with label smoothing + penalty for overly sharp predictions.
    sharpness_penalty penalizes low-entropy output distributions."""
    n_cls = logits.size(-1)
    mask = (targets != ignore_index).float()
    n_valid = mask.sum().clamp(min=1)

    ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index,
                         label_smoothing=label_smoothing)
    ce_loss = (ce * mask).sum() / n_valid

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_ent = math.log(n_cls)
    sharpness_loss = (1.0 - entropy / max_ent).mean()

    return ce_loss + sharpness_penalty * sharpness_loss


# ─── Training ────────────────────────────────────────────────────────────────
def train_model(tokenizer, train_seqs, val_seqs, device, epochs, loss_type="ce",
                loss_kwargs=None, overrides=None, label=""):
    """Train model with specified loss function and hyperparameters."""
    if overrides is None:
        overrides = {}

    # Apply overrides to configs
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TrainingConfig.learning_rate,
        weight_decay=TrainingConfig.weight_decay)

    total_updates = len(train_loader) * epochs
    warmup = max(1, int(total_updates * TrainingConfig.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = {"train": [], "val": []}
    loss_kw = loss_kwargs or {}

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        optimizer.zero_grad()

        for step, (input_ids, targets, time_ids, position_ids, attn_mask) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True).unsqueeze(0)
            targets = targets.to(device, non_blocking=True).unsqueeze(0)
            time_ids = time_ids.to(device, non_blocking=True).unsqueeze(0)
            position_ids = position_ids.to(device, non_blocking=True).unsqueeze(0)
            attn_mask = attn_mask.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                if loss_type == "ce":
                    _, _, loss = model(input_ids, time_ids, position_ids, attn_mask, targets)
                else:
                    logits_coarse, _, _ = model(input_ids, time_ids, position_ids, attn_mask)
                    shift_logits = logits_coarse[:, :-1, :].contiguous()
                    shift_targets = targets.contiguous()
                    if (shift_targets == -100).all():
                        continue
                    loss_fn_map = {
                        "focal": focal_loss,
                        "entropy_reg": entropy_reg_loss,
                        "variance_weighted": variance_weighted_loss,
                        "combined_anti_collapse": combined_anti_collapse,
                        "sharpness_penalty": sharpness_penalty_loss,
                    }
                    loss_fn = loss_fn_map[loss_type]
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_targets.view(-1), **loss_kw)

            if loss is None:
                continue
            (loss / TrainingConfig.accumulation_steps).backward()
            if (step + 1) % TrainingConfig.accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            losses.append(loss.item())

        avg_train = sum(losses) / max(len(losses), 1)

        model.eval()
        vlosses = []
        with torch.no_grad():
            for input_ids, targets, time_ids, position_ids, attn_mask in val_loader:
                input_ids = input_ids.to(device).unsqueeze(0)
                targets = targets.to(device).unsqueeze(0)
                time_ids = time_ids.to(device).unsqueeze(0)
                position_ids = position_ids.to(device).unsqueeze(0)
                attn_mask = attn_mask.to(device)
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    _, _, loss = model(input_ids, time_ids, position_ids, attn_mask, targets)
                if loss is not None:
                    vlosses.append(loss.item())

        avg_val = sum(vlosses) / max(len(vlosses), 1)
        elapsed = time.time() - t0
        history["train"].append(avg_train)
        history["val"].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1}/{epochs}: train={avg_train:.4f} val={avg_val:.4f} "
                  f"best={best_val:.4f}(ep{best_epoch+1}) {elapsed:.0f}s")

    if best_state:
        model.load_state_dict(best_state)

    # Restore config overrides
    for (cls_name, k), v in saved.items():
        cfg = ModelConfig if cls_name == "ModelConfig" else TrainingConfig
        setattr(cfg, k, v)

    return model, best_val, best_epoch, history


# ─── Evaluation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_1step_acc(model, tokenizer, test_stocks, device, max_stocks=EVAL_N_1STEP):
    """1-step next-token accuracy. Fast."""
    vocab = tokenizer.bsq_coarse.vocab_size
    bos_id = vocab
    stocks = test_stocks[:max_stocks]
    total_correct, total_tokens = 0, 0

    for stock in stocks:
        feat = stock["features_raw"]
        day, month, year = stock["day"], stock["month"], stock["year"]
        cutoff = np.datetime64(pd.Timestamp(DataConfig.cutoff_date))
        ci = int(np.searchsorted(stock["dates_dt"], cutoff, side="left"))
        if ci >= len(feat) - 5:
            continue
        feat_t = feat[ci:]
        if len(feat_t) < NormConfig.min_lookback + 5:
            continue
        normed = rolling_normalize(feat_t)
        idx_c, _ = tokenizer.encode(torch.from_numpy(normed).float().unsqueeze(0).to(device))
        token_ids = idx_c[0].cpu().numpy()
        ids = [bos_id] + token_ids.tolist()
        d_l = [day[ci]] + day[ci:].tolist()
        m_l = [month[ci]] + month[ci:].tolist()
        y_l = [year[ci]] + year[ci:].tolist()
        S = len(ids)
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        tids = torch.stack([
            torch.tensor([d_l[:-1]], dtype=torch.long),
            torch.tensor([m_l[:-1]], dtype=torch.long),
            torch.tensor([y_l[:-1]], dtype=torch.long),
        ], dim=-1).to(device)
        pos = torch.arange(S - 1, device=device).unsqueeze(0)
        mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool, device=device))
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            lc, _, _ = model(inp, tids, pos, mask)
        preds = lc.argmax(dim=-1)
        total_correct += (preds == tgt).float().sum().item()
        total_tokens += tgt.shape[1]

    return total_correct / max(total_tokens, 1)


@torch.no_grad()
def eval_collapse_and_mape(model, tokenizer, test_stocks, device, n_stocks=EVAL_N_COLLAPSE):
    """Collapse metric + MAPE + DA + token diversity.

    Collapse metric = mean(|Pred_x| - |Acc_x|) across all stocks/steps.
    Negative = model predicts smaller moves than reality = more collapsed.

    Returns dict with: mape, da, collapse, pred_amplitude, acc_amplitude,
                       unique_pred_tokens, top_token_ratio, zero_collapse
    """
    vocab = tokenizer.bsq_coarse.vocab_size
    bos_id = vocab

    rng = np.random.RandomState(42)
    eval_stocks = rng.choice(test_stocks, min(n_stocks, len(test_stocks)), replace=False)

    all_mape, all_da = [], []
    pred_counts = {}
    all_pred_amp, all_acc_amp = [], []   # amplitude tracking

    for stock in eval_stocks:
        feat = stock["features_raw"]
        day, month, year = stock["day"], stock["month"], stock["year"]
        cutoff = np.datetime64(pd.Timestamp(DataConfig.cutoff_date))
        ci = int(np.searchsorted(stock["dates_dt"], cutoff, side="left"))
        if ci >= len(feat) - 15:
            continue
        feat_t, day_t, month_t, year_t = feat[ci:], day[ci:], month[ci:], year[ci:]
        if len(feat_t) < NormConfig.min_lookback + 15:
            continue

        normed = rolling_normalize(feat_t)
        idx_c, _ = tokenizer.encode(torch.from_numpy(normed).float().unsqueeze(0).to(device))
        token_ids = idx_c[0].cpu().numpy()

        split_point = min(len(token_ids) - GEN_STEPS, CONTEXT_LEN_EVAL)
        if split_point < 10:
            continue

        gt_tokens = token_ids[split_point:split_point + GEN_STEPS].tolist()
        ids = [bos_id] + token_ids[:split_point].tolist()
        d_ctx = [day_t[0]] + day_t[:split_point].tolist()
        m_ctx = [month_t[0]] + month_t[:split_point].tolist()
        y_ctx = [year_t[0]] + year_t[:split_point].tolist()

        generated = []
        for step in range(GEN_STEPS):
            S = len(ids)
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            tids = torch.stack([
                torch.tensor([d_ctx], dtype=torch.long),
                torch.tensor([m_ctx], dtype=torch.long),
                torch.tensor([y_ctx], dtype=torch.long),
            ], dim=-1).to(device)
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
            m_ctx.append(m_ctx[-1])
            y_ctx.append(y_ctx[-1])

        # Decode predictions and ground truth to feature space
        pt = torch.tensor([generated], dtype=torch.long, device=device)
        gt = torch.tensor([gt_tokens], dtype=torch.long, device=device)
        pf = tokenizer.decode_all(pt.unsqueeze(-1).expand(-1, -1, 2).contiguous())[0].cpu().numpy()
        gf = tokenizer.decode_all(gt.unsqueeze(-1).expand(-1, -1, 2).contiguous())[0].cpu().numpy()

        eps = 1e-6
        all_mape.append(np.mean(np.abs(pf - gf) / (np.abs(gf) + eps)) * 100)
        if len(pf) > 1:
            all_da.append(np.mean(np.sign(pf[:, 0]) == np.sign(gf[:, 0])))

        # Collapse metric: Pred_x - Acc_x (amplitude difference)
        pred_amp = np.mean(np.abs(pf[:, 0]))
        acc_amp = np.mean(np.abs(gf[:, 0]))
        all_pred_amp.append(pred_amp)
        all_acc_amp.append(acc_amp)

    total_preds = sum(pred_counts.values())
    top_ratio = max(pred_counts.values()) / max(total_preds, 1) if pred_counts else 0
    collapse = float(np.mean(all_pred_amp) - np.mean(all_acc_amp)) if all_pred_amp else None

    return {
        "mape": float(np.mean(all_mape)) if all_mape else None,
        "da": float(np.mean(all_da)) if all_da else None,
        "collapse": collapse,
        "pred_amplitude": float(np.mean(all_pred_amp)) if all_pred_amp else None,
        "acc_amplitude": float(np.mean(all_acc_amp)) if all_acc_amp else None,
        "unique_pred_tokens": len(pred_counts),
        "top_token_ratio": top_ratio,
        "zero_collapse": top_ratio > 0.5,
        "n_stocks": len(all_mape),
    }


# ─── Reasoning Module (Wave 4) ──────────────────────────────────────────────
class CausalReasoningBlock(nn.Module):
    """Lightweight causal reasoning: cross-attention between token predictions
    and a learned 'reasoning' memory, inspired by Universal Transformer / CoT."""
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, memory):
        # x: [B, N, D], memory: [B, M, D]
        h = self.norm1(x)
        h, _ = self.cross_attn(h, memory, memory)
        x = x + self.gate.tanh() * h
        x = x + self.ffn(self.norm2(x))
        return x


class KronosPreviewWithReasoning(nn.Module):
    """KronosPreview + CausalReasoningBlock applied after transformer blocks.
    Reasoning memory is a set of learned tokens (like Perceiver / latent tokens)."""
    def __init__(self, base_model_state=None, n_reason_tokens=8, n_reason_layers=1, dropout=0.0):
        super().__init__()
        cfg = ModelConfig
        # Copy base model components
        self.token_emb = nn.Embedding(cfg.vocab_size + 2, cfg.dim)
        self.time_emb_day = nn.Embedding(32, cfg.dim)
        self.time_emb_month = nn.Embedding(13, cfg.dim)
        self.time_emb_year = nn.Embedding(100, cfg.dim)
        from model.kronos_preview import TransformerBlock, RotaryEmbedding, RMSNorm
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.num_kv_heads, cfg.ffn_multiplier, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = RMSNorm(cfg.dim)
        self.head_coarse = nn.Linear(cfg.dim, cfg.vocab_size + 2, bias=True)
        self.head_fine = nn.Linear(cfg.dim, cfg.vocab_size + 2, bias=True)
        self.rotary = RotaryEmbedding(cfg.dim // cfg.heads, base=cfg.rope_base)

        # Reasoning module
        self.reason_tokens = nn.Parameter(torch.randn(1, n_reason_tokens, cfg.dim) * 0.02)
        self.reason_blocks = nn.ModuleList([
            CausalReasoningBlock(cfg.dim, heads=cfg.heads, dropout=dropout)
            for _ in range(n_reason_layers)
        ])
        self._gradient_checkpointing = False

        # Load base model weights if provided
        if base_model_state is not None:
            missing, unexpected = self.load_state_dict(base_model_state, strict=False)
            print(f"  Loaded base model. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, input_ids, time_ids, position_ids, attn_mask=None, targets=None):
        no_batch = input_ids.dim() == 1
        if no_batch:
            input_ids = input_ids.unsqueeze(0)
            time_ids = time_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
            if targets is not None:
                targets = targets.unsqueeze(0)

        x = self.token_emb(input_ids)
        x = x + self.time_emb_day(time_ids[..., 0])
        x = x + self.time_emb_month(time_ids[..., 1])
        x = x + self.time_emb_year(time_ids[..., 2])
        sin, cos = self.rotary(position_ids)

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, sin, cos, attn_mask, use_reentrant=False)
            else:
                x = block(x, sin, cos, attn_mask)

        # Reasoning: attend to learned reason tokens
        B = x.size(0)
        memory = self.reason_tokens.expand(B, -1, -1)
        for rblock in self.reason_blocks:
            x = rblock(x, memory)

        x = self.norm(x)
        logits_coarse = self.head_coarse(x)
        logits_fine = self.head_fine(x)

        loss = None
        if targets is not None:
            shift_logits = logits_coarse[:, :-1, :].contiguous()
            shift_targets = targets.contiguous()
            if (shift_targets != -100).any():
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1), ignore_index=-100)

        if no_batch:
            logits_coarse = logits_coarse.squeeze(0)
            logits_fine = logits_fine.squeeze(0)
        return logits_coarse, logits_fine, loss


def train_reasoning_model(base_ckpt_path, tokenizer, train_seqs, val_seqs, device, epochs,
                          n_reason_tokens=8, n_reason_layers=1, loss_type="ce",
                          loss_kwargs=None, overrides=None, label="", freeze_base=False):
    """Train model with reasoning module on top of base model."""
    if overrides is None:
        overrides = {}

    saved = {}
    for cls_name, params in overrides.items():
        cfg = ModelConfig if cls_name == "ModelConfig" else TrainingConfig
        for k, v in params.items():
            saved[(cls_name, k)] = getattr(cfg, k)
            setattr(cfg, k, v)

    # Load base model state
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    base_state = base_ckpt["model_state_dict"]

    model = KronosPreviewWithReasoning(
        base_model_state=base_state,
        n_reason_tokens=n_reason_tokens,
        n_reason_layers=n_reason_layers,
        dropout=overrides.get("ModelConfig", {}).get("dropout", ModelConfig.dropout),
    ).to(device)

    if freeze_base:
        for name, param in model.named_parameters():
            if "reason" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Frozen base. Trainable: {trainable:,} / {total:,}")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  All trainable: {trainable:,}")

    train_loader = make_dataloader(train_seqs, batch_size=TrainingConfig.batch_size, shuffle=True)
    val_loader = make_dataloader(val_seqs, batch_size=TrainingConfig.batch_size, shuffle=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=TrainingConfig.learning_rate,
                                  weight_decay=TrainingConfig.weight_decay)
    total_updates = len(train_loader) * epochs
    warmup = max(1, int(total_updates * TrainingConfig.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = {"train": [], "val": []}
    loss_kw = loss_kwargs or {}

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        optimizer.zero_grad()

        for step, (input_ids, targets, time_ids, position_ids, attn_mask) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True).unsqueeze(0)
            targets = targets.to(device, non_blocking=True).unsqueeze(0)
            time_ids = time_ids.to(device, non_blocking=True).unsqueeze(0)
            position_ids = position_ids.to(device, non_blocking=True).unsqueeze(0)
            attn_mask = attn_mask.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                if loss_type == "ce":
                    _, _, loss = model(input_ids, time_ids, position_ids, attn_mask, targets)
                else:
                    logits_coarse, _, _ = model(input_ids, time_ids, position_ids, attn_mask)
                    shift_logits = logits_coarse[:, :-1, :].contiguous()
                    shift_targets = targets.contiguous()
                    if (shift_targets == -100).all():
                        continue
                    loss_fn_map = {
                        "focal": focal_loss,
                        "entropy_reg": entropy_reg_loss,
                        "combined_anti_collapse": combined_anti_collapse,
                    }
                    loss_fn = loss_fn_map[loss_type]
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_targets.view(-1), **loss_kw)

            if loss is None:
                continue
            (loss / TrainingConfig.accumulation_steps).backward()
            if (step + 1) % TrainingConfig.accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, TrainingConfig.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            losses.append(loss.item())

        avg_train = sum(losses) / max(len(losses), 1)
        model.eval()
        vlosses = []
        with torch.no_grad():
            for input_ids, targets, time_ids, position_ids, attn_mask in val_loader:
                input_ids = input_ids.to(device).unsqueeze(0)
                targets = targets.to(device).unsqueeze(0)
                time_ids = time_ids.to(device).unsqueeze(0)
                position_ids = position_ids.to(device).unsqueeze(0)
                attn_mask = attn_mask.to(device)
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    _, _, loss = model(input_ids, time_ids, position_ids, attn_mask, targets)
                if loss is not None:
                    vlosses.append(loss.item())
        avg_val = sum(vlosses) / max(len(vlosses), 1)
        elapsed = time.time() - t0
        history["train"].append(avg_train)
        history["val"].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1}/{epochs}: train={avg_train:.4f} val={avg_val:.4f} "
                  f"best={best_val:.4f}(ep{best_epoch+1}) {elapsed:.0f}s")

    if best_state:
        model.load_state_dict(best_state)

    for (cls_name, k), v in saved.items():
        cfg = ModelConfig if cls_name == "ModelConfig" else TrainingConfig
        setattr(cfg, k, v)

    return model, best_val, best_epoch, history


# ─── Experiment Definitions ──────────────────────────────────────────────────
WAVE1_EXPERIMENTS = [
    {"name": "w1_baseline_10ep",
     "desc": "Baseline: lr=3e-4, wd=0.01, drop=0.1",
     "overrides": {}, "epochs": 10, "loss_type": "ce"},
    {"name": "w1_lr1e-4",
     "desc": "Lower LR: lr=1e-4",
     "overrides": {"TrainingConfig": {"learning_rate": 1e-4}}, "epochs": 10, "loss_type": "ce"},
    {"name": "w1_lr5e-4",
     "desc": "Higher LR: lr=5e-4",
     "overrides": {"TrainingConfig": {"learning_rate": 5e-4}}, "epochs": 10, "loss_type": "ce"},
    {"name": "w1_drop005",
     "desc": "Low dropout: 0.05",
     "overrides": {"ModelConfig": {"dropout": 0.05}}, "epochs": 10, "loss_type": "ce"},
    {"name": "w1_drop002",
     "desc": "Very low dropout: 0.02",
     "overrides": {"ModelConfig": {"dropout": 0.02}}, "epochs": 10, "loss_type": "ce"},
    {"name": "w1_wd0001",
     "desc": "Low weight decay: 0.001",
     "overrides": {"TrainingConfig": {"weight_decay": 0.001}}, "epochs": 10, "loss_type": "ce"},
    {"name": "w1_wd00001",
     "desc": "Very low weight decay: 0.0001",
     "overrides": {"TrainingConfig": {"weight_decay": 0.0001}}, "epochs": 10, "loss_type": "ce"},
]

WAVE2_EXPERIMENTS = [
    {"name": "w2_entropy_reg_a02",
     "desc": "Entropy-reg loss: alpha=0.2, label_smooth=0.05",
     "overrides": {}, "epochs": 10, "loss_type": "entropy_reg",
     "loss_kwargs": {"alpha": 0.2, "label_smoothing": 0.05}},
    {"name": "w2_entropy_reg_a04",
     "desc": "Entropy-reg loss: alpha=0.4 (stronger)",
     "overrides": {}, "epochs": 10, "loss_type": "entropy_reg",
     "loss_kwargs": {"alpha": 0.4, "label_smoothing": 0.05}},
    {"name": "w2_focal_g2",
     "desc": "Focal loss: gamma=2.0",
     "overrides": {}, "epochs": 10, "loss_type": "focal",
     "loss_kwargs": {"gamma": 2.0}},
    {"name": "w2_focal_g3",
     "desc": "Focal loss: gamma=3.0 (stronger)",
     "overrides": {}, "epochs": 10, "loss_type": "focal",
     "loss_kwargs": {"gamma": 3.0}},
    {"name": "w2_combined_ac",
     "desc": "Combined anti-collapse: focal+entropy+LS",
     "overrides": {}, "epochs": 10, "loss_type": "combined_anti_collapse",
     "loss_kwargs": {"gamma": 1.5, "alpha": 0.2, "label_smoothing": 0.03}},
    {"name": "w2_combined_ac_strong",
     "desc": "Combined anti-collapse (stronger)",
     "overrides": {}, "epochs": 10, "loss_type": "combined_anti_collapse",
     "loss_kwargs": {"gamma": 2.0, "alpha": 0.3, "label_smoothing": 0.05}},
]

WAVE2B_EXPERIMENTS = [
    {"name": "w2b_entropy_drop005",
     "desc": "Entropy-reg + low dropout",
     "overrides": {"ModelConfig": {"dropout": 0.05}}, "epochs": 10,
     "loss_type": "entropy_reg", "loss_kwargs": {"alpha": 0.2, "label_smoothing": 0.05}},
    {"name": "w2b_entropy_wd0001",
     "desc": "Entropy-reg + low WD",
     "overrides": {"TrainingConfig": {"weight_decay": 0.001}}, "epochs": 10,
     "loss_type": "entropy_reg", "loss_kwargs": {"alpha": 0.2, "label_smoothing": 0.05}},
    {"name": "w2b_focal_drop005",
     "desc": "Focal + low dropout",
     "overrides": {"ModelConfig": {"dropout": 0.05}}, "epochs": 10,
     "loss_type": "focal", "loss_kwargs": {"gamma": 2.0}},
    {"name": "w2b_var_weighted",
     "desc": "Variance-weighted loss: beta=0.3",
     "overrides": {}, "epochs": 10, "loss_type": "variance_weighted",
     "loss_kwargs": {"beta": 0.3}},
    {"name": "w2b_sharpness",
     "desc": "Sharpness penalty + label smoothing",
     "overrides": {}, "epochs": 10, "loss_type": "sharpness_penalty",
     "loss_kwargs": {"label_smoothing": 0.05, "sharpness_penalty": 0.15}},
]


def run_experiment(exp, tokenizer, train_seqs, val_seqs, test_stocks, device, elapsed_total):
    """Run a single experiment and return results."""
    name = exp["name"]
    print(f"\n{'='*70}")
    print(f"[{elapsed_total/3600:.2f}h] EXPERIMENT: {name}")
    print(f"  {exp['desc']}")
    overrides = exp.get("overrides", {})
    if overrides:
        for cls_name, params in overrides.items():
            for k, v in params.items():
                print(f"  {cls_name}.{k} = {v}")
    print(f"  Loss: {exp['loss_type']}, Epochs: {exp['epochs']}")
    print(f"{'='*70}")

    t0 = time.time()
    model, best_val, best_epoch, history = train_model(
        tokenizer, train_seqs, val_seqs, device, epochs=exp["epochs"],
        loss_type=exp["loss_type"], loss_kwargs=exp.get("loss_kwargs"),
        overrides=exp.get("overrides", {}), label=name)
    train_time = time.time() - t0

    # Save checkpoint
    save_path = f"checkpoints/{name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"dim": ModelConfig.dim, "depth": ModelConfig.depth,
                   "heads": ModelConfig.heads, "num_kv_heads": ModelConfig.num_kv_heads},
        "val_loss": best_val, "epoch": best_epoch, "experiment": name,
    }, save_path)

    # Quick eval: 1-step accuracy
    print(f"  Evaluating 1-step accuracy...")
    step1_acc = eval_1step_acc(model, tokenizer, test_stocks, device, max_stocks=EVAL_N_1STEP)

    # Quick eval: collapse metric + MAPE
    print(f"  Evaluating collapse metric...")
    collapse_eval = eval_collapse_and_mape(model, tokenizer, test_stocks, device, n_stocks=EVAL_N_COLLAPSE)

    result = {
        "name": name, "desc": exp["desc"],
        "overrides": exp.get("overrides", {}),
        "loss_type": exp["loss_type"],
        "loss_kwargs": exp.get("loss_kwargs", {}),
        "epochs": exp["epochs"],
        "best_val_loss": best_val,
        "best_epoch": best_epoch + 1,
        "mape": collapse_eval["mape"],
        "da": collapse_eval["da"],
        "collapse": collapse_eval["collapse"],
        "pred_amplitude": collapse_eval["pred_amplitude"],
        "acc_amplitude": collapse_eval["acc_amplitude"],
        "step1_accuracy": step1_acc,
        "unique_pred_tokens": collapse_eval["unique_pred_tokens"],
        "top_token_ratio": collapse_eval["top_token_ratio"],
        "zero_collapse": collapse_eval["zero_collapse"],
        "train_time_s": train_time,
        "history": history,
    }

    print(f"\n  RESULTS: val={best_val:.4f}(ep{best_epoch+1}) | "
          f"MAPE={collapse_eval['mape']:.2f}% | DA={collapse_eval['da']:.4f} | "
          f"Collapse={collapse_eval['collapse']:.4f} | "
          f"1step={step1_acc:.4f} | tokens={collapse_eval['unique_pred_tokens']} | "
          f"{'COLLAPSED' if collapse_eval['zero_collapse'] else 'OK'} | {train_time:.0f}s")

    return result


def run_reasoning_experiment(exp, base_ckpt_path, tokenizer, train_seqs, val_seqs,
                             test_stocks, device, elapsed_total):
    """Run reasoning module experiment."""
    name = exp["name"]
    print(f"\n{'='*70}")
    print(f"[{elapsed_total/3600:.2f}h] REASONING MODULE: {name}")
    print(f"  {exp['desc']}")
    print(f"  Reason tokens: {exp.get('n_reason_tokens', 8)}, "
          f"Layers: {exp.get('n_reason_layers', 1)}")
    print(f"{'='*70}")

    t0 = time.time()
    model, best_val, best_epoch, history = train_reasoning_model(
        base_ckpt_path, tokenizer, train_seqs, val_seqs, device,
        epochs=exp["epochs"],
        n_reason_tokens=exp.get("n_reason_tokens", 8),
        n_reason_layers=exp.get("n_reason_layers", 1),
        loss_type=exp.get("loss_type", "ce"),
        loss_kwargs=exp.get("loss_kwargs"),
        overrides=exp.get("overrides", {}),
        freeze_base=exp.get("freeze_base", False),
        label=name)
    train_time = time.time() - t0

    save_path = f"checkpoints/{name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "experiment": name}, save_path)

    print(f"  Evaluating 1-step accuracy...")
    step1_acc = eval_1step_acc(model, tokenizer, test_stocks, device, max_stocks=EVAL_N_1STEP)

    print(f"  Evaluating collapse metric...")
    collapse_eval = eval_collapse_and_mape(model, tokenizer, test_stocks, device, n_stocks=EVAL_N_COLLAPSE)

    result = {
        "name": name, "desc": exp["desc"],
        "loss_type": exp.get("loss_type", "ce"),
        "loss_kwargs": exp.get("loss_kwargs", {}),
        "epochs": exp["epochs"],
        "n_reason_tokens": exp.get("n_reason_tokens", 8),
        "n_reason_layers": exp.get("n_reason_layers", 1),
        "freeze_base": exp.get("freeze_base", False),
        "best_val_loss": best_val,
        "best_epoch": best_epoch + 1,
        "mape": collapse_eval["mape"],
        "da": collapse_eval["da"],
        "collapse": collapse_eval["collapse"],
        "pred_amplitude": collapse_eval["pred_amplitude"],
        "acc_amplitude": collapse_eval["acc_amplitude"],
        "step1_accuracy": step1_acc,
        "unique_pred_tokens": collapse_eval["unique_pred_tokens"],
        "top_token_ratio": collapse_eval["top_token_ratio"],
        "zero_collapse": collapse_eval["zero_collapse"],
        "train_time_s": train_time,
        "history": history,
    }

    print(f"\n  RESULTS: val={best_val:.4f}(ep{best_epoch+1}) | "
          f"MAPE={collapse_eval['mape']:.2f}% | DA={collapse_eval['da']:.4f} | "
          f"Collapse={collapse_eval['collapse']:.4f} | "
          f"1step={step1_acc:.4f} | tokens={collapse_eval['unique_pred_tokens']} | "
          f"{'COLLAPSED' if collapse_eval['zero_collapse'] else 'OK'} | {train_time:.0f}s")

    return result


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    set_global_seed(42, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Budget: {TOTAL_BUDGET_S/3600:.1f} hours")

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

    all_results = []
    global_start = time.time()

    # ── Wave 1: Traditional HPO ──────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"# WAVE 1: Traditional HPO ({len(WAVE1_EXPERIMENTS)} experiments)")
    print(f"{'#'*70}")

    for exp in WAVE1_EXPERIMENTS:
        elapsed = time.time() - global_start
        if elapsed > TOTAL_BUDGET_S * 0.48:  # use max 48% for wave 1
            print(f"\n  [!] Wave 1 time limit reached ({elapsed/3600:.2f}h). Moving to Wave 2.")
            break
        result = run_experiment(exp, tokenizer, train_seqs, val_seqs, test_stocks, device, elapsed)
        all_results.append(result)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Wave 2: Loss Experiments ─────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"# WAVE 2: Loss Function Experiments ({len(WAVE2_EXPERIMENTS)} experiments)")
    print(f"{'#'*70}")

    for exp in WAVE2_EXPERIMENTS:
        elapsed = time.time() - global_start
        if elapsed > TOTAL_BUDGET_S * 0.70:
            print(f"\n  [!] Wave 2 time limit reached ({elapsed/3600:.2f}h). Moving to Wave 2b.")
            break
        result = run_experiment(exp, tokenizer, train_seqs, val_seqs, test_stocks, device, elapsed)
        all_results.append(result)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Wave 2b: Combined experiments ────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"# WAVE 2b: Combined Loss+HPO ({len(WAVE2B_EXPERIMENTS)} experiments)")
    print(f"{'#'*70}")

    for exp in WAVE2B_EXPERIMENTS:
        elapsed = time.time() - global_start
        if elapsed > TOTAL_BUDGET_S * 0.82:
            print(f"\n  [!] Wave 2b time limit reached ({elapsed/3600:.2f}h).")
            break
        result = run_experiment(exp, tokenizer, train_seqs, val_seqs, test_stocks, device, elapsed)
        all_results.append(result)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Wave 3: Fine-tune best configs ───────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"# WAVE 3: Fine-tune Best Configs (15 epochs)")
    print(f"{'#'*70}")

    valid = [r for r in all_results if r["mape"] is not None and not r.get("zero_collapse")]
    if valid:
        def score(r):
            mape_score = r["mape"]
            collapse_score = -min(r.get("collapse", 0) or 0, 0) * 100
            return mape_score + collapse_score

        best_exps = sorted(valid, key=score)[:2]

        for i, best_r in enumerate(best_exps):
            elapsed = time.time() - global_start
            if elapsed > TOTAL_BUDGET_S * 0.90:
                break

            original = next((e for e in WAVE1_EXPERIMENTS + WAVE2_EXPERIMENTS + WAVE2B_EXPERIMENTS
                             if e["name"] == best_r["name"]), None)
            if original is None:
                continue

            fine_tune_exp = {
                "name": f"w3_ft_{original['name']}",
                "desc": f"Fine-tuned from {original['name']} (15 epochs)",
                "overrides": original.get("overrides", {}),
                "epochs": 15,
                "loss_type": original.get("loss_type", "ce"),
                "loss_kwargs": original.get("loss_kwargs"),
            }
            result = run_experiment(fine_tune_exp, tokenizer, train_seqs, val_seqs,
                                    test_stocks, device, elapsed)
            all_results.append(result)
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # ── Wave 4: Reasoning Module ─────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"# WAVE 4: Reasoning Module Experiments")
    print(f"{'#'*70}")

    # Find best base checkpoint for reasoning experiments
    best_ce = [r for r in all_results if r["loss_type"] == "ce" and not r.get("zero_collapse")]
    best_base = min(best_ce, key=lambda r: r["mape"]) if best_ce else all_results[0] if all_results else None

    if best_base:
        base_ckpt = f"checkpoints/{best_base['name']}.pt"
        print(f"  Using base checkpoint: {best_base['name']} (MAPE={best_base['mape']:.2f}%)")

        reasoning_exps = [
            {"name": "w4_reason_frozen",
             "desc": "Reasoning (frozen base, 8 tokens, 1 layer)",
             "epochs": 10, "n_reason_tokens": 8, "n_reason_layers": 1,
             "freeze_base": True, "loss_type": "ce"},
            {"name": "w4_reason_trainable",
             "desc": "Reasoning (trainable, 8 tokens, 1 layer)",
             "epochs": 10, "n_reason_tokens": 8, "n_reason_layers": 1,
             "freeze_base": False, "loss_type": "ce"},
        ]

        for exp in reasoning_exps:
            elapsed = time.time() - global_start
            if elapsed > TOTAL_BUDGET_S * 0.97:
                print(f"\n  [!] Time limit reached. Skipping remaining experiments.")
                break
            result = run_reasoning_experiment(exp, base_ckpt, tokenizer, train_seqs,
                                              val_seqs, test_stocks, device, elapsed)
            all_results.append(result)
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - global_start
    print(f"\n{'='*80}")
    print(f"14-HOUR HPO COMPLETE — Total time: {total_time/3600:.2f} hours")
    print(f"{'='*80}")

    print(f"\n{'Experiment':<35} | {'Val':>7} | {'MAPE':>9} | {'DA':>6} | {'Collapse':>9} | "
          f"{'1step':>6} | {'Tok':>4} | {'Status':>8}")
    print("-" * 105)
    for r in all_results:
        mape_str = f"{r['mape']:.1f}%" if r.get('mape') else "N/A"
        da_str = f"{r['da']:.3f}" if r.get('da') else "N/A"
        collapse_str = f"{r['collapse']:.4f}" if r.get('collapse') is not None else "N/A"
        status = "COLLAPSED" if r.get('zero_collapse') else "OK"
        print(f"{r['name']:<35} | {r['best_val_loss']:>7.4f} | {mape_str:>9} | {da_str:>6} | "
              f"{collapse_str:>9} | {r['step1_accuracy']:>6.4f} | {r['unique_pred_tokens']:>4} | {status:>8}")

    valid = [r for r in all_results if r.get("mape") is not None and not r.get("zero_collapse")]
    if valid:
        best = min(valid, key=lambda r: r["mape"])
        least_collapsed = max(valid, key=lambda r: r.get("collapse", -999))
        print(f"\nBest MAPE: {best['name']} (MAPE={best['mape']:.2f}%, Collapse={best.get('collapse', 'N/A')})")
        print(f"Least Collapsed: {least_collapsed['name']} (Collapse={least_collapsed.get('collapse', 'N/A'):.4f}, "
              f"MAPE={least_collapsed['mape']:.2f}%)")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"total_time_h": total_time/3600, "results": all_results}, f, indent=2, default=str)

    print(f"\nAll results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
