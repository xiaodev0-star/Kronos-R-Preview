"""Branch E (DPO PostTrain): train pi from pi_ref on offline preference pairs.

Reuses ``train_base.py`` building blocks via import (no edits to train_base):
``_pad_batch_causal``, ``EarlyStopping``, ``build_wsd_scheduler``,
``_atomic_torch_save``, and the train_base checkpoint format.  pi initializes
from the Branch A checkpoint (pi_ref == pi_init, the standard DPO implicit-KL
anchor).  A single AdamW (lr <= 1e-5) is used (no Muon).

Because the pairs file precomputes pi_ref's log-probabilities, every optimizer
step needs only ONE forward pass: the policy's, via the new differentiable
``forward_selected_trainable`` (selected-position gather, coarse logits only).

DPO loss (from_logits, log-sigmoid, logit-difference clamp +-50):
    x = beta * ((log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l)))
    loss = -log_sigmoid(x)
plus a lambda_ce * CE anchor at the same positions' true next-day token.
Per-stock equal weighting: each stock in a batch contributes the MEAN of its
pair losses, and the batch sum is accumulated over ``accum`` sequences before
one optimizer step.

Snapshots are written every 0.5 epoch as ``{save_path}_snap{index:02d}.pt`` and
indexed in ``model_checkpoints.json`` (integer epochs 1..N) so the Branch E
driver can run the 400-window red-line evaluation per snapshot.

Usage:
    python experiments/05-cpt/e_train_dpo.py \
        --init-ckpt checkpoints/branchA_dm030_8ceb_ep5.pt \
        --tokenizer checkpoints/tokenizer_v2_ohlc.pt \
        --pairs <pairs_s43.npz> --beta 0.1 --lr 1e-5 --epochs 6 \
        --controlled-loader-seed 43 --tag branchE_b0.1_s43 \
        --save-path checkpoints/branchE_b0.1_s43.pt \
        --metrics-dir server_runs/results/04b-cpt/seed42/trials/branchE_b0.1/s43
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "experiments" / "04" / "b-hpo"))

from config import ModelConfig, set_global_seed  # noqa: E402
from data_processor import load_stocks, pack_stocks_v2, split_stocks  # noqa: E402
from eval_helpers import load_gpt, load_tokenizer  # noqa: E402
from train_base import (  # noqa: E402
    EarlyStopping,
    _atomic_torch_save,
    _pad_batch_causal,
    _to_device,
    build_wsd_scheduler,
)


# ============================================================================
# Data structures
# ============================================================================

class PairIndex:
    """symbol -> list of pair records, loaded from the pairs npz."""

    def __init__(self, pairs_path: Path):
        data = np.load(str(pairs_path), allow_pickle=True)
        self.schema = int(data["schema"][0]) if "schema" in data else 1
        self.temperature = float(
            data["temperature"][0]
        ) if "temperature" in data else 1.0
        symbols = data["symbols"]
        positions = data["position"]
        chosen = data["chosen_id"]
        rejected = data["rejected_id"]
        refw = data["ref_logp_chosen"]
        refl = data["ref_logp_rejected"]
        true_id = data["true_id"]
        by_symbol: dict[str, list[tuple]] = {}
        for i in range(len(symbols)):
            sym = str(symbols[i])
            rec = (
                int(positions[i]),
                int(chosen[i]),
                int(rejected[i]),
                float(refw[i]),
                float(refl[i]),
                int(true_id[i]),
            )
            by_symbol.setdefault(sym, []).append(rec)
        self.by_symbol = by_symbol
        self.n_pairs = len(symbols)
        data.close()

    def has(self, symbol: str) -> bool:
        return symbol in self.by_symbol

    def get(self, symbol: str):
        return self.by_symbol.get(symbol, ())


class DPOLoader:
    """Yields ``(batch_9tuple, symbols, grp)`` matching TokenBudgetLoader order.

    Replicates ``train_base.TokenBudgetLoader``'s grouping (seed/epoch-addressable
    shuffling, band-sorted groups under a token budget) so the DPO run sees the
    same batching as production training, but also exposes each batch's stock
    symbols so pairs can be gathered per sequence.
    """

    def __init__(self, sequences, max_tokens, cap_B=64, shuffle=True,
                 loader_seed=42, band=64):
        self.sequences = sequences
        self.max_tokens = int(max_tokens)
        self.cap_B = int(cap_B)
        self.shuffle = shuffle
        self.loader_seed = int(loader_seed)
        self.band = int(band)
        self._epoch = 0
        self.last_iteration_stats = {}

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def _generator(self, stream):
        generator = torch.Generator()
        seed = (
            int(self.loader_seed) + 1_000_003 * int(self._epoch) + int(stream)
        ) % (2 ** 63 - 1)
        generator.manual_seed(seed)
        return generator

    def _build_groups(self):
        idx = list(range(len(self.sequences)))
        if self.shuffle:
            g = self._generator(0)
            perm = torch.randperm(len(idx), generator=g).tolist()
            idx = [idx[p] for p in perm]
            idx.sort(key=lambda i: self.sequences[i]["input_ids"].shape[0] // self.band)
        else:
            idx.sort(key=lambda i: self.sequences[i]["input_ids"].shape[0])
        groups, cur, cur_max = [], [], 0
        for i in idx:
            L = self.sequences[i]["input_ids"].shape[0]
            new_max = max(cur_max, L)
            if cur and ((len(cur) + 1) * new_max > self.max_tokens
                        or len(cur) + 1 > self.cap_B):
                groups.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            groups.append(cur)
        return groups

    def _reset_stats(self):
        self.last_iteration_stats = {
            "microbatches": 0,
            "sequences": 0,
            "real_tokens": 0,
            "padded_tokens": 0,
        }

    def __iter__(self):
        self._reset_stats()
        groups = self._build_groups()
        if self.shuffle:
            g = self._generator(1)
            perm = torch.randperm(len(groups), generator=g).tolist()
            groups = [groups[p] for p in perm]
        for grp in groups:
            seqs = [self.sequences[i] for i in grp]
            symbols = [str(s["symbol"]) for s in seqs]
            batch = _pad_batch_causal(seqs)
            B = len(seqs)
            stats = self.last_iteration_stats
            stats["microbatches"] += 1
            stats["sequences"] += B
            stats["real_tokens"] += sum(
                int(s["input_ids"].shape[0]) for s in seqs
            )
            stats["padded_tokens"] += B * max(
                int(s["input_ids"].shape[0]) for s in seqs
            )
            yield batch, symbols, grp

    def __len__(self):
        return len(self._build_groups())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-ckpt", type=Path, required=True,
                        help="pi_ref checkpoint; pi initializes from these weights.")
    parser.add_argument("--tokenizer", type=Path,
                        default=ROOT / "checkpoints" / "tokenizer_v2_ohlc.pt")
    parser.add_argument("--pairs", type=Path, required=True,
                        help="Pairs npz produced by e_build_pairs.py")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Single AdamW LR (<= 1e-5 per Branch E design).")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--accum", type=int, default=32,
                        help="Sequences per optimizer step (accumulation).")
    parser.add_argument("--batch-tokens", type=int, default=6144)
    parser.add_argument("--batch-cap", type=int, default=64)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--controlled-loader-seed", type=int, default=42)
    parser.add_argument("--ce-weight", type=float, default=0.1,
                        help="lambda_ce: CE-anchor weight at pair positions.")
    parser.add_argument("--dm-weight", type=float, default=0.0,
                        help="Branch A distribution-match weight. NOTE: the "
                             "selective forward cannot compute the full-sequence "
                             "histogram term; values > 0 are not supported.")
    parser.add_argument("--val-holdout-ratio", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--snapshot-every", type=float, default=0.5,
                        help="Write a snapshot every N epochs (half-epoch default).")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--tag", type=str, default="branchE")
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="Debug: limit train stocks (0 = all).")
    parser.add_argument("--token-cache-dir", type=Path, default=None,
                        help="pack_stocks_v2 token cache dir; defaults to "
                             "checkpoints/token_cache_e.")
    return parser.parse_args()


def collect_pairs_for_batch(
    symbols: list[str],
    pair_index: PairIndex,
    max_len_b: int,
    device: torch.device,
):
    """Return aligned pair tensors for a batch of stocks.

    Sequence position ``p`` maps directly to batch column ``p`` (untruncated
    right-padded _pad_batch_causal, position_ids = arange).  ``rows`` are row
    indices into the batch (possibly repeated), ``ppos`` the column positions.
    """
    rows: list[int] = []
    ppos: list[int] = []
    chosen: list[int] = []
    rejected: list[int] = []
    refw: list[float] = []
    refl: list[float] = []
    true_id: list[int] = []
    for k, sym in enumerate(symbols):
        recs = pair_index.get(sym)
        if not recs:
            continue
        for (p, ch, rej, rw, rl, tid) in recs:
            if p >= max_len_b:
                continue
            rows.append(k)
            ppos.append(p)
            chosen.append(ch)
            rejected.append(rej)
            refw.append(rw)
            refl.append(rl)
            true_id.append(tid)
    if not rows:
        return None
    return {
        "rows": torch.as_tensor(rows, dtype=torch.long, device=device),
        "ppos": torch.as_tensor(ppos, dtype=torch.long, device=device),
        "chosen": torch.as_tensor(chosen, dtype=torch.long, device=device),
        "rejected": torch.as_tensor(rejected, dtype=torch.long, device=device),
        "refw": torch.as_tensor(refw, dtype=torch.float32, device=device),
        "refl": torch.as_tensor(refl, dtype=torch.float32, device=device),
        "true_id": torch.as_tensor(true_id, dtype=torch.long, device=device),
    }


def dpo_pair_loss(
    lps: torch.Tensor,
    pairs: dict,
    beta: float,
):
    """Per-pair DPO loss + CE anchor from temperature-scaled log-probs."""
    pi_w = lps.gather(1, pairs["chosen"].unsqueeze(1)).squeeze(1)
    pi_l = lps.gather(1, pairs["rejected"].unsqueeze(1)).squeeze(1)
    x = beta * ((pi_w - pairs["refw"]) - (pi_l - pairs["refl"]))
    x = x.clamp(-50.0, 50.0)
    dpo = -F.logsigmoid(x)
    ce = -(lps.gather(1, pairs["true_id"].unsqueeze(1)).squeeze(1))
    return dpo, ce


@torch.no_grad()
def evaluate_holdout(
    model,
    holdout_seqs,
    pair_index: PairIndex,
    vocab: int,
    temperature: float,
    beta: float,
    device: torch.device,
    batch_tokens: int = 6144,
    cap_B: int = 64,
):
    """Mean DPO loss + pair accuracy on held-out train stocks (no grad)."""
    model.eval()
    loader = DPOLoader(holdout_seqs, batch_tokens, cap_B=cap_B,
                       shuffle=False, loader_seed=0)
    losses = []
    accs = []
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for batch, symbols, _grp in loader:
            _unpack = _to_device(batch, device)
            inp, _tgt, _ftgt, tids, pos, _mask, va, _rt, _sw = _unpack[:9]
            # Branch F extended _to_device to a 10-tuple (regime_ids last); DPO
            # has no regimes, so drop it.
            _ = _unpack[9] if len(_unpack) > 9 else None
            max_len_b = inp.shape[1]
            pairs = collect_pairs_for_batch(symbols, pair_index, max_len_b, device)
            if pairs is None:
                continue
            coarse, _fine = model.forward_selected_trainable(
                inp, tids, pos, pairs["rows"], pairs["ppos"], va_values=va
            )
            lps = torch.log_softmax(
                coarse[:, :vocab].float() / temperature, dim=-1
            )
            dpo, _ce = dpo_pair_loss(lps, pairs, beta)
            losses.append(dpo.mean().item())
            pi_w = lps.gather(1, pairs["chosen"].unsqueeze(1)).squeeze(1)
            pi_l = lps.gather(1, pairs["rejected"].unsqueeze(1)).squeeze(1)
            accs.append((pi_w > pi_l).float().mean().item())
    model.eval()
    if not losses:
        return float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(accs))


def save_snapshot(model, epoch: float, step: int, args, save_path: Path,
                  history_row: dict) -> None:
    """Write a train_base-compatible inference snapshot."""
    sd = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    payload = {
        "model_state_dict": sd,
        "config": {
            "dim": ModelConfig.dim,
            "depth": ModelConfig.depth,
            "heads": ModelConfig.heads,
            "num_kv_heads": ModelConfig.num_kv_heads,
            "ffn_multiplier": ModelConfig.ffn_multiplier,
            "vocab_size": ModelConfig.vocab_size,
            "vocab_fine": ModelConfig.vocab_fine,
        },
        "val_loss": float(history_row.get("val_dpo", float("nan"))),
        "epoch": epoch,
        "tag": args.tag,
        "beta": args.beta,
        "lr": args.lr,
        "branch": "E",
    }
    _atomic_torch_save(payload, str(save_path))


def write_checkpoint_index(args, entries: list[dict]) -> None:
    """Write model_checkpoints.json in the evaluate_epoch_trajectory format.

    Every entry is normalized with the fields the trajectory evaluator reads
    from the index (train_loss / val_loss / learning_rate / ...).  DPO has no
    canonical token val_loss, so placeholders are recorded; the actual red-line
    metrics come from the per-snapshot 400-window evaluation, not the index.
    """
    normalized = []
    for e in sorted(entries, key=lambda x: int(x["epoch"])):
        train_loss = float(e.get("train_loss", 3.0))
        val_loss = float(e.get("val_loss", 3.6))
        normalized.append({
            "epoch": int(e["epoch"]),
            "path": str(e["path"]),
            "size_bytes": int(e.get("size_bytes", Path(e["path"]).stat().st_size)),
            "train_loss": train_loss,
            "train_coarse_loss": train_loss,
            "train_fine_loss": float(e.get("train_fine_loss", 3.0)),
            "train_het_loss": float(e.get("train_het_loss", 0.1)),
            "val_loss": val_loss,
            "val_coarse_loss": val_loss,
            "val_fine_loss": float(e.get("val_fine_loss", 3.6)),
            "val_het_loss": float(e.get("val_het_loss", 0.1)),
            "learning_rate": float(e.get("learning_rate", 0.0)),
            "learning_rate_adam": float(e.get("learning_rate_adam", 0.0)),
            "optimizer_steps_this_epoch": int(e.get("optimizer_steps_this_epoch", 0)),
            "global_step": int(e.get("global_step", 0)),
            "best_so_far": bool(e.get("best_so_far", False)),
        })
    index = {
        "tag": args.tag,
        "save_path": str(args.save_path.resolve()),
        "updated_epoch": max((e["epoch"] for e in normalized), default=0),
        "checkpoints": normalized,
    }
    metrics_dir = args.metrics_dir or (args.save_path.parent)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "model_checkpoints.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.dm_weight > 0:
        raise NotImplementedError(
            "Branch E's selective forward cannot compute the full-sequence "
            "distribution-match term; keep --dm-weight 0 (the red-line collapse "
            "gate covers the same failure mode)."
        )
    if args.lr > 1e-5:
        print(f"  [warn] lr={args.lr} exceeds the Branch E <= 1e-5 bound", flush=True)
    set_global_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(ROOT)
    print(f"Device: {device}, tag={args.tag}", flush=True)
    print(f"  init={args.init_ckpt}, pairs={args.pairs}, beta={args.beta}, "
          f"lr={args.lr}", flush=True)

    tokenizer = load_tokenizer(str(args.tokenizer), device)
    ModelConfig.vocab_size = tokenizer.vocab_coarse
    ModelConfig.vocab_fine = tokenizer.bsq_fine.vocab_size
    vocab = int(tokenizer.vocab_coarse)

    pair_index = PairIndex(args.pairs)
    temperature = pair_index.temperature
    print(f"  pairs={pair_index.n_pairs}, temperature={temperature}", flush=True)

    # pi = pi_ref weights (implicit KL anchor).  The policy forward runs in EVAL
    # mode (dropout off): with identical weights pi's log-probs equal the stored
    # pi_ref log-probs at init, so the DPO loss starts at exactly log 2 with zero
    # gradient (no spurious signal from dropout noise), and the implicit KL anchor
    # stays well-defined.  ``forward_selected_trainable`` still backprops through
    # the backbone (dropout is identity in eval mode, so gradients are exact).
    model = load_gpt(str(args.init_ckpt), device, tokenizer=tokenizer)
    model.eval()

    stocks = load_stocks(max_stocks=args.max_stocks)
    train_s, _val_s, _test_s = split_stocks(stocks)
    if args.max_stocks > 0:
        train_s = train_s[: args.max_stocks]

    token_cache_dir = args.token_cache_dir or (
        ROOT / "checkpoints" / "token_cache_e"
    )
    train_seqs = pack_stocks_v2(
        train_s, tokenizer, mode="train", cache_dir=str(token_cache_dir)
    )
    # Keep only sequences that actually carry DPO pairs.
    train_seqs = [
        s for s in train_seqs if pair_index.has(str(s["symbol"]))
    ]
    if not train_seqs:
        raise RuntimeError("No train sequences carry DPO pairs; check pair symbols.")
    print(f"  train seqs with pairs: {len(train_seqs)}", flush=True)

    # Holdout split (5% of pair-bearing train stocks, seeded).
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(train_seqs)).tolist()
    n_hold = max(1, int(len(train_seqs) * args.val_holdout_ratio))
    hold_idx = set(order[:n_hold])
    hold_seqs = [train_seqs[i] for i in order[:n_hold]]
    train_sub = [train_seqs[i] for i in order[n_hold:]]
    print(f"  train={len(train_sub)} holdout={len(hold_seqs)}", flush=True)

    loader = DPOLoader(
        train_sub, args.batch_tokens, cap_B=args.batch_cap, shuffle=True,
        loader_seed=args.controlled_loader_seed,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, int(np.ceil(len(train_sub) / max(args.accum, 1))))
    total_updates = max(steps_per_epoch * args.epochs, 1)
    scheduler = build_wsd_scheduler(
        optimizer, total_updates, warmup_ratio=args.warmup_ratio
    )
    early_stop = EarlyStopping(patience=args.early_stop_patience, mode="min")

    metrics_dir = args.metrics_dir or (args.save_path.parent)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "dpo_loss": [],
        "ce_loss": [],
        "val_dpo": [],
        "val_pair_acc": [],
        "lr": [],
        "global_step": [],
        "elapsed_s": [],
    }
    global_step = 0
    saved_snapshot_half = 0
    snapshots_index: list[dict] = []
    cumulative_seqs = 0
    t0 = time.time()

    try:
        for epoch in range(args.epochs):
            loader.set_epoch(epoch)
            loss_acc = 0.0
            ce_acc = 0.0
            n_loss_batches = 0
            seqs_in_accum = 0
            optimizer.zero_grad(set_to_none=True)

            for batch, symbols, grp in loader:
                _unpack = _to_device(batch, device)
                inp, _tgt, _ftgt, tids, pos, _mask, va, _rt, _sw = _unpack[:9]
                # Branch F extended _to_device to a 10-tuple (regime_ids last).
                _ = _unpack[9] if len(_unpack) > 9 else None
                max_len_b = inp.shape[1]
                pairs = collect_pairs_for_batch(
                    symbols, pair_index, max_len_b, device
                )
                batch_count = len(symbols)
                if pairs is not None:
                    with torch.amp.autocast(
                        "cuda", enabled=device.type == "cuda"
                    ):
                        coarse, _fine = model.forward_selected_trainable(
                            inp, tids, pos, pairs["rows"], pairs["ppos"],
                            va_values=va,
                        )
                        lps = torch.log_softmax(
                            coarse[:, :vocab].float() / temperature, dim=-1
                        )
                        dpo, ce = dpo_pair_loss(lps, pairs, args.beta)
                        B = inp.shape[0]
                        row_sums = torch.zeros(B, device=device).index_add_(
                            0, pairs["rows"], dpo
                        )
                        row_counts = torch.zeros(B, device=device).index_add_(
                            0, pairs["rows"], torch.ones_like(dpo)
                        )
                        per_seq = row_sums / row_counts.clamp(min=1.0)
                        per_seq_sum = per_seq.sum()
                        ce_scalar = ce.mean()
                        loss = per_seq_sum / max(args.accum, 1) \
                            + args.ce_weight * ce_scalar
                    loss.backward()
                    loss_acc += float(per_seq_sum.detach())
                    ce_acc += float(ce_scalar.detach())
                    n_loss_batches += 1

                seqs_in_accum += batch_count
                cumulative_seqs += batch_count
                if seqs_in_accum >= args.accum:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    seqs_in_accum = 0

                # Half-epoch snapshots based on cumulative progress.
                progress_epochs = cumulative_seqs / max(len(train_sub), 1)
                target_snapshot_half = int(progress_epochs / args.snapshot_every)
                while saved_snapshot_half < target_snapshot_half:
                    saved_snapshot_half += 1
                    snap_epoch = saved_snapshot_half * args.snapshot_every
                    snap_path = args.save_path.parent / (
                        f"{args.save_path.stem}_snap{saved_snapshot_half:02d}.pt"
                    )
                    save_snapshot(
                        model, snap_epoch, global_step, args, snap_path,
                        {"val_dpo": float("nan")},
                    )
                    snapshots_index.append({
                        "epoch": saved_snapshot_half,
                        "fractional_epoch": snap_epoch,
                        "path": str(snap_path.resolve()),
                        "size_bytes": snap_path.stat().st_size,
                        "global_step": global_step,
                        "tag": args.tag,
                        "beta": args.beta,
                    })
                    print(
                        f"  [snapshot] half={saved_snapshot_half} "
                        f"(ep {snap_epoch:.1f}) -> {snap_path}",
                        flush=True,
                    )

            if seqs_in_accum > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            val_dpo, val_acc = evaluate_holdout(
                model, hold_seqs, pair_index, vocab, temperature, args.beta,
                device, batch_tokens=args.batch_tokens, cap_B=args.batch_cap,
            )
            avg_dpo = loss_acc / max(n_loss_batches, 1)
            avg_ce = ce_acc / max(n_loss_batches, 1)
            history["epoch"].append(epoch + 1)
            history["dpo_loss"].append(avg_dpo)
            history["ce_loss"].append(avg_ce)
            history["val_dpo"].append(val_dpo)
            history["val_pair_acc"].append(val_acc)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["global_step"].append(global_step)
            history["elapsed_s"].append(time.time() - t0)
            print(
                f"[{args.tag}] Epoch {epoch+1}: train_dpo={avg_dpo:.4f} "
                f"ce={avg_ce:.4f} val_dpo={val_dpo:.4f} "
                f"val_pair_acc={val_acc:.3f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"step={global_step}",
                flush=True,
            )
            if early_stop(val_dpo if np.isfinite(val_dpo) else float("inf"), epoch):
                print(f"  [early_stop] patience exhausted at epoch {epoch+1}", flush=True)
                break

        # Final snapshot at the last reached boundary (if any pending).
        progress_epochs = cumulative_seqs / max(len(train_sub), 1)
        final_target_half = int(progress_epochs / args.snapshot_every)
        while saved_snapshot_half < final_target_half:
            saved_snapshot_half += 1
            snap_epoch = saved_snapshot_half * args.snapshot_every
            snap_path = args.save_path.parent / (
                f"{args.save_path.stem}_snap{saved_snapshot_half:02d}.pt"
            )
            save_snapshot(model, snap_epoch, global_step, args, snap_path,
                          {"val_dpo": float("nan")})
            snapshots_index.append({
                "epoch": saved_snapshot_half,
                "fractional_epoch": snap_epoch,
                "path": str(snap_path.resolve()),
                "size_bytes": snap_path.stat().st_size,
                "global_step": global_step,
                "tag": args.tag,
                "beta": args.beta,
            })

    except KeyboardInterrupt:
        print("  [interrupt] saving snapshots index and history so far", flush=True)

    write_checkpoint_index(args, snapshots_index)

    history_path = metrics_dir / f"history_{args.tag}.json"
    temporary = history_path.with_suffix(history_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    os.replace(temporary, history_path)
    print(f"Done. snapshots={len(snapshots_index)}, history={history_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
