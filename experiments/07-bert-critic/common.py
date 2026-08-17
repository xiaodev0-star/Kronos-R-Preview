"""Exp 07 BERT-Critic shared infrastructure (self-contained, no 06 imports).

Merged from: improve_common, score_bert, build_gpt_candidates, eval_critic.
All functions live here so no cross-module imports among 07 scripts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import sys
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import IterableDataset

# ============================================================================
# Paths
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
for _p in (EXP_DIR, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_io import StudyLayout, default_study_roots, file_sha256  # noqa: E402
from config import DataConfig, NormConfig  # noqa: E402
from data_processor import split_stocks, document_normalize, _stock_cutoff_idx  # noqa: E402
from eval_helpers import load_gpt  # noqa: E402
from model import load_tokenizer  # noqa: E402

# ============================================================================
# Constants
# ============================================================================
EXP_KEY = "07-bert-critic"
CUTOFF_DATE = "2024-02-01"
CALIB_START, CALIB_STOP = "2023-02-01", "2024-02-01"
HOLDOUT_OFFSET = 400
HOLDOUT_DAYS = 80
VALIDATION_OFFSETS = tuple(range(0, HOLDOUT_OFFSET, 1))
VOCAB_BASE = 128
MASK_ID = VOCAB_BASE + 2

CENTERS_PATH = ROOT / "checkpoints" / "coarse_logret_centers.npy"
CPT_SELECTION_PATH = ROOT / "server_runs" / "results" / "04b-cpt" / "seed42" / "trials" / "selection.json"

TC = 1.4          # locked calibrated coarse temperature
TF = 1.1
EPS = 1e-12


_EXP06_COMMON = None


def posttrain_artifacts(seed=42):
    """Load Exp 06's canonical artifact map without importing it as ``common``."""
    global _EXP06_COMMON
    if _EXP06_COMMON is None:
        path = ROOT / "experiments" / "06-posttrain" / "common.py"
        spec = importlib.util.spec_from_file_location("kronos_exp06_common", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load Exp 06 helpers from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXP06_COMMON = module
    return _EXP06_COMMON.artifact_paths(seed=seed)


def posttrain_common():
    """Return the loaded Exp 06 helper module for shared data preparation."""
    posttrain_artifacts()
    return _EXP06_COMMON


def _pt06_path(key, seed=42):
    return posttrain_artifacts(seed)[key]


PT01_REC = _pt06_path("records")
EVAL_HIDDEN = _pt06_path("hidden")
P6_HEAD = _pt06_path("head_rank_mlp_spearman")


# ============================================================================
# IO utilities
# ============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_roots(seed: int = 42) -> StudyLayout:
    weights, results = default_study_roots(EXP_KEY, seed=seed)
    return StudyLayout.create(weights, results)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


PREDICTION_FIELD_TYPES = {
    "offset": "int16",
    "true_coarse_id": "int16",
    "gpt_top1_id": "int16",
    "bert_top1_id": "int16",
    "date_key": "string",
    "stock_uid": "string",
    "quality": "bool",
}


class PredictionParquetWriter:
    """Stream model/date/stock prediction rows into one typed Parquet table."""

    def __init__(self, path, fields, *, include_model=True):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.fields = tuple(fields)
        self.include_model = bool(include_model)
        self._pa = pa
        self._pq = pq
        self._writer = None
        self.rows = 0

    def _type_for(self, name):
        pa = self._pa
        kind = PREDICTION_FIELD_TYPES.get(name, "float32")
        return {
            "int16": pa.int16(),
            "string": pa.string(),
            "bool": pa.bool_(),
            "float32": pa.float32(),
        }[kind]

    def write(self, records, *, model=None):
        pa = self._pa
        arrays = {}
        n_rows = None
        for name in self.fields:
            if name in records and records[name] is not None:
                arr = np.asarray(records[name])
                if arr.ndim != 1:
                    raise ValueError(f"prediction field {name!r} is not one-dimensional")
                if n_rows is None:
                    n_rows = len(arr)
                elif len(arr) != n_rows:
                    raise ValueError(
                        f"prediction field {name!r} has {len(arr)} rows; "
                        f"expected {n_rows}"
                    )
        if n_rows is None:
            raise ValueError("no one-dimensional prediction fields to write")

        if self.include_model:
            if model is None:
                raise ValueError("model name is required for the combined prediction table")
            arrays["model"] = pa.array(
                np.full(n_rows, str(model), dtype=object), type=pa.string()
            )
        for name in self.fields:
            dtype = self._type_for(name)
            if name not in records or records[name] is None:
                arrays[name] = pa.nulls(n_rows, type=dtype)
                continue
            arr = np.asarray(records[name])
            arrays[name] = pa.array(arr, type=dtype)

        table = pa.table(arrays)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = self._pq.ParquetWriter(
                self.path, table.schema, compression="zstd"
            )
        elif table.schema != self._writer.schema:
            table = table.cast(self._writer.schema)
        self._writer.write_table(table)
        self.rows += n_rows

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def write_prediction_parquet(path, records, fields, *, model=None):
    """Write one prediction chunk; optionally add its formal model name."""
    with PredictionParquetWriter(path, fields, include_model=model is not None) as writer:
        writer.write(records, model=model)
    return Path(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dict_sha256(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_trial(entry):
    roots = resolve_roots()
    ledger_path = roots.results_root / "run_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entry.setdefault("timestamp", utc_now())
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================================
# Selection / upstream
# ============================================================================

class SelectionError(RuntimeError):
    pass


def load_reviewed_selection(path=CPT_SELECTION_PATH):
    if not Path(path).exists():
        raise SelectionError(f"Selection not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        sel = json.load(f)
    if not (sel.get("upstream_eligible") or sel.get("upstream", {}).get("upstream_eligible")):
        raise SelectionError("Selection not marked upstream_eligible")
    return sel


def upstream_checkpoint_path(sel):
    up = sel.get("upstream") or sel.get("parent_selection", {})
    ckpt = ROOT / Path(up["checkpoint"])
    if not ckpt.exists():
        raise FileNotFoundError(f"Upstream checkpoint missing: {ckpt}")
    return ckpt


def upstream_paths(selection_path=CPT_SELECTION_PATH):
    sel = load_reviewed_selection(selection_path)
    up = sel.get("upstream") or sel.get("parent_selection", {})
    ckpt = ROOT / Path(up["checkpoint"])
    tok = ROOT / Path(up["tokenizer"])
    return ckpt, tok


def roots(seed=42):
    return resolve_roots(seed=seed)


def weights_root(seed=42):
    return roots(seed).weights_root


def results_root(seed=42):
    return roots(seed).results_root


STAGE_DIRS = {"A": "A-train", "B": "B-score", "C": "C-fusion"}

# The three A-stage checkpoints are the formal model variants for Exp 07.
# Every variant must be scored on the same 400 pre-holdout offsets.
MODEL_VARIANTS = (
    {"name": "BERT", "artifact": "bert-base", "suffix": "BERT"},
    {"name": "BERT-FT", "artifact": "bert-ft", "suffix": "BERT-FT"},
    {"name": "BERT-PPS", "artifact": "bert-pps", "suffix": "BERT-PPS"},
)


def stage_weights(stage, seed=42):
    return weights_root(seed) / STAGE_DIRS[stage.upper()]


def stage_results(stage, seed=42):
    path = results_root(seed) / STAGE_DIRS[stage.upper()]
    path.mkdir(parents=True, exist_ok=True)
    return path


def weights_artifact(name, seed=42, **parts):
    """Canonical Exp 07 weight/cache path for a named artifact."""
    root = weights_root(seed)
    mapping = {
        "bert-base": stage_weights("A", seed) / "BERT.pt",
        "bert-ft": stage_weights("A", seed) / "BERT-FT.pt",
        "bert-pps": stage_weights("A", seed) / "BERT-PPS.pt",
        "gpt-proposals": stage_weights("A", seed) / "gpt-proposals-ep100.pt",
        "bert-index": stage_weights("B", seed) / "input-index.pkl",
        "hidden-fit": stage_weights("B", seed) / "hidden-fit-w512.npz",
        "hidden-calib": stage_weights("B", seed) / "hidden-calib-w512.npz",
        "hidden-eval": stage_weights("B", seed) / "hidden-eval-w512.npz",
        "candidates-fit": stage_weights("B", seed) / "candidates-calib-k8.npz",
        "candidates-calib": stage_weights("B", seed) / "candidates-calib-k8.npz",
        "candidates-eval": stage_weights("B", seed) / "candidates-eval-k8.npz",
        "scores-fit": stage_weights("B", seed) / "scores-calib-k8-w512-stride1.npz",
        "scores-calib": stage_weights("B", seed) / "scores-calib-k8-w512-stride1.npz",
        "scores-eval": stage_weights("B", seed) / "scores-eval-k8-w512-stride1.npz",
        "bert-head": stage_weights("B", seed) / f"rank-head-seed{seed}.pt",
        "p6-scores": stage_weights("C", seed) / "p6-eval-scores.npy",
        "fint-scores": stage_weights("C", seed) / "fint-scores-eval.npz",
        "stack-scores": stage_weights("C", seed) / "stack-scores-eval.npz",
    }
    if name == "bert-head":
        model = str(parts.get("model", "BERT")).strip() or "BERT"
        safe_model = model.replace("/", "-").replace("\\", "-").replace(" ", "-")
        mapping[name] = stage_weights("B", seed) / f"rank-head-{safe_model}-seed{int(seed)}.pt"
    if name == "rank-head-scores":
        model = str(parts.get("model", "BERT")).strip() or "BERT"
        safe_model = model.replace("/", "-").replace("\\", "-").replace(" ", "-")
        mapping[name] = stage_weights("B", seed) / f"rank-head-scores-{safe_model}.npz"
    if name in {"fint-scores", "stack-scores"} and parts.get("model"):
        model = str(parts["model"]).strip()
        safe_model = model.replace("/", "-").replace("\\", "-").replace(" ", "-")
        mapping[name] = stage_weights("C", seed) / f"{name}-{safe_model}.npz"
    return mapping[name]


def result_artifact(stage, filename, seed=42):
    return stage_results(stage, seed) / filename


def cand_path(region, seed=42):
    key = "candidates-calib" if region == "calib" else "candidates-eval"
    return weights_artifact(key, seed=seed)


def scores_path(region, seed=42, suffix=""):
    sfx = f"_{suffix}" if suffix else ""
    if not sfx:
        key = "scores-calib" if region == "calib" else "scores-eval"
        return weights_artifact(key, seed=seed)
    return stage_weights("B", seed) / f"scores-{region}-k8-w512-stride1{sfx}.npz"


def model_variant(name):
    """Return the formal model-variant record for a semantic model name."""
    key = str(name).strip().upper()
    for variant in MODEL_VARIANTS:
        if variant["name"].upper() == key or variant["suffix"].upper() == key:
            return variant
    names = ", ".join(v["name"] for v in MODEL_VARIANTS)
    raise ValueError(f"unknown Exp 07 model variant {name!r}; expected one of {names}")


def model_checkpoint(name, seed=42):
    """Return the A-stage checkpoint for a formal model variant."""
    return weights_artifact(model_variant(name)["artifact"], seed=seed)


def require_full_validation_coverage(offset, date_key, *, label="evaluation"):
    """Require one date for every pre-holdout offset 0..399.

    The guard is intentionally based on the row records rather than a summary
    count: a run with 400 dates but one missing offset must fail before it is
    treated as a formal result.  ``offset`` is per-stock window position, so a
    single offset may legitimately map to several calendar dates when stocks
    have different listing histories.
    """
    offsets = np.asarray(offset, dtype=np.int64)
    dates = np.asarray(date_key).astype("<U10")
    if offsets.ndim != 1 or dates.ndim != 1 or len(offsets) != len(dates):
        raise ValueError(f"{label}: offset/date arrays are not aligned")
    expected = np.asarray(VALIDATION_OFFSETS, dtype=np.int64)
    actual = np.unique(offsets)
    missing = np.setdiff1d(expected, actual)
    extra = np.setdiff1d(actual, expected)
    if missing.size or extra.size:
        raise RuntimeError(
            f"{label}: expected validation offsets 0..399; "
            f"missing={missing[:10].tolist()} extra={extra[:10].tolist()}"
        )

    rows_by_offset = np.bincount(offsets, minlength=len(expected))
    if np.any(rows_by_offset == 0):
        missing = np.flatnonzero(rows_by_offset == 0)
        raise RuntimeError(f"{label}: offsets without rows: {missing.tolist()}")
    return {
        "n_rows": int(len(offsets)),
        "n_offsets": int(len(actual)),
        "offset_min": int(actual.min()),
        "offset_max": int(actual.max()),
        "n_dates": int(len(np.unique(dates))),
        "rows_per_offset_min": int(rows_by_offset.min()),
        "rows_per_offset_max": int(rows_by_offset.max()),
    }


def gpt_q_path(region, seed=42):
    return stage_weights("B", seed) / f"gpt-q-{region}-full128.npz"


# ============================================================================
# Model architectures (heads)
# ============================================================================

class MlpRankHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0, loss="soft_spearman"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1))
        self.loss = loss

    def forward(self, h):
        return self.net(h).squeeze(-1)


class LinearReturnHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.loss = "huber"

    def forward(self, h):
        return self.fc(h).squeeze(-1)


class MlpReturnHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1))
        self.loss = "huber"

    def forward(self, h):
        return self.net(h).squeeze(-1)


class LinearDirectionHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h).squeeze(-1))


class MlpDirectionHead(nn.Module):
    def __init__(self, dim=256, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1))

    def forward(self, h):
        return torch.sigmoid(self.net(h).squeeze(-1))


class LinearRankHead(nn.Module):
    def __init__(self, dim=256, loss="pairwise"):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.loss = loss

    def forward(self, h):
        return self.fc(h).squeeze(-1)


# ============================================================================
# Loss functions
# ============================================================================

def soft_rank(scores, tau=1.0):
    d = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    return torch.sigmoid(d / tau).sum(-1)


def soft_spearman_loss(scores, true_ranks, tau=1.0):
    r = soft_rank(scores, tau).float()
    t = true_ranks.float()
    r = r - r.mean()
    t = t - t.mean()
    denom = (r * r).sum().clamp_min(1e-9) * (t * t).sum().clamp_min(1e-9)
    corr = (r * t).sum() / denom.sqrt()
    return (1.0 - corr).clamp_min(0.0)


def pairwise_logistic_loss(scores, true_logret, tau=1.0, dead_zone=None,
                           max_pairs=100_000, seed=42):
    n = scores.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    if dead_zone is None:
        dead_zone = 0.0
    rng = torch.Generator(device=scores.device).manual_seed(seed)
    idx_i = torch.randint(0, n, (max_pairs,), device=scores.device, generator=rng)
    idx_j = torch.randint(0, n, (max_pairs,), device=scores.device, generator=rng)
    keep = (idx_i != idx_j) & ((true_logret[idx_i] - true_logret[idx_j]).abs() >= dead_zone)
    if keep.sum() < 1:
        return torch.tensor(0.0, device=scores.device)
    i, j = idx_i[keep], idx_j[keep]
    target = (true_logret[i] > true_logret[j]).float()
    logit = (scores[i] - scores[j]) / tau
    return F.binary_cross_entropy_with_logits(logit, target)


def huber_loss(pred, target, delta=1.0):
    return F.smooth_l1_loss(pred, target, beta=delta)


def bce_direction(head_out, y):
    return F.binary_cross_entropy(head_out, y, reduction="mean")


def compute_head_loss(head, h, feats, y, loss_kind, tau=1.0, dead_zone=None):
    out = head(feats) if hasattr(head, 'c1') else head(h)
    if loss_kind in ("huber", "mse", "reg"):
        return huber_loss(out, y, delta=1.0)
    if loss_kind in ("bce", "direction"):
        return bce_direction(out, y)
    if loss_kind in ("soft_spearman",):
        ranks = np.argsort(np.argsort(y.detach().cpu().numpy())).astype(np.float32)
        t = torch.from_numpy(ranks).to(y.device)
        return soft_spearman_loss(out, t, tau=tau)
    if loss_kind in ("pairwise",):
        return pairwise_logistic_loss(out, y, tau=tau, dead_zone=dead_zone)
    raise ValueError(loss_kind)


# ============================================================================
# Data loading & cross-section
# ============================================================================

class DailyCrossSectionLoader(IterableDataset):
    def __init__(self, recs, min_stocks=30, seed=42, shuffle_dates=True):
        self.recs = recs
        self.min_stocks = min_stocks
        self.seed = seed
        self.shuffle_dates = shuffle_dates
        self._dates = {}
        for r in recs:
            self._dates.setdefault(r["date_key"], []).append(r)

    def __iter__(self):
        dates = list(self._dates.keys())
        if self.shuffle_dates:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(dates)
        for d in dates:
            rows = self._dates[d]
            if len(rows) >= self.min_stocks:
                yield d, rows


def load_rows(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


FOLDS = {
    "R0": (None, "2020-02-01", "2020-02-01", "2021-02-01"),
    "R1": (None, "2021-02-01", "2021-02-01", "2022-02-01"),
    "R2": (None, "2022-02-01", "2022-02-01", "2023-02-01"),
}
FINAL_FIT_STOP = "2023-02-01"


def in_interval(d, start, stop):
    if start is not None and d < start:
        return False
    if stop is not None and d >= stop:
        return False
    return True


def fold_split(rows, fold):
    start, fstop, vstart, vstop = FOLDS[fold]
    fit_idx = [i for i, d in enumerate(rows["date_key"])
               if in_interval(str(d)[:10], start, fstop)]
    val_idx = [i for i, d in enumerate(rows["date_key"])
               if in_interval(str(d)[:10], vstart, vstop)]
    return np.asarray(fit_idx), np.asarray(val_idx)


def final_fit_split(rows):
    idx = [i for i, d in enumerate(rows["date_key"])
           if str(d)[:10] < FINAL_FIT_STOP]
    return np.asarray(idx)


def _build_cross_section_recs(rows, idx):
    recs = []
    for i in idx:
        recs.append({"date_key": str(rows["date_key"][i])[:10],
                     "stock_uid": str(rows["stock_uid"][i]),
                     "hidden": rows["hidden"][i],
                     "true_logret": float(rows["true_logret"][i]),
                     "raw_logret": float(rows["true_logret"][i])})
    return recs


def _eval_rank_loss(head, rows, val_idx, loss_kind, tau=1.0, dead_zone=None):
    device = next(head.parameters()).device
    recs = _build_cross_section_recs(rows, val_idx)
    loader = DailyCrossSectionLoader(recs, min_stocks=30, seed=0, shuffle_dates=False)
    head.eval()
    total, steps = 0.0, 0
    with torch.no_grad():
        for date, rows_in in loader:
            hb = torch.from_numpy(np.stack([r["hidden"] for r in rows_in])).to(device)
            yb = torch.from_numpy(np.asarray([r["true_logret"] for r in rows_in],
                                             dtype=np.float32)).to(device)
            total += compute_head_loss(head, hb, None, yb, loss_kind, tau=tau,
                                       dead_zone=dead_zone).item()
            steps += 1
    return total / max(1, steps)


def train_rank_per_date(head, rows, fit_idx, val_idx, loss_kind, *, lr, epochs,
                        seed=42, tau=1.0, dead_zone=None):
    device = next(head.parameters()).device
    recs = _build_cross_section_recs(rows, fit_idx)
    loader = DailyCrossSectionLoader(recs, min_stocks=30, seed=seed, shuffle_dates=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        head.train()
        total, steps = 0.0, 0
        for date, rows_in in loader:
            hb = torch.from_numpy(np.stack([r["hidden"] for r in rows_in])).to(device)
            yb = torch.from_numpy(np.asarray([r["true_logret"] for r in rows_in],
                                             dtype=np.float32)).to(device)
            opt.zero_grad()
            loss = compute_head_loss(head, hb, None, yb, loss_kind, tau=tau, dead_zone=dead_zone)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1
        history["train_loss"].append(total / max(1, steps))
        history["val_loss"].append(_eval_rank_loss(
            head, rows, val_idx, loss_kind, tau=tau, dead_zone=dead_zone))
    return head, history


# ============================================================================
# Metrics & bootstrap
# ============================================================================

def _per_date(rec, dense_threshold):
    uniq, inv = np.unique(rec["date_key"], return_inverse=True)
    return uniq, inv


def arm_metrics(rec, score_field, dense_threshold, proper=None):
    score = rec[score_field]
    true = rec["true_logret"]
    quality = rec.get("quality", np.ones(len(score), dtype=bool))
    valid = np.isfinite(score) & np.isfinite(true) & quality
    uniq, inv = _per_date(rec, dense_threshold)
    n_dates = len(uniq)
    da = np.full(n_dates, np.nan)
    ic = np.full(n_dates, np.nan)
    mae_arr = np.full(n_dates, np.nan)
    mape = np.full(n_dates, np.nan)
    cnt = np.zeros(n_dates, dtype=np.int64)
    for i in range(n_dates):
        m = (inv == i) & valid
        c = int(m.sum())
        cnt[i] = c
        if c < dense_threshold:
            continue
        s = score[m]
        t = true[m]
        da[i] = float(np.mean((np.sign(s) > 0) == (t > 0)))
        if c >= 2 and not np.all(s == s[0]):
            ic[i] = float(spearmanr(s, t)[0])
        else:
            ic[i] = 0.0
        mae_arr[i] = float(np.mean(np.abs(s - t)))
        mape[i] = float(np.mean(np.abs(np.exp(s - t) - 1.0)))
    dense = cnt >= dense_threshold
    out = {
        "score_field": score_field,
        "n_dense_dates": int(dense.sum()),
        "avg_daily_rank_ic": float(np.nanmean(ic[dense])) if dense.any() else None,
        "avg_da_per_date": float(np.nanmean(da[dense])) if dense.any() else None,
        "avg_mape": float(np.nanmean(mape[dense])) if dense.any() else None,
        "avg_mae": float(np.nanmean(mae_arr[dense])) if dense.any() else None,
        "per_date": {
            str(uniq[i])[:10]: {"da": da[i], "rank_ic": ic[i], "mae": mae_arr[i], "n": int(cnt[i])}
            for i in range(n_dates)
        },
    }
    if proper is not None:
        out["proper"] = proper
    return out


def daily_rank_ic(score, true, dates, dense_min):
    uniq, inv = np.unique(dates, return_inverse=True)
    n = len(uniq)
    ic = np.full(n, np.nan)
    da = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=np.int64)
    for i in range(n):
        m = inv == i
        c = int(m.sum())
        cnt[i] = c
        if c < dense_min:
            continue
        s = score[m]
        t = true[m]
        if c >= 2 and not np.all(s == s[0]):
            ic[i] = spearmanr(s, t)[0]
        else:
            ic[i] = 0.0
        da[i] = np.mean((np.sign(s) > 0) == (t > 0))
        mae[i] = np.mean(np.abs(s - t))
    dense = cnt >= dense_min
    return ic, da, mae, cnt, dense


def circular_moving_block_bootstrap(deltas, block_length=5, n_replicates=10_000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(deltas)
    if n < block_length:
        return np.array([deltas.mean()])
    starts = rng.randint(0, n, size=n_replicates)
    indices = (starts[:, None] + np.arange(block_length)[None, :]) % n
    return deltas[indices].mean(axis=1)


def paired_bootstrap_ci(candidate_rows, reference_rows, metric, dense_min=None,
                        block_lengths=(5, 10, 20), n_replicates=10_000, seed=42, **kw):
    def _daily(rows, m):
        by_date = defaultdict(list)
        for r in rows:
            by_date[r["date_key"]].append(r)
        out = {}
        for d, rs in by_date.items():
            vals = [r[m] for r in rs if m in r]
            trues = [r["true_logret"] for r in rs if m in r]
            if len(vals) >= (dense_min or 5):
                ic = spearmanr(vals, trues)[0] if len(set(vals)) > 1 else 0.0
                out[d] = ic
        return out
    c_daily = _daily(candidate_rows, metric)
    r_daily = _daily(reference_rows, metric)
    common_dates = sorted(set(c_daily) & set(r_daily))
    if len(common_dates) < 2:
        return {"metric": metric, "n_dates": len(common_dates), "point": None, "error": "too few dates"}
    c_s = np.asarray([c_daily[d] for d in common_dates])
    r_s = np.asarray([r_daily[d] for d in common_dates])
    deltas = c_s - r_s
    point = float(deltas.mean())
    cis = {}
    for L in block_lengths:
        means = circular_moving_block_bootstrap(deltas, L, n_replicates, seed)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        cis[str(L)] = {"block_length": L, "ci_lower": lo, "ci_upper": hi, "significant": lo > 0.0}
    return {"metric": metric, "n_dates": len(common_dates), "point": point,
            "block_cis": cis,
            "block_robust": all(v["significant"] for v in cis.values())}


# ============================================================================
# Coarse posterior -> raw-space return decoding
# ============================================================================

def load_centers():
    return np.load(CENTERS_PATH)


def softmax_rows(logp):
    p = np.exp(np.asarray(logp, dtype=np.float32))
    fin = np.isfinite(p).all(axis=1)
    pn = np.where(fin[:, None], p, np.float32(0.0))
    pn = pn / np.maximum(pn.sum(axis=1, keepdims=True), np.float32(EPS))
    return np.where(fin[:, None], pn, np.nan)


def decode_coarse(p_or_logp, centers, p_mean0, p_std0, log_space=False, chunk=300_000):
    centers = np.asarray(centers, dtype=np.float64)
    pstd = np.maximum(np.asarray(p_std0, dtype=np.float64), EPS)
    pm = np.asarray(p_mean0, dtype=np.float64)
    n = int(np.asarray(p_or_logp).shape[0])
    res = {k: np.full(n, np.nan, dtype=np.float64)
           for k in ("e_mean", "e_median", "p_up_raw", "p_up_naive", "e_norm", "med_norm")}
    order = np.argsort(centers)
    cs = centers[order]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        x = np.asarray(p_or_logp[start:stop], dtype=np.float32)
        if log_space:
            x = np.exp(x)
        fin = np.isfinite(x).all(axis=1)
        pn = np.where(fin[:, None], x, np.float32(0.0))
        pn = pn / np.maximum(pn.sum(axis=1, keepdims=True), np.float32(EPS))
        m = stop - start
        pstd_c = pstd[start:stop]
        pm_c = pm[start:stop]
        e_norm = (pn @ centers).astype(np.float64)
        ps = pn[:, order]
        cdf = np.cumsum(ps, axis=1)
        idx = np.sum(cdf < np.float32(0.5), axis=1).clip(max=len(centers) - 1)
        med_norm = cs[idx]
        thr = -pm_c / pstd_c
        pup_raw = np.sum(pn * (centers[None, :] > thr[:, None]), axis=1).astype(np.float64)
        pup_naive = np.sum(pn * (centers[None, :] > 0.0), axis=1).astype(np.float64)
        mask = np.where(fin, 1.0, np.nan)
        res["e_norm"][start:stop] = e_norm * mask
        res["e_mean"][start:stop] = (e_norm * pstd_c + pm_c) * mask
        res["med_norm"][start:stop] = med_norm * mask
        res["e_median"][start:stop] = (med_norm * pstd_c + pm_c) * mask
        res["p_up_raw"][start:stop] = pup_raw * mask
        res["p_up_naive"][start:stop] = pup_naive * mask
    return res


def poe_fused(logp_bert, logq_gpt, lam, chunk=300_000):
    n = int(np.asarray(logp_bert).shape[0])
    v = int(np.asarray(logp_bert).shape[1])
    pf_out = np.full((n, v), np.nan, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        p_bert = softmax_rows(logp_bert[start:stop])
        fin = np.isfinite(p_bert).all(axis=1)
        p_bert = np.where(fin[:, None], p_bert, np.float32(0.0))
        q = np.asarray(logq_gpt[start:stop], dtype=np.float32)
        q = np.where(fin[:, None], q, np.float32(1.0))
        q = q / q.sum(axis=1, keepdims=True)
        q = np.maximum(q, np.float32(EPS))
        lf = ((1.0 - lam) * np.log(q)
              + lam * np.log(np.maximum(p_bert, np.float32(EPS))))
        lf -= lf.max(axis=1, keepdims=True)
        pf = np.exp(lf)
        pf = pf / pf.sum(axis=1, keepdims=True)
        pf_out[start:stop] = np.where(fin[:, None], pf, np.nan)
    good = np.isfinite(pf_out).all(axis=1)
    if good.any():
        assert np.allclose(pf_out[good].sum(axis=1), 1.0, atol=1e-4), "PoE row sums != 1"
    return pf_out


# ============================================================================
# Record construction & region slices
# ============================================================================

def build_rec(cand, pt01=None, extra=None):
    n = len(cand["stock_uid"])
    rec = {
        "date_key": cand["date_key"],
        "stock_uid": cand["stock_uid"],
        "true_logret": cand["true_logret"].astype(np.float64),
        "quality": cand["quality"].astype(bool),
        "p_mean0": cand["p_mean0"].astype(np.float64),
        "p_std0": cand["p_std0"].astype(np.float64),
        "post_median": cand["post_median"].astype(np.float64),
        "p_up": cand["p_up"].astype(np.float64),
        "post_std": cand["post_std"].astype(np.float64),
    }
    if "offset" in cand.files:
        rec["offset"] = cand["offset"].astype(np.int64)
    if pt01 is not None:
        rec["post_mean"] = pt01["post_mean"].astype(np.float64)
        rec["greedy_return"] = pt01["greedy_return"].astype(np.float64)
    if extra:
        rec.update(extra)
    return rec


def slice_rec(rec, keep_mask):
    out = {}
    for k, v in rec.items():
        out[k] = np.asarray(v)[keep_mask]
    return out


def eval_dev_confirm(rec):
    off = np.asarray(rec["offset"], dtype=np.int64)
    dev = slice_rec(rec, (off >= 0) & (off <= 299))
    conf = slice_rec(rec, (off >= 300) & (off < 400))
    return dev, conf


# ============================================================================
# Array-based metrics & bootstrap
# ============================================================================

def _split_points(dates):
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    split = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    bounds = np.concatenate([[0], split, [len(dates)]]).astype(np.int64)
    uniq = [str(sd[int(bounds[i])]) for i in range(len(bounds) - 1)]
    return order, bounds, uniq


def daily_rank_ic_series(rec, field, dense_min):
    score = np.asarray(rec[field], dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(np.asarray(rec["true_logret"], dtype=np.float64)) \
        & np.asarray(rec["quality"]).astype(bool)
    order, bounds, uniq = _split_points(rec["date_key"])
    s = score[order]
    v = valid[order]
    out = {}
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        m = v[lo:hi]
        c = int(m.sum())
        if c < dense_min:
            continue
        ss = s[lo:hi][m]
        tt = np.asarray(rec["true_logret"], dtype=np.float64)[order[lo:hi]][m]
        if c >= 2 and not np.all(ss == ss[0]):
            ic = float(spearmanr(ss, tt)[0])
        else:
            ic = 0.0
        out[uniq[i]] = ic
    return out


class DailyIcCache:
    def __init__(self, rec, dense_min):
        self.dense_min = dense_min
        self.order, self.bounds, self.uniq = _split_points(rec["date_key"])
        self.true = np.asarray(rec["true_logret"], dtype=np.float64)[self.order]
        self.qual = np.asarray(rec["quality"]).astype(bool)[self.order]

    def series(self, score):
        s = np.asarray(score, dtype=np.float64)[self.order]
        out = {}
        for i in range(len(self.bounds) - 1):
            lo, hi = int(self.bounds[i]), int(self.bounds[i + 1])
            ss, tt, qq = s[lo:hi], self.true[lo:hi], self.qual[lo:hi]
            m = np.isfinite(ss) & np.isfinite(tt) & qq
            c = int(m.sum())
            if c < self.dense_min:
                continue
            sv, tv = ss[m], tt[m]
            if c >= 2 and not np.all(sv == sv[0]):
                ic = float(spearmanr(sv, tv)[0])
            else:
                ic = 0.0
            out[self.uniq[i]] = ic
        return out


def circular_block_means(deltas, block_length, n_replicates=10_000, seed=42):
    return circular_moving_block_bootstrap(
        deltas, block_length=block_length, n_replicates=n_replicates, seed=seed)


def bootstrap_vs(rec, field, ref_rec, ref_field, dense_min, block_lengths=(5, 10, 20),
                 n_replicates=10_000, seed=42, _cand_cache=None, _ref_cache=None):
    cc = _cand_cache if _cand_cache is not None else DailyIcCache(rec, dense_min)
    rc = _ref_cache if _ref_cache is not None else DailyIcCache(ref_rec, dense_min)
    cs = cc.series(rec[field])
    rs = rc.series(ref_rec[ref_field])
    common = sorted(set(cs) & set(rs))
    if len(common) < 2:
        return {"n_dates": len(common), "point": None, "block_cis": None,
                "error": "fewer than 2 common dense dates"}
    c_series = np.asarray([cs[d] for d in common], dtype=float)
    r_series = np.asarray([rs[d] for d in common], dtype=float)
    deltas = c_series - r_series
    point = float(deltas.mean())
    cis = {}
    for L in block_lengths:
        means = circular_block_means(deltas, L, n_replicates, seed)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        cis[str(L)] = {"block_length": int(L), "ci_lower": lo, "ci_upper": hi,
                       "significant_directional": bool(lo > 0.0)}
    return {"metric": "rank_ic", "n_dates": len(common), "point": point,
            "candidate_mean": float(c_series.mean()),
            "reference_mean": float(r_series.mean()),
            "block_cis": cis,
            "block_robust": all(v["significant_directional"] for v in cis.values())}


def metrics_table(rec, fields, dense_threshold):
    out = {}
    for name, field in fields.items():
        if field not in rec or rec[field] is None:
            continue
        try:
            m = arm_metrics(rec, field, dense_threshold)
        except Exception as e:  # noqa: BLE001
            m = {"error": str(e)}
        m.pop("per_date", None)
        out[name] = m
    return out


def region_dense_threshold(cand, frac=0.8, floor=5):
    dates = cand["date_key"]
    _, counts = np.unique(dates, return_counts=True)
    return max(floor, int(np.ceil(frac * int(counts.max()))))


def calib_dense_threshold(seed=42):
    c = np.load(cand_path("calib", seed), allow_pickle=True)
    return region_dense_threshold(c)


def summarize(rec, fields, dense_threshold, refs=None, label=""):
    out = {"label": label, "dense_threshold": dense_threshold,
           "full": metrics_table(rec, fields, dense_threshold)}
    dev, conf = eval_dev_confirm(rec)
    if dev and len(dev["stock_uid"]):
        out["dev_0_299"] = metrics_table(dev, fields, dense_threshold)
        out["confirm_300_399"] = metrics_table(conf, fields, dense_threshold)
    if refs:
        cc = DailyIcCache(rec, dense_threshold)
        out["bootstrap_vs"] = {}
        for ref_name, (ref_rec, ref_field) in refs.items():
            rc = DailyIcCache(ref_rec, dense_threshold)
            out["bootstrap_vs"][ref_name] = {}
            for name, field in fields.items():
                if field not in rec or rec[field] is None:
                    continue
                try:
                    out["bootstrap_vs"][ref_name][name] = bootstrap_vs(
                        rec, field, ref_rec, ref_field, dense_threshold,
                        _cand_cache=cc, _ref_cache=rc)
                except Exception as e:  # noqa: BLE001
                    out["bootstrap_vs"][ref_name][name] = {"error": str(e)}
    return out


def write_json_ledger(path, payload, event, **trial):
    write_json(path, payload)
    append_trial({"event": event, "status": "ok", **trial})
    return path


# ============================================================================
# Protocol guards
# ============================================================================

FORBIDDEN_RANK_FIELDS = ("logp", "margin", "entropy", "electra_acceptance")


def assert_score_semantics(field):
    name = field.lower()
    if any(f in name for f in FORBIDDEN_RANK_FIELDS):
        raise ValueError(
            f"rank score '{field}' violates row-score semantics (E-1): "
            "token-likelihood / entropy / acceptance quantities are forbidden")


def require_ca_not_regressed(ca_path, baseline_avg_rank=3.23, max_regression=0.3):
    if not Path(ca_path).exists():
        raise RuntimeError(f"C-a artifact missing: {ca_path}")
    ca = load_json(ca_path)
    avg_rank = float(ca["c_a"]["avg_true_rank"])
    if avg_rank > baseline_avg_rank + max_regression:
        raise RuntimeError(f"C-a regressed: {avg_rank:.2f} > {baseline_avg_rank + max_regression}")
    return ca


def row_set_fingerprint(stock_uid, date_key, ckpt_path, out_json=None):
    u = np.asarray(stock_uid)
    d = np.asarray(date_key)
    row_hash = hashlib.sha256()
    row_hash.update(str(len(u)).encode())
    for uu, dd in zip(sorted(set(map(str, u))), sorted(set(map(str, d)))):
        row_hash.update(uu.encode())
        row_hash.update(b"|")
        row_hash.update(dd.encode())
    fp = {
        "schema": "bert-hidden-cache-fingerprint-v1",
        "n_rows": int(len(u)),
        "n_unique_uids": int(len(set(map(str, u)))),
        "n_unique_dates": int(len(set(map(str, d)))),
        "row_set_sha256": row_hash.hexdigest(),
        "bert_ckpt_sha256": file_sha256(Path(ckpt_path)),
        "ckpt_path": str(ckpt_path),
    }
    if out_json is not None:
        write_json(Path(out_json), fp)
    return fp


# ============================================================================
# BERT data construction
# ============================================================================

def in_audit_calibration(date_str):
    return CALIB_START <= date_str < CALIB_STOP


def bert_time_for_target_date(date_str):
    y, m, d = (int(x) for x in str(date_str)[:10].split("-"))
    y_enc = max(0, min(99, y - 2010))
    return torch.tensor([d, m, y_enc], dtype=torch.long)


def selection_history_window(input_ids_row, time_ids_row, va_row, pos, window):
    start = max(0, pos + 1 - window)
    return input_ids_row[start:pos + 1], time_ids_row[start:pos + 1], va_row[start:pos + 1]


def build_bert_input(hist_ids, hist_time, hist_va, target_time,
                     vocab_base=VOCAB_BASE, mask_id=MASK_ID):
    ids = torch.cat([hist_ids, torch.tensor([mask_id], dtype=torch.long)])
    tids = torch.cat([hist_time, target_time.unsqueeze(0)], dim=0)
    va = torch.cat([hist_va, torch.zeros(1, 2, dtype=hist_va.dtype)], dim=0)
    pos = torch.arange(ids.shape[0], dtype=torch.long)
    return ids, tids, va, pos


# ============================================================================
# Scoring-aligned masking
# ============================================================================

def _recency_weights(seq_len, recency_window, device):
    dist = seq_len - 1 - torch.arange(seq_len, device=device)
    w = torch.ones(seq_len, device=device)
    if recency_window > 1:
        near = dist < recency_window
        w[near] = 1.0 + (recency_window - 1 - dist[near]).clamp(min=0).float() / float(recency_window - 1)
    return w


def _corrupt_801010(ids, vocab_base, mask_id, mlm_prob, corrupt_fracs,
                    recency_window, generator):
    N = ids.shape[0]
    device = ids.device
    is_special = ids >= vocab_base
    w = _recency_weights(N, recency_window, device)
    rand = torch.rand(N, device=device, generator=generator)
    can_mask = (~is_special) & (rand < (mlm_prob * w).clamp(max=0.5))
    f_mask, f_rnd, f_keep = corrupt_fracs
    pick = torch.rand(N, device=device, generator=generator)
    do_mask = can_mask & (pick < f_mask)
    do_random = can_mask & (pick >= f_mask) & (pick < f_mask + f_rnd)
    out = ids.clone()
    out[do_mask] = mask_id
    if bool(do_random.any()):
        out[do_random] = torch.randint(0, vocab_base, (int(do_random.sum()),), device=device)
    labels = torch.full((N,), -100, dtype=torch.long, device=device)
    labels[can_mask] = ids[can_mask]
    return out, labels


def make_scoring_batch(input_ids, time_id, va, vocab_base, mask_id,
                       window=512, mlm_prob=0.15,
                       corrupt_fracs=(0.8, 0.1, 0.1),
                       final_pos_frac=0.5, va_zero_frac=0.5,
                       recency_window=64, generator=None):
    device = input_ids.device
    r = torch.rand(1, device=device, generator=generator).item() if generator else torch.rand(1, device=device).item()
    day_mask = input_ids < vocab_base
    valid_pos = torch.nonzero(day_mask, as_tuple=False).squeeze(-1)
    if valid_pos.numel() == 0:
        valid_pos = torch.arange(input_ids.shape[0], device=device)
    if r < final_pos_frac and valid_pos.numel() > 1:
        p = int(valid_pos[torch.randint(valid_pos.numel(), (1,), generator=generator).item()])
        start = max(0, p + 1 - window)
        ids = input_ids[start:p + 1].clone()
        tids = time_id[start:p + 1]
        v = va[start:p + 1].clone()
        L = ids.shape[0]
        ids[-1] = mask_id
        labels = torch.full((L,), -100, dtype=torch.long, device=device)
        labels[-1] = input_ids[p]
        if L > 1:
            hist, hlab = _corrupt_801010(input_ids[start:p], vocab_base, mask_id, mlm_prob,
                                         corrupt_fracs, recency_window, generator)
            ids[:-1] = hist
            labels[:-1] = hlab
        v[-1] = 0.0
        pos_out = torch.arange(L, device=device)
        return ids, tids, v, labels, pos_out, start
    ids, labels = _corrupt_801010(input_ids, vocab_base, mask_id, mlm_prob,
                                  corrupt_fracs, recency_window, generator)
    v = va.clone()
    masked = ids == mask_id
    vdrop = torch.rand(v.shape[0], device=device, generator=generator) < va_zero_frac
    v = v.masked_fill((masked & vdrop).unsqueeze(-1), 0.0)
    pos_out = torch.arange(ids.shape[0], device=device)
    return ids, time_id, v, labels, pos_out, 0


# ============================================================================
# BERT checkpoint loading
# ============================================================================

def load_bert(ckpt_path, device):
    from model.kronos_bert import KronosBert
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = types.SimpleNamespace(
        vocab_size=int(cfg["vocab_size"]), vocab_fine=int(cfg.get("vocab_fine", 128)),
        dim=int(cfg["dim"]), depth=int(cfg["depth"]), heads=int(cfg["heads"]),
        num_kv_heads=int(cfg.get("num_kv_heads", 1)),
        ffn_multiplier=int(cfg.get("ffn_multiplier", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        va_hidden_dim=int(cfg.get("va_hidden_dim", 64)),
        rope_base=float(cfg.get("rope_base", 10000.0)),
    )
    model = KronosBert(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_cfg, ckpt


# ============================================================================
# Per-stock prepared-input index
# ============================================================================

def build_index(tokenizer_path, device="cpu", max_stocks=0, cache_path=None):
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            idx = pickle.load(f)
        print(f"[index] loaded cached index {len(idx.by_uid)} stocks")
        return idx
    tokenizer = load_tokenizer(str(tokenizer_path), torch.device(device))
    import importlib.util
    _c06_path = ROOT / "experiments" / "06-posttrain" / "common.py"
    _c06_spec = importlib.util.spec_from_file_location("common06", _c06_path)
    _c06 = importlib.util.module_from_spec(_c06_spec)
    _c06_spec.loader.exec_module(_c06)
    load_stocks_uid = _c06.load_stocks_uid
    attach_close_prices_uid = _c06.attach_close_prices_uid
    prepare_stocks_uid = _c06.prepare_stocks_uid
    stocks = load_stocks_uid(DataConfig.data_dir)
    if max_stocks:
        stocks = stocks[:max_stocks]
    attach_close_prices_uid(stocks)
    prepped = prepare_stocks_uid(stocks, tokenizer, torch.device(device))
    idx = BertInputIndex(prepped)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[index] cached {len(prepped)} stocks -> {cache_path}")
    return idx


class BertInputIndex:
    def __init__(self, prepped):
        self.by_uid = {}
        for p in prepped:
            uid = p["stock_uid"]
            day = np.asarray(p["day"], dtype=np.int64)
            month = np.asarray(p["month"], dtype=np.int64)
            year = np.asarray(p["year"], dtype=np.int64)
            time_ids = torch.from_numpy(
                np.stack([day, month, year], axis=-1)).to(torch.long)
            inp_ids = torch.as_tensor(p["inp_ids"], dtype=torch.long)
            va = torch.as_tensor(p["va"], dtype=torch.float32)
            dates_int = np.asarray(
                [int(str(d)[:10].replace("-", "")) for d in p["dates_raw"]],
                dtype=np.int32)
            self.by_uid[uid] = {"inp_ids": inp_ids, "time_ids": time_ids,
                                "va": va, "dates_int": dates_int}

    def position_for(self, uid, date_key):
        p = self.by_uid.get(uid)
        if p is None:
            return None
        d_int = int(str(date_key)[:10].replace("-", ""))
        i = int(np.searchsorted(p["dates_int"], d_int, side="left"))
        if i >= len(p["dates_int"]) or int(p["dates_int"][i]) != d_int:
            return None
        return i

    def bert_input(self, uid, date_key, window, vocab_base=VOCAB_BASE,
                   mask_id=VOCAB_BASE + 2, shuffle_seed=None, position=None):
        p = self.by_uid[uid]
        if position is not None:
            pos = int(position)
        else:
            pos = self.position_for(uid, date_key)
        if pos is None or pos < 0 or pos >= p["inp_ids"].shape[0]:
            return None
        start = max(0, pos + 1 - window)
        hist_ids = p["inp_ids"][start:pos + 1]
        hist_time = p["time_ids"][start:pos + 1]
        hist_va = p["va"][start:pos + 1]
        if shuffle_seed is not None:
            rng = np.random.RandomState(int(shuffle_seed))
            perm = rng.permutation(hist_ids.shape[0])
            hist_ids = hist_ids[perm]
            hist_time = hist_time[perm]
            hist_va = hist_va[perm]
        target_time = bert_time_for_target_date(date_key)
        return build_bert_input(hist_ids, hist_time, hist_va, target_time,
                                vocab_base=vocab_base, mask_id=mask_id)

    def __contains__(self, uid):
        return uid in self.by_uid


# ============================================================================
# BERT scoring
# ============================================================================

@torch.no_grad()
def score_rows(index, rows, model, *, window=512, batch_size=32, device="cuda",
               shuffle_history=False):
    n = len(rows)
    K = rows.topk_width() if hasattr(rows, "topk_width") else 0
    logp_full = np.full((n, VOCAB_BASE), np.nan, dtype=np.float32)
    logp_topk = np.full((n, max(K, 1)), np.nan, dtype=np.float32)
    top1_id = np.full(n, -1, dtype=np.int32)
    margin = np.full(n, np.nan, dtype=np.float32)
    dev = torch.device(device)
    idx = 0
    while idx < n:
        stop = min(idx + batch_size, n)
        built = []
        for j in range(idx, stop):
            shuf_seed = j if shuffle_history else None
            b = index.bert_input(rows.stock_uid(j), rows.date_key(j), window,
                                 shuffle_seed=shuf_seed,
                                 position=rows.position(j))
            built.append(b)
        valid = [b is not None for b in built]
        if not any(valid):
            idx = stop
            continue
        valid_sel = [b for b, ok in zip(built, valid) if ok]
        max_len = max(b[0].shape[0] for b in valid_sel)
        B = len(valid_sel)
        inp = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        tids = torch.zeros(B, max_len, 3, dtype=torch.long, device=dev)
        va = torch.zeros(B, max_len, 2, dtype=torch.float32, device=dev)
        maskpos = torch.empty(B, dtype=torch.long, device=dev)
        for i, (ids, ti, v, _) in enumerate(valid_sel):
            L = ids.shape[0]
            inp[i, :L] = ids
            tids[i, :L] = ti
            va[i, :L] = v
            maskpos[i] = L - 1
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda"),
                                dtype=torch.bfloat16):
            logits = model(inp, tids, torch.arange(max_len, device=dev)
                           .unsqueeze(0).expand(B, -1), va_values=va)
        mp = maskpos.view(B, 1, 1).expand(B, 1, logits.shape[-1])
        ml = torch.gather(logits, 1, mp).squeeze(1).float()
        lp = torch.log_softmax(ml, dim=-1).cpu().numpy()
        g = 0
        for j in range(idx, stop):
            if not valid[j - idx]:
                continue
            logp_full[j] = lp[g]
            top1_id[j] = int(np.argmax(lp[g]))
            slp = -np.sort(-lp[g])
            margin[j] = float(slp[0] - slp[1])
            tk = rows.topk_ids(j)
            if tk is not None and len(tk) > 0:
                logp_topk[j, :len(tk)] = lp[g][tk]
            g += 1
        idx = stop
    return {"logp_bert_topk": logp_topk, "bert_top1_id": top1_id,
            "bert_margin": margin, "logp_bert_full": logp_full}


class NumpyRowTable:
    def __init__(self, stock_uid, date_key, position=None, topk_ids=None):
        self._u = stock_uid
        self._d = date_key
        self._p = position
        self._t = topk_ids
        self._n = len(stock_uid)

    def __len__(self):
        return self._n

    def stock_uid(self, i):
        return str(self._u[i])

    def date_key(self, i):
        return str(self._d[i])

    def position(self, i):
        return int(self._p[i]) if self._p is not None else None

    def topk_ids(self, i):
        return self._t[i] if self._t is not None else None

    def topk_width(self):
        return int(self._t.shape[1]) if self._t is not None else 0

    def select(self, keep_mask):
        return NumpyRowTable(self._u[keep_mask], self._d[keep_mask],
                             position=self._p[keep_mask] if self._p is not None else None,
                             topk_ids=self._t[keep_mask] if self._t is not None else None)


# ============================================================================
# GPT coarse-q for hidden
# ============================================================================

def load_model_cpu(ckpt, tok_path):
    tok = load_tokenizer(str(tok_path), torch.device("cpu"))
    model = load_gpt(str(ckpt), torch.device("cpu"), tokenizer=tok)
    model.eval()
    return model, tok


@torch.no_grad()
def coarse_q_for_hidden(model, hidden, t_c=TC, vocab_base=128):
    logits = model.coarse_logits_from_hidden(hidden)
    p_full = torch.softmax(logits / t_c, dim=-1)
    special_mass = p_full[:, vocab_base:].sum(dim=-1)
    q = p_full[:, :vocab_base] / (1.0 - special_mass).clamp_min(EPS).unsqueeze(-1)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return q, special_mass


# ============================================================================
# P6 scoring
# ============================================================================

def load_p6_scores(head_path, hidden):
    ck = torch.load(str(head_path), map_location="cpu", weights_only=False)
    sd = ck["head_state"]
    net = nn.Sequential(
        nn.Linear(256, 64), nn.SiLU(), nn.Identity(), nn.Linear(64, 1))
    net.load_state_dict({k.replace("net.", ""): v for k, v in sd.items()})
    net.eval()
    with torch.no_grad():
        s = net(torch.from_numpy(hidden).float()).squeeze(-1).numpy()
    return s


def require_controls(controls_path):
    if not Path(controls_path).exists():
        raise RuntimeError(
            f"C-a/C-b controls must run before B1-B5 (missing {controls_path})")
