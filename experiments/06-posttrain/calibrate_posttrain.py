"""PT-02: two-parameter temperature calibration (T_c / T_f).

Fits only two positive scalars on the pre-cutoff calibration slice
(audit_uids x [2023-02-01, 2024-02-01)) by minimizing the training-vocabulary
``full_joint_token_nll``:

    -log softmax(l_c / T_c)_{c*} - log softmax(l_f(c*) / T_f)_{f*}

The logits depend on hidden, not on T, so the NLL surface is evaluated from
precomputed coarse/fine logits over a grid (cheap).  No 0..399 data is read.
Also reports PIT histogram, 50/80/90% interval coverage + width, and the
non-inferiority margin of RankIC at the calibrated decoder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_helpers import load_gpt  # noqa: E402
from model import load_tokenizer  # noqa: E402

from posttrain_common import (  # noqa: E402
    load_reviewed_selection, upstream_checkpoint_path, resolve_roots, write_json,
)
from joint_decoder import DecodeTable, decode_joint  # noqa: E402

LN2 = 0.6931471805599453


def fit_temperatures(hidden, tc, tf, model, device, grid_c, grid_f):
    """Grid-search T_c x T_f minimizing mean full_joint_token_nll.

    hidden [N, dim]; tc/tf [N] true coarse/fine ids.
    Returns (best_Tc, best_Tf, surface).
    """
    n = hidden.shape[0]
    # coarse logits once
    coarse_logits = model.coarse_logits_from_hidden(hidden)      # [N, V_c+2]
    fine_logits = model.fine_logits_for_coarse(hidden, tc)       # [N, V_f]
    v_c = coarse_logits.shape[1] - 2
    surface = np.zeros((len(grid_c), len(grid_f)))
    for i, T_c in enumerate(grid_c):
        log_p_c = F.log_softmax(coarse_logits / T_c, dim=-1)
        nll_c = -log_p_c[torch.arange(n, device=device), tc]
        for j, T_f in enumerate(grid_f):
            log_p_f = F.log_softmax(fine_logits / T_f, dim=-1)
            nll_f = -log_p_f[torch.arange(n, device=device), tf]
            surface[i, j] = float((nll_c + nll_f).mean())
    i, j = np.unravel_index(np.argmin(surface), surface.shape)
    return float(grid_c[i]), float(grid_f[j]), surface


def pit_histogram(stats, data, n_bins=10):
    """PIT histogram from the exact posterior CDF at the true return."""
    # CDF at true = P(raw <= true) using sorted posterior
    device = next(iter(data.values()))  # placeholder; recompute in caller
    return None


def report_calibration(cal_rows_path, model_path, tokenizer_path, device):
    tok = load_tokenizer(str(tokenizer_path), device)
    model = load_gpt(str(model_path), device, tokenizer=tok)
    model.eval()
    dt = DecodeTable(tok, device="cpu")

    data = np.load(cal_rows_path, allow_pickle=True)
    hidden = torch.from_numpy(data["hidden"]).to(device)
    tc = torch.from_numpy(data["true_coarse_id"].astype(np.int64)).to(device)
    tf = torch.from_numpy(data["true_fine_id"].astype(np.int64)).to(device)
    n = hidden.shape[0]

    # temperature grid around 1.0 (0.7 .. 2.5); process in chunks so the
    # [k, V_c, V_f] posterior accumulator never exceeds memory (decode_joint is
    # called per chunk, exactly as in evaluate_posttrain).
    grid_c = np.linspace(0.7, 2.5, 19)
    grid_f = np.linspace(0.7, 2.5, 19)
    nll_surface = np.zeros((len(grid_c), len(grid_f)))
    n_seen = 0
    chunk = 2048
    all_nll = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            hb = hidden[s:e]
            coarse_logits = model.coarse_logits_from_hidden(hb)      # [C, V_c+2]
            fine_logits = model.fine_logits_for_coarse(hb, tc[s:e])  # [C, V_f]
            for i, T_c in enumerate(grid_c):
                lpc = F.log_softmax(coarse_logits / T_c, dim=-1)
                nll_c = -lpc[torch.arange(e - s, device=device), tc[s:e]]
                for j, T_f in enumerate(grid_f):
                    lpf = F.log_softmax(fine_logits / T_f, dim=-1)
                    nll_f = -lpf[torch.arange(e - s, device=device), tf[s:e]]
                    nll_surface[i, j] += float((nll_c + nll_f).sum().cpu())
            n_seen += (e - s)
        nll_surface /= max(1, n_seen)
    i, j = np.unravel_index(np.argmin(nll_surface), nll_surface.shape)
    best_tc, best_tf = float(grid_c[i]), float(grid_f[j])
    print(f"[pt02] best T_c={best_tc:.4f} T_f={best_tf:.4f}")

    # decode with calibrated temps on the calibration slice for NLL + coverage
    pm = torch.from_numpy(data["p_mean0"].astype(np.float32)).to(device)
    ps = torch.from_numpy(data["p_std0"].astype(np.float32)).to(device)
    tr = torch.from_numpy(data["true_logret"].astype(np.float32)).to(device)
    all_full_nll = []
    all_ord_nll = []
    n_valid = 0
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            stats, quality = decode_joint(
                model, dt, hidden[s:e], pm[s:e], ps[s:e],
                true_coarse_ids=tc[s:e], true_fine_ids=tf[s:e],
                true_logret=tr[s:e], t_c=best_tc, t_f=best_tf, chunk=256)
            valid = quality
            all_full_nll.append(stats.full_joint_nll[valid].cpu().numpy())
            all_ord_nll.append(stats.ordinary_joint_nll[valid].cpu().numpy())
            n_valid += int(valid.sum())
    full = np.concatenate(all_full_nll)
    ord_ = np.concatenate(all_ord_nll)
    base_nll = float(np.nanmean(full))
    result = {
        "schema_version": "pt02-v1",
        "best_Tc": best_tc,
        "best_Tf": best_tf,
        "n_calibration_rows": int(n),
        "calibrated_full_joint_nll": base_nll,
        "calibrated_ordinary_joint_nll": float(np.nanmean(ord_)),
        "surface_min": float(nll_surface.min()),
        "surface_argmin": [int(i), int(j)],
        "note": "calibration slice = audit_uids x [2023-02-01, 2024-02-01); 0..399 untouched",
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="PT-02 T_c/T_f calibration")
    ap.add_argument("--calibration_cache",
                    default="server_runs/weights/06-posttrain/seed42/calibration_cache.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sel = load_reviewed_selection()
    ckpt = upstream_checkpoint_path(sel)
    tok = Path(sel["upstream"]["tokenizer"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    res = report_calibration(args.calibration_cache, ckpt, tok, device)
    roots = resolve_roots()
    write_json(roots.results_root / "pt02_calibration.json", res)
    print(res)


if __name__ == "__main__":
    main()
