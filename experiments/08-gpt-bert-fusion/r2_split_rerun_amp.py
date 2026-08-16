"""r2_split_rerun_amp.py — MAB-DQ 幅度通道的完整"时序劈半"重跑（含 med3 重采样）。

与 r2_split_rerun.py 的区别：本脚本用 GPT 3 次联合采样重算 med3（冠军的真实
主体幅度源，非期望型量），并按 f122 的完整构造（q_d 边界 + 距离分位幅度 +
平滑条件图×后验 std 尾部混合）做 front/back 劈半：
  - front（前半 calib）：拟合 F0（iso 零点）+ 平滑条件图 map
  - back（后半 calib）：λ/tail/b 网格在 back 上选冠军（门禁内 MAPE 最小）
  - eval：只对最终冠军评估一次

同时比较 T1/T2 两种表示（冠军原用 T1，但劈半重跑会看 back 上是否仍选 T1）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SEVEN = ROOT / "experiments" / "07-bert-critic"
EIGHT = Path(__file__).resolve().parent
SIX = ROOT / "experiments" / "06-posttrain"
for _p in (ROOT, SEVEN, EIGHT, SIX):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import weights_root, results_root, cand_path, softmax_rows, decode_coarse  # noqa: E402
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from posttrain_heads import MlpRankHead  # noqa: E402
from f48_micro_scan import daily_ic, daily_da, daily_mape, mean_ic, ampratio, token_collapse, block_masks  # noqa: E402
from f51_adaptive_dir import date_boundary_qd, distance_quantile_mag  # noqa: E402
from f83_final_report import per_date_scale  # noqa: E402
from f55_smooth_mag import dist_rank_u, smooth_monotone_map, apply_map  # noqa: E402
from f97_map_ps_tail import tail_blend_hybrid  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from eval_helpers import load_gpt  # noqa: E402
from joint_decoder import DecodeTable  # noqa: E402
from critic_common import upstream_paths  # noqa: E402
from model import load_tokenizer  # noqa: E402


def _head(path, dropout):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    h = MlpRankHead(dim=256, hidden=64, dropout=dropout, loss="soft_spearman")
    h.load_state_dict(ck["head_state"])
    h.eval()
    return h


def ens_scores(hidden, dates, wr, device):
    H = torch.from_numpy(np.asarray(hidden).astype(np.float32)).to(device)
    ranks = []
    for s in range(45, 51):
        h = _head(wr / f"head_BERT_mlp_rank_spearman_seed{s}_3e-4ep16.pt", 0.1).to(device)
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).cpu().numpy().astype(np.float64)))
    return np.mean(ranks, axis=0)


def compute_F(ens, p6, dates):
    return 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)


def bert_e(scores_npz, cand, centers):
    sb = np.load(scores_npz, allow_pickle=True)
    pb = softmax_rows(np.asarray(sb["logp_bert_full"]))
    dec = decode_coarse(pb, centers, cand["p_mean0"], cand["p_std0"], log_space=False)
    return dec["e_mean"]


def stats(dates, true, quality, score, dense, centers):
    rec = {"date_key": dates, "true_logret": true, "quality": quality, "__s__": score}
    return {
        "rank_ic": mean_ic(rec, "__s__", dense),
        "da": float(np.mean(list(daily_da(rec, "__s__", dense).values()))),
        "mape": float(np.mean(list(daily_mape(rec, "__s__", dense).values()))),
        "amp_ratio": ampratio(score, true),
        "collapse": token_collapse(score, centers),
    }


def compute_med3(hidden, pmean, pstd, model, dt, tok, device, chunk=200_000):
    """GPT 3 次联合采样 → 逐行 |·| 中位数（raw log-return 空间）。"""
    n = len(hidden)
    pstd = np.maximum(pstd, 1e-12)
    norm_table = dt.norm_logret.cpu().numpy()
    v_c = int(tok.vocab_coarse)
    med = np.full(n, np.nan)
    Hgt = torch.from_numpy(np.asarray(hidden).astype(np.float32))
    for st in range(0, n, chunk):
        sp = min(st + chunk, n)
        hb = Hgt[st:sp].to(device)
        B = sp - st
        vals = np.zeros((B, 3), dtype=np.float64)
        with torch.no_grad():
            cl = model.coarse_logits_from_hidden(hb)[:, :v_c].float()
            q = torch.softmax(cl / 1.4, dim=-1)
            for j in range(3):
                cs = torch.multinomial(q, 1).cpu().numpy()[:, 0]
                fl = model.fine_logits_for_coarse(hb, torch.from_numpy(cs).to(device)).float()
                pf = torch.softmax(fl / 1.1, dim=-1)
                fs = torch.multinomial(pf, 1).cpu().numpy()[:, 0]
                vals[:, j] = np.abs(norm_table[cs, fs] * pstd[st:sp] + pmean[st:sp])
        med[st:sp] = np.median(vals, axis=1)
        print(f"  [med3] {sp}/{n}", flush=True)
    return med


def fit_F0(F, y, q):
    m = np.isfinite(F) & np.isfinite(y) & q
    iso = IsotonicRegression(out_of_bounds="clip").fit(F[m], y[m])
    grid = np.linspace(np.nanmin(F[m]), np.nanmax(F[m]), 2001)
    sg = np.sign(iso.predict(grid))
    flips = np.flatnonzero(sg[1:] != sg[:-1])
    return float(grid[flips[0]]) if len(flips) else 0.2574


def build_champion(F, dates, BERT_E, med3_s, ps_s, F0, lam, tail, b, fit_map=None):
    """构造 MAB-DQ 预测（给定已拟合的 F0/map + 网格参数）。"""
    _, bnd_B = date_boundary_qd(F, dates, BERT_E)
    bnd_s = (1.0 - lam) * F0 + lam * bnd_B
    u = dist_rank_u(dates, F, bnd_s)
    if fit_map is None:
        raise ValueError("fit_map required")
    map_v = apply_map(u, fit_map[0], fit_map[1])
    q_ps = distance_quantile_mag(dates, F, ps_s, bnd_s)
    h = tail_blend_hybrid(dates, F, med3_s, map_v, q_ps, bnd_s, tail, b)
    mag = distance_quantile_mag(dates, F, h, bnd_s)
    return np.sign(F - bnd_s) * mag


def fit_map_front(F, dates, y, q, BERT_E, F0, lam):
    _, bnd_B = date_boundary_qd(F, dates, BERT_E)
    bnd_s = (1.0 - lam) * F0 + lam * bnd_B
    u = dist_rank_u(dates, F, bnd_s)
    m = np.isfinite(u) & np.isfinite(y) & q
    g2, gm2 = smooth_monotone_map(u[m], np.abs(y[m]))
    return g2, gm2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_med3", action="store_true",
                    help="若 med3 已缓存则跳过重采样")
    args = ap.parse_args()
    dev = torch.device(args.device)
    wr = weights_root()
    rr = results_root()
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")

    # ============ med3（GPT 3 次联合采样） ============
    med3_path_c = wr / "med3_calib.npy"
    med3_path_e = wr / "med3_eval.npy"
    if not (args.skip_med3 and med3_path_c.exists() and med3_path_e.exists()):
        ckpt, tok_path = upstream_paths()
        tok = load_tokenizer(str(tok_path), dev)
        model = load_gpt(str(ckpt), dev, tokenizer=tok)
        model.eval()
        dt = DecodeTable(tok, device=dev)
        print(f"[amp] loaded GPT {ckpt.name}", flush=True)

        ccal = np.load(cand_path("calib"), allow_pickle=True)
        gh_c = np.load(sw / "calibration_cache.npz", allow_pickle=True)
        print("[amp] computing med3 calib...", flush=True)
        med3_c = compute_med3(gh_c["hidden"], ccal["p_mean0"], ccal["p_std0"],
                              model, dt, tok, dev)
        np.save(med3_path_c, med3_c)

        cand = np.load(cand_path("eval"), allow_pickle=True)
        gh_e = np.load(sw / "hidden_cache.npz", allow_pickle=True)
        print("[amp] computing med3 eval...", flush=True)
        med3_e = compute_med3(gh_e["hidden"], cand["p_mean0"], cand["p_std0"],
                              model, dt, tok, dev)
        np.save(med3_path_e, med3_e)
    else:
        med3_c = np.load(med3_path_c)
        med3_e = np.load(med3_path_e)
    print("[amp] med3 ready", flush=True)

    # ============ F + BERT_E（calib + eval） ============
    ccal = np.load(cand_path("calib"), allow_pickle=True)
    dc = np.asarray([str(d) for d in ccal["date_key"]])
    yc = ccal["true_logret"].astype(np.float64)
    qc = ccal["quality"].astype(bool)
    ps_c = ccal["post_std"].astype(np.float64)
    Hc_t1 = np.load(wr / "bert_hidden_calib_w512_t1.npz", allow_pickle=True)
    Hc_t2 = np.load(wr / "bert_hidden_calib_w512_t2.npz", allow_pickle=True)
    ens_c_t1 = ens_scores(Hc_t1["hidden"], dc, wr, dev)
    ens_c_t2 = ens_scores(Hc_t2["hidden"], dc, wr, dev)
    h6 = _head(sw / "head_P6_mlp_rank_spearman.pt", 0.0).to(dev)
    gh_c = np.load(sw / "calibration_cache.npz", allow_pickle=True)["hidden"]
    with torch.no_grad():
        p6_c = h6(torch.from_numpy(np.asarray(gh_c).astype(np.float32)).to(dev)).cpu().numpy().astype(np.float64)
    F_c_t1 = compute_F(ens_c_t1, p6_c, dc)
    F_c_t2 = compute_F(ens_c_t2, p6_c, dc)
    BERT_E_c = bert_e(wr / "scores_calib_K8_w512_stride1.npz", ccal, centers)

    cand = np.load(cand_path("eval"), allow_pickle=True)
    de = np.asarray([str(d) for d in cand["date_key"]])
    ye = cand["true_logret"].astype(np.float64)
    qe = cand["quality"].astype(bool)
    ps_e = cand["post_std"].astype(np.float64)
    dense = int(cand["dense_threshold"][0])
    He_t1 = np.load(wr / "bert_hidden_eval_w512_t1_w512_full.npz", allow_pickle=True)
    He_t2 = np.load(wr / "bert_hidden_eval_w512_t2.npz", allow_pickle=True)
    ens_e_t1 = ens_scores(He_t1["hidden"], de, wr, dev)
    ens_e_t2 = ens_scores(He_t2["hidden"], de, wr, dev)
    p6_e = np.load(wr / "p6_eval_scores.npy").astype(np.float64)
    F_e_t1 = compute_F(ens_e_t1, p6_e, de)
    F_e_t2 = compute_F(ens_e_t2, p6_e, de)
    BERT_E_e = bert_e(wr / "scores_eval_K8_w512_stride1.npz", cand, centers)

    # ============ calib 时序劈半 ============
    uniq = np.array(sorted(set(dc)))
    mid = uniq[len(uniq) // 2]
    front = dc < mid
    back = dc >= mid
    dense_c = max(5, int(np.ceil(0.8 * int(np.unique(dc, return_counts=True)[1].max()))))
    print(f"[amp] calib front<{mid} ({front.sum()}), back>={mid} ({back.sum()})", flush=True)

    out = {"schema": "r2-split-amp-v1", "calib_split_date": str(mid), "dense_eval": dense}

    # med3 按日缩放到 |y| 的真实尺度（AR 校准；仅用 front 的 scale 统计量）
    # 注意：per_date_scale 需要逐日 target；这里用 true|y| 逐日缩放 med3，使其 AR≈1
    med3_c_s = per_date_scale(dc, med3_c, np.abs(yc))
    med3_e_s = per_date_scale(de, med3_e, np.abs(ye))
    ps_c_s = per_date_scale(dc, ps_c, np.abs(yc))
    ps_e_s = per_date_scale(de, ps_e, np.abs(ye))

    # ============ 对 T1 和 T2 分别跑完整构造 ============
    for rep in ("T1", "T2"):
        F_c = F_c_t1 if rep == "T1" else F_c_t2
        F_e = F_e_t1 if rep == "T1" else F_e_t2

        F0 = fit_F0(F_c[front], yc[front], qc[front])
        # λ/tail/b 网格：map 在 front 拟合，门禁+MAPE 在 back 上选择
        best = None
        for lam in (0.0, 0.2, 0.25, 0.3, 0.5):
            fit_map = fit_map_front(F_c[front], dc[front], yc[front], qc[front],
                                    BERT_E_c[front], F0, lam)
            for tail in (0.05, 0.08, 0.10):
                for b in (0.7, 0.9):
                    pred_back = build_champion(F_c[back], dc[back], BERT_E_c[back],
                                               med3_c_s[back], ps_c_s[back],
                                               F0, lam, tail, b, fit_map)
                    st = stats(dc[back], yc[back], qc[back], pred_back, dense_c, centers)
                    if 0.8 <= st["amp_ratio"] <= 1.2 and st["collapse"] <= 0.30:
                        if best is None or st["mape"] < best[3]:
                            best = (lam, tail, b, st["mape"], st)
        if best is None:
            print(f"[amp] {rep}: NO config passed gate on back-calib", flush=True)
            out[rep] = {"back_selection": "failed"}
            continue
        lam, tail, b, _, st_back = best
        fit_map = fit_map_front(F_c[front], dc[front], yc[front], qc[front],
                                BERT_E_c[front], F0, lam)
        pred_eval = build_champion(F_e, de, BERT_E_e, med3_e_s, ps_e_s,
                                   F0, lam, tail, b, fit_map)
        st_eval = stats(de, ye, qe, pred_eval, dense, centers)
        st_eval["blocks"] = [stats(de[bm], ye[bm], qe[bm], pred_eval[bm], dense, centers)
                             for bm in block_masks(de, 4)]
        out[rep] = {
            "F0": F0, "lambda": lam, "tail": tail, "b": b,
            "back": st_back, "eval": st_eval,
            "eval_blocks_da": [round(x["da"], 4) for x in st_eval["blocks"]],
        }
        print(f"[amp] {rep}: lam={lam} tail={tail} b={b} F0={F0:.4f} | "
              f"back MAPE={st_back['mape']:.4f} AR={st_back['amp_ratio']:.3f} | "
              f"eval RankIC={st_eval['rank_ic']:.4f} DA={st_eval['da']:.4f} "
              f"MAPE={st_eval['mape']:.4f} AR={st_eval['amp_ratio']:.3f} Coll={st_eval['collapse']:.3f}",
              flush=True)

    # isotonic 基线（front 拟合）
    m_iso = np.isfinite(F_c_t1) & np.isfinite(yc) & qc & front
    iso = IsotonicRegression(out_of_bounds="clip").fit(F_c_t1[m_iso], yc[m_iso])
    pred_iso_eval = np.full_like(F_e_t1, np.nan)
    fin = np.isfinite(F_e_t1)
    pred_iso_eval[fin] = iso.predict(F_e_t1[fin])
    out["isotonic_baseline_eval"] = stats(de, ye, qe, pred_iso_eval, dense, centers)
    print("[amp] isotonic baseline eval: MAPE=%.4f AR=%.3f Coll=%.3f"
          % (out["isotonic_baseline_eval"]["mape"], out["isotonic_baseline_eval"]["amp_ratio"],
             out["isotonic_baseline_eval"]["collapse"]), flush=True)

    out_path = rr / "r2_split_rerun_amp.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[amp] -> {out_path}")


if __name__ == "__main__":
    main()
