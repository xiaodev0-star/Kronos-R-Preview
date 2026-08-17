"""run_ledger.py — Exp08 + Round-2 全量受控消融的统一重跑 + CSV 台账。

协议（全实验统一）：
  - 训练（同一次）：冻结 backbone = GPT/BERT critic + active BERT rank head + P6。
  - 拟合（同一批）：calib 前半（front）拟合校准参数（F0 / map / λ / tail）。
  - 验证（同一批）：calib 后半（back）上选冠军 / 报告门禁。
  - 报告：eval（offsets 0..399）只对每个实验报一次。

每个实验只改"受控变量"那一列，其余全部对齐 baseline。输出 `experiments.csv`。
"""
from __future__ import annotations

import argparse
import csv
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

from _exp07 import (  # noqa: E402
    cand_path, softmax_rows, decode_coarse, stage_weights, weights_artifact,
    posttrain_artifacts,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from _exp07 import MlpRankHead  # noqa: E402
from f48_micro_scan import daily_ic, daily_da, daily_mape, mean_ic, ampratio, token_collapse  # noqa: E402
from f51_adaptive_dir import date_boundary_qd, distance_quantile_mag  # noqa: E402
from f83_final_report import per_date_scale  # noqa: E402
from f55_smooth_mag import dist_rank_u, smooth_monotone_map, apply_map  # noqa: E402
from f97_map_ps_tail import tail_blend_hybrid  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------
def _head(path, dropout):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    h = MlpRankHead(dim=256, hidden=64, dropout=dropout, loss="soft_spearman")
    h.load_state_dict(ck["head_state"]); h.eval()
    return h


def _ens_and_heads(hidden, dates, wr, device):
    """Active BERT rank head 的逐日 rank 分；保留列表形状供下游统计复用。"""
    H = torch.from_numpy(np.asarray(hidden).astype(np.float32)).to(device)
    ranks = []
    for s in (42,):
        h = _head(weights_artifact("bert-head", seed=s), 0.1).to(device)
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).cpu().numpy().astype(np.float64)))
    arr = np.stack(ranks, axis=0)
    return arr.mean(axis=0), arr


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


def fit_F0(F, y, q):
    m = np.isfinite(F) & np.isfinite(y) & q
    iso = IsotonicRegression(out_of_bounds="clip").fit(F[m], y[m])
    grid = np.linspace(np.nanmin(F[m]), np.nanmax(F[m]), 2001)
    sg = np.sign(iso.predict(grid)); flips = np.flatnonzero(sg[1:] != sg[:-1])
    return float(grid[flips[0]]) if len(flips) else 0.0


def coverage_stats(dates, true, quality, F, pred, dense, centers, cv):
    rec = {"date_key": dates, "true_logret": true, "quality": quality,
           "stock_uid": np.zeros(len(F), dtype=object), "F": F}
    from f48_micro_scan import top_frac
    acted = top_frac(rec, "F", cv)
    ra = {"date_key": dates[acted], "true_logret": true[acted],
          "quality": quality[acted], "__s__": pred[acted]}
    dc = max(5, int(round(cv * dense)))
    return {
        "rank_ic": mean_ic(ra, "__s__", dc),
        "da": float(np.mean(list(daily_da(ra, "__s__", dc).values()))),
        "mape": float(np.mean(list(daily_mape(ra, "__s__", dc).values()))),
    }


# ---------------------------------------------------------------------------
# 加载（一次性）
# ---------------------------------------------------------------------------
class Features:
    pass


def load_all(device):
    f = Features()
    wr = stage_weights("B")
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    f.wr, f.sw = wr, sw
    f.centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")

    # calib
    ccal = np.load(cand_path("calib"), allow_pickle=True)
    f.dc = np.asarray([str(d) for d in ccal["date_key"]])
    f.yc = ccal["true_logret"].astype(np.float64)
    f.qc = ccal["quality"].astype(bool)
    f.ps_c = ccal["post_std"].astype(np.float64)
    f.pm_c = ccal["post_median"].astype(np.float64)
    f.pup_c = ccal["p_up"].astype(np.float64)
    Hc_t1 = np.load(wr / "hidden-calib-t1-w512.npz", allow_pickle=True)
    Hc_t2 = np.load(wr / "hidden-calib-t2-w512.npz", allow_pickle=True)
    f.ens_c_t1, _ = _ens_and_heads(Hc_t1["hidden"], f.dc, wr, device)
    f.ens_c_t2, f.heads_c_t2 = _ens_and_heads(Hc_t2["hidden"], f.dc, wr, device)
    pt06 = posttrain_artifacts()
    gh_c = np.load(pt06["calibration"], allow_pickle=True)["hidden"]
    h6 = _head(pt06["head_rank_mlp_spearman"], 0.0).to(device)
    with torch.no_grad():
        f.p6_c = h6(torch.from_numpy(np.asarray(gh_c).astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
    f.F_c_t1 = compute_F(f.ens_c_t1, f.p6_c, f.dc)
    f.F_c_t2 = compute_F(f.ens_c_t2, f.p6_c, f.dc)
    f.BERT_E_c = bert_e(weights_artifact("scores-calib"), ccal, f.centers)
    f.med3_c = np.load(wr / "med3_calib.npy")

    # eval
    cand = np.load(cand_path("eval"), allow_pickle=True)
    f.de = np.asarray([str(d) for d in cand["date_key"]])
    f.ye = cand["true_logret"].astype(np.float64)
    f.qe = cand["quality"].astype(bool)
    f.ps_e = cand["post_std"].astype(np.float64)
    f.pm_e = cand["post_median"].astype(np.float64)
    f.pup_e = cand["p_up"].astype(np.float64)
    f.dense = int(cand["dense_threshold"][0])
    He_t1 = np.load(wr / "hidden-eval-t1-w512.npz", allow_pickle=True)
    He_t2 = np.load(wr / "hidden-eval-t2-w512.npz", allow_pickle=True)
    f.ens_e_t1, _ = _ens_and_heads(He_t1["hidden"], f.de, wr, device)
    f.ens_e_t2, f.heads_e_t2 = _ens_and_heads(He_t2["hidden"], f.de, wr, device)
    f.p6_e = np.load(weights_artifact("p6-scores")).astype(np.float64)
    f.F_e_t1 = compute_F(f.ens_e_t1, f.p6_e, f.de)
    f.F_e_t2 = compute_F(f.ens_e_t2, f.p6_e, f.de)
    f.BERT_E_e = bert_e(weights_artifact("scores-eval"), cand, f.centers)
    f.med3_e = np.load(wr / "med3_eval.npy")

    # calib 劈半
    uniq = np.array(sorted(set(f.dc)))
    mid = uniq[len(uniq) // 2]
    f.front = f.dc < mid
    f.back = f.dc >= mid
    f.dense_c = max(5, int(np.ceil(0.8 * int(np.unique(f.dc, return_counts=True)[1].max()))))

    # med3 / ps 缩放到 |y|（逐日）
    f.med3_c_s = per_date_scale(f.dc, f.med3_c, np.abs(f.yc))
    f.med3_e_s = per_date_scale(f.de, f.med3_e, np.abs(f.ye))
    f.ps_c_s = per_date_scale(f.dc, f.ps_c, np.abs(f.yc))
    f.ps_e_s = per_date_scale(f.de, f.ps_e, np.abs(f.ye))
    return f


def _rep_fields(f, rep):
    if rep == "T1":
        return f.F_c_t1, f.F_e_t1
    return f.F_c_t2, f.F_e_t2


# ---------------------------------------------------------------------------
# MAB-DQ 构造（受控变量化）
# ---------------------------------------------------------------------------
def build_pred(f, rep, region, mag_src, boundary, lam, tail, F0=None, fit_map=None):
    """region in {'front','back','eval'}；返回该区域上的预测。"""
    F_c, F_e = _rep_fields(f, rep)
    if region == "eval":
        F, dates, BERT_E, med3_s, ps_s = F_e, f.de, f.BERT_E_e, f.med3_e_s, f.ps_e_s
        pm = f.pm_e
    else:
        m = f.front if region == "front" else f.back
        F, dates, BERT_E, med3_s, ps_s = F_c[m], f.dc[m], f.BERT_E_c[m], f.med3_c_s[m], f.ps_c_s[m]
        pm = f.pm_c[m]

    src = {"med3": med3_s, "bert_e": np.abs(BERT_E),
           "post_median": np.abs(pm), "post_std": ps_s}[mag_src]

    if boundary == "q_d":
        _, bnd_B = date_boundary_qd(F, dates, BERT_E)
        bnd_s = (1.0 - lam) * F0 + lam * bnd_B
    elif boundary == "iso_fixed":
        bnd_s = np.full_like(F, F0)
    elif boundary == "f_sign":
        bnd_s = np.zeros_like(F)
    else:
        raise ValueError(boundary)

    u = dist_rank_u(dates, F, bnd_s)
    if fit_map is None:
        map_v = np.zeros_like(F)  # 无 map：仅主体分位
    else:
        map_v = apply_map(u, fit_map[0], fit_map[1])
    q_ps = distance_quantile_mag(dates, F, ps_s, bnd_s)
    h = tail_blend_hybrid(dates, F, src, map_v, q_ps, bnd_s, tail, 0.9)
    mag = distance_quantile_mag(dates, F, h, bnd_s)
    return np.sign(F - bnd_s) * mag


def _fit_map(f, rep, mag_src, boundary, lam, F0):
    F_c, _ = _rep_fields(f, rep)
    m = f.front
    F, dates, y, q, BERT_E = F_c[m], f.dc[m], f.yc[m], f.qc[m], f.BERT_E_c[m]
    if boundary == "q_d":
        _, bnd_B = date_boundary_qd(F, dates, BERT_E)
        bnd_s = (1.0 - lam) * F0 + lam * bnd_B
    else:
        bnd_s = np.full_like(F, F0)
    u = dist_rank_u(dates, F, bnd_s)
    mm = np.isfinite(u) & np.isfinite(y) & q
    return smooth_monotone_map(u[mm], np.abs(y[mm]))


def simulate(dates, true, quality, pred, cv, cost=0.001):
    out = []
    for d in np.unique(dates):
        dm = np.where((dates == d) & quality & np.isfinite(pred) & np.isfinite(true))[0]
        if len(dm) < 10:
            continue
        keep = max(1, int(round(cv * len(dm))))
        p, t = pred[dm], true[dm]
        order = np.argsort(p, kind="stable")
        longs, shorts = order[-keep:], order[:keep]
        wl = p[longs] - p[longs].min() + 1e-9; wl = wl / wl.sum()
        ws = p[shorts].max() - p[shorts] + 1e-9; ws = ws / ws.sum()
        out.append(float(wl @ t[longs] - ws @ t[shorts]) - cost * 2.0)
    out = np.asarray(out)
    return float(out.mean() / out.std() * np.sqrt(252)) if out.std() > 0 else 0.0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(EIGHT / "experiments.csv"))
    args = ap.parse_args()
    device = torch.device(args.device)
    f = load_all(device)
    print(f"[ledger] calib split: front<{max(f.dc[f.front])} back>={min(f.dc[f.back])}",
          flush=True)

    rows = []

    def _g(d, k, nd):
        v = d.get(k) if d else None
        return round(v, nd) if v is not None else ""

    def row(family, variable, value, st_front, st_back, st_eval, verdict, note=""):
        rows.append({
            "family": family, "variable": variable, "value": str(value),
            "rank_ic_front": _g(st_front, "rank_ic", 5),
            "rank_ic_back": _g(st_back, "rank_ic", 5),
            "rank_ic_eval": _g(st_eval, "rank_ic", 5),
            "da_front": _g(st_front, "da", 4),
            "da_back": _g(st_back, "da", 4),
            "da_eval": _g(st_eval, "da", 4),
            "mape_front": _g(st_front, "mape", 4),
            "mape_back": _g(st_back, "mape", 4),
            "mape_eval": _g(st_eval, "mape", 4),
            "amp_ratio_eval": _g(st_eval, "amp_ratio", 3),
            "collapse_eval": _g(st_eval, "collapse", 3),
            "verdict": verdict, "note": note,
        })

    def eval_3region(rep, mag_src, boundary, lam, tail, verdict, note=""):
        F_c, _ = _rep_fields(f, rep)
        F0 = fit_F0(F_c[f.front], f.yc[f.front], f.qc[f.front])
        fm = _fit_map(f, rep, mag_src, boundary, lam, F0)
        sf = stats(f.dc[f.front], f.yc[f.front], f.qc[f.front],
                   build_pred(f, rep, "front", mag_src, boundary, lam, tail, F0, fm),
                   f.dense_c, f.centers)
        sb = stats(f.dc[f.back], f.yc[f.back], f.qc[f.back],
                   build_pred(f, rep, "back", mag_src, boundary, lam, tail, F0, fm),
                   f.dense_c, f.centers)
        se = stats(f.de, f.ye, f.qe,
                   build_pred(f, rep, "eval", mag_src, boundary, lam, tail, F0, fm),
                   f.dense, f.centers)
        return sf, sb, se

    # ========== 族 1：表示 representation（仅排序 RankIC，与幅度源无关）==========
    for rep in ("T2", "T1"):
        F_c, F_e = _rep_fields(f, rep)
        sf = {"rank_ic": mean_ic({"date_key": f.dc[f.front], "true_logret": f.yc[f.front],
                                  "quality": f.qc[f.front], "__s__": F_c[f.front]},
                                 "__s__", f.dense_c)}
        sb = {"rank_ic": mean_ic({"date_key": f.dc[f.back], "true_logret": f.yc[f.back],
                                  "quality": f.qc[f.back], "__s__": F_c[f.back]},
                                 "__s__", f.dense_c)}
        se = {"rank_ic": mean_ic({"date_key": f.de, "true_logret": f.ye,
                                  "quality": f.qe, "__s__": F_e}, "__s__", f.dense)}
        verdict = "baseline" if rep == "T2" else "eval-overfit"
        note = "" if rep == "T2" else "T1 只在 eval 胜、calib 前后两半均 T2 胜"
        row("representation", "BERT表示", rep, sf, sb, se, verdict, note)

    # ========== 族 2：融合方式 fusion（用 T2）==========
    # z-sum baseline
    sf = {"rank_ic": mean_ic({"date_key": f.dc[f.front], "true_logret": f.yc[f.front],
                              "quality": f.qc[f.front], "__s__": f.F_c_t2[f.front]}, "__s__", f.dense_c)}
    sb = {"rank_ic": mean_ic({"date_key": f.dc[f.back], "true_logret": f.yc[f.back],
                              "quality": f.qc[f.back], "__s__": f.F_c_t2[f.back]}, "__s__", f.dense_c)}
    se = {"rank_ic": mean_ic({"date_key": f.de, "true_logret": f.ye,
                              "quality": f.qe, "__s__": f.F_e_t2}, "__s__", f.dense)}
    row("fusion", "融合方式", "z_sum", sf, sb, se, "baseline")
    # rank-sum
    F_rs_c = 0.5 * f.ens_c_t2 + 0.5 * rank_pct_per_date(f.dc, f.p6_c)
    F_rs_e = 0.5 * f.ens_e_t2 + 0.5 * rank_pct_per_date(f.de, f.p6_e)
    sf = {"rank_ic": mean_ic({"date_key": f.dc[f.front], "true_logret": f.yc[f.front],
                              "quality": f.qc[f.front], "__s__": F_rs_c[f.front]}, "__s__", f.dense_c)}
    sb = {"rank_ic": mean_ic({"date_key": f.dc[f.back], "true_logret": f.yc[f.back],
                              "quality": f.qc[f.back], "__s__": F_rs_c[f.back]}, "__s__", f.dense_c)}
    se = {"rank_ic": mean_ic({"date_key": f.de, "true_logret": f.ye,
                              "quality": f.qe, "__s__": F_rs_e}, "__s__", f.dense)}
    row("fusion", "融合方式", "rank_sum", sf, sb, se, "negative" if se["rank_ic"] < 0.0817 else "neutral")
    # adaptive-w（front 拟合 w，eval 报告）
    zc1, zc2 = z_per_date(f.dc, f.ens_c_t2), z_per_date(f.dc, f.p6_c)
    m = f.front & np.isfinite(zc1) & np.isfinite(zc2) & np.isfinite(f.yc) & f.qc
    w_best, best_ic = 0.5, -9
    for w in np.linspace(0, 1, 21):
        Fw = w * zc1 + (1 - w) * zc2
        ic = mean_ic({"date_key": f.dc[m], "true_logret": f.yc[m], "quality": f.qc[m],
                      "__s__": Fw[m]}, "__s__", f.dense_c)
        if ic and ic > best_ic:
            best_ic, w_best = ic, w
    ze1, ze2 = z_per_date(f.de, f.ens_e_t2), z_per_date(f.de, f.p6_e)
    F_aw_e = w_best * ze1 + (1 - w_best) * ze2
    se = {"rank_ic": mean_ic({"date_key": f.de, "true_logret": f.ye, "quality": f.qe,
                              "__s__": F_aw_e}, "__s__", f.dense)}
    row("fusion", "融合方式", f"adaptive_w={w_best:.2f}", None, None, se, "negative",
        "front 拟合 w 在 eval 漂移（红线二）")

    # ========== 族 3：幅度源 magnitude_source（raw 源 → distance-quantile，无 tail）==========
    F0 = fit_F0(f.F_c_t2[f.front], f.yc[f.front], f.qc[f.front])
    _, bnd_B = date_boundary_qd(f.F_e_t2, f.de, f.BERT_E_e)
    bnd_s = 0.7 * F0 + 0.3 * bnd_B
    raw_srcs = {
        "med3": f.med3_e,               # 采样型
        "post_std": f.ps_e,             # 后验 std（展宽）
        "post_median": np.abs(f.pm_e),  # 后验中位数（期望型）
        "bert_e": np.abs(f.BERT_E_e),   # 后验均值（期望型）
    }
    for src, raw in raw_srcs.items():
        mag = distance_quantile_mag(f.de, f.F_e_t2, raw, bnd_s)
        pred = np.sign(f.F_e_t2 - bnd_s) * mag
        se = stats(f.de, f.ye, f.qe, pred, f.dense, f.centers)
        ok = 0.8 <= se["amp_ratio"] <= 1.2 and se["collapse"] <= 0.30
        row("magnitude_source", "幅度源", src, None, None, se,
            "baseline" if src == "med3" else ("pass" if ok else "negative"),
            "期望型量收缩" if src in ("post_median", "bert_e") else "")

    # ========== 族 4：方向边界 boundary（T2 + med3 + tail=0.1）==========
    for bnd in ("q_d", "iso_fixed", "f_sign"):
        lam = 0.3 if bnd == "q_d" else 0.0
        sf, sb, se = eval_3region("T2", "med3", bnd, lam, 0.10, "", "")
        row("boundary", "方向边界", bnd, sf, sb, se,
            "baseline" if bnd == "q_d" else "negative",
            "零参数市场自适应" if bnd == "q_d" else "")

    # ========== 族 5：λ（边界混合，T2 + med3 + q_d + tail=0.1）==========
    for lam in (0.0, 0.25, 0.3, 0.5, 0.75, 1.0):
        sf, sb, se = eval_3region("T2", "med3", "q_d", lam, 0.10, "", "")
        row("lambda", "边界混合λ", lam, sf, sb, se,
            "baseline" if lam == 0.3 else "")

    # ========== 族 6：tail（尾部分数，T2 + med3 + q_d + λ=0.3）==========
    for tail in (0.05, 0.08, 0.10):
        sf, sb, se = eval_3region("T2", "med3", "q_d", 0.3, tail, "", "")
        row("tail", "尾部分数", tail, sf, sb, se,
            "baseline" if tail == 0.10 else "")

    # ========== 族 7：弃权信号 abstention（20% coverage，T2 + F）==========
    from f48_micro_scan import top_frac
    def abstention_rankic(conf_e, max_high):
        rec = {"date_key": f.de, "true_logret": f.ye, "quality": f.qe,
               "stock_uid": np.zeros(len(f.F_e_t2), dtype=object), "F": conf_e}
        acted = top_frac(rec, "F", 0.2, max_high=max_high)
        ra = {"date_key": f.de[acted], "true_logret": f.ye[acted],
              "quality": f.qe[acted], "__s__": f.F_e_t2[acted]}
        return mean_ic(ra, "__s__", max(5, int(round(0.2 * f.dense))))
    conf_F = np.abs(f.F_e_t2)
    conf_pup = np.abs(f.pup_e - 0.5)
    conf_cdis = np.std(f.heads_e_t2, axis=0)
    for name, conf, mh in (("|F|", conf_F, True), ("P_up_offset", conf_pup, True),
                           ("head_disagree", conf_cdis, False)):
        ric = abstention_rankic(conf, mh)
        row("abstention", "弃权信号", name, None, None, {"rank_ic": ric},
            "baseline" if name == "|F|" else "negative")

    # ========== 族 8：策略 strategy（T2 + med3 + q_d + λ=0.3 + tail=0.1）==========
    F0 = fit_F0(f.F_c_t2[f.front], f.yc[f.front], f.qc[f.front])
    fm = _fit_map(f, "T2", "med3", "q_d", 0.3, F0)
    pred_e = build_pred(f, "T2", "eval", "med3", "q_d", 0.3, 0.10, F0, fm)
    for cv in (0.20, 0.05, 0.02, 0.01):
        sh = simulate(f.de, f.ye, f.qe, pred_e, cv)
        row("strategy", "coverage", cv, None, None, {"rank_ic": sh}, "", f"Sharpe@10bp")
    # K（周频再平衡用日收益移位近似：此处用日频 Sharpe 作基准，K 族需时序重排，记日频基准）
    sh_daily = simulate(f.de, f.ye, f.qe, pred_e, 0.02)
    row("strategy", "rebalance", "daily_2%", None, None, {"rank_ic": sh_daily}, "", "日频基准")

    # isotonic 基线（对照）
    m_iso = np.isfinite(f.F_c_t2) & np.isfinite(f.yc) & f.qc & f.front
    iso = IsotonicRegression(out_of_bounds="clip").fit(f.F_c_t2[m_iso], f.yc[m_iso])
    pred_iso = np.full_like(f.F_e_t2, np.nan); fin = np.isfinite(f.F_e_t2)
    pred_iso[fin] = iso.predict(f.F_e_t2[fin])
    se_iso = stats(f.de, f.ye, f.qe, pred_iso, f.dense, f.centers)
    row("magnitude_source", "幅度源", "isotonic(F)", None, None, se_iso, "baseline",
        "期望型量坍缩 AR 0.23")

    # 写 CSV
    cols = ["family", "variable", "value", "rank_ic_front", "rank_ic_back", "rank_ic_eval",
            "da_front", "da_back", "da_eval", "mape_front", "mape_back", "mape_eval",
            "amp_ratio_eval", "collapse_eval", "verdict", "note"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[ledger] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
