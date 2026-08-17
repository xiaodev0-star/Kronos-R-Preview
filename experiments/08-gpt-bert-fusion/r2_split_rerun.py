"""r2_split_rerun.py — Round-2 微观校准的"时序劈半"重跑。

目标：原 Round-2 的冠军筛选是在 eval（400 日）上做的，存在泄漏。本脚本按
"calib 时序劈半"协议重跑：
  - 前半 calib（front）  = 拟合校准参数（F0 / 边界 / 幅度）
  - 后半 calib（back）   = 选择冠军（T1 vs T2 表示、λ/tail 网格）
  - eval（offsets 0..399）= 只对最终冠军评估一次

核心裁决问题：T1（去噪前）表示在原轮中"在 eval 上探测选出、calib 反而偏好 T2"。
这里直接比较 F_T1 vs F_T2 的 RankIC 在 front/back/eval 三个切片上是否一致。

幅度源用 |post_std|（GPT 后验 std，calib/eval 均有缓存）作为 med3 的代理；
med3 需 GPT 3 次联合采样（昂贵），此处不重算——RankIC 结论与幅度源无关
（任意保序映射精确保留 RankIC），DA/MAPE 为近似值。
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
for _p in (ROOT, SEVEN, EIGHT, ROOT / "experiments" / "06-posttrain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _exp07 import (  # noqa: E402
    weights_root, results_root, cand_path, softmax_rows, decode_coarse,
    stage_weights, weights_artifact, posttrain_artifacts,
)
from f0_scores import rank_pct_per_date, z_per_date  # noqa: E402
from _exp07 import MlpRankHead  # noqa: E402
from f48_micro_scan import (  # noqa: E402
    daily_ic, daily_da, daily_mape, mean_ic, ampratio, token_collapse, block_masks,
)
from f51_adaptive_dir import date_boundary_qd, distance_quantile_mag  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402


def _head(path, dropout):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    h = MlpRankHead(dim=256, hidden=64, dropout=dropout, loss="soft_spearman")
    h.load_state_dict(ck["head_state"])
    h.eval()
    return h


def ens_scores(hidden, dates, wr, device):
    """Active BERT rank head → per-date rank-percentile."""
    H = torch.from_numpy(np.asarray(hidden).astype(np.float32)).to(device)
    ranks = []
    for s in (42,):
        h = _head(weights_artifact("bert-head", seed=s), 0.1).to(device)
        with torch.no_grad():
            ranks.append(rank_pct_per_date(dates, h(H).cpu().numpy().astype(np.float64)))
    return np.mean(ranks, axis=0)


def compute_F(ens, p6, dates):
    return 0.5 * z_per_date(dates, ens) + 0.5 * z_per_date(dates, p6)


def bert_e(scores_npz, cand, centers, device):
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


def fit_boundary_iso(F, y, q, dates):
    """Isotonic zero-crossing F0 (fit on given mask)."""
    m = np.isfinite(F) & np.isfinite(y) & q
    iso = IsotonicRegression(out_of_bounds="clip").fit(F[m], y[m])
    grid = np.linspace(np.nanmin(F[m]), np.nanmax(F[m]), 2001)
    sg = np.sign(iso.predict(grid))
    flips = np.flatnonzero(sg[1:] != sg[:-1])
    F0 = float(grid[flips[0]]) if len(flips) else 0.2574
    return F0


def champion_pred(F, dates, BERT_E, mag_abs, lam, F0):
    """MAB-DQ 简化版：市场自适应边界 + 距离分位幅度（|post_std| 作幅度源）。"""
    _, bnd_B = date_boundary_qd(F, dates, BERT_E)
    bnd_s = (1.0 - lam) * F0 + lam * bnd_B
    mag = distance_quantile_mag(dates, F, mag_abs, bnd_s)
    return np.sign(F - bnd_s) * mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    wr = stage_weights("B")
    rr = results_root()
    sw = ROOT / "server_runs" / "weights" / "06-posttrain" / "seed42"
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")

    # ============ calib ============
    ccal = np.load(cand_path("calib"), allow_pickle=True)
    dc = np.asarray([str(d) for d in ccal["date_key"]])
    yc = ccal["true_logret"].astype(np.float64)
    qc = ccal["quality"].astype(bool)
    ps_c = ccal["post_std"].astype(np.float64)

    Hc_t1 = np.load(wr / "hidden-calib-t1-w512.npz", allow_pickle=True)
    Hc_t2 = np.load(wr / "hidden-calib-t2-w512.npz", allow_pickle=True)
    ens_c_t1 = ens_scores(Hc_t1["hidden"], dc, wr, dev)
    ens_c_t2 = ens_scores(Hc_t2["hidden"], dc, wr, dev)
    gh_c = np.load(posttrain_artifacts()["calibration"], allow_pickle=True)["hidden"]
    h6 = _head(posttrain_artifacts()["head_rank_mlp_spearman"], 0.0).to(dev)
    with torch.no_grad():
        p6_c = h6(torch.from_numpy(np.asarray(gh_c).astype(np.float32)).to(dev)).cpu().numpy().astype(np.float64)
    F_c_t1 = compute_F(ens_c_t1, p6_c, dc)
    F_c_t2 = compute_F(ens_c_t2, p6_c, dc)
    BERT_E_c = bert_e(weights_artifact("scores-calib"), ccal, centers, dev)

    # ============ eval ============
    cand = np.load(cand_path("eval"), allow_pickle=True)
    de = np.asarray([str(d) for d in cand["date_key"]])
    ye = cand["true_logret"].astype(np.float64)
    qe = cand["quality"].astype(bool)
    ps_e = cand["post_std"].astype(np.float64)
    dense = int(cand["dense_threshold"][0])

    He_t1 = np.load(wr / "hidden-eval-t1-w512.npz", allow_pickle=True)
    He_t2 = np.load(wr / "hidden-eval-t2-w512.npz", allow_pickle=True)
    ens_e_t1 = ens_scores(He_t1["hidden"], de, wr, dev)
    ens_e_t2 = ens_scores(He_t2["hidden"], de, wr, dev)
    p6_e = np.load(weights_artifact("p6-scores")).astype(np.float64)
    F_e_t1 = compute_F(ens_e_t1, p6_e, de)
    F_e_t2 = compute_F(ens_e_t2, p6_e, de)
    BERT_E_e = bert_e(weights_artifact("scores-eval"), cand, centers, dev)

    # ============ calib 时序劈半 ============
    uniq = np.array(sorted(set(dc)))
    mid = uniq[len(uniq) // 2]
    front = dc < mid
    back = dc >= mid
    print(f"[r2] calib {len(uniq)} dates; front<{mid} ({front.sum()} rows), "
          f"back>={mid} ({back.sum()} rows)", flush=True)

    out = {"schema": "r2-split-rerun-v1", "calib_split_date": str(mid)}

    # ============ 裁决：F_T1 vs F_T2 RankIC（与幅度源无关） ============
    dense_c = max(5, int(np.ceil(0.8 * int(np.unique(dc, return_counts=True)[1].max()))))
    out["rankic"] = {}
    for name, F in (("F_T1", F_c_t1), ("F_T2", F_c_t2)):
        out["rankic"][name] = {
            "front": stats(dc[front], yc[front], qc[front], F[front], dense_c, centers)["rank_ic"],
            "back": stats(dc[back], yc[back], qc[back], F[back], dense_c, centers)["rank_ic"],
            "eval": stats(de, ye, qe, (F_e_t1 if name == "F_T1" else F_e_t2), dense, centers)["rank_ic"],
        }
        print(f"[r2] {name} RankIC front/back/eval = "
              f"{out['rankic'][name]['front']:.4f} / {out['rankic'][name]['back']:.4f} / "
              f"{out['rankic'][name]['eval']:.4f}", flush=True)

    # ============ eval 分块：T1 vs T2 RankIC 是否一致 ============
    out["rankic"]["eval_blocks"] = {}
    for name, F in (("F_T1", F_e_t1), ("F_T2", F_e_t2)):
        out["rankic"]["eval_blocks"][name] = [
            stats(de[bm], ye[bm], qe[bm], F[bm], dense, centers)["rank_ic"]
            for bm in block_masks(de, 4)]
    print("[r2] eval blocks RankIC T1:", [round(x, 4) for x in out["rankic"]["eval_blocks"]["F_T1"]])
    print("[r2] eval blocks RankIC T2:", [round(x, 4) for x in out["rankic"]["eval_blocks"]["F_T2"]])

    # ============ 冠军构造（T1）：front 拟合 / back 选择 / eval 评估 ============
    # 幅度源用 |post_median| 代理 med3（注意：期望型量会收缩，AR 偏低——真实冠军用 med3 采样）
    mag_c = np.abs(ccal["post_median"].astype(np.float64))
    mag_e = np.abs(cand["post_median"].astype(np.float64))
    # 边界混合 λ 网格：在 front 拟合、back 选择（门禁内 DA 最大）
    best = None
    for lam in (0.0, 0.2, 0.25, 0.3, 0.5):
        F0 = fit_boundary_iso(F_c_t1[front], yc[front], qc[front], dc[front])
        pred_back = champion_pred(F_c_t1[back], dc[back], BERT_E_c[back], mag_c[back], lam, F0)
        st = stats(dc[back], yc[back], qc[back], pred_back, dense_c, centers)
        if 0.8 <= st["amp_ratio"] <= 1.2 and st["collapse"] <= 0.30:
            if best is None or st["da"] > best[3]:
                best = (lam, F0, st["da"], st)
    if best is None:
        lam = 0.25
        print("[r2] WARNING: no lambda passed the gate on back-calib (|post_std| 源)",
              flush=True)
    else:
        lam, _, _, _ = best
    F0 = fit_boundary_iso(F_c_t1[front], yc[front], qc[front], dc[front])
    print(f"[r2] back-selected lambda={lam}, F0={F0:.4f}", flush=True)

    # 用 back 选出的 λ、front 拟合的 F0 在 eval 上做最终评估（一次性）
    F0 = fit_boundary_iso(F_c_t1[front], yc[front], qc[front], dc[front])
    pred_eval = champion_pred(F_e_t1, de, BERT_E_e, mag_e, lam, F0)
    st_eval = stats(de, ye, qe, pred_eval, dense, centers)
    # 基线对照：纯 isotonic（PathA，原轮 MAPE 0.0229 但坍缩）
    m_iso = np.isfinite(F_c_t1) & np.isfinite(yc) & qc & front
    iso = IsotonicRegression(out_of_bounds="clip").fit(F_c_t1[m_iso], yc[m_iso])
    pred_iso_eval = np.full_like(F_e_t1, np.nan)
    fin = np.isfinite(F_e_t1)
    pred_iso_eval[fin] = iso.predict(F_e_t1[fin])
    st_iso = stats(de, ye, qe, pred_iso_eval, dense, centers)

    out["champion_T1"] = {
        "lambda": lam, "F0": F0,
        "back": stats(dc[back], yc[back], qc[back],
                      champion_pred(F_c_t1[back], dc[back], BERT_E_c[back], mag_c[back], lam, F0),
                      dense_c, centers),
        "eval": st_eval,
        "eval_isotonic_baseline": st_iso,
        "blocks_eval": [stats(de[bm], ye[bm], qe[bm], pred_eval[bm], dense, centers)
                        for bm in block_masks(de, 4)],
    }
    print(f"[r2] champion eval: RankIC={st_eval['rank_ic']:.4f} DA={st_eval['da']:.4f} "
          f"MAPE={st_eval['mape']:.4f} AR={st_eval['amp_ratio']:.3f} Coll={st_eval['collapse']:.3f}", flush=True)
    print(f"[r2] isotonic baseline eval: MAPE={st_iso['mape']:.4f} AR={st_iso['amp_ratio']:.3f} "
          f"Coll={st_iso['collapse']:.3f}", flush=True)
    print("[r2] blocks eval DA:", [round(b["da"], 4) for b in out["champion_T1"]["blocks_eval"]], flush=True)

    out_path = rr / "r2_split_rerun.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[r2] -> {out_path}")


if __name__ == "__main__":
    main()
