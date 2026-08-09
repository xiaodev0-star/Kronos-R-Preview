"""t3_calibration.py — plan §4 T3: calibration of the BERT critic.

Fits ONLY on the audit/calibration slice (audit_uids x [2023-02-01, 2024-02-01));
0..399 is pure inference.  Two calibrations, both inheriting the 06 protocol:

  T_b temperature  single temperature on p_BERT minimizing the NLL of the
                   realized coarse token on the calib slice (mirrors 06 T_c/T_f).
  P(up) calibration isotonic regression of P_BERT(raw r > 0) against the
                   realized direction (true_logret > 0); Platt as fallback.

Acceptance (plan §4 T3): NLL/ECE improvement on calib + fusion arms (R2/R5)
non-inferior after substituting the calibrated P(up).  Outputs are written to
``results_root / t3_calibration.json`` and the fitted T_b / isotonic mapping
are the only calibration parameters that may enter later fusion/stacking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEVEN = Path(__file__).resolve().parent
SIX = ROOT / "experiments" / "06-posttrain"
for _p in (ROOT, SEVEN, SIX):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from improve_common import (  # noqa: E402
    EPS, load_centers, softmax_rows, decode_coarse, cand_path, scores_path,
    results_root, weights_root, build_rec, write_json_ledger,
)
from critic_common import append_trial  # noqa: E402


def _nll(logp_true, T):
    """NLL of realized tokens under temperature T: -mean(log_softmax(logp/T)[true])."""
    l = logp_true.astype(np.float64) / float(T)
    m = l.max()
    logz = m + np.log(np.exp(l - m).sum())
    return float(-(l - logz).mean())


def _ece(prob, y, n_bins=10):
    prob = np.clip(prob, 0.0, 1.0)
    idx = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        pbar = prob[m].mean()
        fbar = y[m].mean()
        ece += (m.sum() / len(prob)) * abs(pbar - fbar)
    return float(ece)


def fit_tb(logp_calib, true_coarse, n_grid=400):
    """Single temperature on the calib slice (minimize realized-token NLL)."""
    from scipy.optimize import minimize_scalar
    good = np.isfinite(logp_calib).all(axis=1) & (true_coarse >= 0) & (true_coarse < 128)
    lp_true = logp_calib[good][np.arange(good.sum()), true_coarse[good].astype(int)]
    nll_before = _nll(lp_true, 1.0)
    res = minimize_scalar(lambda t: _nll(lp_true, t), bounds=(0.05, 10.0),
                          method="bounded", options={"xatol": 1e-4})
    Tb = float(res.x)
    nll_after = _nll(lp_true, Tb)
    return Tb, nll_before, nll_after, int(good.sum())


def _ece_brier(ll, yy, Tb=None, mapping=None):
    """ECE + Brier for a P(up) array (optionally post-transform)."""
    p = ll if mapping is None else mapping.predict(np.asarray(ll)[:, None])
    if Tb is not None:
        p = np.clip(p, 0.0, 1.0)
    ece = _ece(p, yy)
    brier = float(np.mean((p - yy) ** 2))
    eps = 1e-12
    llv = -float(np.mean(yy * np.log(np.clip(p, eps, 1))
                         + (1 - yy) * np.log(np.clip(1 - p, eps, 1))))
    return ece, brier, llv


def run_t3(out_json=None):
    centers = load_centers()
    cc = np.load(cand_path("calib"), allow_pickle=True)
    cs = np.load(scores_path("calib"), allow_pickle=True)
    rec = build_rec(cc)
    logp = cs["logp_bert_full"]
    true_c = np.asarray(cc["true_coarse_id"], dtype=np.int64)
    dec = decode_coarse(softmax_rows(logp), centers, rec["p_mean0"], rec["p_std0"])
    pup_raw = dec["p_up_raw"]
    y = (np.asarray(rec["true_logret"]) > 0).astype(np.float64)
    valid = np.isfinite(pup_raw) & np.isfinite(y)

    # ---- T_b temperature (realized-token NLL) ----
    Tb, nll0, nll1, n_valid = fit_tb(logp, true_c)
    print(f"[t3] T_b = {Tb:.4f}  NLL 1.0 -> {nll0:.4f} / T_b -> {nll1:.4f} "
          f"(n={n_valid})")

    # ---- P(up) isotonic / Platt ----
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    X = pup_raw[valid].astype(np.float64)[:, None]
    yy = y[valid]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(X.ravel(), yy)
    platt = LogisticRegression(C=1e6)
    platt.fit(X, yy)
    pup_cal_iso = iso.predict(X)
    pup_cal_platt = platt.predict_proba(X)[:, 1]

    ece0, brier0, ll0 = _ece_brier(pup_raw[valid], yy)
    ece_iso, brier_iso, ll_iso = _ece_brier(pup_raw[valid], yy, mapping=iso)
    ece_platt, brier_platt, ll_platt = _ece_brier(pup_raw[valid], yy, mapping=platt)
    print(f"[t3] P(up): raw  ECE {ece0:.4f} Brier {brier0:.4f} LL {ll0:.4f}")
    print(f"[t3]        iso  ECE {ece_iso:.4f} Brier {brier_iso:.4f} LL {ll_iso:.4f}")
    print(f"[t3]        platt ECE {ece_platt:.4f} Brier {brier_platt:.4f} LL {ll_platt:.4f}")

    # ---- eval (0..399) pure-inference ECE/Brier of P_BERT(up) raw vs calibrated
    eval_res = None
    cand_e = np.load(cand_path("eval"), allow_pickle=True)
    sc_e = np.load(scores_path("eval"), allow_pickle=True)
    rec_e = build_rec(cand_e)
    logp_e = sc_e["logp_bert_full"]
    dec_e = decode_coarse(softmax_rows(logp_e), centers, rec_e["p_mean0"], rec_e["p_std0"])
    pup_e = dec_e["p_up_raw"]
    y_e = (np.asarray(rec_e["true_logret"]) > 0).astype(np.float64)
    v_e = np.isfinite(pup_e) & np.isfinite(y_e)
    eval_res = {
        "raw": dict(zip(("ece", "brier", "logloss"),
                        _ece_brier(pup_e[v_e], y_e[v_e]))),
        "isotonic": dict(zip(("ece", "brier", "logloss"),
                             _ece_brier(pup_e[v_e], y_e[v_e], mapping=iso))),
        "n_valid": int(v_e.sum()),
    }
    print(f"[t3] eval P(up): raw {eval_res['raw']}  iso {eval_res['isotonic']}")

    result = {
        "schema": "t3-calibration-v1",
        "fitted_on": "calibration_slice_audit_2023_02_2024_02",
        "T_b": {"value": Tb, "nll_at_1_0": nll0, "nll_at_Tb": nll1,
                "n_realized_tokens": n_valid},
        "P_up": {
            "raw": {"ece": ece0, "brier": brier0, "logloss": ll0},
            "isotonic": {"ece": ece_iso, "brier": brier_iso, "logloss": ll_iso,
                         "calib_n_fit_rows": int(valid.sum())},
            "platt": {"ece": ece_platt, "brier": brier_platt, "logloss": ll_platt},
        },
        "eval_0_399_inference": eval_res,
        "holdout_used": False,
    }
    if out_json:
        write_json_ledger(out_json, result, "t3_calibration", T_b=Tb)
    # persist fitted params for downstream fusion (R5-v2 / stacking)
    import pickle
    fitted_path = weights_root() / "t3_fitted_params.pkl"
    with open(fitted_path, "wb") as f:
        pickle.dump({"T_b": Tb, "isotonic": iso}, f)
    print(f"[t3] fitted params -> {fitted_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description="T3 calibration (T_b + P(up))")
    args = ap.parse_args()
    res = run_t3(out_json=results_root() / "t3_calibration.json")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
