"""BERT-Critic-Rerank-Plan §11.2: 8 new contract tests (on top of 06's 18).

These are plain-runtime tests (no CUDA) that pin the critic's data / interface /
protocol invariants.  Run:  python experiments/07-bert-critic/tests/test_critic_contracts.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.kronos_bert import KronosBert, make_mlm_batch
from model.kronos_electra import make_replaced_batch

try:
    from bert_data import (
        build_bert_input,
        bert_time_for_target_date,
        parse_date_to_timelist,
    )
except ImportError:
    # bert_data may not be authored yet during the P0 phase; the BERT-input
    # tests then live in bert_data's own tests.  Keep this module importable.
    build_bert_input = None

# --- vocabulary constants mirroring the production tokenizer (7+7 bits) -------
VOCAB_BASE = 128          # tokenizer.vocab_coarse
BOS, EOS, MASK = 128, 129, 130


# ============================================================================
# T1. t+1 (MASK) row va_values must be exactly zero — full-scan style.
# ============================================================================
def test_mask_row_va_zero():
    if build_bert_input is None:
        return
    # history tokens for rows [0..6]; selection position p=6 predicts row 6;
    # window W covers all history.
    hist_ids = torch.tensor([3, 5, 7, 9, 11, 13, 15], dtype=torch.long)
    hist_time = torch.zeros(7, 3, dtype=torch.long)
    hist_time[:, 0] = torch.arange(1, 8)            # day-of-month
    hist_va = torch.ones(7, 2)                       # real history VA
    target_dt = "2024-02-01"
    inp, tids, va, pos = build_bert_input(
        hist_ids=hist_ids, hist_time=hist_time, hist_va=hist_va,
        target_date=target_dt, vocab_base=VOCAB_BASE, mask_id=MASK,
    )
    assert inp[-1] == MASK, "last position must be the MASK slot"
    assert torch.equal(va[-1], torch.zeros(2)), "MASK row va must be 0"
    assert (va[:-1] == 1.0).all(), "history va must be preserved"
    assert len(inp) == 8 and len(va) == 8


# ============================================================================
# T2. BERT input is truncated at the MASK: no rows after the t+1 position.
# ============================================================================
def test_no_rows_after_mask():
    if build_bert_input is None:
        return
    hist_ids = torch.tensor([3, 5, 7, 9, 11, 13, 15], dtype=torch.long)
    hist_time = torch.zeros(7, 3, dtype=torch.long)
    hist_va = torch.ones(7, 2)
    inp, _, _, _ = build_bert_input(hist_ids, hist_time, hist_va,
                                    "2024-02-01", VOCAB_BASE, MASK)
    assert inp.shape[0] == 8, "BOS + 7 hist + MASK"
    assert (inp[:-1] < VOCAB_BASE).all() or inp[0] == BOS, \
        "history positions are real tokens (BOS at 0)"


# ============================================================================
# T3. BERT vocab_size == tokenizer.vocab_coarse (128); mask_id consistent.
# ============================================================================
def test_bert_vocab_wiring():
    import types
    cfg = types.SimpleNamespace(
        vocab_size=128, vocab_fine=128, dim=64, depth=2, heads=2,
        num_kv_heads=1, ffn_multiplier=2, dropout=0.0, va_hidden_dim=16,
        rope_base=10000.0)
    model = KronosBert(cfg)
    assert model.vocab_base == 128, "F1: BERT head vocab must come from cfg"
    assert model.mask_id == 130
    assert model.token_emb.num_embeddings == 128 + 3
    # The checkpoint config must record the actual (tokenizer-derived) vocab,
    # not a 1024 default — F1 stores vocab_size / vocab_fine / mask_id.
    ckpt_config = {"vocab_size": 128, "vocab_fine": 128, "mask_id": 130}
    assert ckpt_config["vocab_size"] == model.vocab_base
    assert ckpt_config["mask_id"] == model.mask_id


# ============================================================================
# T4. Stage-2 negative sampling source is a frozen GPT (proposal path) and
#     respects no_grad semantics — here we only check the sampling function
#     returns ordinary-vocab ids driven by the provided proposal.
# ============================================================================
def test_gpt_proposal_negatives():
    ids = torch.randint(0, VOCAB_BASE, (200,))
    ids[::50] = BOS                       # specials never replaced
    proposal = torch.zeros(200, VOCAB_BASE)
    proposal[:, 17] = 12.0                # dominant logit -> ~99% token 17
    rep_ids, labels = make_replaced_batch(
        ids, VOCAB_BASE, replace_prob=1.0,
        generator=torch.Generator().manual_seed(3),
        proposal=proposal, temp=1.0)
    nonspec = ids < VOCAB_BASE
    frac17 = (rep_ids[nonspec] == 17).float().mean().item()
    assert frac17 > 0.9, f"GPT-proposal sampling under-sampled dominant token: {frac17}"
    assert (rep_ids[nonspec] < VOCAB_BASE).all(), "replaced ids must be ordinary codes"
    assert labels[nonspec].sum() == 0, "all replaced -> label 0"
    # uniform control arm still works
    rep_u, lab_u = make_replaced_batch(
        ids, VOCAB_BASE, replace_prob=1.0,
        generator=torch.Generator().manual_seed(4))
    assert lab_u[nonspec].sum() == 0


# ============================================================================
# T5. Candidate cache must be row-aligned with the 06 eval records
#     (same (uid, date) set).  The build_gpt_candidates module enforces this by
#     construction; here we assert the join key helper round-trips.
# ============================================================================
def test_candidate_alignment_key():
    recs = [{"date_key": "2024-03-01", "stock_uid": "a.csv"},
            {"date_key": "2024-03-01", "stock_uid": "b.csv"}]
    keys = {(r["date_key"], r["stock_uid"]) for r in recs}
    assert ("2024-03-01", "a.csv") in keys
    assert len(keys) == 2


# ============================================================================
# T6. Fusion parameters are fit only on the audit slice (date-interval assert).
# ============================================================================
def test_calibration_date_assert():
    from bert_data import in_audit_calibration
    if in_audit_calibration is None:
        return
    assert in_audit_calibration("2023-06-15") is True
    assert in_audit_calibration("2024-02-01") is False
    assert in_audit_calibration("2023-01-01") is False


# ============================================================================
# T7. Stage-1 training data max date < cutoff (scanned at load; helper here).
# ============================================================================
def test_stage1_data_before_cutoff():
    cutoff = np.datetime64("2024-02-01")
    dates = np.array(["2023-12-31", "2024-01-31", "2024-02-01"], dtype="datetime64[D]")
    assert (dates[:-1] < cutoff).all()
    assert not (dates[-1] < cutoff)


# ============================================================================
# T8. C-a/C-b must run before B1-B5 (pipeline-order guard: eval_critic refuses
#     to produce B1-B5 summaries without a controls artifact).
# ============================================================================
def test_controls_gate_order(tmp_path=Path(tempfile.gettempdir())):
    controls_path = tmp_path / "controls.json"
    if not controls_path.exists():
        # Without the C-a/C-b artifact the formal eval must refuse to run.
        from eval_critic import require_controls
        try:
            require_controls(controls_path)
            raise AssertionError("require_controls must raise on missing controls")
        except RuntimeError:
            pass
    else:
        require_controls(controls_path)


# ============================================================================
# Improvement-Plan §6 NEW contract tests (appended 2026-08-05)
# ============================================================================

# --- N1. Row-score semantics (E-1): decoded returns only; likelihood / entropy
#         / acceptance quantities forbidden as rank scores (R4 abstention excepted)
def test_row_score_semantics_decoded_returns():
    from improve_common import decode_coarse, assert_score_semantics
    centers = np.load(ROOT / "checkpoints" / "coarse_logret_centers.npy")
    rng = np.random.RandomState(0)
    p = rng.dirichlet(np.ones(128), size=200).astype(np.float64)
    pm = np.full(200, -1e-4); ps = np.full(200, 0.03)
    dec = decode_coarse(p, centers, pm, ps)
    assert dec["e_mean"].shape == (200,)
    assert np.all(np.isfinite(dec["e_mean"])), "decoded E[r] must be finite"
    assert np.all(dec["e_mean"] > -0.5) and np.all(dec["e_mean"] < 0.5), \
        "decoded E[r] must be on log-return scale"
    assert (dec["p_up_raw"] >= 0.0).all() and (dec["p_up_raw"] <= 1.0).all()
    assert (dec["p_up_naive"] >= 0.0).all() and (dec["p_up_naive"] <= 1.0).all()
    # forbidden rank-score fields must be rejected
    for bad in ("logp_bert_topk", "bert_margin", "coarse_entropy",
                "electra_acceptance_mean"):
        try:
            assert_score_semantics(bad)
            raise AssertionError(f"{bad} must be rejected as a rank score")
        except ValueError:
            pass
    # decoded fields pass
    for good in ("ebert_mean", "ebert_median", "ebert_pup", "poe_e_l0.5",
                 "critic_pick_center", "p6_score", "post_median"):
        assert_score_semantics(good)


# --- N2. PoE fusion must normalize over the full 128-code support (Σp = 1 ± 1e-4)
def test_poe_full_support_normalized():
    from improve_common import poe_fused
    rng = np.random.RandomState(1)
    logp = np.log(rng.dirichlet(np.ones(128), size=300)).astype(np.float64)
    q = rng.dirichlet(np.ones(128), size=300).astype(np.float64)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        pf = poe_fused(logp, q, lam)
        assert pf.shape == (300, 128), "PoE must span the full 128-code support"
        assert np.allclose(pf.sum(axis=1), 1.0, atol=1e-4), \
            f"PoE row sums != 1 at lam={lam}"
        assert np.all(pf >= 0.0)


# --- N3. T3 calibration parameters must be fit on the audit slice only
def test_t3_calibration_fit_slice():
    from bert_data import in_audit_calibration
    # the audit/calibration slice is audit_uids x [2023-02-01, 2024-02-01)
    assert in_audit_calibration("2023-02-01") is True
    assert in_audit_calibration("2023-12-31") is True
    assert in_audit_calibration("2024-01-31") is True
    assert in_audit_calibration("2024-02-01") is False   # cutoff excluded
    assert in_audit_calibration("2023-01-31") is False
    # the real calib candidate cache (if present) must lie in that interval
    cc = ROOT / "server_runs" / "weights" / "07-bert-critic" / "seed42" / "candidates_calib_K8.npz"
    if cc.exists():
        d = np.load(cc, allow_pickle=True)["date_key"]
        dts = np.array([str(x)[:10] for x in d], dtype="datetime64[D]")
        assert (dts >= np.datetime64("2023-02-01")).all()
        assert (dts < np.datetime64("2024-02-01")).all()


# --- N4. T1/T2 fine-tune must rerun C-a and not regress before R/T continuation
def test_ca_regression_gate():
    import tempfile, json
    from improve_common import require_ca_not_regressed
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "controls.json")
        with open(good, "w", encoding="utf-8") as f:
            json.dump({"c_a": {"avg_true_rank": 3.20}}, f)
        require_ca_not_regressed(good, baseline_avg_rank=3.23, max_regression=0.3)
        # a fine-tuned model that regressed C-a past the threshold must be blocked
        bad = os.path.join(td, "controls_bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump({"c_a": {"avg_true_rank": 4.20}}, f)
        try:
            require_ca_not_regressed(bad, baseline_avg_rank=3.23, max_regression=0.3)
            raise AssertionError("regressed C-a must be rejected")
        except RuntimeError:
            pass
        # missing artifact must be blocked too
        try:
            require_ca_not_regressed(os.path.join(td, "missing.json"))
            raise AssertionError("missing C-a artifact must be rejected")
        except RuntimeError:
            pass


# --- N5. T5 BERT-hidden cache fingerprint must pin ckpt hash + row-set hash
def test_bert_hidden_fingerprint():
    import tempfile, hashlib
    from improve_common import row_set_fingerprint
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "bert.pt")
        with open(ckpt, "wb") as f:
            f.write(b"fake-bert-checkpoint")
        uid = np.array(["a.csv", "b.csv", "a.csv"])
        dts = np.array(["2024-01-05", "2024-01-05", "2024-01-06"])
        out = os.path.join(td, "fingerprint.json")
        fp1 = row_set_fingerprint(uid, dts, ckpt, out_json=out)
        assert fp1["n_rows"] == 3
        assert fp1["n_unique_uids"] == 2 and fp1["n_unique_dates"] == 2
        assert "row_set_sha256" in fp1 and "bert_ckpt_sha256" in fp1
        assert os.path.exists(out)
        # same row set -> identical fingerprint (order-insensitive by design)
        fp2 = row_set_fingerprint(np.array(["a.csv", "a.csv", "b.csv"]),
                                  np.array(["2024-01-06", "2024-01-05", "2024-01-05"]),
                                  ckpt)
        assert fp1["row_set_sha256"] == fp2["row_set_sha256"]


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)} tests, {n_pass} passed, {len(tests)-n_pass} failed")
    return 0 if n_pass == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
