# AGENTS.md

## Active experiment line

Historical outputs were removed after two numerical training bugs were fixed.
Do not restore or compare old checkpoints.

```text
Exp 01 bits
  -> reviewed selection
Exp 02 tokenizer architecture
  -> reviewed selection
Exp 03 controlled capacity
  -> reviewed selection
Exp 04-A loss
  -> reviewed selection
Exp 04-B optimizer
  -> reviewed selection
Exp 04-C HPO
```

## Measurement invariants

- Raw daily `Collapse` and raw daily `Unique` token counts are **not** comparable
  across arms whose codebook size differs. A larger codebook slices the same
  data more finely, so Collapse falls and Unique rises mechanically. They are
  comparable only when the bit split is held fixed, and even then they drift
  with how much of the codebook the tokenizer fills.
- Codebook-size-invariant quality: `avg_da_per_date`, `avg_daily_rank_ic`,
  `avg_mape`, and their per-window floors `min_window_da` /
  `min_window_daily_rank_ic`.
- Capacity-normalized behaviour: `median_daily_{collapse,unique_token,
  effective_token,distribution}_alignment`, `median_daily_token_support_f1`, and
  `median_daily_codebook_balance_score` with its `p10_daily_*` worst-decile
  form. These compare a prediction against its **same-day target**.
- Learning evidence: `I_learn = nominal_codebook_bits - CE` in bits, computed
  as `(bits_l1 + bits_l2) - (val_coarse_loss + val_fine_loss)/log(2)`.
  The old `H(target) - CE` form is invalid because CE >= empirical target
  entropy. Exp 01 measured ~1.3 bits/token of joint learned information across
  a 64x range of joint vocabularies, so codebook size is not the capacity
  bottleneck. The flatness is an empirical saturation finding, not an a-priori
  vocabulary-invariance property of the metric.
- `avg_da_above_baseline` compares against a per-day oracle that already knows
  the majority direction. Negative values are expected and are not a failure.

## Numerical invariants

- `RotaryEmbedding.forward` constructs frequency angles and `sin`/`cos` in
  FP32 with autocast disabled. `_apply_rope` alone casts them to Q/K dtype.
- Training fine token `t` is teacher-conditioned on current target coarse
  token `t`; inference uses current predicted coarse token `t`.
- Fine code `0` is valid. Only `-100` is ignored for fine EOS/padding.

Run before formal work:

```bash
python experiment_io.py --require-cuda
```

## Output invariant

All runners use two roots:

- `KRONOS_WEIGHTS_ROOT`: server-only checkpoints and caches.
- `KRONOS_RESULTS_ROOT`: downloadable JSON/CSV/NPZ, plots, and logs.

The results tree must never contain `.pt`, `.pth`, `.ckpt`, or prepared-input
caches. Export-oriented runners may write `download_manifest.json` to inventory
this boundary; local-only Exp 07 runs must enforce the same boundary in memory
without generating a download manifest.

Every formal run must record source hashes, runtime/GPU/package metadata,
dataset fingerprint, training history, update schedule, exact dataset target
distributions, full evaluation distributions, and per-stock/date prediction
records.

## Protocol invariants

- Every runner evaluates the full pre-holdout region at single-day resolution:
  400 contiguous 1-day windows (offsets 0,1,...,399) score all 400 pre-holdout
  days, so no post-cutoff day before the holdout is skipped. `VALIDATION_OFFSETS`
  is derived as `range(0, HOLDOUT_OFFSET, EVAL_DAYS)` with `EVAL_DAYS=1`; do not
  revert to the old sparse 0/100/200/300 sampling. The window count only sets the
  robustness-stat granularity; per-date records preserve daily detail regardless.
- Holdout offset 400 stays sealed until the explicit final command.
- All bootstrap comparisons use root `bootstrap_utils.py`; experiment packages
  must not duplicate bootstrap code or import it from another experiment.
- Follow `PROTOCOL_LOCK.md` for validation/holdout use: fit/calib selects, the
  400-day window validates the final locked model once, and holdout is unsealed
  exactly once.
- Controlled comparisons use a local loader seed and exact accumulation
  boundaries.
- Do not consume a proposal as an upstream decision. Downstream scripts require
  reviewed `selection.json` with `upstream_eligible=true`.
- Compare arms only when data fingerprint, source fingerprint, RoPE/fine-head
  invariants, optimizer-step count, and validation windows match.
- Token/prepared caches live on the weights side and are disposable.

The GPT line is active. BERT and ELECTRA remain deferred.
