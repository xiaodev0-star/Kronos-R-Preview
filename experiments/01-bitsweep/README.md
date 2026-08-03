# Exp 01: tokenizer bit-width sweep

```bash
python experiments/01-bitsweep/run_bitsweep.py
```

The runner trains every registered bit pair, tokenizer, and controlled
downstream GPT. Every GPT epoch is evaluated over the full pre-holdout region at
single-day resolution: 400 contiguous 1-day windows (offsets 0,1,...,399) score
all 400 pre-holdout days, so every trading day before the sealed offset-400
holdout is scored as its own window.
The formal defaults use a 65,536-token adaptive GPT microbatch and evaluation
batch size 32, tuned conservatively for a 24 GiB RTX 4090. The effective
optimizer batch remains fixed in sequences, so these settings improve GPU
occupancy without changing the controlled update schedule.

Heavy artifacts:

```text
$KRONOS_WEIGHTS_ROOT/01-bitsweep/seed42/
```

Downloadable bundle:

```text
$KRONOS_RESULTS_ROOT/01-bitsweep/seed42/
```

The bundle includes tokenizer training history and exact tokenizer
coarse/fine/joint counts, GPT target distributions, every epoch's full
prediction counts, per-date metrics, and per-stock/date records.

The formal run does not silently choose bits. After external review:

```bash
python experiments/01-bitsweep/narrate_bitsweep.py \
  --select_config 7+7 --rationale "reviewed rationale"
```

This creates the reviewed `selection.json` required by Exp 02. Re-running the
command with a different `--select_config` overwrites the decision, so a
re-review after a measurement correction is a single command.

## Decision report

```bash
python experiments/01-bitsweep/report_bitsweep.py
```

Writes `REPORT.md` plus `plots/report/fig1..fig8`, which recompute the whole
decision from `combined_epoch_summary.json`, `configs/*/tokenizer_metrics.json`
and `configs/*/dataset_token_summary.json`. The report is the argument for the
chosen bit split; `NARRATIVE.md` only records it.

## Reading the bundle without a codebook-size confound

Raw daily `Collapse` and raw daily `Unique` counts are **not** comparable
between bit configurations. A larger codebook slices the same data more finely,
which mechanically lowers Collapse and raises Unique regardless of whether the
GPT learned anything. Compare instead:

- `avg_da_per_date`, `avg_daily_rank_ic`, and `avg_mape` — invariant to
  vocabulary size — together with their per-window floors `min_window_da` and
  `min_window_daily_rank_ic`;
- `median_daily_*_alignment`, `median_daily_token_support_f1`, and
  `median_daily_codebook_balance_score` (plus its `p10_*` worst-decile form),
  which measure the prediction against the **same-day target** distribution;
- `H(target) - CE` in bits, using `dataset_token_summary.json` for the marginal
  target entropy and `val_coarse_loss`/`val_fine_loss` for the cross-entropy.
  This is the vocabulary-invariant statement of how much structure the GPT
  actually captured.

Every one of these fields is already present in
`combined_epoch_summary.json`.
