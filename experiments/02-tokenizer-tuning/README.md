# Exp 02: tokenizer architecture sweep

```bash
python experiments/02-tokenizer-tuning/run_tokenizer_sweep.py
```

Exp 02 requires the reviewed Exp 01 bit selection. It fixes those bits, sweeps
the tokenizer encoder/decoder capacity, and records the complete tokenizer and
downstream GPT trajectories. Its 24 GiB server defaults match Exp 01: 65,536
tokens per adaptive GPT microbatch and evaluation batch size 32.

## Grid

`embedding_dim x hidden_dim`:

```text
48x192  48x256  64x192  64x256  96x192  96x256      original grid
96x384  128x256 128x384                             capacity-extension tier
```

The extension tier exists because Exp 01 showed the binding constraint at fixed
bits is **codebook occupancy**, not nominal vocabulary size: at
`64x192` the coarse layer of the selected split filled only part of its nominal
codes. The sweep has to reach far enough to distinguish "the encoder is too
small" from "this codebook is intrinsically hard to fill". Use `--configs` to run
a subset.

## What Exp 02 measures, and why not raw Collapse/Unique

Bits are identical across arms, so `avg_da_per_date`, `avg_daily_rank_ic`, and
`avg_mape` are directly comparable. Behaviour is not: each architecture fills the
fixed codebook to a different degree, which moves the **target** distribution as
well as the prediction. The analyzer therefore reads four separate groups and
never merges them into a score:

1. **Codebook-invariant quality** — late-median DA / daily RankIC / MAPE plus the
   per-window floors `min_window_da` and `min_window_daily_rank_ic`.
2. **Capacity-normalized behaviour** — collapse / unique / effective / distribution
   alignment against the same-day target, support F1, and the codebook balance
   score with its worst-decile `p10` form. Raw Collapse and raw Unique counts are
   still exported but only as descriptive context.
3. **Tokenizer sufficiency** — reconstruction MAE/RMSE plus per-layer code
   utilization and effective code counts, exported for coarse, fine, **and**
   joint levels. Exp 01 exported only the joint aggregate, which is exactly why a
   badly under-filled coarse layer stayed invisible.
4. **Predictive information** — `H(target) - CE` in bits, which is invariant to
   vocabulary size. Exp 01 found it saturated near 1.3 bits/token across a 64x
   range of joint vocabularies, so it is the reference for judging whether better
   occupancy is signal or a longer unpredictable tail.

`analysis.json` also stores a **staged screen**: Stage A applies preregistered
quality floors on every window, Stage B ranks the survivors on normalized
behaviour, and Stage C reports tokenizer sufficiency. It is an ordered shortlist
for review, not a winner. Selecting an arm that failed Stage A is allowed but
prints a warning and is recorded in `selection.json`.

Heavy and downloadable roots are:

```text
$KRONOS_WEIGHTS_ROOT/02-tokenizer-tuning/seed42/
$KRONOS_RESULTS_ROOT/02-tokenizer-tuning/seed42/
```

After reviewing the downloaded results:

```bash
python experiments/02-tokenizer-tuning/analyze_tokenizer_epochwise.py \
  --select_config 64x192 --rationale "reviewed rationale"
```

The resulting reviewed `selection.json` is consumed by Exp 03.
