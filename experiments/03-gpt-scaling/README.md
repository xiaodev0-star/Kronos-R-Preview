# Exp 03: controlled GPT capacity

```bash
python experiments/03-gpt-scaling/run_gpt_capacity.py
```

The controlled grid is:

| Config | dim | depth | heads | KV heads |
|---|---:|---:|---:|---:|
| deep | 256 | 4 | 4 | 1 |
| depth6 | 256 | 6 | 4 | 1 |
| depth8 | 256 | 8 | 4 | 1 |
| width384_d4 | 384 | 4 | 6 | 1 |
| width512_d4_kv1 | 512 | 4 | 8 | 1 |
| xlarge | 512 | 4 | 8 | 2 |

All configurations share the reviewed Exp 02 tokenizer, exact data ordering,
full sequences, and optimizer-step schedule. Coarse, fine, and joint targets
and predictions are recorded exactly for every checkpoint. Per-stock/date
records permit offline confusion and cross-sectional analysis.

```text
$KRONOS_WEIGHTS_ROOT/03-gpt-scaling/seed42/
$KRONOS_RESULTS_ROOT/03-gpt-scaling/seed42/
```

The run writes `selection_proposal.json`. After downloading and reviewing the
bundle, promote it explicitly:

```bash
python experiments/03-gpt-scaling/analyze_gpt_capacity.py \
  --record_selection --rationale "reviewed rationale"
```

Only reviewed `selection.json` is accepted by Exp 04.
