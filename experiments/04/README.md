# Exp 04: optimizer ablation and HPO

Exp 04 requires the reviewed Exp 02 and Exp 03 selections. Cross-entropy is
fixed (decided without a separate loss ablation), so the series starts at the
optimizer ablation.

Run one stage at a time:

```bash
python experiments/04/a-optimizer-ablation/sweep_optimizer.py
python experiments/04/b-hpo/sweep_hpo.py
```

The A run writes `selection_proposal.json`, not an automatically eligible
decision. Download its result tree, review all mature-window, per-date, token
distribution, and per-stock records, then record the chosen arm:

```bash
python experiments/04/a-optimizer-ablation/sweep_optimizer.py \
  --analyze_only --record_selection \
  --select_arm muon --selection_rationale "reviewed rationale"
```

Default result directories:

```text
$KRONOS_RESULTS_ROOT/04a-optimizer-ablation/seed42/
$KRONOS_RESULTS_ROOT/04b-hpo/seed42/
```

Corresponding checkpoints and caches stay below `KRONOS_WEIGHTS_ROOT`.
