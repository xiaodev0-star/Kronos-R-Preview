# Exp 04-A: optimizer ablation

```bash
python experiments/04/a-optimizer-ablation/sweep_optimizer.py
```

Cross-entropy is fixed (decided without a separate loss ablation). Both arms
inherit the Exp 02 tokenizer and Exp 03 architecture and compare AdamW with
Muon+AdamW.

## Selection rule (aligned with Exp 03)

Each arm is summarized over its **full epoch trajectory** (50 epochs) as a
single window. The primary ranking is **token-balance dominated**:

1. `median_daily_codebook_balance_score` (coarse)
2. `p10_daily_codebook_balance_score`
3. `median_daily_token_support_f1`
4. `median_daily_token_jsd` (lower better)
5. `median_daily_effective_token_alignment`
6. `median_daily_collapse_rate` (lower better)
7. `median_daily_unique_tokens`

Downstream DA / RankIC / MAPE / AmpRatio serve only as **guardrails** (must
not regress beyond `max(floor, 2x reference within-window SD)` vs the AdamW
reference arm). Fine and joint codebook balance are reported as secondary
diagnostics.

Paired moving-block bootstrap (10000 replicates, 5-day block) is computed
over all epochs; it is diagnostic only and does not affect ranking.

## AdamW arm reuse

The AdamW arm reuses the Exp 03 depth6 trajectory (50 epochs, identical
recipe: `lr=3e-4, dropout=0.1, wd=0.01, fine_weight=0.3, het_weight=0.1,
warmup=0.05, batch_tokens=6144`). Weights are linked via NTFS junction;
trajectory JSONs are copied under `arms/adamw/epoch_trajectory/`. Only the
Muon arm needs to be trained (50 epochs for like-for-like comparison).

No arm becomes a downstream dependency until it is explicitly recorded after
reviewing the results:

```bash
python experiments/04/a-optimizer-ablation/sweep_optimizer.py \
  --analyze_only --record_selection \
  --select_arm muon --selection_rationale "reviewed rationale"
```
