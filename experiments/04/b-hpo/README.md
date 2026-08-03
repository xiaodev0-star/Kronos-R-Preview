# Exp 04-B: full-data HPO

```bash
python experiments/04/b-hpo/sweep_hpo.py
```

Requires the reviewed Exp 04-A (optimizer) selection. Every trial uses full
data, retains every epoch checkpoint on the weights side, and writes the full
trajectory and per-stock/date diagnostics on the results side.

## Selection rule (aligned with Exp 03)

Trials are ranked by **token-balance dominated** lexicographic key:
codebook balance (median, p10) → token support F1 → token JSD → effective
token alignment → collapse → unique tokens. Downstream DA/RankIC/MAPE/AmpRatio
serve only as guardrails; the baseline (Exp 04-A selected optimizer) must not
be dominated.

The search follows the optimizer recipe selected by Exp 04-A. Review the
Exp 04-A bundle before starting it.

The final holdout is opened only by:

```bash
python experiments/04/b-hpo/sweep_hpo.py --holdout
```
