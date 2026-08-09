# T3：采样自洽推理结论

> 对应 `ContinuePreTrain-ToDo.md` T3。纯推理、零训练：对 400 窗协议每个
> (stock, date) 位置从 coarse logits 多项采样 K=8 条路径，对比三种策略：
> argmax（单路径贪心）、vote（K 路径方向多数）、mean（K 路径收益均值）。

## 协议

- 模型：`checkpoints/exp04b_best_ep100.pt`（CPT 正式产出）
- 采样：K=8，temperature=1.0，`torch.multinomial`（无重复采样 from softmax）
- 位置：1,798,899 个 (stock, date) 预测（4543 股票 × 400 窗单日）
- 覆盖：447 个交易日（聚合去重后 443 个可用）
- 解码：coarse token 采样 + fine argmax → decode_all → 还原价格空间 log_ret
- DA/IC 口径与 `compute_windowed_metrics` 一致（per-date sign 匹配 + 横截面 Spearman）
- 统计：paired bootstrap（2000 次重采样，per-date delta 的 95% CI）
- **holdout 未触碰**（offsets 0-399 仅，与正式协议一致）

## 结果（per-date 均值）

| 策略 | mean_DA | median_DA | mean_IC | 相对 argmax |
|------|---------|-----------|---------|-------------|
| argmax（基线） | 49.17% | 49.11% | +0.0323 | — |
| vote（方向多数） | 49.27% | 49.31% | +0.0368 | DA +0.18pp, IC +0.0052 |
| mean（收益均值） | 49.21% | 48.98% | **+0.0419** | DA +0.19pp, **IC +0.0129** |

## Paired bootstrap（443 日期，2000 次重采样）

| 对比 | DA delta | 95% CI | 显著？ | IC delta | 95% CI | 显著？ |
|------|----------|--------|--------|----------|--------|--------|
| vote − argmax | +0.18pp | [−0.12, +0.48] | ❌ | +0.0052 | [−0.0052, +0.0157] | ❌ |
| mean − argmax | +0.19pp | [−0.31, +0.71] | ❌ | **+0.0129** | **[+0.0043, +0.0224]** | ✅ |

## 结论

1. **mean（K 路径收益均值）采样自洽：RankIC 提升显著**（+0.0129，95% CI 不含零），
   DA 提升不显著（+0.19pp，CI 跨零）。
2. **vote（方向多数）两项均不显著**，不推荐。
3. 按 ToDo T3 判据（"acted DA/RankIC 提升显著即成为默认推理配置"）：
   **mean-采样作为默认推理配置被接受（RankIC 维度满足判据）**，DA 维度如实标注不显著。

## 局限与说明

- **单 seed**（seed 42）：DA 单 seed 是 coin-flip 的已知事实（04-A bootstrap），
  这里的显著性检验是**同 seed 内的 paired 对比**（443 日期重采样），
  不是跨 seed 泛化。正式跨 seed 结论需多 seed 复核。
- mean 采样的 IC 提升来自平滑单路径噪声——把 softmax 的次优峰纳入均值，
  减小单路径的离群方向误差。这与模型预测分布的熵（collapse 42%）一致的机制。
- 作为默认推理配置，**对后续 Branch 的推理侧一律采用 mean-采样**（K=8）。

## 产物

- 逐位置明细：`server_runs/results/04b-cpt/seed42/trials/local_cpt/t3_self_consistency/per_position.csv`
  （1,798,899 行：argmax_lr / vote_lr / mean_lr / true_lr）
- 汇总：`.../t3_self_consistency/summary.json`
- 脚本：`experiments/05-cpt/t3_sampling_self_consistency.py`
