# Exp 03 · GPT 容量与 9+7 Tokenizer 的匹配

> **状态**：正式实验已完成（2026-07-26~27，seed=42；5 个架构 × 50 epoch = 250 个 checkpoint 全部评估）。
>
> **修订结论**：Exp 03 的首要问题不是“哪个模型 DA 最高”，而是“哪个 GPT 容量最能利用 Exp 01/02 已确定的 tokenizer，同时没有把多样性改善换成明显的泛化损失”。
>
> **旧协议下的领先候选**：`xlarge`（dim=512 / depth=4 / heads=8 / kv_heads=2，16.96M 参数）的成熟区间 **epoch 16–20**，代表 checkpoint 为 **epoch 18**。
>
> **三个不同角色**：`deep@46` 是计算效率拐点；`xlarge@18` 是容量—利用—泛化平衡点；`xlarge@44` 是纯码本利用率峰值。
>
> **重要否定结论**：没有任何受测配置真正“驾驭”了 9+7 tokenizer。`xlarge@18` 也只覆盖真实日目标 support 的约 24%，有效 token 数约为真实分布的 19%，日 collapse 中位数仍约为真实目标的 3 倍。
>
> **2026-07-27 协议审计**：旧 adaptive microbatch 会跨越梯度累积边界，loader 顺序还受架构相关的全局 RNG 状态影响。五个配置实际执行了 3934–4104 个 optimizer steps，并非相同的 4160；因此本页的架构排序是有价值但带混杂的历史结果，必须由 [`../03-gpt-scaling-sup/`](../03-gpt-scaling-sup/) 的六点统一重训复核后，才能冻结容量。

历史选型记录见 [`run_seed42/selection.json`](run_seed42/selection.json)，完整自动分析见 [`run_seed42/ANALYSIS.md`](run_seed42/ANALYSIS.md)。它们未因事后审计而改写原始数值，但其“selected”状态应解释为 Sup 复核前的候选。Holdout offset 400 未开启。

---

## 1. 对本次质疑的判断

| 观点 | 评估 | 必要修正 |
|---|---|---|
| Exp 03 应回答 GPT 能否利用既定 tokenizer，而不是再次用 DA/MAPE 选型 | **基本正确** | DA/MAPE 不应进入码本利用分数，但不能完全丢弃；它们要作为“新增多样性是否有用”的 guardrail |
| Tokenizer 对 GPT 的主要影响应体现在 Collapse 与 Unique | **方向正确，但指标不充分** | 只看 Unique 会奖励随机撒 token；只看 Collapse 不知道众数是否正确。必须相对真实 token 分布加入 support、熵/有效词表与 JSD |
| AmpRatio 更应由 GPT 训练/HPO 改进 | **正确** | 它不属于码本利用率本身，但仍是输出校准 guardrail；不能因为后续会 HPO 就把严重失真当作无关 |
| 应从全部 epoch 中找最理想 checkpoint，而不是只读最终 epoch | **正确** | 不能直接从 50 次观察中 cherry-pick 单点；应先定义成熟区间，再用连续多 epoch 平台选代表点 |
| Exp 03 应确定最佳 GPT 尺寸 | **目标正确，当前实验只能给“最佳受测尺寸”** | 五个点同时混合了宽度、深度和 kv_heads，且 4.48M 到 16.96M 中间很稀疏，不能推断连续意义上的全局最优 |

因此，原报告把 `deep` 写成总选型并不严谨：它回答的是“谁在成熟 DA、MAPE、行为和计算量之间最实用”，不是“谁最能利用 9+7 tokenizer”。重新分析后，`deep` 是旧网格的效率拐点，`xlarge@18` 是旧网格的 coarse 容量候选；两者都需要在 Sup 的严格同 step、同数据顺序协议下复核。

## 2. 实验协议

| 项目 | 固定值 |
|---|---|
| Python | 3.12.10，`D:\conda_envs\llm-t\Scripts\python.exe` |
| seed | 42 |
| 数据 | 全量 A 股、完整股票序列 |
| Tokenizer | Exp 02 选型：64x192，bits 9+7 |
| GPT 配方 | CE + AdamW，lr=3e-4，50 epochs，无 early stop |
| Checkpoint | 每个 epoch 保存并评估 |
| 评估窗口 | offset 0/100/200/300，各 20 个高覆盖交易日 |
| 每个 checkpoint | 80 个日截面、约 35.6 万个预测 |
| Holdout | offset 400 起 80 日封存 |
| 旧健康门 | 任一有效日 collapse ≤35%，且 unique ≥32 |

所有架构都使用完整序列；`deep/large/xlarge` 用梯度检查点控制显存。训练时原本意图让不同 `batch_tokens` 只承担吞吐与显存控制，但事后发现 microbatch 跨累积边界后会整体触发 step，导致架构间实际更新数不同；模型初始化消耗的随机数还会改变后续 loader shuffle。

| Config | Scheduler 预算 | 实际 optimizer steps | 相对 4160 的缺口 |
|---|---:|---:|---:|
| `baseline` | 4160 | 3934 | −226 |
| `wide` | 4160 | 3981 | −179 |
| `deep` | 4160 | 3984 | −176 |
| `large` | 4160 | 4039 | −121 |
| `xlarge` | 4160 | 4104 | −56 |

这意味着各架构不仅训练更新数不同，而且在同一 epoch 上处于不同 LR step；原排序不能被视为纯容量因果效应。Sup 使用架构无关 loader seed、统一 `batch_tokens=6144`、精确累积块和逐 epoch step 断言，将六个受控点全部重训到同一 4160-step 轨迹。

## 3. 为什么旧的 Unique/Collapse 还不够

### 3.1 分母不能直接用理论码本 512

逐日真实目标并不会覆盖全部 512 个 coarse token。80 个保留日截面的真实分布为：

| 真实 coarse 目标 | Min | P10 | Median | Mean | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Unique | 87 | 134.9 | **174.5** | 165.8 | 186.0 | 193 |
| Collapse | 4.65% | 6.05% | **10.45%** | 14.45% | 24.65% | 69.77% |
| 有效 token 数 `2^H` | 5.8 | 18.6 | **55.3** | 51.7 | 72.3 | 83.3 |

因此：

- `pred_unique / 512` 会低估一个本来就稀疏的日分布；
- raw Unique 会被罕见的一次性错误 token 夸大；
- Collapse 和 Unique 高度相关，不能证明模型用了“正确的” token；
- 当前实验只保留了 **coarse 512-way 主 AR 头**，不能声称测到了完整的 65,536 joint code 利用率。

为 Exp 03-Sup 接入 fine/joint 评估时还发现：现有训练路径在位置 `t` 用 `coarse_(t-1)` 作为 fine head 的 teacher condition，却监督 `fine_t`；推理路径则用当前预测的 `coarse_t` 作 condition。这是继承配方中的条件错位。Sup 为隔离容量变量仍保留现有语义；容量主排名使用 coarse target-relative 指标，fine/joint 完整分布只作审计。修复必须另立质量实验并统一重训完整网格。

### 3.2 两阶段诊断

对全部 250 个已有 epoch，先用现有逐日结果计算目标矩对齐：

```text
A_unique   = min(U_pred / U_true, U_true / U_pred)
A_collapse = min(C_pred / C_true, C_true / C_pred)
TMA        = sqrt(A_unique × A_collapse)
```

TMA 只用于扫描完整轨迹。它仍不能判断 token ID 是否正确，所以代表 checkpoint 重新跑了一次精确评估，并增加：

- Target support recall：真实日 support 中有多少 token 被预测分布使用；
- Prediction support precision：预测出的 token 有多少属于真实日 support；
- 有效 token 数：`2^H`，避免 one-off token 虚增 Unique；
- Jensen–Shannon divergence：预测与真实 token 频率分布的距离；
- Coarse token accuracy；
- 目标相对 Collapse。

为了便于定位折中点，定义诊断性分数：

```text
CB score = (
    support_F1
    × (1 - JSD)
    × effective_token_alignment
    × collapse_alignment
) ^ (1/4)
```

这个分数不是新的最终 KPI。所有组成项、DA、RankIC、MAPE、AmpRatio、val loss 和计算量仍分别报告。

## 4. 成熟 checkpoint 选择规则

“看最佳 epoch”这个方向是对的，但直接在 50 个单点里挑最大值会产生选择偏差。本报告采用：

1. 每个配置的成熟起点 = 首次满足 `val_loss <= 1.01 × 该配置最小 val_loss`；
2. 枚举连续 5 epoch 窗口；
3. 工作平衡窗要求 5 个 epoch 都仍在上述 near-best-loss 区间内；
4. 先最大化窗口的 epoch-level median TMA，再比较 P10 TMA；
5. 用窗口中心作为代表 checkpoint，并做精确 token 分布复评；
6. 另外保留不受 loss 约束的纯利用率峰值，作为敏感性分析。

这里的 loss 只定义“尚未明显离开泛化盆地”，不参与码本分数。这样既不把最低 loss 当成下游最优，也不会把过拟合后的无条件分布扩张直接判成胜利。

| Config | 最小 val loss | 成熟起点 | 平衡窗 | 代表 ep | 纯利用率稳定窗 | 单点 TMA 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3.7131 | 16 | 39–43 | 41 | 39–43 | ep39 |
| deep | 3.6648 | 16 | 44–48 | 46 | 44–48 | ep44 |
| wide | 3.6647 | 15 | 43–47 | 45 | 43–47 | ep32 |
| large | 3.6549 | 13 | 44–48 | 46 | 44–48 | ep41 |
| xlarge | 3.6559 | 8 | **16–20** | **18** | 45–49 | **ep44** |

## 5. 精确码本诊断结果

### 5.1 各架构的成熟平衡 checkpoint

| Config@ep | 参数 | CB score | Target recall | 有效 token（预测/真实） | JSD | Collapse（预测/真实） | Unique | DA | RankIC | MAPE | Amp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline@41 | 2.58M | 0.247 | 18.4% | 6.7 / 55.3 | 0.668 | 36.5% / 10.4% | 31.5 | 53.22% | 0.0442 | 3.360 | 1.164 |
| deep@46 | 4.48M | 0.280 | 21.9% | 8.5 / 55.3 | 0.588 | 35.6% / 10.4% | 38.0 | **53.40%** | 0.0387 | **3.033** | 0.928 |
| wide@45 | 5.40M | 0.273 | 20.6% | 7.2 / 55.3 | 0.650 | 38.3% / 10.4% | 35.0 | 49.15% | 0.0362 | 3.211 | 0.912 |
| large@46 | 7.51M | 0.254 | 21.6% | 7.1 / 55.3 | 0.644 | 39.0% / 10.4% | 37.5 | 49.81% | 0.0398 | 3.262 | **0.979** |
| **xlarge@18** | **16.96M** | **0.319** | **24.4%** | **10.2 / 55.3** | **0.572** | **31.8% / 10.4%** | **43.0** | 51.80% | 0.0328 | 3.056 | 0.877 |

几个重要现象：

- `deep` 在参数更少的同时，CB score、JSD 和 collapse 都优于 `wide/large`，所以“加深比单纯加宽有效”的结论仍成立；
- `deep` 是参数—码本收益的明显拐点：baseline→deep 每新增 1M 参数带来约 0.0177 CB score，而 deep→xlarge 只有约 0.0031，边际效率相差约 **5.75 倍**；
- `xlarge@18` 在全部组成指标上给出当前最强的总体码本处理，但提升幅度远不足以宣称已充分利用 tokenizer；
- 所有代表点的 prediction support precision 接近 1，说明 Unique 增加主要来自真实日 support 内的 token，不是简单随机撒到完全无关的 token；
- 即便如此，`xlarge@18` 的有效 token 只有 10.2，而真实中位数是 55.3；它仍严重集中在少数 token 上。

### 5.2 `xlarge@18` 相对 `deep@46`

逐日配对差值为 `xlarge@18 − deep@46`。95% 区间使用四个验证窗内的 5 日循环 moving-block bootstrap（10,000 次）：

| 指标 | 差值 | 95% 区间 | 结论 |
|---|---:|---:|---|
| CB score | +0.0433 | [+0.0283, +0.0581] | 码本平衡改善有逐日分辨力 |
| Target support recall | +3.45pp | [+2.55, +4.32]pp | 覆盖改善有分辨力 |
| 有效 token | +1.66 | [+0.99, +2.35] | 有效多样性改善有分辨力 |
| JSD | −0.0308 | [−0.0512, −0.0086] | 目标分布更接近 |
| Collapse | −2.98pp | [−5.96, +0.12]pp | 方向更好，区间刚跨 0 |
| DA | −1.60pp | [−3.84, +0.57]pp | 无法区分 |
| Daily RankIC | −0.0059 | [−0.0357, +0.0217] | 无法区分 |
| MAPE | +0.022pp | [−0.084, +0.134]pp | 无法区分 |

这正是“容量—利用—表现平衡”的证据：`xlarge@18` 的码本利用改善清楚，而 DA/MAPE/RankIC 尚没有可分辨退化。它不是 DA 冠军，但 DA 也没有提供足够证据否决这次容量提升。

### 5.3 为什么不选纯利用率更高的 `xlarge@44`

| xlarge checkpoint | val loss | CB score | Target recall | 有效 token | JSD | Collapse | DA | MAPE | Amp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ep18，平衡点** | 3.674 | 0.319 | 24.4% | 10.2 | 0.572 | 31.8% | **51.80%** | **3.056** | 0.877 |
| **ep44，利用峰值** | 3.858 | **0.368** | **29.9%** | **12.4** | **0.526** | **29.0%** | 50.57% | 3.126 | 0.897 |

`ep44 − ep18` 的逐日诊断显示：

- CB score +0.0402，95% CI [+0.0316, +0.0483]；
- support recall +5.79pp，CI [+5.31, +6.27]pp；
- collapse −3.32pp，CI [−4.56, −1.98]pp；
- 但 DA −1.24pp，CI [−1.99, −0.45]pp；
- MAPE +0.070pp，CI [+0.030, +0.113]pp。

所以后期确实继续学会了“使用更多正确 support 内的 token”，这支持“下游指标与 loss/利用率会解耦”的观察；但它同时已付出可分辨的 DA 与 MAPE 代价。若 Exp 03 的目标是找平衡点，ep18 比 ep44 更合适。若目标只是测容量上限，ep44 应保留为上界，而不是删除。

## 6. 修订后的 Exp 03 结论

### 6.1 最佳受测容量：`xlarge`

在固定 64x192 @ 9+7 tokenizer 的五个受测架构中，`xlarge` 最能改善：

- 真实 target support 覆盖；
- 熵意义上的有效 token 数；
- 预测—真实 token 分布 JSD；
- Collapse；
- 目标相对的综合码本平衡。

若只阅读旧协议结果，代表候选是 **ep18**，对应成熟平衡窗 **ep16–20**。它替代旧报告中“最后 10 epoch + DA/MAPE 主导”的 `deep@50`，但在 Sup 完成前不应作为跨实验的冻结工作 checkpoint。

### 6.2 计算效率拐点：`deep`

`deep` 仍然非常重要：

- 只有 4.48M 参数，约为 xlarge 的 26%；
- 明确支配 `wide/large`；
- 相对 baseline 的码本改善、MAPE 与幅度校准都稳定；
- 相同硬件上 50 epoch 训练 + 全轨迹评估约 77.8 分钟，而 xlarge 约 277 分钟。

因此 `deep` 是低成本实验和部署的合理效率基线，但不再被描述为“最能驾驭 tokenizer 的尺寸”。

### 6.3 没有配置真正驾驭 tokenizer

`xlarge@18` 仍存在：

- target support recall 仅 24.4%；
- 有效 token 对齐仅 18.8%；
- 日 collapse 中位数 31.8%，目标仅 10.4%；
- 旧健康门仍不通过；
- fine/joint codebook 利用没有被测量。

所以严谨结论是：

> **旧网格中 xlarge 最接近所需容量，但 17M 参数仍未充分建模 9+7 tokenizer；Exp 03 找到了带训练预算混杂的历史领先点，没有找到“已经足够”或可冻结的容量。**

## 7. 对 Exp 04 的直接影响

1. Exp 04 的正式容量依赖应等待 Exp 03-Sup 冻结，不能直接从旧 [`selection.json`](run_seed42/selection.json) 读取 `xlarge`；
2. 若只复现旧结果或做诊断，可使用 `model_ep18.pt`；它不是 Sup 完成后的正式 warm-start 承诺；
3. HPO 的主要目标可以继续放在 AmpRatio、优化充分性、dropout/LR 与下游 DA/RankIC/MAPE，但每个候选必须同步保留本次新增的 target-relative codebook 指标；
4. 不能把 DA/MAPE 纳入码本分数，否则 Exp 04 的优化目标会反向污染 Exp 03 的容量结论；
5. xlarge 的正式 HPO 成本显著更高，应采用分阶段筛选；`deep` 可继续作为廉价对照，但不能替代最终容量复核；
6. 后续 evaluator 已能输出 `true_coarse_id`、support、有效 token、JSD 与 CB score，新的 epoch 轨迹不再只保存 Unique/Collapse。

## 8. 下一轮容量实验建议

本节建议现已落地为 [`../03-gpt-scaling-sup/`](../03-gpt-scaling-sup/)：新增两个纯深度点、两个纯宽度点，并加入 `deep/xlarge` 两个端点。由于上述 optimizer-step/data-order 审计，六个点会在修正协议下全部重训，而不是复用旧端点 checkpoint。正式单-seed 运行尚未开始，以下内容保留为 Sup 的设计依据。

当前五点无法定位 4.48M–16.96M 之间的拐点。下一轮应：

- 在 `deep` 与 `xlarge` 之间增加至少两个受控点；
- 分开做“只加深”和“只加宽”，不要同时改变 dim/depth/kv_heads；
- 保持 optimizer step、完整序列、tokenizer、seed 和评估窗口一致；
- 主排名使用成熟 5-epoch 窗口的 target-relative 指标；
- DA/RankIC/MAPE/AmpRatio 作 guardrail；
- 至少对入围点补多 seed；
- 若要讨论完整 65,536 joint codebook，必须另外记录 fine-token 真实/预测分布。

## 9. 计算成本

| Config | 训练 | 50-checkpoint 评估 | 总计 | 相对 baseline |
|---|---:|---:|---:|---:|
| baseline | 28.0 min | 11.3 min | 39.4 min | 1.00x |
| wide | 44.9 min | 16.3 min | 61.3 min | 1.56x |
| deep | 59.5 min | 18.3 min | 77.8 min | 1.98x |
| large | 72.7 min | 22.7 min | 95.5 min | 2.42x |
| xlarge | 137.3 min | 139.6 min | 277.0 min | 7.03x |

旧 xlarge 评估时间包含 batch=1 的完整 50-checkpoint 轨迹；这不改变质量比较，但说明把 xlarge 带入 Exp 04 时必须重新核算实验预算。

## 10. 产物与复现

```text
03-gpt-scaling/
├── gpt_scaling_epochwise.py
├── analyze_gpt_scaling_epochwise.py
├── README.md
└── run_seed42/
    ├── study_manifest.json
    ├── selection.json
    ├── analysis.json / ANALYSIS.md
    ├── target_distribution_summary.json
    ├── target_daily_distribution.csv
    ├── codebook_epoch_summary.{csv,json}
    ├── capacity_windows.csv
    ├── codebook_representative_checkpoints.csv
    ├── codebook_parameter_pareto.csv
    ├── plots/
    └── configs/<arch>/codebook_diagnostics/
```

```powershell
# 重建分析并发布 xlarge@18 选型
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\03-gpt-scaling\analyze_gpt_scaling_epochwise.py `
  --select_config xlarge
```

本报告的分析脚本会从 prepared cache 恢复真实 coarse 目标分布，扫描全部 250 个旧 epoch，并读取代表 checkpoint 的精确复评。任何涉及完整 joint codebook 的结论都不能从本报告外推；Sup 已加入逐 epoch 的 fine/joint 真实与预测完整计数 sidecar。
