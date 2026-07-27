# Exp 01 · BitSweep：BSQ 双层码本大小选择

> **状态**：已完成（2026-07-24 重跑 seed=42；2026-07-26 复核重析并清洗目录）
> **结论**：选定 **bits = 9+7**（联合词表 65,536）作为 Exp 02 / Exp 03 的上游依赖。9+7 在成熟期（后 10 epoch）六项指标中拿下 DA（53.34%）、P90 日坍缩（55.8%）、日 unique（32）三项第一，RankIC 第三、MAPE 第四，无明显短板。
> **限定**：0/500 个 checkpoint 通过行为健康门、holdout 从未开封、单 seed——这是一次**依赖选型**，不是"码本问题已解决"的最终结论。选型正文见 [`rerun_seed42/selection.json`](rerun_seed42/selection.json)。

---

## 1. 实验问题

BSQ 双层量化码本大小由 $(L_1, L_2)$ 决定，联合词表 $V = 2^{L_1+L_2}$。本实验回答：**对 2.7M GPT + A 股全量 OHLC 数据，码本该多大、粗细两层怎么分配？**

本实验存在两代版本，回答的问题并不相同：

| | 历史版（2026-07-11，已退役） | 重跑版（2026-07-24，当前有效） |
|---|---|---|
| 判据 | 码本**效率**（最小充分码本，明确不以 DA 为判据） | 成熟期**下游质量 + 预测行为**（非合成分数分组判读） |
| 评估 | 仅最终 checkpoint，493 交易日单次全量推理 | **每个 GPT epoch** × 4 个 20 日验证窗 |
| 结论 | 8+6（16K）Util% 最高 | **9+7**（65,536）成熟期最均衡 |

## 2. 历史实验为何退役

### 2.1 历史结论存档（数据源已不在盘上，数字出自 git 历史中的 `Exp-SweepBit.md`）

历史协议：tokenizer embedding_dim=48、focal γ=4 + heteroscedastic + Muon、GPT 40 epochs、终点单权重推理 493 日；筛选规则 = 在 Unique≥64 且 Collapse≤30% 的配置中取 Util% 最高。

| Config | Vocab | Unique | Util% | Collapse% | DA% | 重建 MAE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 6+6 | 4,096 | 61 | 1.489 | 30.6 | 50.07 | 0.267 |
| **8+6**（旧推荐） | 16,384 | 144 | 0.879 | 26.0 | 49.04 | 0.228 |
| 9+6 | 32,768 | 212 | 0.647 | 16.9 | 49.24 | 0.193 |
| 9+7 | 65,536 | 186 | 0.284 | 21.2 | 49.48 | 0.209 |
| 9+9 | 262,144 | 218 | 0.083 | 15.2 | 49.61 | 0.215 |

历史核心论断："DA 全部 49.04–50.07%（极差 1.04pp），码本大小不是预测质量杠杆；所有配置 `da_above_baseline ≈ −21pp`"。

### 2.2 退役原因（2026-07-24 脚本审计要点）

P0 级（直接影响结论有效性）：

1. **只评估终点权重** —— 看不到 epoch 动力学；本次重跑证明多数配置 DA 峰值在 epoch 2–3、之后各配置分化，终点单评正是"DA 平坦"错觉的来源；
2. 旧配方 focal+Muon 与当前 CE+AdamW 基线不一致，码本效应与训练配方混杂；
3. 旧评估无验证/holdout 边界，全部日期参与选择；
4. Collapse/Unique/RankIC 采用全期 pooled 口径，掩盖单日严重坍缩，无法表达横截面可用性；
5. 历史 MAPE 曾有一日错位，绝对值不可与当前口径直接比较。

P1 级：tokenizer resume 语义损坏（`.ckpt` 写出但从不加载）、warmup 有未计数的 `optimizer.step()`、缓存身份不校验、同进程修改全局 Config 可能跨配置污染、OOM 静默跳过股票仍标记 completed 等。

历史脚本（`sweep_bits.py`、`sweep_full_eval.py`、`gen_sweep_plots.py`）与历史报告（`Exp-SweepBit.md`）已于 2026-07-26 从工作区移除，可在 git 历史中找回。

## 3. 重跑协议（rerun_seed42）

| 项目 | 固定值 |
|---|---|
| Python | 3.12.10，`D:\conda_envs\llm-t\Scripts\python.exe` |
| seed | 42（单 seed） |
| bits | 6+6、7+6、7+7、8+6、8+7、8+8、9+6、9+7、9+8、9+9 |
| Tokenizer | embedding=64、hidden=192、100 epochs、取验证损失最优（无 early stop） |
| GPT | 2.7M（dim=256/depth=2/heads=4）、**CE + AdamW**、50 epochs、全量 4695 股、完整序列 |
| Checkpoint | 每个 GPT epoch 保存 `model_epN.pt` 并写入校验索引 |
| 评估 | 每 epoch × 4 个验证窗（offset 0/100/200/300，各 20 观测日）× 全部可用 test 股票 |
| Holdout | offset 400 起 80 日**封存未启用** |
| 判读 | 无加权总分；质量组（DA/MAPE/日级横截面 RankIC）与行为组（P90 日坍缩/日 unique/AmpRatio）分别判读 + Pareto 集合 |
| 健康门 | 日坍缩率 ≤35% 且日 unique ≥32（对全部日取最严值） |

运行审计链（`study_manifest.json`）：中途一次 `eval_helpers.py` 推理吞吐优化迁移（严格等价验证通过，`aggregate_exact: true`，加速 2.20x）；9+8 在 epoch 13 处一次 KeyboardInterrupt 后断点续跑完成。均不影响数据有效性。

## 4. 重跑结果

### 4.1 成熟期总表（后 10 epoch 中位数；括号内为全程峰值）

| Config | Vocab | 成熟 DA | 峰值 DA (ep) | 成熟 RankIC | 成熟 MAPE | 成熟 P90 坍缩 | 成熟 unique | 成熟 Amp | Tok MAE |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6+6 | 4,096 | 49.29% | 51.12% (2) | 0.0248 | 3.334 | 81.4% | 15.0 | 1.047 | 0.2354 |
| 7+6 | 8,192 | 51.37% | 52.15% (3) | 0.0408 | 3.378 | 71.7% | 23.0 | 1.087 | 0.2251 |
| 7+7 | 16,384 | 53.22% | 53.62% (18) | 0.0456 | 3.516 | 70.9% | 18.0 | 1.209 | 0.2183 |
| 8+6 | 16,384 | 52.06% | 54.21% (3) | **0.0638** | 3.563 | 67.7% | 24.0 | 1.257 | 0.2121 |
| 8+7 | 32,768 | 49.37% | 49.75% (2) | 0.0189 | 3.259 | 57.8% | 25.0 | **0.941** | 0.2056 |
| 8+8 | 65,536 | 50.94% | 50.96% (49) | 0.0546 | 3.722 | 65.3% | 22.2 | 1.318 | 0.1969 |
| 9+6 | 32,768 | 51.99% | **55.06% (3)** | 0.0266 | 3.410 | 75.8% | 19.2 | 1.154 | 0.2165 |
| **9+7** | 65,536 | **53.34%** | 53.91% (3) | 0.0458 | 3.363 | **55.8%** | **32.0** | 1.171 | 0.1942 |
| 9+8 | 131,072 | 49.74% | 50.00% (11) | 0.0206 | 3.765 | 73.9% | 31.0 | 1.207 | 0.2085 |
| 9+9 | 262,144 | 52.35% | 53.66% (3) | 0.0390 | **3.324** | 64.7% | 25.2 | 0.992 | 0.1828 |

统计口径：500 个 config-epoch 点；joint Pareto 146、quality Pareto 38、behaviour Pareto 25；**健康门通过 0/500**。

### 4.2 epoch 动力学

- **早期 DA 尖峰是普遍模式**：10 个配置中 8 个的峰值 DA 出现在 epoch 2–3，随后回落（如 9+6：e3 55.06% → ~52% 平台）。**9+7 是唯一"峰值 ≈ 成熟值"的高分配置**（e3 53.91% → e10 低点 51.39% → e50 缓慢恢复至 53.35%），成熟读数不依赖早停运气。
- **坍缩随训练单调改善**：e1 的 P90 日坍缩为 76%–100%，e50 降至 55%–82%；9+7 全程最低点 54.4%（e46）。**行为健康是训练后期属性**，早停选 DA 峰值会拿到行为最差的模型。
- **多样性从近乎坍缩起步**：e1 日 unique 中位数仅 1–6，e50 升至 15–32；9+7 以 32 收官，恰压在健康门槛线上。

图表：[quality_trajectories](rerun_seed42/plots/quality_trajectories.png) ·
[behaviour_trajectories](rerun_seed42/plots/behaviour_trajectories.png) ·
[capacity_story](rerun_seed42/plots/capacity_story.png) ·
[mature_metric_dashboard](rerun_seed42/plots/mature_metric_dashboard.png) ·
[mature_quality_behaviour_map](rerun_seed42/plots/mature_quality_behaviour_map.png) ·
[da_vs_collapse_pareto](rerun_seed42/plots/da_vs_collapse_pareto.png) ·
[tokenizer_capacity](rerun_seed42/plots/tokenizer_capacity.png)

### 4.3 重建质量与下游的真实关系

- Tokenizer MAE ↔ 成熟 DA：Spearman ρ = −0.345（p=0.33，**不显著**；500 点层面 ρ=−0.149，极弱）。反例：9+8 MAE 不差但 DA 垫底；9+6 MAE 较差但峰值 DA 全场第一。
- Tokenizer MAE ↔ 成熟 P90 坍缩：ρ = **+0.806**（p=0.005）；MAE ↔ 成熟 unique：ρ = **−0.745**（p=0.013）。
- GPT val_loss ↔ DA：ρ = 0.0026（500 点，完全无关）；但**配置内** val_loss ↔ unique 一致为 −0.90 ~ −0.96、↔ 坍缩强正（9+7: +0.956）。跨配置方向反转是 Simpson 效应（大词表配置损失天然更高）。

**结论：重建质量预测的是下游行为健康（低坍缩、高多样性），不预测方向准确率**；配置内多训有益，跨配置比 val_loss 是陷阱。

## 5. 选型：9+7

选型规则为分组分级判断（无加权总分），完整记录于 [`rerun_seed42/selection.json`](rerun_seed42/selection.json)：

- 9+7 成熟期 DA 第一（53.34%）、P90 日坍缩第一（55.8%）、日 unique 第一（32）、RankIC 第三（0.0458）、MAPE 第四（3.363），无明显短板；
- **对 9+9**（最强反候选）：词表小 4 倍，DA +0.99pp、坍缩 −8.91pp、unique +6.75、RankIC +0.0068；接受的代价是幅度校准更差（1.171 vs 0.992）、MAPE +0.039；
- **对 7+7**：DA 仅差 0.12pp 且词表小 4 倍，但 unique 18 vs 32、坍缩 70.9% vs 55.8%、MAPE 3.516 vs 3.363，行为组全面落后；
- **对 8+6**（旧推荐）：8+6 仅存王牌是 RankIC 全场第一（0.0638），但 DA 低 1.28pp、坍缩高 11.9pp、unique 少 8。若后续把横截面选股（RankIC）提为首要目标，8+6 值得重新入场。

## 6. 与历史结论的关系

1. **"DA ~49% 平坦、码本不是杠杆"——被推翻**。成熟期 DA 极差 4.05pp、峰值极差 5.3pp；小码本（6+6/7+6）受到真实惩罚。历史的"平坦"是终点单评 + 旧配方（focal+Muon）的产物。
2. **"8+6 最小充分"——在旧问题（效率）内自洽，在新问题（成熟期下游）下被 9+7 取代**。两代结论并不直接矛盾：判据变了。
3. **旧健康判据失效**：旧口径 8+6 报 Unique=144 / Collapse=26%（全期 pooled），新日级口径下无任何配置健康——"健康"结论是度量定义的产物，日级口径才反映横截面可用性。

## 7. 合理性与可靠性评估

### 7.1 设计合理性

相对历史协议的改进是实质的：逐 epoch 评估（暴露动力学）、四个不重叠验证窗（统计量）、日级坍缩/多样性/RankIC 口径（横截面可用性）、holdout 制度（防选择性过拟合）、manifest 指纹化（防混跑）、非合成分数（防加权黑箱）。设计上这是一次合格的对照实验。

遗留的合理性问题（知情使用）：

- **自举依赖**：tokenizer 架构固定为 64x192，该选择源于旧 Exp 02 在 bits=8+6 下的结论。Exp 02 重跑已在 9+7 下复验（64x192 仍最优），循环已闭合，但两个实验共享这一先验；
- **配方条件性**：重跑用 CE+AdamW（为与 Exp 04 消融基线一致），与历史生产配方（focal γ=4 + het）不同。9+7 的优势严格来说是"CE 协议下的"；
- **评估窗覆盖**：4 个窗口均在 2024-02-01 之后的连续 80 个观测日内，市场 regime 覆盖有限。

### 7.2 结论可信度分级

| 级别 | 结论 | 依据 |
|---|---|---|
| 高 | 小码本（6+6/7+6）系统性受罚；MAE 预测行为健康而非 DA；早峰/坍缩后期改善的动力学模式 | 效应量大、跨配置一致、有机制解释 |
| 高 | 整条管线可确定性复现 | Exp 02 重跑以相同 seed 重训 64x192@9+7，tokenizer MAE **逐位相同**（0.1941523…），GPT 成熟 DA 差异 ≈0.0004pp |
| 中 | 9+7 优于 7+7 / 9+9 / 8+6 的相对排序 | 领先幅度 0.12–1.28pp；**8+7、9+8 两个"哑弹"（DA 贴 50%）打破词表单调性，是单 seed 训练方差的直接内部证据**，头部差距在同一噪声量级 |
| 低 | DA 的绝对水平（53%） | 健康门 0/500：一个 P90 日坍缩 56% 的模型可从"当日多数方向"式退化输出获得 DA；历史全量评估 `da_above_baseline ≈ −21pp` 的教训未被本协议推翻 |

### 7.3 升级可靠性的动作（按性价比排序）

1. **3-seed 复验**头部三组（9+7 / 7+7 / 8+6），确认排序稳定性——哑弹现象说明这不是可选项；
2. **holdout 一次性揭盲**（offset 400 起 80 日）：仅对最终选型执行一次，作为发表级确认；
3. 坍缩治理属 GPT 主线工作（损失函数 / 采样 / 校准，Exp 04+）：码本选择只能保证不雪上加霜，9+7 即该保证下的最优。

## 8. 目录结构与复现

2026-07-26 清洗：移除历史脚本/报告/图表（git 历史可找回）与可再生缓存（各 config 的 token `cache/`、`shared/tokenizer_features/`，重跑时自动重建），保留全部权重、日志、逐 epoch 评估数据与分析产物。

```text
01-bitsweep/
├── README.md                        ← 本报告（唯一实验报告）
├── rerun_bits_epochwise.py          ← 重跑主脚本（断点续跑、manifest 指纹化）
├── evaluate_tokenizer.py            ← tokenizer 重建/码字诊断（rerun 依赖，Exp 02 复用）
├── analyze_bitsweep_epochwise.py    ← 非合成分数分析（可独立重跑）
├── narrate_bitsweep.py              ← 成熟期摘要 + selection.json 生成
└── rerun_seed42/
    ├── study_manifest.json          ← 参数/数据/源码指纹与运行审计链
    ├── selection.json               ← 9+7 选型（Exp 02 依赖入口，勿手改）
    ├── combined_epoch_summary.{csv,json}  ← 500 config-epoch 评估数据
    ├── analysis.json / *.csv        ← Pareto、包络、代表点、成熟期摘要
    ├── plots/                       ← 报告引用的全部图表
    ├── unattended_resume.*.log      ← 无人值守运行日志
    └── configs/bits_LL_LL/          ← 每配置：tokenizer.pt、model_ep1..50.pt、
                                        epoch_trajectory/、logs/、run.json
```

```powershell
# 正式重跑（自动断点续跑；数据齐全时不会重复训练）
& 'D:\conda_envs\llm-t\Scripts\python.exe' .\experiments\01-bitsweep\rerun_bits_epochwise.py

# 仅重建分析与图表
& 'D:\conda_envs\llm-t\Scripts\python.exe' .\experiments\01-bitsweep\analyze_bitsweep_epochwise.py

# 仅重建成熟期摘要/选型叙事（会重写 selection.json，慎用）
& 'D:\conda_envs\llm-t\Scripts\python.exe' .\experiments\01-bitsweep\narrate_bitsweep.py --select_config 9+7
```
