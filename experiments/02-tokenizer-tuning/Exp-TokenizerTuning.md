# Tokenizer 超参调优实验报告

> **一句话摘要**：在 bits=[8,6] 固定后，对 embedding_dim × hidden_dim 做 3×2 扫描。**embedding_dim=64, hidden_dim=192** 的重建 MAE 比 baseline（48x192）低 14.5%，达到 9+6 bits 配置的重建水平；GPT 下游验证通过（DA +0.14pp, Collapse +2.3pp，均在阈值内）。

| 实验属性 | 内容 |
|----------|------|
| 前置实验 | `experiments/01-bitsweep/`（确定 bits=[8,6]） |
| 扫描范围 | embedding_dim ∈ {48, 64, 96} × hidden_dim ∈ {192, 256} |
| 评估方法 | 阶段一：tokenizer 重建 MAE；阶段二：全量 GPT 下游验证 |
| 数据 | `tok_sweep_results.json` + `tok_sweep_gpt_validation.json` |
| 核心结论 | 推荐 **embedding_dim=64, hidden_dim=192** |

---

## 1. 实验目的

BitSweep 实验确定了 bits=[8,6]（16K 词表）为最优码本大小。但发现一个异常：8+7 的重建 MAE 比 8+6 更差。本实验探究是否可以通过调整 tokenizer 的结构参数（embedding_dim、hidden_dim）来改善重建质量。

**核心问题**：增大 embedding_dim 能否提升 latent 空间表达容量，降低重建下限？

## 2. 实验设计

### 2.1 阶段一：tokenizer 重建扫描

固定 bits=[8,6]，扫描 6 个组合：

| 参数 | 当前值 | 候选值 |
|------|--------|--------|
| `embedding_dim` | 48 | 48, 64, 96 |
| `hidden_dim` | 192 | 192, 256 |

每个组合训一个 tokenizer（100 epochs, early stop patience=15），评估 val MAE/MSE + tokenizer 级 Unique/Collapse。

### 2.2 阶段二：GPT 下游验证（硬门槛）

对阶段一选出的最优 tokenizer，与 baseline 各训一个 GPT（全量数据，early stop patience=5），用 windowed eval 对比下游指标。

判定标准：DA 下降 ≤ 1pp 且 Collapse 恶化 ≤ 10pp → 采用新 tokenizer。

## 3. 阶段一结果：重建质量

![MAE 排名](plt/mae_ranking.png)

| Config | MAE | MSE | ΔMAE | Uniq | Coll% |
|--------|-----|-----|------|------|-------|
| **64x192** | **0.1947** | **0.1083** | **−14.5%** | 217 | 6.6% |
| 48x256 | 0.2114 | 0.1334 | −7.2% | 211 | 6.1% |
| 96x192 | 0.2125 | 0.1245 | −6.7% | 237 | 7.0% |
| 96x256 | 0.2219 | 0.1386 | −2.6% | 219 | 5.5% |
| **48x192** | **0.2277** | **0.1598** | baseline | 209 | 8.8% |
| 64x256 | 0.2321 | 0.1557 | +1.9% | 220 | 8.0% |

![热力图](plt/heatmap.png)

**关键发现**：

1. **64x192 是明确最优**：MAE=0.195，比 baseline 低 14.5%，与 9+6 bits 的 MAE（0.193）持平——用 16K 词表达到了 32K 词表的重建质量。
2. **hidden_dim=256 不稳定**：64x256 的 MAE（0.232）比 64x192（0.195）差 19%，说明更大 hidden_dim 不一定更好，可能过拟合。
3. **embedding_dim 是真正杠杆**：48→64 带来 14.5% 改善，64→96 仅 6.7%（递减）。
4. **所有配置的 tokenizer 级 Unique/Collapse 健康**：Unique 209-237, Collapse 5.5-8.8%。

## 4. 阶段二结果：GPT 下游验证

![GPT 验证](plt/gpt_validation.png)

| 指标 | baseline 48x192 | candidate 64x192 | Δ | 阈值 | 判定 |
|------|-----------------|-------------------|------|------|------|
| DA | 49.15% | 49.29% | +0.14pp | ±1pp | ✅ |
| DA above baseline | −19.97% | −19.83% | +0.14pp | — | — |
| Collapse | 24.9% | 27.2% | +2.3pp | ±10pp | ✅ |
| Unique | 101 | 86 | −15 | — | — |
| RankIC | 0.0208 | 0.0343 | +0.0135 | — | 改善 |
| AmpRatio | 1.025 | 1.295 | +0.270 | — | 更激进 |

**判定：✅ PASS**。DA 和 Collapse 均在阈值内。RankIC 有实质改善（+65%）。

### 4.1 Tokenizer 级 vs GPT 级指标方向反转

![Tokenizer vs GPT](plt/tok_vs_gpt.png)

一个值得注意的现象：tokenizer 级和 GPT 级的 Unique/Collapse **方向相反**：

| 层级 | 48x192 | 64x192 | 趋势 |
|------|--------|--------|------|
| Tokenizer 级 Unique | 209 | 217 | 64x 更好 |
| Tokenizer 级 Collapse | 8.8% | 6.6% | 64x 更好 |
| GPT 级 Unique | 101 | 86 | 48x 更好 |
| GPT 级 Collapse | 24.9% | 27.2% | 48x 更好 |

tokenizer 本身更健康（更多 unique、更低 collapse），但 GPT 学出来的分布反而更集中。这说明 64x192 的 token 分布对 GPT 来说有不同的学习难度——token 更"干净"但 GPT 反而更倾向于少数高频模式。这一现象值得在后续 GPT 调参阶段关注。

## 5. 与 BitSweep 基准对比

| 指标 | BitSweep 8+6 (48x192) | 本次 48x192 | 本次 64x192 |
|------|----------------------|-------------|-------------|
| DA | 49.04% | 49.15% | 49.29% |
| Collapse | 26.0% | 24.9% | 27.2% |
| Unique | 144 | 101 | 86 |

三个值非常接近，再次验证了 **tokenizer 结构参数对 GPT 下游性能的影响极小**。真正决定 DA 的是 GPT 模型本身。

## 6. 结论

### 6.1 推荐配置

**embedding_dim=64, hidden_dim=192**（bits=[8,6] 不变）。

| 约束 | 读数 | 判定 |
|------|------|------|
| 重建 MAE 显著改善 | 0.195 (−14.5%) | 通过 |
| GPT DA 不退化 | +0.14pp | 通过（阈值 ±1pp） |
| GPT Collapse 不显著恶化 | +2.3pp | 通过（阈值 ±10pp） |
| RankIC 改善 | +0.0135 (+65%) | 额外收益 |

### 6.2 关键洞察

1. **embedding_dim 是 tokenizer 的真正杠杆**：从 48→64 带来 14.5% 的重建改善，且达到了 9+6 bits（32K 词表）的重建水平——用 16K 词表达到了 32K 词表的效果。
2. **hidden_dim 不是越大越好**：256 在 64x 配置下反而比 192 差 19%，可能因过拟合。
3. **tokenizer 超参对 GPT 下游影响极小**：DA 变化仅 0.14pp，远小于 GPT 自身超参的影响。下一步应聚焦 GPT 扩容。

### 6.3 局限

1. **单 seed**：仅 seed=42，结果的方差未知。
2. **未测试 encoder 深度**：当前 encoder 只有 2 层 Linear，更深的 encoder 可能在相同 embedding_dim 下进一步改善 MAE。
3. **commitment_cost / entropy_weight 未调**：留待 GPT 调参阶段。
4. **Collapse 方向反转未完全解释**：tokenizer 级更好的 Unique/Collapse 为何导致 GPT 级更差，需要进一步分析。

---

## 附录 A：图表索引

| 图表 | 文件 | 内容 |
|------|------|------|
| MAE 排名 | [plt/mae_ranking.png](plt/mae_ranking.png) | 6 组配置的重建 MAE 排名及改善百分比 |
| 热力图 | [plt/heatmap.png](plt/heatmap.png) | embedding_dim × hidden_dim 对 MAE 和 Collapse 的影响 |
| GPT 验证 | [plt/gpt_validation.png](plt/gpt_validation.png) | DA、RankIC、Collapse 的下游对比 |
| 层级对比 | [plt/tok_vs_gpt.png](plt/tok_vs_gpt.png) | Tokenizer 级 vs GPT 级 Unique/Collapse 方向反转 |

## 附录 B：复现

```bash
cd experiments/02-tokenizer-tuning

# 阶段一：tokenizer 扫描（~1 小时）
python sweep_tokenizer.py

# 阶段二：GPT 验证（~2-3 小时）
python sweep_tokenizer_validate.py

# 生成图表
python gen_plots.py
```
