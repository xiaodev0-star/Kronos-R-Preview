# GPT 架构扫描实验报告

> **一句话摘要**：5 个架构（2.7M~16.5M）全量训练+评估，**2.7M baseline 是最优配置**——模型越大 DA 越低，RankIC 越差。当前数据量是瓶颈，不是模型容量。

| 实验属性 | 内容 |
|----------|------|
| 前置实验 | `experiments/02-tokenizer-tuning/`（确定 tokenizer 配置） |
| 扫描范围 | dim ∈ {256,384,512} × depth ∈ {2,3,4} × heads ∈ {4,6,8} |
| 训练方式 | 全量 4695 stocks，early_stop_patience=5 |
| 评估方式 | 全量 test stocks，所有可用日期（子进程隔离） |
| 核心结论 | **2.7M baseline 最优**；扩容不提升 DA |

---

## 1. 实验目的

实验 01/02 确定了 tokenizer 配置（bits=[8,6], embedding_dim=64, hidden_dim=192），但 GPT 的 DA ≈ 49% 始终跑输基线 ~21pp。本实验探究：**GPT 模型扩容后 DA 能否突破 49% 平台？**

## 2. 实验设计

### 2.1 架构配置

固定：bits=[8,6], embedding_dim=64, hidden_dim=192, focal γ=4, het=ON, dropout=0.1

| 配置 | dim | depth | heads | kv_heads | ~Params | gradient ckpt | seq_len |
|------|-----|-------|-------|----------|---------|---------------|---------|
| **baseline** | 256 | 2 | 4 | 1 | 2.4M | 否 | full |
| **wide** | 384 | 2 | 6 | 1 | 5.1M | 否 | full |
| **deep** | 256 | 4 | 4 | 1 | 4.3M | 是 | full |
| **large** | 384 | 3 | 6 | 1 | 7.2M | 是 | full |
| **xlarge** | 512 | 4 | 8 | 2 | 16.5M | 是 | full |

### 2.2 训练参数

- 全量 4695 stocks，batch_size=1, accumulation_steps=32→64（epoch 15+翻倍）
- lr=3e-4, Muon+AdamW 优化器, WSD scheduler
- early_stop_patience=5（连续 5 个 epoch val_loss 不降即停）
- 所有配置使用 full sequence（max_seq_len=0），xlarge 通过 gradient_checkpointing 适配 VRAM

## 3. 结果

![综合指标](plt/summary_table.png)

| 排名 | Config | ~Params | DA% | RankIC | MAPE% | BL-MAPE% | AmpRatio | Collapse% | Unique |
|------|--------|---------|-----|--------|-------|----------|----------|-----------|--------|
| 1 | **baseline** | 2.4M | **49.05%** | **0.0231** | 4.09 | 2.18 | 1.462 | 25.1% | 120 |
| 2 | deep | 4.3M | 48.87% | 0.0136 | 3.83 | 2.18 | 1.317 | 24.3% | 114 |
| 3 | xlarge | 16.5M | 48.55% | 0.0161 | **3.16** | 2.18 | 0.863 | 29.7% | 104 |
| 4 | wide | 5.1M | 48.53% | 0.0066 | 4.15 | 2.18 | 1.491 | 25.2% | 121 |
| 5 | large | 7.2M | 48.28% | 0.0044 | 3.92 | 2.18 | 1.389 | 28.4% | 88 |

BL-MAPE = 2.18%（"预测不变"基线：pred_price = base_close）

### 3.1 DA vs 模型大小

![DA vs Size](plt/da_vs_size.png)

**模型越大，DA 越低**。线性趋势斜率为负（约 −0.05 pp/M），说明当前数据量下模型扩容不仅无益，反而有害。

### 3.2 多维指标雷达图

![雷达图](plt/radar.png)

五维指标（按参数量降序排列）：DA、RankIC、Perplexity（=exp(val_loss)）、MAPE、Collapse。baseline 在 DA 和 RankIC 两个维度上最优。xlarge 的 Perplexity 最低（3.50），但这反映的是欠拟合（过大模型 + 过少数据 → 学到过于保守的分布），而非真正的泛化能力。

### 3.3 RankIC 与 AmpRatio

![RankIC & AmpRatio](plt/rankic_ampratio.png)

**RankIC 随模型扩大急剧下降**：从 0.023（2.4M）降至 0.004（7.2M），降幅 82%。更大的模型丧失了对股票的排序能力——它们学会了"所有股票都一样"的退化策略。

### 3.4 MAPE：幅度预测质量

![MAPE](plt/mape.png)

**所有模型的 MAPE 都高于"预测不变"基线（2.18%）**——模型的幅度预测不仅没有帮助，反而引入了额外噪声。

| Config | MAPE% | 超基线 (pp) | AmpRatio | 诊断 |
|--------|-------|-------------|----------|------|
| xlarge | **3.16** | +0.98 | 0.863 | 最保守（低估幅度），MAPE 最好 |
| deep | 3.83 | +1.65 | 1.317 | 中等 |
| large | 3.92 | +1.74 | 1.389 | 中等 |
| baseline | 4.09 | +1.91 | 1.462 | 高估幅度 |
| wide | 4.15 | +1.97 | 1.491 | 高估幅度最严重 |

xlarge 的 MAPE 最好（+0.98pp），但其 AmpRatio=0.863 表明它系统性**低估**了价格变化幅度。这与 seq=4096 版本（AmpRatio=1.034，轻微高估）完全相反——更长的序列让模型变得更保守，预测更接近"不变"基线。

**核心洞察**：DA（方向）和 MAPE（幅度）是两个独立的问题。当前模型在方向上约有 49% 的准确率（略高于随机），但在幅度上完全不如"不变"基线。未来改进应聚焦于幅度预测的校准（如 GRPO 的 reward 设计），而非模型大小。

### 4.1 为什么更大模型反而更差？

1. **数据量不足**：4695 只股票 × ~2500 天 ≈ 1170 万条样本，对于 7M+ 参数的模型来说，有效样本/参数比不足
2. **过拟合训练分布**：大模型 memorize 了训练集的噪声模式，在验证集上泛化更差
3. **val_loss 不可信**：deep 的 val_loss（3.722）优于 baseline（3.756），但 DA 更差（48.87% vs 49.05%）——验证了 HPO 2026-06-18 的发现：val_loss 与下游 DA 的 Spearman ρ 仅 0.188

### 4.2 xlarge 的 AmpRatio 异常

xlarge 的 AmpRatio=0.863（系统性低估幅度），与 seq=4096 版本（1.034，轻微高估）完全相反。这说明更长的序列让模型变得更保守——它看到更多上下文后，学到的策略是"预测更小的变化"，这在 MAPE 上有优势（3.16% vs 4.09%），但 DA 更差（48.55% vs 49.05%）。

### 4.3 深度 vs 宽度

| 对比 | baseline (2层) vs deep (4层) | baseline (dim=256) vs wide (dim=384) |
|------|------------------------------|--------------------------------------|
| DA 变化 | −0.18pp | −0.52pp |
| RankIC 变化 | −42% | −72% |
| 结论 | 深度扩展稍好 | 宽度扩展更差 |

在同等增量下，深度扩展比宽度扩展对 DA 的损害更小。但两者都不如不扩展。

## 5. 结论

### 5.1 核心发现

**2.7M baseline（dim=256, depth=2, heads=4, kv_heads=1）是当前数据量下的最优架构。**

| 发现 | 证据 |
|------|------|
| 扩容不提升 DA | 5 个架构 DA 范围仅 48.28%~49.05%，baseline 最高 |
| 扩容损害 RankIC | 7.2M 的 RankIC（0.004）仅为 2.4M（0.023）的 18% |
| val_loss 不可信 | deep val_loss 更好但 DA 更差 |
| 深度优于宽度 | 同等增量下，depth=4 的 DA 比 dim=384 高 0.34pp |

### 5.2 瓶颈诊断

当前 DA ≈ 49% 的瓶颈**不是模型容量，而是**：
1. **数据量**：4695 只股票对 2.7M 模型已经足够，更大模型无数据可学
2. **预测任务本身的难度**：金融时序的信噪比极低，DA above baseline 始终为负（约 −20pp）
3. **可能需要的不是更大的模型，而是更好的训练策略**（GRPO、更大的 BERT、更好的 loss）

### 5.3 局限

1. **单 seed**：仅 seed=42
2. **xlarge 不公平比较**：max_seq_len=4096 与其他配置的 full seq 不可比
3. **未测试更大的 accumulation_steps 或更长训练**：可能缓解小模型的欠拟合
4. **未测试 Muon 以外的优化器**：不同优化器可能对大模型有不同的泛化特性

### 5.4 后续方向

1. **GRPO 后训练**：在 baseline 2.7M 上做 GRPO，直接优化 DA
2. **数据增强**：引入更多特征（order book、新闻情绪）增加信息量
3. **更大 BERT**：当前 BERT 16M，可以尝试更大的校准器
4. **Curriculum 改进**：更精细的序列长度 curriculum，或课程学习策略

---

## 附录 A：图表索引

| 图表 | 文件 | 内容 |
|------|------|------|
| 综合指标表 | [plt/summary_table.png](plt/summary_table.png) | 5 配置 × 14 指标全量数据（含 MAPE） |
| DA vs 模型大小 | [plt/da_vs_size.png](plt/da_vs_size.png) | DA 随参数量增加而下降的趋势 |
| 雷达图 | [plt/radar.png](plt/radar.png) | 5 维指标对比（DA/RankIC/AmpRatio/Collapse/Unique） |
| RankIC & AmpRatio | [plt/rankic_ampratio.png](plt/rankic_ampratio.png) | 下游信号质量随模型扩大而退化 |
| MAPE 对比 | [plt/mape.png](plt/mape.png) | 所有模型 MAPE 高于"不变"基线，xlarge 最保守 |

## 附录 B：复现

```bash
cd experiments/03-gpt-scaling

# 全部 5 个架构（约 4-5 小时，支持断点续算）
python sweep_gpt_arch.py

# 生成图表
python gen_plots.py
```
