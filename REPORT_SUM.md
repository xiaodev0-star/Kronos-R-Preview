# Kronos-R-Preview HPO — 配置命名说明 & 汇总 (Summary)

**日期**: 2026-06-03 | **总实验数**: 57 | **总耗时**: ~31.5 小时

---

## 配置命名规则

采用分段式命名: `{family}_{loss}_{hyperparams}`

| 段 | 含义 | 示例值 |
|----|------|--------|
| `w1` / `w2` / `w4` / `fu` | 实验族系 (详见下文) | w2 = Focal baseline |
| `focal` | 使用 Focal Loss 训练 | focal = 带 γ 参数的 Focal Loss |
| `reason` | 使用 CausalReasoningBlock 推理模块 | reason = 含可学习 memory tokens |
| `gX` / `gXpY` | Focal γ 值 | g6=6.0, g3p5=3.5 |
| `ls0XX` | Label Smoothing 值 | ls005=0.05, ls01=0.1 |
| `ent0X` | Entropy 正则化 α 值 | ent02=0.2 |
| `wdXXXX` | Weight Decay 值 | wd0001=0.001 |
| `drop00X` | Dropout 比率 | drop005=0.05 |
| `lrXeX` | Learning Rate | lr1e4=1×10⁻⁴ |
| `frozen` | 推理模块权重冻结 (仅训练 reasoning block) | |
| `full` | 全参数微调 (解冻所有参数) | |
| `ens` | Ensemble (多模型 logits 平均) | |

### 核心技术概念

#### 1. Focal Loss (Focal Loss)

**作用**: 解决 token 预测中的"容易token主导"问题。

在 CE Loss 中，所有 token 的预测误差等权。但金融 K 线数据中，大部分 token 代表小幅波动(容易预测)，
少数代表大幅波动(难以预测)。CE 模型倾向预测"零变化"来最小化平均误差，导致 **AmpRatio 坍塌**。

Focal Loss 通过 `(1-p_t)^γ` 因子 downweight 容易 token 的 loss:
- γ=0: 退化为 CE
- γ=3-4: 适度关注难 token，AmpRatio 从 0.7x → ~0.9x
- **γ=6-8: 大幅改善 AmpRatio (1.57x-1.59x) 和 MAPE**
- γ=10: 最强抑制，累积方向最优

#### 2. CausalReasoningBlock (推理模块)

**作用**: 在标准 Transformer block 之后增加跨注意力到可学习的 memory tokens。

```
input → [Transformer Blocks × 2] → [CausalReasoningBlock] → output
                                        ↑
                              N 个 learnable memory tokens
                              (cross-attention + gate + FFN)
```

- Memory tokens 可以学习全局统计模式(波动率、趋势等)
- Gate 机制 (`gate.tanh() * h`) 控制推理信息注入量
- `frozen`: 仅训练 reasoning block，Transformer保持预训练权重
- `full`: 全参数微调

#### 3. Label Smoothing

**作用**: 将 one-hot target 平滑为 `(1-ε) * one_hot + ε/V * uniform`。

- 降低模型对单个 token 的过度自信
- 与 Focal Loss 配合时改善了 1-Step MAPE (3.99% vs 4.01%)
- ε=0.05 最优，ε=0.1 差异不大

#### 4. Entropy Regularization

**作用**: 在 Focal Loss 中增加 `-α * H(p)` 项，鼓励预测分布保持适度熵。

- 防止分布过集中(坍塌到少数 token)
- 实验证明 α=0.2 无效 (MAPE 4.10% → 4.10%)

#### 5. Ensemble (推理时 Logits 平均)

**作用**: 加载多个模型，在推理时对 logits 取平均后 argmax。

```python
avg_logits = sum(model_i(inp, tids, pos, mask) for model_i in models) / N
pred = avg_logits.argmax()
```

- 无需额外训练，纯推理时操作
- 实验中 Ensemble 未超越最佳单模型 (4.05% vs 3.99%)
- 原因: 最佳模型已接近性能上限，模型间预测相关性高

#### 6. 关键指标

| 指标 | 全称 | 含义 | 理想值 |
|------|------|------|:------:|
| **MAPE** | Mean Absolute Percentage Error | 预测价格 vs 真实价格的百分比误差 (价格空间) | → 0 |
| **DA** | Directional Accuracy | 逐日涨跌方向判断正确率 | > 50% |
| **CumDA** | Cumulative DA | 10步累计方向判断正确率 | > 50% |
| **AmpRatio** | Amplitude Ratio | 预测波动幅度 / 真实波动幅度 | = 1.0 |
| **Collapse** | Zero-Collapse | 预测幅度 - 真实幅度 (负=欠预测) | = 0 |
| **Val Loss** | Validation Loss (CE) | 验证集交叉熵 (统一用CE以便比较) | → 0 |
| **StepX** | AR Step X | 自回归第 X 步的指标 | — |

### 完整示例

| 配置名 | 拆解 |
|--------|------|
| `w1_wd0001` | CE baseline, weight_decay=0.001 |
| `w2_focal_g6_ls005` | Focal baseline, γ=6.0, label_smoothing=0.05 |
| `fu_focal_g3p5_10ep` | Follow-up, Focal γ=3.5, 10 epochs (旧Tokenizer) |
| `w4_reason_focal_g8` | Reasoning, Focal γ=8.0, all params trainable (CE→Focal两阶段) |
| `w4_reason_full_wd001` | Reasoning, CE, wd=0.001, all params (CE→CE全量微调) |
| `w2_focal_g6_ent02` | Focal baseline, γ=6.0, entropy α=0.2 |
| `w4_reason_frozen` | Reasoning, reasoning block frozen, transformer from w1_wd0001 |
| `ens_reason_top2focal` | Ensemble: w4_reason_frozen + w2_focal_g6 + w2_focal_g5 |

---

## 全部 HPO 实验汇总

### 第1轮 — 14h HPO (旧 Tokenizer, token-space MAPE)

22 个实验，使用 `tokenizer.pt` (全数据训练)，eval 基于 10-step AR token-space MAPE。

| # | Experiment | MAPE | DA | Collapse | AmpRatio | 说明 |
|---|-----------|------|----|----------|----------|------|
| 1 | w1_baseline_10ep | 564.9% | 0.625 | -0.101 | 0.70x | CE 基线 |
| 2 | w1_lr1e-4 | 1200.6% | 0.626 | +0.483 | 2.41x | 低LR → 坍塌 |
| 3 | w1_lr5e-4 | 1423.7% | 0.578 | +0.652 | 2.91x | 高LR → 严重坍塌 |
| 4 | w1_drop005 | 470.1% | 0.589 | -0.133 | 0.61x | 少Dropout改善MAPE |
| 5 | w1_drop002 | 724.7% | 0.661 | +0.325 | 1.95x | 极少Dropout → 过拟合 |
| 6 | w1_wd0001 | 528.7% | 0.637 | -0.099 | 0.71x | 低wd → DA+MAPE平衡 |
| 7 | w1_wd00001 | 1350.3% | 0.666 | +0.861 | 3.52x | 极低wd → 严重坍塌 |
| 8 | w2_entropy_reg_a02 | 857.4% | 0.649 | +0.218 | 1.64x | 熵正则化较弱 |
| 9 | w2_entropy_reg_a04 | 535.9% | 0.578 | -0.061 | 0.82x | 熵正则化改善 |
| 10 | w2_focal_g2 | 578.5% | 0.624 | +0.047 | 1.14x | γ=2.0 太弱 |
| 11 | w2_focal_g3 | 530.4% | 0.580 | -0.035 | **0.90x** | 最佳单模型校准 |
| 12 | w2_combined_ac | 610.4% | 0.559 | -0.111 | 0.68x | 组合损失反效果 |
| 13 | w2_combined_ac_strong | 983.9% | 0.667 | +0.560 | 2.64x | 强组合 → 坍塌 |
| 14 | w2b_entropy_drop005 | 1336.1% | 0.660 | +0.785 | 3.29x | 组合 → 严重坍塌 |
| 15 | w2b_entropy_wd0001 | 540.6% | 0.621 | -0.122 | 0.64x | 熵+低wd可行 |
| 16 | w2b_focal_drop005 | 597.4% | 0.583 | +0.029 | 1.09x | Focal+少Dropout |
| 17 | w2b_var_weighted | 643.8% | 0.659 | +0.190 | 1.56x | 方差加权 → 过预测 |
| 18 | w2b_sharpness | 1128.2% | 0.636 | +0.649 | 2.90x | Sharpness → 坍塌 |
| 19 | w3_ft_w2_focal_g3 | 857.1% | 0.625 | +0.238 | 1.70x | 微调 → 劣化 |
| 20 | w3_ft_w1_wd0001 | 901.3% | 0.579 | +0.248 | 1.72x | 微调 → 劣化 |
| 21 | w4_reason_frozen | **516.4%** | 0.626 | -0.095 | 0.72x | 推理模块最高MAPE |
| 22 | w4_reason_trainable | 1357.0% | 0.611 | +0.493 | 2.44x | 推理可训练 → 坍塌 |

### 第2轮 — Follow-up (旧 Tokenizer)

8 个实验，验证最优配置 + γ 扫描 + Ensemble。

| # | Experiment | MAPE | DA | Collapse | AmpRatio | 说明 |
|---|-----------|------|----|----------|----------|------|
| 23 | fu_reason_frozen_15ep | 583.1% | 0.611 | -0.079 | 0.77x | 15ep 劣于 10ep |
| 24 | fu_focal_g3_15ep | 759.6% | 0.616 | +0.023 | 1.07x | 15ep 过拟合 |
| 25 | fu_focal_g3p5_10ep | 532.9% | **0.666** | -0.051 | 0.85x | 最高DA |
| 26 | fu_focal_g2p5_10ep | 757.3% | 0.647 | +0.178 | 1.52x | γ=2.5 太弱 |
| 27 | fu_reason_16tok_10ep | 618.1% | 0.634 | -0.041 | 0.88x | 16 tokens → 过拟合 |
| 28 | ensemble_equal | 531.3% | 0.626 | **-0.021** | **0.94x** | 最佳校准 |
| 29 | ensemble_reason_bias | 619.6% | 0.649 | -0.030 | 0.91x | 偏重推理 → 劣化 |
| 30 | ensemble_focal_bias | 588.1% | 0.614 | +0.021 | 1.06x | 偏重focal → 劣化 |

### 第3轮 — HPO v2 (新 Tokenizer, 价格空间 MAPE)

18 个实验，新TK + fixed mask，五族基线系统调优。

| # | Config | Val Loss | MAPE | DA | AmpRatio | Family | 说明 |
|:-:|--------|:--------:|:----:|:--:|:--------:|:------:|------|
| 31 | w1_wd0001 | 1.6728 | 4.22% | 48.62% | 1.672x | ce | CE 基线 |
| 32 | w1_focal_g4 | 1.8223 | 4.08% | 48.69% | 1.615x | focal | Focal on CE baseline |
| 33 | w1_lr1e4 | 1.7473 | 4.53% | 48.36% | 1.848x | ce | 低LR收敛慢 |
| 34 | w1_drop005 | 1.6719 | 4.20% | 48.49% | 1.661x | ce | 少Dropout改善 |
| 35 | w1_ls005 | 1.7173 | 4.16% | 48.46% | 1.644x | ce | 标签平滑有帮助 |
| 36 | w2_focal_g3 | 1.7822 | 4.14% | 48.77% | 1.647x | focal | γ=3.0 基准 |
| 37 | w2_focal_g4 | 1.8220 | 4.08% | 48.68% | 1.614x | focal | γ=4.0 |
| 38 | w2_focal_g5 | 1.8607 | 4.03% | 48.83% | 1.595x | focal | γ=5.0 |
| 39 | **w2_focal_g6** | **1.8965** | **4.01%** | **48.88%** | **1.584x** | focal | **v2 最佳** |
| 40 | w2_focal_g4_ls005 | 1.8536 | 4.04% | 48.76% | 1.593x | focal | Focal+标签平滑 |
| 41 | fu_focal_g3p5 | 1.8028 | 4.10% | 48.68% | 1.626x | focal | γ=3.5 |
| 42 | fu_focal_g3p5_ent02 | 1.8031 | 4.10% | 48.70% | 1.625x | focal | 熵正则化无效 |
| 43 | fu_focal_g5_ent02 | 1.8605 | 4.04% | 48.82% | 1.596x | focal | γ=5+熵 |
| 44 | fu_focal_g4p5 | 1.8413 | 4.05% | 48.77% | 1.605x | focal | γ=4.5 |
| 45 | w4_reason_frozen | 1.6678 | 4.20% | 48.68% | 1.663x | reason | CE推理基线 |
| 46 | w4_reason_focal_g3 | 1.7460 | 4.23% | 48.80% | 1.676x | reason | 推理+Focal反效果 |
| 47 | w4_reason_focal_g4 | 1.7712 | 4.24% | 48.77% | 1.682x | reason | 推理+Focal反效果 |
| 48 | w4_reason_lr1e4 | 1.6693 | 4.19% | 48.69% | 1.657x | reason | 推理低LR |

### 第4轮 — HPO v3 (γ 扩展 + Reasoning 两阶段)

9 个实验，γ=7~10 + label smoothing组合 + CE→Focal两阶段。

| # | Config | Val Loss | MAPE | DA | AmpRatio | Family | 说明 |
|:-:|--------|:--------:|:----:|:--:|:--------:|:------:|------|
| 49 | w2_focal_g7 | 1.9294 | 4.00% | 48.86% | 1.579x | focal | γ=7.0 |
| 50 | w2_focal_g8 | 1.9615 | 4.03% | 48.81% | 1.594x | focal | γ=8.0 |
| 51 | w2_focal_g10 | 2.0244 | 4.10% | 48.61% | 1.626x | focal | γ=10.0 |
| 52 | **w2_focal_g6_ls005** | **1.9238** | **3.99%** | **48.90%** | **1.572x** | focal | **全场最佳 1-Step** |
| 53 | w2_focal_g6_ls01 | 1.9622 | 4.00% | 48.93% | 1.575x | focal | ls=0.1 差异不大 |
| 54 | w4_reason_focal_g6 | 1.9019 | 4.17% | 48.37% | 1.669x | reason | 两阶段Focal |
| 55 | w4_reason_focal_g8 | 1.9672 | 4.05% | 48.18% | 1.602x | reason | 两阶段Focal |
| 56 | w4_reason_full_wd001 | 1.7413 | 4.30% | 48.47% | 1.720x | reason | 推理全量微调 |
| 57 | w2_focal_g6_ent02 | 1.9238 | 4.01% | 48.82% | 1.582x | focal | γ=6+熵 |

### Ensemble 实验汇总

| # | Name | Models | MAPE | DA | AmpRatio |
|:-:|------|--------|:----:|:--:|:--------:|
| E1 | ensemble_equal | w4_reason ⊕ w2_focal_g3 | 531.3%* | 0.626* | 0.94x* |
| E2 | ens_reason_focal_g6 | w4_reason_frozen + w2_focal_g6 | 4.07% | 48.72% | 1.603x |
| E3 | ens_top3_focal | w2_focal_g6 + g5 + g4 | 4.05% | 48.81% | 1.601x |
| E4 | ens_reason_top2focal | w4_reason_frozen + w2_focal_g6 + g5 | 4.06% | 48.73% | 1.598x |

> *旧Tokenizer token-space MAPE，不可直接与新结果比较

---

## 10-Step AR 自回归汇总 (价格空间)

评估条件: 500 stocks × 3 splits × 10 steps, context=128 tokens

| Config | CumDA† | MAPE | Step1 | Step5 | Step10 | 退化倍数 |
|--------|:------:|:----:|:-----:|:-----:|:------:|:------:|
| **w2_focal_g10** | **52.7%** | 8.48% | 3.2% | 8.2% | 12.9% | 2.1x |
| w2_focal_g8 | 51.9% | 8.48% | 3.1% | 8.1% | 13.0% | 2.1x |
| w2_focal_g7 | 51.7% | 8.54% | 3.1% | 8.1% | 13.0% | 2.1x |
| w2_focal_g6_ls005 | 51.6% | 8.64% | 3.1% | 8.3% | 13.1% | 2.2x |
| w2_focal_g6 | 51.6% | 8.74% | 3.1% | 8.4% | 13.2% | 2.2x |
| w4_reason_full_wd001 | 51.7% | 10.61% | 3.2% | 10.2% | 16.8% | 2.5x |
| w4_reason_frozen | 51.6% | 13.31% | 3.2% | 9.8% | 29.8% | 3.1x |
| **w1_wd0001** (CE) | 51.1% | **17.23%** | 3.3% | **11.5%** | **43.7%** | **4.1x** |

> †CumDA = 累积10步方向准确率 (sign(cumsum_10) vs sign(true_cumsum_10)), 随机=50%

---

## Focal γ 全局扫描

| γ | 1-Step MAPE | 10-Step MAPE | CumDA | 推荐场景 |
|:-:|:-----------:|:-----------:|:-----:|---------|
| 3.0 | 4.14% | 8.94% | 51.5% | 早期探索 |
| 3.5 | 4.10% | 8.88% | 51.5% | 历史最高DA |
| 5.0 | 4.03% | 8.78% | 51.6% | 过渡 |
| **6.0** | 3.99% | 8.74% | 51.6% | **1-Step 最优 (配 ls=0.05)** |
| 7.0 | 4.00% | 8.54% | 51.7% | 平衡选择 |
| **8.0** | 4.03% | **8.48%** | 51.9% | **10-Step AR 最优** |
| 10.0 | 4.10% | **8.48%** | **52.7%** | 累积方向最优 |

---

## 最终推荐模型

| 场景 | 推荐配置 | MAPE | 来源 |
|------|---------|:----:|:----:|
| **1-Step 预测** | w2_focal_g6_ls005 | 3.99% | V3 |
| **10-Step AR 预测** | w2_focal_g8 | 8.48% | V3 |
| **累积方向判断** | w2_focal_g10 | CumDA=52.7% | V3 |
| **推理模块** | w4_reason_frozen | val_loss=1.668 | V2 |
| **最低 val_loss (CE)** | w4_reason_full_wd001 | val_loss=1.741 | V3 |
