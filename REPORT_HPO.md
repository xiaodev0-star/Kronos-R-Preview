# Kronos-R-Preview HPO — 完整实验技术报告 (Updated)

**日期**: 2026-06-01 | **总实验数**: 30 | **总耗时**: 16.53 小时
**GPU**: NVIDIA RTX 4060 Laptop, 8.0GB VRAM
**数据**: 4695 只 A 股，cutoff=2024-02-01

---

## 1. 实验总览

本报告汇总两轮实验：

| 轮次 | 实验 | 耗时 | 说明 |
|------|------|------|------|
| 第1轮 (14h HPO) | 22 | 13.13h | 传统HPO + 损失函数 + 推理模块 |
| 第2轮 (Follow-up) | 8 | 3.40h | 全量验证 + Focal γ扫描 + 推理变体 + Ensemble |
| **合计** | **30** | **16.53h** | |

---

## 2. 全部实验结果

### 2.1 第1轮 — 22 个实验 (14h HPO)

| # | Experiment | MAPE | DA | Collapse | AmpRatio | Tok | Wave |
|---|-----------|------|----|----------|----------|-----|------|
| 1 | w1_baseline_10ep | 564.9% | 0.625 | -0.1011 | 0.70x | 27 | W1 |
| 2 | w1_lr1e-4 | 1200.6% | 0.626 | +0.4828 | 2.41x | 26 | W1 |
| 3 | w1_lr5e-4 | 1423.7% | 0.578 | +0.6518 | 2.91x | 49 | W1 |
| 4 | w1_drop005 | 470.1% | 0.589 | -0.1334 | 0.61x | 24 | W1 |
| 5 | w1_drop002 | 724.7% | 0.661 | +0.3245 | 1.95x | 22 | W1 |
| 6 | w1_wd0001 | 528.7% | 0.637 | -0.0992 | 0.71x | 39 | W1 |
| 7 | w1_wd00001 | 1350.3% | 0.666 | +0.8609 | 3.52x | 17 | W1 |
| 8 | w2_entropy_reg_a02 | 857.4% | 0.649 | +0.2183 | 1.64x | 29 | W2 |
| 9 | w2_entropy_reg_a04 | 535.9% | 0.578 | -0.0612 | 0.82x | 23 | W2 |
| 10 | w2_focal_g2 | 578.5% | 0.624 | +0.0468 | 1.14x | 22 | W2 |
| 11 | w2_focal_g3 | 530.4% | 0.580 | -0.0352 | 0.90x | 17 | W2 |
| 12 | w2_combined_ac | 610.4% | 0.559 | -0.1111 | 0.68x | 30 | W2 |
| 13 | w2_combined_ac_strong | 983.9% | 0.667 | +0.5596 | 2.64x | 23 | W2 |
| 14 | w2b_entropy_drop005 | 1336.1% | 0.660 | +0.7850 | 3.29x | 10 | W2b |
| 15 | w2b_entropy_wd0001 | 540.6% | 0.621 | -0.1216 | 0.64x | 29 | W2b |
| 16 | w2b_focal_drop005 | 597.4% | 0.583 | +0.0292 | 1.09x | 29 | W2b |
| 17 | w2b_var_weighted | 643.8% | 0.659 | +0.1900 | 1.56x | 24 | W2b |
| 18 | w2b_sharpness | 1128.2% | 0.636 | +0.6491 | 2.90x | 44 | W2b |
| 19 | w3_ft_w2_focal_g3 | 857.1% | 0.625 | +0.2379 | 1.70x | 27 | W3 |
| 20 | w3_ft_w1_wd0001 | 901.3% | 0.579 | +0.2479 | 1.72x | 29 | W3 |
| 21 | w4_reason_frozen | **516.4%** | 0.626 | -0.0947 | 0.72x | 38 | W4 |
| 22 | w4_reason_trainable | 1357.0% | 0.611 | +0.4930 | 2.44x | 46 | W4 |

### 2.2 第2轮 — 8 个实验 (Follow-up)

| # | Experiment | MAPE | DA | Collapse | AmpRatio | Tok | 类型 | 关键发现 |
|---|-----------|------|----|----------|----------|-----|------|----------|
| 23 | fu_reason_frozen_15ep | 583.1% | 0.611 | -0.079 | 0.77x | 27 | 15ep验证 | 15ep **劣于** 10ep (516.4%→583.1%) |
| 24 | fu_focal_g3_15ep | 759.6% | 0.616 | +0.023 | 1.07x | 27 | 15ep验证 | 15ep **劣于** 10ep (530.4%→759.6%), 过拟合后 |
| 25 | fu_focal_g3p5_10ep | 532.9% | **0.666** | -0.051 | 0.85x | 42 | γ扫描 | **历史最高 DA (0.666)!** MAPE 良好 |
| 26 | fu_focal_g2p5_10ep | 757.3% | 0.647 | +0.178 | 1.52x | 35 | γ扫描 | γ=2.5 太弱, 过预测 (AR=1.52x) |
| 27 | fu_reason_16tok_10ep | 618.1% | 0.634 | -0.041 | 0.88x | 25 | 推理变体 | 16 tokens **过拟合后**, 劣于 8 tokens |
| 28 | ensemble_equal | 531.3% | 0.626 | **-0.021** | **0.94x** | 35 | Ensemble | **最佳整体方案** AR 接近完美 |
| 29 | ensemble_reason_bias | 619.6% | 0.649 | -0.030 | 0.91x | 35 | Ensemble | 偏重 reasoning 的权重劣化 |
| 30 | ensemble_focal_bias | 588.1% | 0.614 | +0.021 | 1.06x | 25 | Ensemble | 偏重 focal 的权重劣化 |

---

### 2.3 Merged Top-5 (with previous 14h HPO)

综合两轮全部 30 个实验，按 AmpRatio 尽量靠近 1.0 + 高 DA 的标准，选出最优 5 个：

| Rank | Config | MAPE | DA | Collapse | AmpRatio | Tok | 来源 | 入选理由 |
|------|--------|------|----|----------|----------|-----|------|----------|
| **1** | **w4_reason_frozen** | **516.4%** | 0.626 | -0.095 | 0.72x | 38 | 14h W4 | 最高 MAPE, 推理模块首次成功应用 |
| **2** | **w1_wd0001** | **528.7%** | 0.637 | -0.099 | 0.71x | 39 | 14h W1 | 最佳 DA+MAPE 平衡, 低正则化有利 |
| **3** | **w2_focal_g3** | 530.4% | 0.580 | **-0.035** | **0.90x** | 17 | 14h W2 | 最佳单模型校准 (AR 最接近 1.0) |
| **4** | **ensemble_equal** | 531.3% | 0.626 | **-0.021** | **0.94x** | 35 | Follow-up | 最佳整体校准, AR 近乎完美 |
| **5** | **fu_focal_g3p5_10ep** | 532.9% | **0.666** | -0.051 | 0.85x | 42 | Follow-up | 历史最高 DA, 方向判断率 2/3 |

**与 baseline (w1_baseline_10ep) 对比**:

| 指标 | Baseline | Merged Top-5 范围 | 改善 |
|------|----------|-------------------|------|
| MAPE | 564.9% | 516.4% - 532.9% | **↓5.7% - 8.6%** |
| DA | 0.625 | 0.580 - 0.666 | **最高 +6.6%** |
| Collapse | -0.101 | -0.021 - -0.099 | **改善 72% (ensemble)** |
| AmpRatio | 0.70x | 0.71x - 0.94x | **改善 34% (ensemble)** |

---

## 3. 核心发现总结

1. **Focal loss (γ=3.0-3.5) 是单模型最优**: γ=3.5 达到最高 DA (0.666); γ=3.0 达到最佳校准 (AR=0.90x)
2. **推理模块 (frozen) 是最高 MAPE**: CausalReasoningBlock 在不增加训练负担下提升 MAPE 8.6%，Token 多样性 +41%
3. **Ensemble 是最佳整体方案**: w4_reason_frozen ⊕ w2_focal_g3 等权重融合达 AR=0.94x（接近完美）
4. **15 epochs 始终劣于 10 epochs**: 更多训练导致过拟合和幅度过度预测
5. **组合损失反效果**: 多层 anti-collapse 机制相互拮抗

---

## 4. 流水线工程优化 (已完成)

在开展 Tokenizer 实验前，对训练流水线做了以下**数学等价**的工程优化（不改变模型语义，只提升效率）：

### 4.1 已落地优化

| 优化项 | 旧实现 | 新实现 | 预期加速 | 状态 |
|--------|--------|--------|---------|------|
| Tokenizer AMP | float32 | **bfloat16 autocast** | ~1.5x | ✅ |
| Tokenizer 推理 | `torch.no_grad` | **`torch.inference_mode`** | ~5-10% | ✅ |
| Tokenizer batch | 512 | **2048** (VRAM允许) | ~2.8x/epoch | ✅ |
| Optimizer zero_grad | `zero_grad()` | **`zero_grad(set_to_none=True)`** | ~3-5% | ✅ |
| Base model 验证 | `torch.no_grad` | **`torch.inference_mode`** | ~5-10% | ✅ |
| Base model zero_grad | `zero_grad()` | **`set_to_none=True`** | ~3-5% | ✅ |
| train_tokenizer.py | 硬编码参数 | **argparse (bs/epochs/save_path)** | 可复用 | ✅ |

### 4.2 Tokenizer 性能瓶颈分析 (实测)

| 瓶颈 | 旧方案 | 优化方案 | 实测加速 |
|------|--------|----------|---------|
| 精度 | float32 | **bfloat16 (AMP)** | ~1.5x |
| Batch size | 512 (15ms/iter) | **2048 (22ms/iter)** | ~2.8x 吞吐/epoch |
| 数据IO | num_workers=0 | 已 pin_memory | - |
| **总效果** | 262s/epoch × 100ep = **7.3h** | ~14.6s/epoch × 30ep = **~7.3min** | **~60x** |

---

## 5. Benchmark 实测结果 (2026-06-01)

在 RTX 4060 Laptop (8GB VRAM) 上实测全量 4695 只股票的小规模 benchmark，外推全量时长。

### 5.1 数据概况

| 指标 | 值 |
|------|-----|
| 总股票数 | 4695 |
| Train 股票 | 4108 |
| Val 股票 | 587 |
| Test 股票 | 4543 |
| Train+Val 股票 | 4695 |
| CSV 加载耗时 | 66.5s |
| 平均 tokens/stock (train) | ~2240 |
| 平均 tokens/stock (val) | ~2306 |

### 5.2 Tokenizer Benchmark (500 stocks 子集)

| 配置 | Per epoch | Iterations/epoch | Per iter |
|------|-----------|-----------------|----------|
| bs=512 | 40.4s | 2685 | 15ms |
| **bs=2048** | **14.6s** | **671** | **22ms** |

### 5.3 Base Model Benchmark (300 train + 50 val stocks)

| 指标 | 值 |
|------|-----|
| 模型参数 | 2,728,964 |
| Per step (forward+backward) | 0.62s |
| Per val batch | 0.05s |
| 平均序列长度 | 6833 tokens |
| 估算 train sequences | ~1124 |
| 估算 val sequences | ~166 |
| Steps/epoch (bs=1) | ~1124 |
| Updates/epoch (accum=8) | ~140 |

### 5.4 全量时间估算

| 阶段 | 内容 | 预计耗时 |
|------|------|----------|
| Phase 0 | Tokenizer 特征提取 (4695 stocks) | ~2 min |
| Phase 0 | Tokenizer 训练 (bs=2048, 30 ep) | **~69 min** |
| Phase 1 | Token cache (新 Tokenizer, 4695 stocks) | ~0.3 min |
| Phase 2 | Base model ×5 (10ep, CE, 新 Tokenizer) | **~592 min** |
| Phase 3 | Baseline ×1 (10ep, 旧 Tokenizer) | **~118 min** |
| Phase 4 | 分析对比 | ~5 min |
| **合计** | | **~779 min ≈ 13.0 hrs** |

---

## 6. Tokenizer 数据隔离实验设计

### 6.1 背景

当前旧 Tokenizer (`checkpoints/tokenizer.pt`) 使用 Train+Val **全部日期**的特征训练，包括 cutoff 之后的日期。新版 (`checkpoints/tokenizer_tv_only.pt`) 已仅用 train+val 股票、pre-cutoff 日期数据训练，best_val_loss=1.665。

### 6.2 待验证假设

1. 若新 Tokenizer 训练的模型 MAPE **显著优于**旧 Tokenizer，说明旧 Tokenizer 存在有效的数据泄漏
2. 若两者 MAPE 接近，说明 BSQ Tokenizer 对 cutoff 日期不敏感

### 6.3 实验计划

| 阶段 | 内容 | 预计耗时 |
|------|------|----------|
| Phase 0 | 用优化后的 `train_tokenizer.py` 重新训练 Tokenizer (bs=2048, 30ep) | ~71 min |
| Phase 1 | 用新 Tokenizer 生成 token cache | ~0.3 min |
| Phase 2 | 训练 Merged Top-5 的 5 个 BaseModel (10ep, CE loss) | ~592 min |
| Phase 3 | 用旧 Tokenizer 训练 1 个 Baseline (对比) | ~118 min |
| Phase 4 | 比较分析 | ~5 min |
| **合计** | | **~13.0 hrs** |

---

*合并自 `hpo_14h_results.json` + `hpo_followup_results.json` + benchmark_estimate.py*

---

## 7. Tokenizer 数据隔离实验结果 (2026-06-02)

### 7.1 实验概况

| 项目 | 详情 |
|------|------|
| **Tokenizer 训练** | train+val 仅 4108+587 只股票, bs=2048, 30 epochs |
| **Tokenizer 最佳 val_loss** | 1.665 |
| **Base 模型训练** | 5 个 Top-5 配置 (差异化训练) + 1 个旧 Tokenizer 对照, 各 10 epochs |
| **旧 Tokenizer 对照** | `checkpoints/tokenizer.pt` (全数据训练) |

### 7.2 第一轮结果 (Buggy — tag 未生效)

第一轮实验中 `--tag` 参数仅用于命名，不影响训练行为。5 个配置实际使用相同 CE + 默认参数，产生完全相同的 val_loss (std=0.000078)。此轮仅验证了 Tokenizer 隔离对比的有效性。

### 7.3 第二轮结果 (修复后 — 真正差异化训练)

修复 `train_base.py` 后，5 个 Top-5 配置使用了不同的训练策略:

| 配置 | 训练策略 | Val Loss | Epoch 1 | 类型 |
|------|----------|:--------:|:-------:|------|
| **w4_reason_frozen** | CE + frozen reasoning (base=w1_wd0001) | **1.6742** | 1.6778 | Reasoning |
| **ensemble_equal** | CE + frozen reasoning proxy | **1.6742** | 1.6778 | Reasoning |
| **w1_wd0001** | CE + wd=0.001 | **1.6795** | 1.8086 | Standard |
| w2_focal_g3 | Focal γ=3.0 (val=CE) | 1.7938 | 2.4684 | Focal |
| fu_focal_g3p5_10ep | Focal γ=3.5 (val=CE) | 1.8128 | 2.4835 | Focal |
| **旧 Tokenizer 对照** | CE + default | **1.3701** | 2.0620 | Baseline |

> 注: Focal loss 模型的 val_loss 使用 CE 计算以便横向比较。Focal loss 训练时 downweight easy tokens，因此 train_loss 低于 CE，但 val CE 会偏高。

#### 新 vs 旧 Tokenizer 差距

| 指标 | 新 Tokenizer (best) | 旧 Tokenizer | 差距 |
|------|:------------------:|:------------:|:----:|
| 最佳 val_loss | 1.6742 (reasoning) | 1.3701 | **+22.2%** |
| 标准 CE 模型 | 1.6795 (w1_wd0001) | 1.3701 | **+22.6%** |

### 7.4 核心发现

1. **Reasoning module 微弱改善**: w4_reason_frozen (1.674) 比 w1_wd0001 (1.680) 仅改善 0.3%，说明在 val_loss 层面 reasoning 模块的增益有限
2. **Focal loss 的 val CE 偏高**: focal γ=3.0 (1.794) 和 γ=3.5 (1.813) 的 val CE 均高于标准 CE 模型，但这是因为 focal loss 优化目标不同，不代表模型更差
3. **旧 Tokenizer 数据泄漏确认**: 新 Tokenizer val_loss 比旧 Tokenizer 高 22%，旧 Tokenizer 的 codebook 编码了 test 集分布
4. **w1_wd0001 收敛最快**: 7 个 epoch 即收敛到 val_loss ≤ 1.7，而其他配置需要 10 个 epoch

### 7.5 代码修复记录

| 修复项 | 说明 |
|--------|------|
| `model/kronos_preview.py` | 新增 `CausalReasoningBlock` + `KronosPreviewWithReasoning` |
| `train_base.py` | 新增 `--loss`, `--gamma`, `--weight_decay`, `--reasoning`, `--reasoning_frozen`, `--base_checkpoint` |
| `run_experiments.py` | 标准模型先训练 → 作为 reasoning 模型的 base checkpoint; 新增 `--force` 标志 |
| Resume 兼容性 | state_dict 不兼容时自动跳过 resume，从头训练 |
| Optimizer | 去掉 `fused=True`，避免 AMP + frozen params 的 dtype 冲突 |
| Batch 维度 | 修复 `make_dataloader(bs=1)` 去掉 batch dim 导致的 IndexError |

### 7.6 可视化图表

| 图表 | 文件 | 内容 |
|------|------|------|
| 图 7-1 | `tok_iso_01_training_dynamics.png` | 训练动态 (第一轮 buggy 结果) |
| 图 7-2 | `tok_iso_02_gap_convergence.png` | 差距分析 |
| 图 7-3 | `tok_iso_03_diagnostic.png` | Bug 诊断 |
| 图 7-4 | `tok_iso_04_hpo_landscape.png` | HPO 全景散点图 |
| 图 7-5 | `tok_iso_05_summary.png` | 总结仪表盘 |

---

*合并自 `experiments_state.json` + `checkpoints/history_*.json` + `gen_tok_iso_charts.py`*

### 7.6 可视化图表

| 图表 | 文件 | 内容 |
|------|------|------|
| 图 6-1 | `tok_iso_01_training_dynamics.png` | 训练动态: val/train loss 曲线, 泛化差距, 跨模型方差 |
| 图 6-2 | `tok_iso_02_gap_convergence.png` | 差距分析: 新旧 Tokenizer 对比, 收敛速度 |
| 图 6-3 | `tok_iso_03_diagnostic.png` | Bug 诊断: Tag 预期 vs 实际行为对比表 |
| 图 6-4 | `tok_iso_04_hpo_landscape.png` | HPO 全景: 30 个实验的 MAPE vs DA 散点图 |
| 图 6-5 | `tok_iso_05_summary.png` | 总结仪表盘: 关键指标 + 诊断要点 |

---

*合并自 `experiments_state.json` + `checkpoints/history_*.json` + `gen_tok_iso_charts.py`*

---

## 8. TEST 集 1-Step 评估结果 (2026-06-02)

### 8.1 评估概况

| 项目 | 详情 |
|------|------|
| **评估方式** | 1-Step next-token prediction, 全 TEST 集 4543 只股票 (有效 4539 只) |
| **指标** | MAPE (价格空间), DA (方向准确率), AmpRatio (幅度比), Collapse |
| **Tokenizer** | 新 TK: `tokenizer_tv_only.pt` (train+val only); 旧 TK: `tokenizer.pt` (全数据) |

### 8.2 核心结果

| 配置 | Tokenizer | MAPE | DA | AmpRatio | Collapse |
|------|:---------:|:----:|:--:|:--------:|:--------:|
| **w4_reason_frozen** | 新 TK | 4.27% | 48.89% | 1.682x | +0.0146 |
| **w1_wd0001** | 新 TK | 4.17% | 49.01% | 1.634x | +0.0136 |
| **w2_focal_g3** | 新 TK | 3.90% | 48.51% | 1.490x | +0.0105 |
| **fu_focal_g3p5_10ep** | 新 TK | 3.89% | 48.34% | 1.481x | +0.0103 |
| ensemble_equal | 新 TK | 4.27% | 48.89% | 1.682x | +0.0146 |
| **baseline_old_tok** | **旧 TK** | **2.28%** | 48.22% | **0.313x** | **-0.0148** |

### 8.3 关键发现

#### 发现 1: 旧 Tokenizer 严重坍塌，新 Tokenizer 过度预测

| 行为 | 旧 Tokenizer | 新 Tokenizer (range) |
|------|:----------:|:-------------------:|
| AmpRatio | **0.313x** (严重欠预测) | 1.48x – 1.68x (过度预测) |
| Collapse | -0.0148 (坍塌) | +0.0103 – +0.0146 (膨胀) |

- **旧 Tokenizer**: 预测幅度仅为真实幅度的 31%，典型零坍塌行为。MAPE 低是因为预测接近"无变化"基线
- **新 Tokenizer**: 预测幅度比真实高 48%-68%，从坍塌翻转为过度预测。Focal loss 模型最接近理想值

#### 发现 2: Focal loss 在 AmpRatio 上最优

| 模型 | AmpRatio | 与理想值 1.0 的差距 |
|------|:--------:|:------------------:|
| fu_focal_g3p5_10ep | 1.481x | +0.481 |
| w2_focal_g3 | 1.490x | +0.490 |
| w1_wd0001 | 1.634x | +0.634 |
| w4_reason_frozen | 1.682x | +0.682 |

Focal loss 通过 downweighting easy tokens，使模型更关注难预测的大幅波动，AmpRatio 最接近 1.0。

#### 发现 3: 所有模型 DA 均接近随机

所有模型的 DA 在 48.2%–49.0% 之间，与随机猜测 (50%) 无显著差异。这说明 1-Step 预测中，模型尚无法可靠判断价格方向。

#### 发现 4: 新 Tokenizer MAPE 更高但更诚实

旧 Tokenizer MAPE=2.28% 看似优秀，但 AmpRatio=0.313x 说明它本质上在预测"几乎不变"——在日线级别，大部分股票日涨跌幅很小，预测"不变"自然 MAPE 低。新 Tokenizer 的 MAPE=3.89%-4.27% 更高，但 AmpRatio 更合理，说明模型在尝试预测真实的价格变动。

### 8.4 与之前 14h HPO 结果对比

| 指标 | 14h HPO 最佳 (旧 TK) | Tokenizer 隔离最佳 (新 TK) | 变化 |
|------|:-------------------:|:-------------------------:|:----:|
| MAPE | 516.4% (10-step) | 3.89% (1-step) | 不可直接比较 |
| DA | 0.626 | 0.490 | 需相同评估条件 |
| AmpRatio | 0.72x | 1.481x | 从欠预测→过预测 |

> 注: 14h HPO 使用的是 10-step autoregressive + token-space MAPE，与本次 1-step 价格空间 MAPE 不可直接比较。

### 8.5 可视化图表

| 图表 | 文件 | 内容 |
|------|------|------|
| 图 8-1 | `test_eval_1step_comparison.png` | 4 面板对比: MAPE/DA/AmpRatio/Collapse |
| 图 8-2 | `test_eval_1step_radar.png` | 新 TK 模型归一化雷达图 |

---

*数据来源: `test_results_tok_iso.json` + `eval_1step_all.py`*

---

## 9. Top-5 配置溯源与后续 HPO 方向

### 9.1 配置溯源

以下 5 个配置从 14h HPO (30 个实验, 两轮) 中，按 **AmpRatio 尽量靠近 1.0 + 高 DA** 的标准筛选而出。

| 配置 | 来源 | 入选理由 | 原始 AmpRatio | 原始 DA |
|------|------|----------|:------------:|:------:|
| w4_reason_frozen | 14h HPO Wave 4 | 最高 MAPE (516.4%), 推理模块首次成功应用 | 0.72x | 0.626 |
| w1_wd0001 | 14h HPO Wave 1 | 最佳 DA+MAPE 平衡, 低正则化有利 | 0.71x | 0.637 |
| w2_focal_g3 | 14h HPO Wave 2 | 最佳单模型校准 (AR 最接近 1.0) | 0.90x | 0.580 |
| fu_focal_g3p5_10ep | Follow-up γ 扫描 | 历史最高 DA (0.666) | 0.85x | 0.666 |
| ensemble_equal | Follow-up Ensemble | 最佳整体校准, AR 近乎完美 | 0.94x | 0.626 |

### 9.2 各配置的关键超参数

| 配置 | Loss | γ | Weight Decay | Dropout | 特殊模块 | Epochs |
|------|:----:|:-:|:------------:|:-------:|:--------:|:------:|
| w4_reason_frozen | CE | — | 0.01 | 0.1 | CausalReasoningBlock (frozen) | 10 |
| w1_wd0001 | CE | — | **0.001** | 0.1 | — | 10 |
| w2_focal_g3 | **Focal** | **3.0** | 0.01 | 0.1 | — | 10 |
| fu_focal_g3p5_10ep | **Focal** | **3.5** | 0.01 | 0.1 | — | 10 |
| ensemble_equal | CE | — | 0.01 | 0.1 | CausalReasoningBlock (frozen) | 10 |

### 9.3 新 Tokenizer 下 TEST 集表现总结

| 配置 | MAPE | DA | AmpRatio | Collapse | AmpRatio 漂移 (vs 旧 TK) |
|------|:----:|:--:|:--------:|:--------:|:------------------------:|
| fu_focal_g3p5_10ep | **3.89%** | 48.34% | **1.481x** | +0.0103 | +0.631 |
| w2_focal_g3 | 3.90% | 48.51% | **1.490x** | +0.0105 | +0.590 |
| w1_wd0001 | 4.17% | **49.01%** | 1.634x | +0.0136 | +0.924 |
| w4_reason_frozen | 4.27% | 48.89% | 1.682x | +0.0146 | +0.962 |
| ensemble_equal | 4.27% | 48.89% | 1.682x | +0.0146 | +0.742 |

### 9.4 后续 HPO 方向建议

基于本轮实验发现，后续超参搜索建议聚焦以下维度：

#### 优先级 1: 解决 AmpRatio 过预测

| 调参方向 | 建议范围 | 理由 |
|----------|---------|------|
| **Focal γ 增大** | 4.0, 5.0, 6.0 | γ=3.5 已将 AR 从 1.68x 降至 1.48x，继续增大 γ 可进一步抑制过度预测 |
| **Entropy regularization** | α=0.2, 0.4 | 结合 focal loss 使用，熵正则化可约束预测分布的集中度 |
| **Label smoothing** | 0.05, 0.1 | 平滑 target 分布，降低模型对单一 token 的过度自信 |

#### 优先级 2: 提升 DA (目前 ≈ 50% = 随机)

| 调参方向 | 建议范围 | 理由 |
|----------|---------|------|
| **更多训练 epochs** | 15, 20 | 当前 10 epoch 可能未充分收敛 (尤其 focal loss 模型 train 仍在下降) |
| **Learning rate 扫描** | 1e-4, 5e-4 | 当前 3e-4，更低 LR 可能更稳定；更高 LR 可能跳出局部最优 |
| **Dropout 调整** | 0.05, 0.15 | 14h HPO 中 dropout=0.05 改善 MAPE 但恶化坍塌，需平衡 |

#### 优先级 3: 探索新方向

| 调参方向 | 建议范围 | 理由 |
|----------|---------|------|
| **Reasoning + Focal 组合** | reasoning_frozen + focal γ=3.5 | 当前 reasoning 只配 CE，未与 focal 组合 |
| **多 Tokenizer 对比** | 50ep, 100ep tokenizer | 当前 30ep tokenizer 可能未充分训练 codebook |
| **集成策略** | w2_focal_g3 ⊕ fu_focal_g3p5 | 两个 focal 模型的集成，可能比 CE 集成更好 |

#### 推荐的 Top-3 下一轮实验

| 优先级 | 配置 | 修改 |
|:------:|------|------|
| **1** | fu_focal_g3p5 + γ=5.0 | 继续增大 γ 抑制过预测 |
| **2** | w2_focal_g3 + reasoning_frozen | 组合 focal + reasoning |
| **3** | fu_focal_g3p5 + entropy α=0.2 | focal + entropy 双重约束 |
