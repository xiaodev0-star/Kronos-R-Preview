# Kronos-R-Preview：基于 LLM 范式的金融时序因果预测

## 1. 项目动机

Kronos-R 原项目采用固定 1024 滑动窗口训练 BaseModel，经过 12 轮实验，10-step AR 方向准确率始终 ≈50%（随机水平）。根本原因：

1. **固定窗口截断长期依赖**：模型永远只能看到最近 ~4 年，无法理解股票的完整生命周期
2. **信息泄漏**：每窗口独立 Z-score 归一化使用了未来数据
3. **模型偏大**：17M 参数 vs 14M 训练 token，Chinchilla 比例失衡
4. **辅助模块干扰**：LatentReasoner 在 Pack 序列中跨股票聚合信息，与隔离设计目标矛盾

本项目从根本上重新设计训练范式：**每只股票 = 一篇文档，从头读到尾，标准因果 next-token prediction**。

---

## 2. 核心设计

### 2.1 训练范式

```
原项目: [1024 tokens] → predict next 1024 tokens（滑动窗口）
Preview: [整个股票历史] → predict 每个位置的 next token（LLM 范式）
```

每只股票从第一个交易日读到 cutoff 日期（2024-02-01），模型在每个位置都预测下一个 token。每只股票作为独立序列，用纯因果 mask 训练。归一化采用 per-stock historical Z-Score（统计量仅来自该股票截止日前的训练数据）。

### 2.2 信息泄漏防控

| 环节 | 原项目 | Preview |
|------|--------|---------|
| 归一化 | 每 1024 窗口全局 Z-score | per-stock historical Z-Score (统计量仅来自 train); 首日基线 + Z-Score (VA) |
| 位置编码 | 全局递增（跨股票混合） | per-stock 重置（每只股票从 0 开始） |
| 注意力 | 全连接 | causal（严格屏蔽未来位置） |
| 数据切分 | 按日期 | 按 CSV 文件 + 时间 cutoff |

**归一化安全性说明**：

Historical Z-Score 的统计量 (mean/std) 仅从每支股票截止日前的训练数据计算。模型直接接收的是 BSQ 量化后的离散 token IDs，而非归一化值本身——量化是有损压缩，模型无法从 token 反推出原始统计量。

Causal attention mask 保证 position t 只能 attend 到 [0..t]。position 0 的 hidden state 在训练和推理时的计算**完全相同**（相同的输入、相同的 mask、相同的参数）。梯度虽然从整个序列回传，但只影响参数更新，不改变 forward 计算——参数固定后训推一致。

### 2.3 模型架构

```
Kronos-Preview (2.7M 参数):
  dim=256, depth=2, heads=4, num_kv_heads=1

  Token Embedding (BSQ 10-bit, vocab=1024)
  + Time Embedding (day/month/year, learned)
  + VA Embedding (Volume/Amount, continuous MLP: Linear(2→64)→GELU→Linear(64→256))
      ↓
  Transformer Block × 2:
      RMSNorm → F.scaled_dot_product_attention (GQA + RoPE)
      RMSNorm → SiLU-gated FFN
      ↓
  [可选: CausalReasoningBlock — N 个 learnable memory tokens, cross-attn + gate + FFN]
      ↓
  RMSNorm → Linear → logits (next-token prediction)
```

**Tokenizer**: BSQ (Binary Spherical Quantization), 2-level hierarchical.
输入 4D OHLC 价格特征 → Encoder MLP → 2×BSQ (10-bit each) → vocab=1024.

**VA Embedding**: Volume/Amount 作为连续值直接注入 Transformer，而非量化进 token。
训练时归一化: `log1p(vol) - log1p(vol_day0)` 再 Z-Score。
推理时使用 cutoff 前最后已知 VA 值。

**关键特性**：
- `F.scaled_dot_product_attention`：自动使用 Flash Attention，内存 O(N)
- RoPE：position_ids 外部传入，每只股票重置
- RMSNorm + SiLU-gated FFN：LLaMA 风格
- **CausalReasoningBlock** (可选)：跨注意力到可学习 memory tokens，learnable gating
- Gradient Checkpointing：进一步降低显存

### 2.4 训练特性

**支持 Loss 函数**:
| Loss | 说明 | CLI |
|------|------|-----|
| CE (Cross Entropy) | 标准 next-token prediction | `--loss ce` |
| Focal Loss | 通过 (1-p)ᵞ downweight easy tokens → 抑制零坍塌 | `--loss focal --gamma 6.0` |
| + Label Smoothing | 平滑 target → 降低过度自信 | `--label_smoothing 0.05` |
| + Entropy Reg | 鼓励预测分布保持适度熵 | `--entropy_alpha 0.2` |

**支持模块**:
| 模块 | 说明 | CLI |
|------|------|-----|
| CausalReasoningBlock | N memory tokens + gate + cross-attn + FFN | `--reasoning` |
| Frozen reasoning | 仅训练 reasoning block，transformer 冻结 | `--reasoning_frozen` |
| Base checkpoint | 从预训练权重初始化 | `--base_checkpoint path/to/model.pt` |

### 2.5 数据切分

```
时间线:  2010 ────────── 2024-02-01 ──── 2026.2
              ├─ Train/Val ─┤  ├── Test ──┤

CSV 隔离（空间泛化）:
  Train: 87.5% 的股票（所有 ≤ cutoff 的数据）
  Val:   12.5% 的股票（所有 ≤ cutoff 的数据）

时间隔离（时间泛化）:
  Test:  所有股票在 (cutoff, 2026.2] 的数据
```

---

## 3. HPO 结果摘要 (57 实验, ~31.5h)

详见 `REPORT_SUM.md` 和 `REPORT_HPO.md`。

### 1-Step 预测 (价格空间 MAPE)

| Rank | Config | MAPE | DA | AmpRatio |
|:----:|--------|:----:|:--:|:--------:|
| 1 | w2_focal_g6_ls005 | **3.99%** | 48.90% | **1.572x** |
| 2 | w2_focal_g6_ls01 | 4.00% | 48.93% | 1.575x |
| 3 | w2_focal_g7 | 4.00% | 48.86% | 1.579x |
| 4 | w2_focal_g6 | 4.01% | 48.88% | 1.584x |
| 5 | w4_reason_focal_g8 | 4.05% | 48.18% | 1.602x |

### 10-Step AR (自回归预测, 500 stocks × 3 splits)

| Config | CumDA | MAPE | Step10 MAPE | 退化倍数 |
|--------|:-----:|:----:|:-----------:|:------:|
| w2_focal_g10 | **52.7%** | 8.48% | 12.9% | 2.1x |
| w2_focal_g8 | 51.9% | 8.48% | 13.0% | 2.1x |
| w2_focal_g6_ls005 | 51.6% | 8.64% | 13.1% | 2.2x |
| w4_reason_frozen | 51.6% | 13.31% | 29.8% | 3.1x |
| w1_wd0001 (CE) | 51.1% | **17.23%** | **43.7%** | **4.1x** |

### 核心发现

1. **Focal Loss 是 anti-collapse 最优工具**：γ=6-8 将 AmpRatio 从 1.67x 降至 **1.57x**
2. **Label Smoothing + Focal 协同**：w2_focal_g6_ls005 达全场最优 MAPE=3.99%
3. **CE 模型 AR 灾难退化**：1-step MAPE=4.2% → 10-step=**17.2%** (4.1x)
4. **Focal 模型 AR 稳定**：1-step=4.0% → 10-step=**8.5%** (2.1x)
5. **累积方向 ≈ 随机**：CumDA 51-53%，单步DA=62%不转化
6. **CausalReasoningBlock 最佳用途**：CE预训练→全量微调，val_loss=1.68

---

## 4. 与原项目对比

| 维度 | Kronos-R_Full | Kronos-R-Preview |
|------|--------------|-----------------|
| 模型参数 | 17M | 2.7M（-84%） |
| 序列长度 | 1024 | 8192（+8×） |
| 训练范式 | 滑动窗口 | 文档式因果 LM |
| 注意力 | 手写 softmax | Flash Attention |
| 显存需求 | ~2 GB | ~0.5 GB |
| 信息泄漏 | 有（归一化） | 无 |
| Loss 函数 | CE only | CE / Focal / +Label Smoothing / +Entropy |
| 推理模块 | LatentReasoner (跨股票) | CausalReasoningBlock (per-stock memory) |
| 1-Step MAPE | — | **3.99%** |
| 10-Step AR MAPE | — | **8.48%** |

---

## 5. 快速开始

```bash
cd Kronos-R-Preview

# 1. 训练 Tokenizer (4D OHLC)
python train_tokenizer.py

# 2. 训练模型 (推荐 Focal)
python train_base.py --loss focal --gamma 6.0 --label_smoothing 0.05

# 3. 评估
python TEMP/eval_v2.py --n_stocks 30 --max_steps 10
```

---

## 6. 项目结构

```
Kronos-R-Preview/
├── README.md                      # 快速入门
├── main.md                        # 本文件
├── CODE_WIKI.md                   # 代码文档
├── config.py                      # 全局配置
├── reproducibility.py             # 随机种子
├── data_processor.py              # 数据管道 (document_normalize + pack_stocks_v2)
├── train_tokenizer.py             # Stage A: Tokenizer训练 (4D OHLC)
├── train_base.py                  # Stage B: 模型训练 (Focal/Reasoning/...)
├── model/
│   ├── kronos_preview.py          # KronosPreview + VA embedding + CausalReasoningBlock
│   ├── tokenizer.py               # BSQ Hierarchical Tokenizer
│   └── tokenizer_config.py        # Tokenizer配置工具
├── checkpoints/
│   ├── tokenizer_v2_ohlc.pt       # Tokenizer (4D OHLC)
│   └── v2_model.pt                # 训练好的模型
├── dataset/                       # CSV 数据
└── TEMP/                          # 历史实验归档 + 评估脚本
    ├── README.md                   # 完整实验记录
    ├── eval_v2.py                  # 评估脚本
    ├── eval_v2_results.json        # 评估结果
    ├── REPORT_HPO.md               # HPO 技术报告
    └── ...
```
