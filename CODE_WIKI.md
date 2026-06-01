# Kronos-R-Preview — Code Wiki

## 1. 项目概述

Kronos-R-Preview 是 Kronos-R 的重新设计版本，采用**标准 LLM 因果预测范式**。与原项目的根本区别：

| 维度 | 原项目 Kronos-R_Full | Preview |
|------|---------------------|---------|
| 序列构建 | 固定 1024 滑动窗口 | 每只股票从头到尾，多股票打包 |
| 归一化 | 每窗口独立 Z-score（泄漏未来） | 滚动窗口 W=252（无未来泄漏） |
| 注意力 | 手写 softmax O(N²) | F.scaled_dot_product_attention O(N) |
| 位置编码 | 全局 RoPE | per-stock 重置 RoPE |
| 辅助模块 | LatentReasoner + HorizonDecoder | 无（纯 LLM） |
| sector_id | 有 | 无（纯市场内生学习） |
| 模型规模 | dim=384, depth=3, ~17M | dim=256, depth=2, ~2.7M |

---

## 2. 项目结构

```
Kronos-R-Preview/
├── config.py                      # 全局配置
├── reproducibility.py             # 随机种子控制
├── data_processor.py              # 数据加载 + 滚动归一化 + 打包
├── model/
│   ├── __init__.py
│   ├── tokenizer.py               # BSQ Tokenizer（与原项目相同）
│   ├── tokenizer_config.py         # Tokenizer 构建工具
│   └── kronos_preview.py          # 新模型：SDPA + RoPE
├── train_tokenizer.py             # Stage A: 训练 Tokenizer
├── train_base.py                  # Stage B: 训练 Base Model
├── dataset/                       # A 股 CSV（symlink → 原项目）
├── checkpoints/                   # 模型权重
├── outputs/                       # 实验输出
├── CODE_WIKI.md                   # 本文件
└── main.md                        # 项目说明
```

---

## 3. 核心设计

### 3.1 数据切分

```
时间线:  2010 ────────── 2024-02-01 ──── 2026.2
              ├─ Train/Val ─┤  ├── Test ──┤

CSV 隔离（空间泛化）:
  Train: 87.5% 的股票（所有 ≤ cutoff 的数据）
  Val:   12.5% 的股票（所有 ≤ cutoff 的数据）

时间隔离（时间泛化）:
  Test:  所有股票在 (cutoff, 2026.2] 的数据
```

### 3.2 滚动归一化

对每只股票的每个位置 `i`，使用前 `W` 步（W=252）的均值/标准差归一化：

```python
window = features[max(0, i - W + 1) : i + 1]
mean = window.mean()
std = window.std()
normalized[i] = (features[i] - mean) / std
```

前 `min_lookback`（20）步输出为 0（数据不足）。

### 3.3 多股票打包

```
[<BOS>] stock_A_token₁ ... stock_A_tokenₙ [<EOS>]
[<BOS>] stock_B_token₁ ... stock_B_tokenₘ [<EOS>]
[<BOS>] stock_C_token₁ ...                   [<PAD>]
←───────────── context_len = 8192 ──────────────→
```

- `position_ids`：每只股票从 0 开始（不全局递增）
- `attention_mask`：causal（下三角）× segment（block-diagonal），防止跨股票注意力
- 特殊 token：`BOS = 2^bits`，`EOS = 2^bits + 1`

### 3.4 模型架构

```
Embedding: token_emb(ids) + time_emb(day) + time_emb(month) + time_emb(year)
    ↓
Transformer Block × 2:
    RMSNorm → F.scaled_dot_product_attention → Residual
    RMSNorm → SiLU-gated FFN → Residual
    ↓
RMSNorm → head_coarse / head_fine
```

位置编码：RoPE，position_ids 由数据集传入（per-stock 重置）。

---

## 4. 配置系统

| 配置类 | 关键参数 | 说明 |
|--------|----------|------|
| `NormConfig` | `lookback_window=252`, `min_lookback=20` | 滚动归一化 |
| `DataConfig` | `cutoff_date=2024-02-01`, `context_len=8192` | 数据切分与打包 |
| `TokenizerConfig` | `bits=10`, `hidden_dim=192` | 与原项目相同 |
| `ModelConfig` | `dim=256`, `depth=2`, `heads=4` | 缩小后的模型 |
| `TrainingConfig` | `batch_size=1`, `accumulation=16`, `lr=3e-4` | 适配 8GB GPU |

---

## 5. 运行方式

### Stage A: 训练 Tokenizer

```bash
python train_tokenizer.py
```

输出：`checkpoints/tokenizer.pt`

### Stage B: 训练 Base Model

```bash
python train_base.py
```

输出：`checkpoints/base_model.pt`

---

## 6. 初步验证结果

### 6.1 Tokenizer（100 只股票，50 epochs）

| 指标 | 数值 |
|------|------|
| 训练 Loss | 0.35 |
| 验证 Loss | 0.36 |
| Coarse 码本利用率 | 332/1024 (32%) |
| Fine 码本利用率 | 339/1024 (33%) |
| 唯一 token pair | 2249 |

### 6.2 Base Model（200 只股票，5 epochs）

| 指标 | 数值 |
|------|------|
| 模型参数 | 2,728,964 |
| 训练序列数 | 60 |
| 平均每序列股票数 | 2.9 |
| 序列利用率 | 100% |
| Epoch 1 → 5 Train Loss | 11.53 → 8.87 |
| Epoch 1 → 5 Val Loss | 10.38 → 8.79 |
| 峰值 GPU 显存 | 0.48 GB |
| 单 epoch 耗时 | ~6 秒 |

### 6.3 关键发现

- **Flash Attention**：峰值显存 0.48 GB（vs 手写 softmax 的 OOM 风险），8GB GPU 完全无压力
- **多股票打包**：avg 2.9 stocks/seq，100% GPU 利用率，无浪费
- **Loss 正常收敛**：无 NaN，无振荡，train/val 同步下降
- **2.7M 参数**：比原项目 17M 小 6 倍，但数据量（14M tokens）相对有限，此规模更合理

---

## 7. 与原项目的依赖关系

| 文件 | 来源 | 是否修改 |
|------|------|----------|
| `model/tokenizer.py` | 复制 | 不改 |
| `model/tokenizer_config.py` | 复制 + 适配 | config 引用改 |
| `reproducibility.py` | 直接复制 | 不改 |
| `dataset/*.csv` | symlink | 不改 |
| `config.py` | 全新 | — |
| `data_processor.py` | 全新 | — |
| `model/kronos_preview.py` | 全新 | — |
| `train_tokenizer.py` | 全新 | — |
| `train_base.py` | 全新 | — |
