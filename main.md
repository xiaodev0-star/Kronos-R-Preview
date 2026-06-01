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

每只股票从第一个交易日读到 cutoff 日期（2024-02-01），模型在每个位置都预测下一个 token。多只股票打包到同一序列（context_len=8192），用 block-diagonal attention mask 隔离。

### 2.2 信息泄漏防控

| 环节 | 原项目 | Preview |
|------|--------|---------|
| 归一化 | 每 1024 窗口全局 Z-score | 滚动窗口 W=252，只用过去数据 |
| 位置编码 | 全局递增（跨股票混合） | per-stock 重置（每只股票从 0 开始） |
| 注意力 | 全连接 | causal × segment（股票间完全隔离） |
| 数据切分 | 按日期 | 按 CSV 文件 + 时间 cutoff |

### 2.3 模型架构

```
Kronos-Preview (2.7M 参数):
  dim=256, depth=2, heads=4, num_kv_heads=1

  Token Embedding + Time Embedding (day/month/year)
      ↓
  Transformer Block × 2:
      RMSNorm → F.scaled_dot_product_attention (Flash Attention)
      RMSNorm → SiLU-gated FFN
      ↓
  RMSNorm → Linear → logits
```

**关键特性**：
- `F.scaled_dot_product_attention`：自动使用 Flash Attention，内存 O(N)
- RoPE：position_ids 外部传入，每只股票重置
- RMSNorm + SiLU-gated FFN：LLaMA 风格
- Gradient Checkpointing：进一步降低显存

### 2.4 数据切分

```
Train: 87.5% 的股票（随机选择，≤ cutoff_date 的数据）
Val:   12.5% 的股票（完全不同的股票，≤ cutoff_date 的数据）
Test:  所有股票在 (cutoff_date, 最新日期] 的数据
```

空间泛化（未见过的股票）+ 时间泛化（未来数据），零信息泄漏。

---

## 3. 初步验证结果

### 3.1 硬件验证

| 配置 | 峰值 GPU 显存 | 8GB 可行性 |
|------|-------------|-----------|
| dim=128, depth=1, ctx=2048 | 0.08 GB | 轻松 |
| dim=256, depth=2, ctx=8192 | **0.48 GB** | **轻松** |

Flash Attention + gradient checkpointing 使 8GB GPU 可以轻松处理 context=8192。

### 3.2 训练收敛

200 只股票 × 5 epochs：
- Train Loss: 11.53 → 8.87（-23%）
- Val Loss: 10.38 → 8.79（-15%）
- 单 epoch 耗时: ~6 秒
- 无 NaN，无振荡

### 3.3 打包效率

| 指标 | 数值 |
|------|------|
| 平均每序列股票数 | 2.9 |
| 序列利用率 | 100% |
| 总训练 token | ~392K（200 只股票） |

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
| 辅助模块 | LatentReasoner 等 | 无（纯 LLM） |

---

## 5. 快速开始

```bash
cd Kronos-R-Preview

# 1. 训练 Tokenizer
python train_tokenizer.py

# 2. 训练 Base Model
python train_base.py
```

### 调整配置

```python
# config.py 中修改
DataConfig.max_stocks = 200      # 限制股票数（调试用）
DataConfig.cutoff_date = "2024-02-01"
ModelConfig.dim = 256
ModelConfig.depth = 2
TrainingConfig.epochs = 10
TrainingConfig.learning_rate = 3e-4
```

或通过环境变量覆盖：

```bash
export KRONOS_PREVIEW_OVERRIDE_JSON=overrides.json
python train_base.py
```

---

## 6. 后续计划

1. **全量训练**：4695 只股票 × 10+ epochs，验证大规模收敛
2. **Rollout 评估**：在 test 期（2024-02-01 之后）做 10-step AR 预测
3. **与原项目对比**：同一 test 集上比较 path_mape / DA
4. **模型缩放**：测试 dim=128/dim=384 的效果差异
5. **后训练**：ExPO / GRPO 方向优化

---

## 7. 项目结构

```
Kronos-R-Preview/
├── config.py                      # 全局配置
├── reproducibility.py             # 随机种子
├── data_processor.py              # 数据管道
├── model/
│   ├── tokenizer.py               # BSQ Tokenizer
│   ├── tokenizer_config.py        # Tokenizer 工具
│   └── kronos_preview.py          # 新模型
├── train_tokenizer.py             # Stage A
├── train_base.py                  # Stage B
├── dataset/                       # CSV 数据（symlink）
├── CODE_WIKI.md                   # 代码文档
└── main.md                        # 本文件
```
