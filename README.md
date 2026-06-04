# Kronos-R-Preview

基于 LLM 范式的金融时序因果预测项目。每只股票 = 一篇文档，从头读到尾，标准因果 next-token prediction。

## 核心结果

1200 支测试股票，全量 pre-cutoff 历史作为 context 的 AR 评估：

| Day | MAPE | DA | AmpRatio |
|:---:|:----:|:--:|:--------:|
| 1 | 5.05% | **74.33%** | 1.543x |
| 10 | 9.88% | 52.33% | 0.572x |

**方向准确率 74.33%** 远超随机水平 (50%)，Volume/Amount 连续 embedding 贡献了关键方向信号。

## 快速开始

```bash
# 1. 训练 Tokenizer (4D OHLC)
python train_tokenizer.py

# 2. 训练模型
python train_base.py --loss focal --gamma 6.0 --label_smoothing 0.05

# 3. 评估
python TEMP/eval_v2.py --n_stocks 30 --max_steps 10
```

## 模型架构

- **2.7M 参数**: dim=256, depth=2, heads=4, GQA (kv_heads=1)
- **BSQ Tokenizer**: 2-level hierarchical, 输入 4D OHLC → vocab=1024
- **VA Embedding**: Volume/Amount 连续 MLP 注入 (`Linear(2,64)→GELU→Linear(64,256)`)
- **可选模块**: CausalReasoningBlock (cross-attention to memory tokens)
- **Loss**: CE / Focal (γ=3-10) / +Label Smoothing / +Entropy Reg
- **归一化**: 价格 historical Z-Score (per-stock train-history); VA 首日基线 + Z-Score。统计量仅来自 train 数据，causal mask 保证训推一致。

## 关键超参 (CLI)

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| `--loss` | `focal` | Focal loss 抑制零坍塌 |
| `--gamma` | `6.0` | 1-Step 最优; 8.0 为 10-Step AR 最优 |
| `--label_smoothing` | `0.05` | 配合 Focal 进一步改善 |
| `--weight_decay` | `0.001` | CE 模型推荐; Focal 可用默认 0.01 |
| `--reasoning` | flag | 启用 CausalReasoningBlock |

## 文档

| 文档 | 内容 |
|------|------|
| `main.md` | 项目设计文档 |
| `CODE_WIKI.md` | 代码架构文档 |
| `TEMP/README.md` | 历史实验记录 + 结果汇总 |

## 项目结构

```
Kronos-R-Preview/
├── config.py               # 全局配置
├── data_processor.py       # 数据管道 (归一化 + 打包)
├── train_tokenizer.py      # Stage A: Tokenizer 训练
├── train_base.py           # Stage B: Transformer 训练
├── model/
│   ├── kronos_preview.py   # 模型定义 + VA embedding
│   ├── tokenizer.py        # BSQ Tokenizer
│   └── tokenizer_config.py
├── checkpoints/            # 模型权重
├── dataset/                # CSV 数据
└── TEMP/                   # 历史实验归档 + 评估脚本
```
