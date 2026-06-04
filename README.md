# Kronos-R-Preview

基于 LLM 范式的金融时序因果预测项目。每只股票 = 一篇文档，从头读到尾，标准因果 next-token prediction。

## 核心结果

### HPO 最优模型 (57 实验, ~31.5h)

| 指标 | 最佳模型 | 数值 |
|------|---------|:----:|
| 1-Step MAPE | w2_focal_g6_ls005 | **3.99%** |
| 1-Step AmpRatio | w2_focal_g6_ls005 | **1.572x** |
| 10-Step AR MAPE | w2_focal_g8 | **8.48%** |
| 累积方向 (CumDA) | w2_focal_g10 | **52.7%** |

### 与 Baseline 对比 (299-300 只测试股票)

使用全部 pre-cutoff 历史数据作为 context 的正确评估协议：

| Model | 1-Step MAPE | 10-Step MAPE | 方向准确率 |
|-------|:-----------:|:------------:|:---------:|
| **Kronos w2_focal_g6_ls005** | **1.61%** | 8.97% | 47.5% |
| **Kronos w2_focal_g8** | **1.63%** | 8.95% | — |
| NaiveDrift | 2.20% | 7.13% | 50.4% (随机) |
| XGBoost (AR) | 2.30% | 7.24% | — |
| ARIMA(1,0,0) | 2.47% | 7.39% | — |
| EWMA(0.94) | 2.98% | 14.19% | — |

**关键发现**：
- Kronos 1-step MAPE **优于所有 baseline 27%+**（1.61% vs 2.20%）
- Baseline 的长程累积 MAPE 优势来自"预测不变"（DA=50%，无信息量），并非真正的预测能力
- Baseline 的累积 MAPE 呈振荡而非单调增长——这是常数预测碰巧接近真实价格的统计现象

## 快速开始

```bash
# 1. 训练 Tokenizer
python train_tokenizer.py

# 2. 训练 Focal 模型 (推荐)
python train_base.py --loss focal --gamma 6.0 --label_smoothing 0.05

# 3. 运行模型对比 (含 Baseline)
cd TEMP && python run_model_comparison.py
```

## 模型架构

- **2.7M 参数**: dim=256, depth=2, heads=4, GQA (kv_heads=1)
- **BSQ Tokenizer**: 2-level hierarchical, coarse(10-bit) + fine(10-bit) → vocab=1024
- **可选模块**: CausalReasoningBlock (cross-attention to memory tokens)
- **Loss**: CE / Focal (γ=3-10) / +Label Smoothing / +Entropy Reg

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
| `REPORT_HPO.md` | 完整实验技术报告 (57实验, 31.5h) |
| `REPORT_SUM.md` | 配置命名说明 + 全部结果汇总 (速查表) |

## 项目结构

```
Kronos-R-Preview/
├── train_base.py           # 训练脚本 (CLI完整)
├── train_tokenizer.py      # Tokenizer训练脚本
├── config.py               # 全局配置
├── data_processor.py       # 数据管道
├── reproducibility.py      # 随机种子
├── model/
│   ├── kronos_preview.py   # KronosPreview + CausalReasoningBlock
│   ├── tokenizer.py        # BSQ Hierarchical Tokenizer
│   └── tokenizer_config.py
├── checkpoints/
│   └── hpo_v3/             # 最优模型权重
├── dataset/                # CSV数据
└── TEMP/                   # 实验归档
    ├── run_model_comparison.py  # 统一对比脚本 (含缓存)
    └── comparison_cache/        # 逐股结果缓存
```
