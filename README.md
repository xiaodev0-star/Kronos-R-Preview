# Kronos-R-Preview

基于 LLM 范式的金融时序因果预测项目。每只股票 = 一篇文档，从头读到尾，标准因果 next-token prediction。

## 核心结果 (57 实验 HPO 后)

| 指标 | 最佳模型 | 数值 |
|------|---------|:----:|
| 1-Step MAPE | w2_focal_g6_ls005 | **3.99%** |
| 1-Step AmpRatio | w2_focal_g6_ls005 | **1.572x** |
| 10-Step AR MAPE | w2_focal_g8 | **8.48%** |
| 累积方向 (CumDA) | w2_focal_g10 | **52.7%** |

## 快速开始

```bash
# 训练 (使用训练好的 Tokenizer)
python train_base.py --loss focal --gamma 6.0 --label_smoothing 0.05

# 推理 (需实现 embedding → next-token loop)
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
| `TEMP/README.md` | 历史实验文件清单 |

## 项目结构

```
Kronos-R-Preview/
├── train_base.py           # 训练脚本 (CLI完整)
├── train_tokenizer.py      # Tokenizer训练脚本
├── config.py               # 全局配置
├── data_processor.py       # 数据管道
├── model/
│   ├── kronos_preview.py   # KronosPreview + CausalReasoningBlock
│   ├── tokenizer.py        # BSQ Hierarchical Tokenizer
│   └── tokenizer_config.py
├── checkpoints/
│   └── tokenizer_tv_only.pt  # 主力Tokenizer
├── dataset/                # CSV数据
└── TEMP/                   # 历史实验归档 (权重/脚本/图表)
```
