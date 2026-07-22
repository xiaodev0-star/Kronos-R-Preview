# Experiment 04-B: Optimizer Ablation — AdamW vs Muon

> 前置：Exp 04-A (Loss Ablation) 确定最优 loss 后，本实验使用该配置。

## 概要

在完全相同的超参下，对比 **AdamW** 和 **Muon+AdamW** 两种优化器的训练收敛行为和下游预测质量。

**动机**：Exp 03 (GPT Scaling) 的扫描脚本默认使用 Muon，但生产 baseline (Exp 01/02 + HPO) 使用 AdamW。两者从未在相同配置下做过严格对比。

**设计原则**：
- 除 optimizer 外，所有超参完全一致（focal γ=4, dropout=0.1, het_weight=0.1, fine_weight=0.3）
- 不限 seq_len，不设 max_stocks，让模型自然收敛
- Early stopping patience=5：连续 5 个 epoch val_loss 不降即停止
- 最多 50 个 epoch，足够长以观察完整收敛曲线

## 文件结构

```
04-optimizer-ablation/
├── README.md               ← 本文件
├── sweep_optimizer.py      ← 主实验脚本
├── gpt_adamw.pt            ← AdamW 最优 checkpoint
├── gpt_muon.pt             ← Muon 最优 checkpoint
├── history_exp04a_adamw.json  ← AdamW 训练历史
├── history_exp04a_muon.json   ← Muon 训练历史
├── eval_adamw.json         ← AdamW windowed eval
├── eval_muon.json          ← Muon windowed eval
├── comparison.json         ← 对比结果汇总
└── plt/                    ← 生成的图表
    ├── loss_curves.png
    └── metric_comparison.png
```

## 复现

```bash
cd experiments/04-optimizer-ablation

# 两个 arm 依次训练 + 评估（主要命令）
python sweep_optimizer.py

# 只跑一个 arm
python sweep_optimizer.py --arm adamw
python sweep_optimizer.py --arm muon

# 断点续算
python sweep_optimizer.py --resume

# 只做评估（训练完成后）
python sweep_optimizer.py --eval_only
```

## 超参配置

| 参数 | AdamW | Muon | 说明 |
|------|-------|------|------|
| optimizer | AdamW | Muon+AdamW | **唯一变量** |
| lr (AdamW) | 3e-4 | 3e-4 | 相同 |
| lr_muon | — | 0.02 | Muon 专用 LR |
| gamma | 4.0 | 4.0 | focal loss |
| dropout | 0.1 | 0.1 | |
| weight_decay | 0.01 | 0.01 | AdamW 部分 |
| het_weight | 0.1 | 0.1 | heteroscedastic |
| fine_weight | 0.3 | 0.3 | fine-token aux |
| epochs | 50 (max) | 50 (max) | early stop 决定实际 |
| early_stop_patience | 5 | 5 | |
| batch_tokens | 12288 | 12288 | 自适应 batching |
| max_stocks | 0 (all) | 0 (all) | 4695 只 |
| max_seq_len | 0 (unlimited) | 0 (unlimited) | 保留全部上下文 |

## 评估指标

- **训练侧**：converged epoch、best val_loss、train/val loss 曲线
- **下游侧**（windowed eval, 20 days）：DA、RankIC、Collapse、Unique、AmpRatio、MAPE

## 通过标准

选择哪个 optimizer 取决于：
1. **收敛速度**：谁用更少 epoch 达到更低 val_loss
2. **最终质量**：val_loss 和 DA 的绝对值
3. **训练稳定性**：loss 曲线是否平滑，有无震荡
