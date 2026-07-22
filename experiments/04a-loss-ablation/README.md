# Experiment 04-A: Loss Ablation — Focal (γ=4) vs Cross-Entropy

## 概要

对比 **Focal Loss (γ=4)** 和 **标准 Cross-Entropy** 在完全相同训练条件下的收敛行为和预测质量。

**动机**：Focal Loss 源自目标检测（Lin et al., 2017），核心思想是通过 `(1-p_t)^γ` 下调 easy sample 的权重、聚焦 hard example。用于金融时序预测存在两个根本性疑问：
1. 金融时序不存在目标检测那样的极端类别不平衡（up/down 近似 50/50）
2. "Hard example" 在金融语境下可能是噪声而非信号——聚焦噪声可能有害

HPO 只在 focal 内部扫了 γ ∈ {2,3,3.5,4,5,6}，**从未与标准 CE 做过公平对比**。本实验填补这一空白。

## 设计原则

- **唯一变量**：loss 函数（focal γ=4 vs ce）
- 其余超参完全一致，均为 HPO 确认的最优默认值
- 不限 seq_len，不设 max_stocks，保留全部历史上下文
- Early stopping patience=5，最多 50 epoch，让模型自然收敛
- 不加 label smoothing 或 entropy 正则，隔离 loss 变量

## 文件结构

```
04a-loss-ablation/
├── README.md               ← 本文件
├── sweep_loss.py           ← 主实验脚本
├── gen_plots.py            ← 绘图脚本
├── gpt_focal.pt            ← Focal 最优 checkpoint
├── gpt_ce.pt               ← CE 最优 checkpoint
├── history_exp04a_focal.json  ← Focal 训练历史
├── history_exp04a_ce.json     ← CE 训练历史
├── eval_focal.json         ← Focal windowed eval
├── eval_ce.json            ← CE windowed eval
├── comparison.json         ← 对比结果汇总
└── plt/                    ← 生成的图表
    └── loss_comparison.png
```

## 复现

```bash
cd experiments/04a-loss-ablation

# 两个 arm 依次训练 + 评估
python sweep_loss.py

# 只跑一个 arm
python sweep_loss.py --arm focal
python sweep_loss.py --arm ce

# 断点续算 / 只做评估
python sweep_loss.py --resume
python sweep_loss.py --eval_only
```

## 超参配置

| 参数 | Focal | CE | 说明 |
|------|-------|-----|------|
| **loss** | **focal** | **ce** | **唯一变量** |
| gamma | 4.0 | — | CE 不使用 gamma |
| label_smoothing | 0.0 | 0.0 | 隔离变量 |
| entropy_alpha | 0.0 | 0.0 | 隔离变量 |
| optimizer | adamw | adamw | |
| lr | 3e-4 | 3e-4 | |
| dropout | 0.1 | 0.1 | |
| weight_decay | 0.01 | 0.01 | |
| het_weight | 0.1 | 0.1 | |
| fine_weight | 0.3 | 0.3 | |
| epochs | 50 (max) | 50 (max) | early stop 决定 |
| early_stop_patience | 5 | 5 | |
| max_stocks | 0 (all) | 0 (all) | 4695 只 |
| max_seq_len | 0 | 0 | 不限 |

## 通过标准

| 判定 | 条件 |
|------|------|
| **Focal 有效** | DA ≥ CE + 0.5pp 且 val_loss ≤ CE |
| **CE 更优** | DA ≥ Focal + 0.5pp 或 val_loss 显著更低 |
| **无差异** | DA 差异 < 0.5pp 且 val_loss 接近 |

## 理论背景

**Focal Loss**（Lin et al., CVPR 2017）：
```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```
- γ=0 退化为标准 CE
- γ>0 下调 easy sample 权重，聚焦 hard sample
- 原始论文用于 one-stage 目标检测（前景/背景极度不平衡）

**Cross-Entropy**：
```
CE(p) = -log(p_t)
```
- 所有 token 等权重，无聚焦机制
- LLM 领域的默认 loss（GPT、LLaMA 等均使用 CE）

**核心问题**：金融时序中，"hard sample" 是噪声还是信息？如果 γ=4 聚焦的是日内剧烈波动（往往是事件驱动噪声），可能损害泛化。
