# Experiment 02: Tokenizer Hyperparameter Tuning

## 概要

在 bits=[8,6] 固定后，对 tokenizer 的 embedding_dim × hidden_dim 做 3×2 扫描。
通过两阶段评估（重建 MAE + GPT 下游验证）确定最优结构参数。

**结论**：推荐 **embedding_dim=64, hidden_dim=192**，重建 MAE 降低 14.5%，GPT 下游不退化。

## 文件结构

```
02-tokenizer-tuning/
├── README.md                        ← 本文件
├── Exp-TokenizerTuning.md           ← 实验报告（含图表引用）
├── plan-tokenizer.md                ← 实验前的调参计划
├── sweep_tokenizer.py               ← 阶段一：embedding_dim × hidden_dim 扫描
├── sweep_tokenizer_validate.py      ← 阶段二：GPT 下游验证
├── gen_plots.py                     ← 绘图脚本
├── tok_sweep_results.json           ← 阶段一原始数据（6 组 MAE/MSE/Unique/Collapse）
├── tok_sweep_gpt_validation.json    ← 阶段二原始数据（2 组 GPT 下游指标）
├── tok_sweep_emb48_hid192.pt        ← baseline tokenizer 权重
├── tok_sweep_emb64_hid192.pt        ← 最优 tokenizer 权重
└── plt/                             ← 生成的图表
    ├── mae_ranking.png
    ├── heatmap.png
    ├── gpt_validation.png
    └── tok_vs_gpt.png
```

## 复现

```bash
cd experiments/02-tokenizer-tuning

# 阶段一：tokenizer 扫描（约 1 小时）
python sweep_tokenizer.py

# 阶段二：GPT 验证（约 2-3 小时）
python sweep_tokenizer_validate.py

# 生成图表
python gen_plots.py
```

## 关键结果

| 阶段 | 指标 | 64x192 | 48x192 (baseline) | 判定 |
|------|------|--------|---------------------|------|
| Tokenizer | MAE | 0.195 | 0.228 | −14.5% |
| GPT | DA | 49.29% | 49.15% | +0.14pp ✅ |
| GPT | Collapse | 27.2% | 24.9% | +2.3pp ✅ |
| GPT | RankIC | 0.0343 | 0.0208 | +65% |

## 依赖

- 项目根目录框架代码：`config.py`, `data_processor.py`, `model/`, `eval_helpers.py`
- `train_base.py`（GPT 训练）
- `train_tokenizer.py`（tokenizer 训练，需含 `--embedding_dim` / `--hidden_dim` 参数）
- `eval_windowed.py`（窗口评估）
