# Experiment 01: BitSweep — BSQ Codebook Size Selection

## 概要

扫描 10 组 $(L_1, L_2)$ 配置（bits 6-9），确定 BSQ 双层量化码本的最优大小。

**结论**：推荐 **8+6**（16K 联合词表），是最小充分码本。

## 文件结构

```
01-bitsweep/
├── README.md              ← 本文件
├── Exp-SweepBit.md        ← 实验报告（含图表引用）
├── sweep_bits.py          ← 主实验脚本：tokenizer 训练 + GPT 训练 + 评估
├── sweep_full_eval.py     ← 全量推理：全部 test stocks × 全部日期
├── gen_sweep_plots.py     ← 绘图脚本（从 metrics.json 生成 9 张图表）
├── plt/                   ← 生成的图表
│   ├── summary_table.png
│   ├── effective_bits_saturation.png
│   ├── unique_vs_vocab.png
│   ├── utilization_vs_collapse.png
│   ├── utilization_ladder.png
│   ├── radar_comparison.png
│   ├── paired_comparison.png
│   ├── da_convergence.png
│   └── reconstruction_diminishing_returns.png
└── checkpoints/           ← 实验产出（4.4GB）
    ├── seed42/            ← 10 组 tokenizer + GPT 权重
    ├── full_eval_seed42/  ← 全量推理结果（metrics.json + npz）
    └── feature_cache/     ← tokenizer 训练特征缓存
```

## 复现

```bash
cd experiments/01-bitsweep

# 1. 训练 10 组 tokenizer + GPT + 评估（约 10-14 小时）
python sweep_bits.py

# 2. 全量推理（约 4-6 小时）
python sweep_full_eval.py --resume

# 3. 生成图表
python gen_sweep_plots.py
```

所有脚本通过 `sys.path` 引用项目根目录的框架代码（`config.py`, `data_processor.py`, `model/`, `eval_helpers.py` 等），无需额外安装。

## 关键结果

| 指标 | 8+6 读数 | 说明 |
|------|----------|------|
| Unique | 144 | 满足 ≥64 硬约束 |
| Collapse | 26.0% | 满足 ≤30% 硬约束 |
| Util% | 0.879% | 所有满足约束配置中最高 |
| Eff Bits | 7.17 | 接近数据内在维度饱和边界 |
| 重建 MAE | 0.228 | 处于递减曲线拐点 |
| DA | 49.04% | 与其他配置无实质差异（极差 1.04 pp） |

## 依赖

- 项目根目录的框架代码：`config.py`, `data_processor.py`, `model/`, `eval_helpers.py`
- `train_base.py`（GPT 训练，由 `sweep_bits.py` 调用）
- `train_tokenizer.py`（tokenizer 训练逻辑，由 `sweep_bits.py` 内联调用）
- `eval_windowed.py`（窗口评估，由 `sweep_bits.py` 调用）
