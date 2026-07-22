# Experiment 03: GPT Model Scaling

## 概要

5 个架构（2.4M~16.5M）全量训练+评估，验证 GPT 扩容是否能提升 DA。

**结论**：2.7M baseline 最优——模型越大 DA 越低，RankIC 越差。数据量是瓶颈，不是模型容量。

## 文件结构

```
03-gpt-scaling/
├── README.md                     ← 本文件
├── Exp-GPTScaling.md             ← 实验报告
├── plan-gpt-scaling.md           ← 实验前计划
├── sweep_gpt_arch.py             ← 扫描脚本（支持断点续算）
├── gpt_arch_sweep_results.json   ← 原始评估数据
├── gpt_arch_sweep_state.json     ← 断点续算状态
├── gpt_arch_baseline.pt          ← baseline 权重 (2.4M)
├── gpt_arch_wide.pt              ← wide 权重 (5.1M)
├── gpt_arch_deep.pt              ← deep 权重 (4.3M)
├── gpt_arch_large.pt             ← large 权重 (7.2M)
├── gpt_arch_xlarge.pt            ← xlarge 权重 (16.5M)
└── plt/
    ├── summary_table.png
    ├── da_vs_size.png
    ├── radar.png
    └── rankic_ampratio.png
```

## 复现

```bash
cd experiments/03-gpt-scaling

# 全部 5 个架构（约 4-5 小时，支持断点续算）
python sweep_gpt_arch.py

# 指定子集
python sweep_gpt_arch.py --configs "baseline,deep"

# 生成图表
python gen_plots.py
```

## 关键结果

| Config | ~Params | DA% | RankIC | 结论 |
|--------|---------|-----|--------|------|
| baseline | 2.4M | **49.05%** | **0.0229** | 最优 |
| xlarge* | 16.5M | 48.99% | 0.0203 | seq=4096 |
| deep | 4.3M | 48.87% | 0.0132 | 深度扩展优于宽度 |
| wide | 5.1M | 48.53% | 0.0065 | 宽度扩展更差 |
| large | 7.2M | 48.28% | 0.0041 | 过拟合 |

## 依赖

- 项目根目录框架代码
- `train_base.py`（需含 `--dim`, `--depth`, `--heads`, `--gradient_checkpointing` 参数）
- `eval.py windowed`（子进程评估）
- `experiments/02-tokenizer-tuning/` 的生产 tokenizer（tok_sweep_emb64_hid192.pt）
