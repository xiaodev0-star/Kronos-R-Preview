# Experiment 04-A：Loss Ablation（Focal γ=4 vs CE）

> 状态：**刷新后的正式脚本已就绪；等待 Exp 03-Sup 冻结容量依赖后再运行正式全量实验。**
>
> 历史协议曾选择 CE，但历史绝对指标来自旧 tokenizer、2-layer baseline 和旧评估流程，不能直接当作本次结果。本次实验会在 Exp 02/03 的正式选型上重新验证这一结论。

## 实验问题

在其余条件完全相同时，比较 Focal Loss（γ=4）和标准 Cross-Entropy。`train_base.py` 的 validation 统一使用 CE，因此 `val_loss` 可作为共同收敛诊断；决策仍以逐日 DA、Daily RankIC、MAPE，以及 collapse、token 多样性和 AmpRatio 为主，不能只按 loss 排名。

## 固定协议

| 项目 | 设置 |
|---|---|
| Python | `D:\conda_envs\llm-t\Scripts\python.exe`（3.12.10） |
| Tokenizer | Exp 02 `selection.json`：64x192 @ 9+7 |
| GPT | Exp 03-Sup `selection.json` 的受控容量选型；正式结果待运行 |
| 优化器 | AdamW，lr=3e-4 |
| 唯一 arm 变量 | `focal, γ=4` vs `ce` |
| 数据 | 全量股票、完整序列 |
| 训练 | 30 epochs，无 early stop，accumulation 恒为 32；精确 128 steps/epoch、3840 total |
| 评估 | 每个 epoch；offset 0/100/200/300，各 20 个有效交易日 |
| 主读数 | 扫描全部 epoch；在 1%-of-best val-loss 区间内选择连续 5 epoch 成熟窗 |
| 健康门 | 每日 collapse ≤35%，每日 unique token ≥32 |
| Holdout | offset 400 起始区间继续封存 |

选择文件中的 CE 是供下游串联使用的**预注册工作 arm**，不是自动宣称的生产赢家。正式运行后必须同时查看 `arm_envelopes.csv`、`ANALYSIS.md` 和健康门通过数。分析器会先为每个 arm 选出稳定 5-epoch 成熟窗，再对逐日指标取均值并在每个 validation window 内做 5 日 moving-block bootstrap；该区间只诊断日期不确定性，不代表多 seed 训练不确定性。

## 运行

```powershell
# 先跑临时端到端检查；产物自动删除
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\a-loss-ablation\sweep_loss.py --smoke

# 正式运行两个 arm
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\a-loss-ablation\sweep_loss.py

# 断点后重复同一命令即可；manifest 会拒绝协议漂移

# 只重建汇总、图和 selection.json
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\a-loss-ablation\gen_plots.py
```

输出写入 `run_seed42/`。每个 arm 保存 30 个 epoch checkpoint、四窗口轨迹、运行日志和可恢复训练状态；两个 arm 共享本研究内的 token cache。

`--arms` 可用于 smoke 或单 arm 诊断；正式子集必须指定另一个 `--output_root`，而且生成的 selection 不具备下游资格，不能静默解锁 04-B。

## 与历史结果的关系

旧实验在旧协议下明显偏向 CE，因此本次仍把 CE 设为下游工作依赖。但 tokenizer 容量、GPT 容量、更新步数和评估口径都已变化，本次结果应被视为新的受控复验，不能把旧的 DA/RankIC 数值复制到新结论中。
