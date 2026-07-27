# Experiment 04-B：Optimizer Ablation（AdamW vs Muon）

> 状态：**刷新后的正式脚本已就绪，等待 Exp 03-Sup 冻结容量依赖并完成 04-A 后运行。**
>
> 脚本会校验 04-A 的工作选择仍为 CE；若不是，会停止并要求显式修改协议，防止下游静默使用错误依赖。

## 实验问题

固定 CE 和全部其他条件，只比较：

- AdamW：`lr=3e-4`；
- Muon+AdamW：二维权重使用 Muon（`lr_muon=0.02`），其余参数使用 AdamW。

这是对两个**固定配置**的受控比较，不代表穷举调优后两类优化器的理论上限。

## 固定协议

| 项目 | 设置 |
|---|---|
| Tokenizer | Exp 02：64x192 @ 9+7 |
| GPT | Exp 03-Sup `selection.json` 的受控容量选型；正式结果待运行 |
| Loss | 04-A 工作依赖：CE |
| 唯一 arm 变量 | AdamW vs Muon+AdamW（含 Muon 专用 LR） |
| 数据 | 全量股票、完整序列 |
| 训练 | 30 epochs，无 early stop，accumulation 恒为 32；精确 128 steps/epoch、3840 total |
| 评估 | 每个 epoch × 四个 validation windows |
| 主读数 | 全 epoch 扫描后的稳定 5-epoch 成熟窗；质量、校准与码本指标分组报告 |
| Holdout | 不使用 |

历史旧协议中 AdamW 的下游行为明显优于 `Muon(lr_muon=0.02)`，因此 AdamW 是预注册的下游工作 arm；刷新实验仍需以新产物为准。正式分析会附带所选 5-epoch 成熟窗逐日均值的 5 日 moving-block bootstrap 配对诊断，但不会把它误写成多 seed 不确定性。

## 运行

```powershell
# 端到端 smoke test
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\b-optimizer-ablation\sweep_optimizer.py --smoke

# 正式运行；缺少 04-A selection.json 时会主动停止
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\b-optimizer-ablation\sweep_optimizer.py

# 只重建分析产物
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\b-optimizer-ablation\gen_plots.py
```

正式输出写入 `run_seed42/`。重复同一命令可复用已完成阶段；源码、数据、上游 selection 或协议变化时，fingerprint 会拒绝混用旧结果。

单 arm `--arms` 运行只用于诊断，必须写入单独目录；其 selection 会标记为不可供 04-C 使用。
