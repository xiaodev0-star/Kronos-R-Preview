# Experiment 04-C：全量 GPT HPO

> 状态：**刷新脚本已就绪；等待 Exp 03-Sup 冻结容量依赖及 04-A/B 完成后开始正式搜索。**
>
> 正式运行依赖 04-A=`ce`、04-B=`adamw` 的刷新后 `selection.json`。最终 holdout 继续封存；只有所有计划 trial 完成且至少一个候选通过成熟期健康门时，显式 `--holdout` 才能打开。

## 为什么重做

历史 04-C 完成了 15 个 50-epoch、seed=42 的全量 trial，但使用的是旧 8+6 tokenizer 和 2-layer baseline。0/15 trial 通过健康门，DA leader 相对 baseline 的提升区间跨零，holdout 未打开。历史结果仍提供两个重要先验：

1. 更低的 LR 往往带来更好的 DA/RankIC，尽管高 LR 的 `val_loss` 更低；
2. 质量在约 epoch 28–37 进入平台，epoch 50 并非唯一合理读数。

新脚本把这些先验转化为可解释的预注册搜索，而不是继续做混杂的随机抽样。

## 刷新后的固定协议

| 项目 | 设置 |
|---|---|
| Python | 3.12.10，固定 `llm-t` 环境 |
| Tokenizer | Exp 02：64x192 @ 9+7 |
| GPT | Exp 03-Sup `selection.json` 的受控容量选型；正式结果待运行 |
| Loss / optimizer | 04-A/B：CE + AdamW |
| 数据 | 全量 4695 股票、完整序列 |
| 训练 | 默认 30 epochs，无 early stop，accumulation 恒为 32；精确 128 steps/epoch、3840 total |
| 评估 | 每个 trial 的每个 epoch × offset 0/100/200/300 × 20 日 |
| 主读数 | 扫描全部 epoch 后选出的连续 5-epoch 成熟窗中位数 |
| 合法候选 | 所选 5-epoch 成熟窗 **全部**满足每日 collapse ≤35%、每日 unique ≥32 |
| Holdout | offset 400 起 80 日，仅显式命令打开一次 |

30 epochs × 恒定 accumulation=32 会由训练器硬校验为 3,840 次 optimizer update，接近 Sup 的 50-epoch 动态 accumulation 4,160 次，但只需 60% 的前后向数据遍历。分析会扫描全部 epoch，并在首次成熟后优先选择仍位于 1%-of-best val-loss 区间内的连续 5 epoch；最后 10 epoch 不再被默认视为成熟答案。

## 搜索计划

候选顺序是确定的，`--n_trials N` 永远取下面计划的前 N 项。Sup 尚未选型时，默认 10 小时预算继续按较保守的旧 `xlarge` 实测成本估算：30 epoch 训练 + 全 epoch 四窗评估约 **166 分钟/trial**，另留 30 分钟缓冲，因此默认只计划 **3 个 trial**。若 Sup 选出更小架构，该估计只会更保守；若要覆盖更多候选，必须显式增加 `--time_budget_hours` 或 `--n_trials`。

| 顺序 | 相对 baseline 的改动 |
|---:|---|
| 1 | baseline：lr=3e-4，dropout=.10，wd=.01，fine=.30，het=.10，warmup=.05 |
| 2–6 | 单独扫描 lr：1e-4、5e-5、7.5e-5、1.5e-4、2e-4 |
| 7–8 | lr=1e-4，dropout=0 / .05 |
| 9–10 | lr=1e-4，label smoothing=.03 / .05 |
| 11–12 | lr=1e-4，dropout=.15 / .20 |
| 13–20 | 更强 smoothing，以及 fine/het/wd/warmup 的单因素探针 |

先完整扫描 LR，是因为历史证据显示 LR 对 `val_loss` 与下游质量的方向相反；随后围绕低 LR 做正则化探针，可避免无法归因的随机组合。

## 排序与 holdout

不使用加权总分。leaderboard 依次使用：

1. 所选成熟窗 5/5 epoch 通过健康门；
2. 所选窗中位 daily DA；
3. 所选窗中位 Daily RankIC；
4. 所选窗中位 MAPE；
5. AmpRatio 与 1 的距离；
6. 最坏单日 collapse；
7. target-relative codebook balance（最终 guardrail，不与 DA 加权）。

如果没有健康候选，实验结论是“无合法赢家”，holdout 保持封存。若有合法赢家，`--holdout` 对所选 5-epoch 窗口的中心 checkpoint 与 baseline 对应中心 checkpoint 各评一次，并报告逐日配对 moving-block bootstrap 区间。

## 运行

```powershell
# 不写正式目录的端到端检查
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\c-hpo\sweep_hpo.py --smoke

# 默认：10 小时规划预算、30 epochs，按 xlarge 成本自动取计划前 3 项
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\c-hpo\sweep_hpo.py

# 显式运行计划前 6 项；重复命令可断点复用
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\c-hpo\sweep_hpo.py --n_trials 6

# 只有正式搜索完整结束且存在健康赢家时才允许执行
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\04\c-hpo\sweep_hpo.py --holdout
```

输出位于 `run_seed42/`。`study_manifest.json` 固化数据签名、源码哈希、上游 selections、完整候选顺序和协议；协议变化必须换输出目录，不能静默复用旧结果。所有 metadata 一致的 trial 共用一份 prepared-eval cache，避免为每个 trial 重复保存约 500 MiB 的相同输入张量；正式启动前还会按 checkpoint 预算检查可用磁盘。

## 辅助分析

`evaluate_epoch_trajectory.py` 现在可从新 leaderboard 自动定位 leader，也可由 A/B/C 显式传入任意 trial/arm。`analyze_epoch_trajectory.py` 默认使用 leaderboard 所选成熟窗的中心 checkpoint 作为参考，并按实际训练长度解析阶段与候选 epoch，不再硬编码 50 epochs 或最终 checkpoint。
