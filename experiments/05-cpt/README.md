# Exp 05: Continue PreTrain (CPT) 阶段收官报告

> 执行周期：2026-08-04。依据 `ContinuePreTrain-ToDo.md` 公共 Trunk T1/T2/T3/T5
> 与 Branch 选取。全程无人值守完成。

## 任务清单与结果

| 任务 | 结果 | 详情 |
|------|------|------|
| **T1** 续训空间确认 + 超参收尾 | ✅ 完成，锁定 CPT recipe | `CPT_RECIPE.md` |
| **T2** 模型汤 | ⚠️ 判负（barrier 存在即弃）；8ceb ep100 成新最佳单 | `T2_MODEL_SOUP.md` |
| **T3** 采样自洽推理 | ✅ 完成，mean 采样 IC 显著 | `T3_SELF_CONSISTENCY.md` |
| **T5** 数据扩容评估 | ✅ 完成，判负（不可行） | `T5_data_expansion.md` |
| **Branch 选取** | ✅ Branch A（分布匹配）通过 | `BRANCH_A.md` |

## 核心产物

- **CPT 正式 checkpoint**：`checkpoints/exp04b_best_ep100.pt`
  - balance 0.576, JSD 0.319, collapse 42.2%, support F1 0.797, unique 63
  - recipe：muon, lr_muon=0.005, lr=3e-4, dropout=0.1, wd=0.01, fine_w=0.3,
    het_w=0.1, warmup=0.05, CE；accumulation 32；5% warmup + cosine to 0
- **T2 补训 8ceb leg**：`checkpoints/exp04b_8ceb_ep100.pt`（lr_muon=0.01，
  其余 recipe 相同）——**新的最佳单 checkpoint**：balance 0.581, JSD 0.308,
  collapse 42.8%, unique 61, DA 49.6%
- **Branch A 产出**：`checkpoints/branchA_dm030_ep6.pt`（基于 4c72 ep100 起步）
  - balance **0.710**（+0.134），collapse **24.4%**（-17.8pp），JSD 0.316
  - 400 窗协议 + paired bootstrap 验收通过（CI [+0.088, +0.117]）

## 关键决策记录

1. **CPT recipe 锁定**：100-epoch from-scratch 达成 token 质量目标，
   ep100 平衡饱和，无需续训。ToDo §1.3 确认——raw val_loss 被坍塌污染，
   禁用；token 质量全程单调改善。
2. **T2 模型汤**：补训 8ceb leg（lr_muon=0.01，3.55h）后完整执行原始
   跨轨迹 soup——**判负**：连线中点（soup）balance −0.065/JSD +0.062
   （bootstrap CI 均显著），插值剖面中部凹陷、JSD 破红线，barrier 存在
   即弃（ToDo）。**附带发现：8ceb ep100 为新的最佳单 checkpoint**（token
   质量全面反超 4c72 ep100），CPT 起点候选建议切换（Branch A 是否重训待确认）。
3. **T3 mean-采样**：RankIC +0.0129 显著（CI 不含零），成为默认推理配置；
   vote 策略不显著弃用。DA 单 seed 为 coin-flip 已知限制。
4. **T5 判负**：数据扩容在本项目约束下不可行（历史用尽、cutoff 后受
   验证协议约束、跨市场不可得）。1.5 tokens/param 数据受限留待后续。
5. **Branch A 通过**：直接针对 CPT 确认缺口（unique 保守、collapse 高），
   分布匹配 loss 显著改善 token 质量且过 CI。下游不劣化。

## 诊断/脚本

- `analyze_cpt_100ep.py` — CPT 曲线诊断（loss + token 质量 + H-CE）
- `t2_checkpoint_soup.py` — checkpoint soup + barrier 检查
- `t3_sampling_self_consistency.py` — K=8 采样自洽推理
- `eval_branchA.py` — Branch A 400 窗评估
- `bootstrap_compare.py` — paired bootstrap 验收
- 分析输出：`server_runs/results/04b-cpt/seed42/trials/local_cpt/diagnostics/`

## 未决事项（留待后续）

- **是否以 8ceb ep100 为新 CPT 基线重训 Branch A**（现 branchA 基于 4c72 ep100；
  8ceb 基线可能进一步抬升 token 质量起点，需用户确认）
- λ ∈ {0.05, 0.3} 补扫（λ=0.3 验证中）
- Branch B（横截面排序）未执行（A 已显著改善 token 质量）
- 多 seed 复核（DA coin-flip 已知限制）
- holdout 400 全程未触碰——留待显式最终命令（ToDo §1.4）
