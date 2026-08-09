# Exp 05: Continue PreTrain (CPT) 阶段收官报告

> 执行周期：2026-08-04 ~ 2026-08-05。依据 `ContinuePreTrain-ToDo.md` 公共 Trunk
> T1/T2/T3/T5 与全 Branch（A-G）。全程无人值守完成。

## 任务清单与结果

| 任务 | 结果 | 详情 |
|------|------|------|
| **T1** 续训空间确认 + 超参收尾 | ✅ 完成，锁定 CPT recipe | `CPT_RECIPE.md` |
| **T2** 模型汤 | ⚠️ 判负（barrier 存在即弃）；8ceb ep100 成新最佳单 | `T2_MODEL_SOUP.md` |
| **T3** 采样自洽推理 | ✅ 完成，mean 采样 IC 显著 | `T3_SELF_CONSISTENCY.md` |
| **T5** 数据扩容评估 | ✅ 完成，判负（不可行） | `T5_data_expansion.md` |
| **Branch A** 日级边际分布匹配 | ✅ **通过**（基于 8ceb 复跑） | `BRANCH_A.md` |
| **Branch B** 横截面排序（ListNet）| ⚠️ 判负（token 坍缩破红线）| `BRANCH_B.md` |
| **Branch C** 非平稳适应 | ⚠️ 判负（recency MAPE 恶化/regime 无改善）| `BRANCH_C.md` |
| **Branch D** 多 token 预测头（MTP）| ⚠️ 判负（一致性过滤恶化）| `BRANCH_D.md` |
| **Branch E** 市场反馈偏好（DPO）| ⚠️ 判负（红线破位）| `BRANCH_E.md` |
| **Branch F** LoRA-per-regime | ⚠️ 判负（条件化无收益）→ **G 冻结** | `BRANCH_F.md` |

## 核心产物

- **CPT 新基线**：`checkpoints/exp04b_8ceb_ep100.pt`（lr_muon=0.01）
  - balance 0.581, JSD 0.308, collapse 42.8%, unique 61, DA 49.6%
  - recipe：muon, lr_muon=0.01, lr=3e-4, dropout=0.1, wd=0.01, fine_w=0.3,
    het_w=0.1, warmup=0.05, CE；accumulation 32；5% warmup + cosine to 0
- **Branch A 产出**：`checkpoints/branchA_dm030_8ceb_ep5.pt`（基于 8ceb ep100）
  - balance **0.698**（+0.117 vs 8ceb 基线），collapse **25.6%**（-17.2pp），unique 78
  - 400 窗 + paired bootstrap 通过（CI [+0.095, +0.120]）；DA 持平、MAPE 改善
- **全分支裁定汇总**：`server_runs/results/04b-cpt/seed42/trials/selection.json`

## 关键决策记录

1. **CPT recipe 锁定**：100-epoch from-scratch；T2 附带发现 lr_muon=0.01（8ceb）反超
   4c72（balance 0.581 vs 0.576），CPT 新基线切 8ceb ep100（用户确认）。
2. **Branch A 通过**：分布匹配（λ=0.3）显著改善 token 质量（balance +0.117,
   collapse -17.2pp），下游不劣化。E 的 π_ref 用此产出。
3. **B/C/D/E/F 全部判负**（机制各不同）：
   - **B（ListNet）**：直接优化横截面排序 → coarse token 坍缩（balance -0.27, JSD 破
     红线 0.366, unique 61→23），RankIC 无改善。
   - **C（recency/regime）**：从收敛终态微调时重采样影响极小；C2 MAPE 显著恶化、
     C3 近端不显著。
   - **D（MTP）**：主头红线不破（balance 0.590 略升），但一致性过滤 acted DA 显著
     恶化（-0.63pp）——未来头无有效一致性信号。
   - **E（DPO）**：β=0.1 第一个快照红线破位（balance 0.646、collapse 35.1%）——
     偏好目标与 token 质量红线根本冲突。
   - **F（LoRA-per-regime）**：冻结 backbone 只训 0.79M LoRA 几乎无效果（learned≈0.003
     bit、ΔW≈0）；条件化判负，**G 永久冻结**。
4. **symbol 冲突已知限制**：4 组指数/ETF 与股票同名 symbol（1/688/852/905），评估
   数据里 4 只股票被指数覆盖（<0.1% 影响）。用户判定不影响，暂不修（见 memory）。
5. **训练-评估数据一致性核查**：cache（2013 起长历史）与当前 CSV（2019 起）起点差异
   是 symbol 冲突的次生表现；其余 4687 只股票训练/评估数据一致。

## 诊断/脚本

- `analyze_cpt_100ep.py` / `t2_cross_config_soup.py` / `t3_sampling_self_consistency.py`
- `eval_branchA/B/C/D/F.py`、`bootstrap_compare.py`、`bootstrap_regime_compare.py`
- `d_consistency_filter.py`（D）、`e_build_pairs.py`/`e_train_dpo.py`/`e_branchE_driver.py`（E）、
  `eval_branchF_specialization.py`（F/G 门槛）
- Branch B/C/D/F 的模型/训练代码已合入 train_base.py / data_processor.py / model/（默认关闭）

## 未决事项（留待后续）

- **holdout 400 全程未触碰**——留待显式最终命令（ToDo §1.4）
- multi-seed 复核（DA/IC 单 seed coin-flip 已知限制；token 质量单 seed 可判）
- symbol 冲突彻底修复（load_stocks 用文件名做 symbol）——影响 <0.1%，用户判定暂不修
- λ 敏感性（Branch A 未测 0.05）等次要项
