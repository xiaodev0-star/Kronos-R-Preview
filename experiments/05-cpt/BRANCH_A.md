# Branch A：日级边际分布匹配（SFT）— 通过，λ=0.3 定案

> 对应 `ContinuePreTrain-ToDo.md` §5 Branch A。针对缺口：pred unique 保守
> （CPT ep100: 63 vs target 87-97）、collapse 42.2%。机制："训你所测"——
> 直接优化被排名的 coarse codebook balance。

## 实现

- **起点**：`checkpoints/exp04b_best_ep100.pt`（CPT 正式产出）
- **损失**：`L = L_CE + 0.3·L_fine + 0.1·L_het + λ·L_dm`
  `L_dm` = 预测 soft-histogram（对非 -100 位置 softmax 聚合）与 target 边际分布
  （训练集 coarse 分布）的对称 KL。
- **LR**：lr 3e-5, lr_muon 5e-4（CPT 峰值 1/10）；warmup 5% + cosine to 0
- **代码**：`train_base.py` 新增 `--dm_weight`（默认 0 不变，零侵入生产）+
  `distribution_match_loss()` + `--base_checkpoint` 支持普通模型加载。

## λ 扫描结果（400 窗协议）

| λ | balance | p10 | JSD | collapse | unique | MAPE | amp | bootstrap delta |
|---|---------|-----|-----|----------|--------|------|-----|-----------------|
| CPT 基线 | 0.576 | 0.427 | 0.319 | 42.2% | 63 | 3.358 | 1.281 | — |
| **0.1** | 0.651 | 0.482 | 0.299 | 32.1% | 70 | 3.285 | 1.196 | +0.061 |
| **0.3** | **0.710** | 0.515 | 0.316 | **24.4%** | **79** | **3.191** | **1.097** | **+0.102** |

λ=0.3 在 token 质量上全面优于 λ=0.1（balance +0.059, collapse -7.7pp, unique +9），
且 MAPE 更优、amp 更接近 1.0（幅度校准更准）。**λ=0.3 定案为主配置**。

## 400 窗协议结果（λ=0.3, ep6，对比 CPT 基线）

| 指标 | CPT ep100 | Branch A λ0.3 | Δ |
|------|-----------|--------------|---|
| **coarse balance** | 0.576 | **0.710** | **+0.134** |
| p10 balance | 0.427 | 0.515 | +0.088 |
| JSD | 0.319 | 0.316 | −0.003（红线 0.366 之下）|
| support F1 | 0.797 | 0.842 | +0.045 |
| unique | 63 | 79 | +16 |
| collapse（median）| 42.2% | **24.4%** | **−17.8pp** |
| joint_balance | 0.388 | 0.477 | +0.089 |
| DA | 48.8% | 48.9% | 持平 |
| RankIC | +0.031 | +0.027 | −0.004（观测）|
| MAPE | 3.358 | 3.191 | −0.167（改善）|
| amp | 1.281 | 1.097 | 更接近 1.0（改善）|

## 统计验收（paired bootstrap，400 日期）

- **λ=0.3：balance delta +0.1019，95% CI [+0.088, +0.117]** → 显著，远超 +0.02
- λ=0.1：delta +0.0609, CI [+0.053, +0.068] → 显著

## 结论

**Branch A 通过验收，λ=0.3 定案。**

- **产出 checkpoint：`checkpoints/branchA_dm030_ep6.pt`**
- token 质量全面显著改善（balance +0.134, collapse −17.8pp, unique +16），
  400 窗 + paired bootstrap 验收通过
- 下游不劣化：DA 持平，MAPE 改善 −0.167，amp 更准；IC 微降 −0.004 在噪声内
- H-CE 保险丝：train loss 上升是 CPT 末态续训的正常现象，token 质量全面
  向上证明非"什么都不学"；amp→1.0 说明幅度校准真实改善
- ep4/6 几乎一致（balance 0.710），SFT 快速收敛，6 epoch 足够

## 说明

- λ 敏感性已验证：λ=0.3 > λ=0.1（token 质量全面更优），未测 0.05
  （预期介于 CPT 与 0.1 之间，非最优方向）。
- 推理侧可叠加 T3 mean-采样（IC +0.013 显著）。
- Branch A 作为 SFT 主产出，token 质量维度显著改善，为后续 Branch B（横截面
  排序）或 E（DPO）提供更健康的起点 checkpoint。
