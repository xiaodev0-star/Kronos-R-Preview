# CPT 正式 Recipe（锁定）

> 依据 `ContinuePreTrain-ToDo.md` T1（续训空间确认 + 超参收尾），
> 本地实施为 full 100-epoch from-scratch retrain（seed 42）。
> 分析基于 400 窗协议（offset 0-399）逐 epoch 评估（每 5 epoch 采样，21 点）。

## 锁定结论

**CPT 正式产出 checkpoint：`checkpoints/exp04b_best_ep100.pt`**

| 项 | 值 |
|----|----|
| 训练形态 | full 100-epoch from-scratch（seed 42） |
| optimizer | Muon + AdamW（muon params / non-muon） |
| lr_muon / lr | 0.005 / 3e-4 |
| dropout / wd / ls | 0.1 / 0.01 / 0.0 |
| fine_w / het_w | 0.3 / 0.1 |
| warmup_ratio | 0.05（5% warmup + cosine decay to 0 @ ep100） |
| loss | CE（coarse + fine + heteroscedastic NLL） |
| accumulation_steps | 32（有效 batch 更大，同 step 下优于 04-B 的 8） |
| batch_tokens | 6144 |
| 总 step | 12,800（128 steps/epoch × 100） |

## 400 窗协议指标（ep100，对比 04-B 锚点）

| 指标 | CPT ep100 | 04-B 锚点 | 状态 |
|------|-----------|-----------|------|
| coarse balance | **0.576** | 0.513 | ✅ 超锚 +0.063 |
| p10 balance | 0.427 | 0.383 | ✅ 超锚 |
| JSD | **0.319** | 0.366 | ✅ 低于锚点（更优） |
| support F1 | 0.797 | 0.634 | ✅ 大幅提升 |
| unique tokens | 63 | 47 | ✅ 接近 target 97 |
| collapse（median） | 42.2% | 28.5%* | ⚠️ 见下 |
| DA | 48.8% | 49.7%* | 观测锚点 |
| RankIC | +0.031 | +0.039* | 观测锚点 |

\* 4c72 参考 ep50（HPO arm，accumulation=8）；CPT 用 accumulation=32，
step 数不同，标注为观测锚点（ToDo §0：下游仅观测，非选型产物）。

## 曲线诊断要点

1. **token 质量全程单调改善**：balance 0.190→0.576，JSD 0.664→0.319，
   collapse 88.5%→42.2%，unique 11→63。与 4c72 参考（ep50 balance 0.608）同趋势。
2. **val_loss 在 ep13 触底后回升（2.89→3.61）**，但 token 质量**持续改善**——
   再次确认 ToDo §1.3 的判断：raw val_loss 被坍塌污染，不可作健康指标。
3. **step 对齐下 CPT 优于 4c72**：同 global_step，本地 balance 全面更高
   （如 10240 steps: 0.567 vs 0.460），更大有效 batch 训练更稳。
4. **ep100 平衡已饱和**：ep85→100 仅 +0.004，collapse 平稳。预算已吃满，
   T1 续训空间确认：无需进一步续训。
5. **H-CE 在 ep10-15 触顶（0.43 bits）后回落**——模型在后期把似然预算
   更多用于分布匹配而非单 token 预测，这符合 CPT 目标（token 质量优先）。

## 正式 recipe 判据

- ✅ 红线（token 质量）全面达标：balance/p10/JSD/support F1 均超 04-B 锚点
- ✅ 续训收益曲线已确认：100 epoch 从零训练达成 token 质量目标
- ✅ LR schedule 收尾完成：5% warmup + cosine 到 0，充分收敛
- ✅ 超参开放点已封口：维持 04-B 定案（lr_muon=0.005 等），未发现需要调整

**CPT 阶段收工。产出 checkpoint 即 `exp04b_best_ep100.pt`，供 T2/T3 与后续分支使用。**

> T2 补训后的注记（2026-08-04）：`exp04b_8ceb_ep100.pt`（同 recipe 仅
> lr_muon=0.01）token 质量全面反超（balance 0.581 vs 0.576, JSD 0.308 vs
> 0.319），为新的最佳单 checkpoint；跨轨迹 soup 判负（barrier）。是否将
> 8ceb ep100 定为 CPT 新基线并重训 Branch A 待确认（见 `T2_MODEL_SOUP.md`）。
