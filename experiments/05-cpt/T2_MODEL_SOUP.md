# T2：模型汤（跨 HPO top-2 轨迹）— 判负（barrier 存在即弃）

> 对应 `ContinuePreTrain-ToDo.md` T2。原始设计：HPO top trials
> `4c721141ab`（lr_muon=0.005）+ `8ceb4d575b`（lr_muon=0.01）权重平均。
> 两条轨迹优化步进不同，属**不同优化路径**；ToDo 要求先做线性插值
> barrier 检查，barrier 存在即弃。本版补训了缺失的 8ceb 权重，
> 以两腿各自 token 质量最优权重执行完整流程。

## 执行

1. **补训 8ceb leg**（临时脚本，命令归档于此，脚本已删）：
   ```
   KRONOS_PREVIEW_OVERRIDE_JSON=server_runs/results/04b-cpt/seed42/trials/8ceb_cpt/override.json \
   python train_base.py --epochs 100 --lr_muon 0.01 \
       --save_path checkpoints/exp04b_8ceb.pt --tag exp04b_8ceb
   ```
   与 4c72 leg（`exp04b_best_ep*.pt`）唯一差异：lr_muon=0.01 vs 0.005。
   数据/模型/tokenizer 配置与 batch 顺序完全一致（seed 42，同一 override.json）。
   - 100 ep × 128 opt-step，3.55 h（RTX 4060 Laptop）
   - 产出：`checkpoints/exp04b_8ceb_ep{1..100}.pt` + `history_exp04b_8ceb.json`
   - val_loss 谷底 2.8612 @ ep20（4c72 谷底 2.8837 @ ep13）；ep100 val=3.488（4c72: 3.609）
2. **两腿各选 token 质量最优权重**（400 窗协议 21 点采样，primary =
   median daily coarse balance，tie-break JSD/collapse）：
   - 4c72 leg（`local_cpt` 轨迹）→ **ep100**：balance 0.5757, JSD 0.3193, collapse 42.2%, unique 63
   - 8ceb leg（`8ceb_cpt` 轨迹）→ **ep100**：balance 0.5806, JSD 0.3079, collapse 42.8%, unique 61
   - 选点记录：`server_runs/results/04b-cpt/seed42/trials/t2_selection.json`
3. **Barrier 检查**：soup 即连线中点（两权重等权平均）。全 400 窗评估
   中点 + α=0.25/0.75 两个插值点，描出连线剖面。

## 结果（400 窗协议）

| 模型 | balance | JSD | collapse | unique | DA | IC |
|------|---------|-----|----------|--------|----|----|
| 4c72 ep100（α=0）| **0.5757** | 0.3193 | 42.2% | 63 | 48.8% | +0.031 |
| interp α=0.25 | 0.5179 | **0.3800** ⚠️ | 41.2% | 44 | 49.0% | +0.043 |
| **soup（中点）** | 0.5105 | 0.3725 ⚠️ | 42.8% | 43 | 49.4% | +0.040 |
| interp α=0.75 | 0.5287 | 0.3340 | 44.2% | 49 | 49.5% | +0.034 |
| 8ceb ep100（α=1）| **0.5806** | 0.3079 | 42.8% | 61 | 49.6% | +0.029 |

⚠️ = 超红线 JSD 0.366。

**连线中部整体凹陷**（balance −0.06~−0.07、unique 塌 18–20 个、JSD 破红线），
两端高——经典 barrier 剖面。paired bootstrap（400 日期，2000 次）：

- balance：soup vs 8ceb ep100 **delta −0.0646，95% CI [−0.0716, −0.0574]**（显著负）
- JSD：delta **+0.0615，CI [+0.0542, +0.0683]**（显著劣化）

## 裁定

**T2 判负：barrier 存在即弃（ToDo T2），soup 不作为 CPT 新起点。**

- 跨 lr_muon 优化轨迹（0.005 vs 0.01）的权重连线中部落在"两模式皆非"
  的高损失区：模型在 Muon 步进下收敛到不相容的局部结构，线性平均失效。
- 与旧结论的关系：本版取代先前"同轨迹近邻 soup 无增益"的本地适配版
  （该版无法获得 8ceb 权重，仅在同轨迹内验证 checkpoint 平均）；
  原始 ToDo 设想的跨轨迹 soup 现已用真实权重完整执行，判负依据充分。

## 附带发现（重要）

**8ceb ep100（lr_muon=0.01）是新的最佳单 checkpoint**：balance 0.5806（
+0.0049 vs 4c72 ep100）、JSD 0.3079（−0.0114）、collapse 42.8%（+0.6pp）、
unique 61（−2）、DA 49.6%（+0.8pp）、MAPE 3.366（−0.008）。val_loss 全程
更低（谷底 −0.023、末态 −0.121）。

- 按 ToDo"最佳单 trial 即留用为 CPT 新起点"的语义，**CPT 起点候选
  应切换为 `checkpoints/exp04b_8ceb_ep100.pt`**。
- 注意：现有 Branch A 产出（`branchA_dm030_ep6.pt`）是从 4c72 ep100 起步的；
  是否以 8ceb ep100 为新基线重训 Branch A 需另行确认（本次不自动执行）。
- 单 seed 观察：lr_muon 上探（0.005→0.01）在该尺度模型上 token 质量仍为
  正方向，与 Exp 04-B"lr_muon 主导且单调"结论一致。

## 产物

- `checkpoints/t2_soup_exp04b_bestep100_exp04b_8cebep100.pt`（中点，判负，保留备查）
- `checkpoints/exp04b_8ceb_ep{1..100}.pt`（8ceb leg 全轨迹）
- `checkpoints/exp04b_8ceb_checkpoints.json` / `history_exp04b_8ceb.json`
- `server_runs/.../trials/8ceb_cpt/`（8ceb 轨迹评估）、`t2_soup/`、`t2_interp_a0.25/`、
  `t2_interp_a0.75/`（barrier 剖面评估）、`t2_selection.json`（选点记录）
- 脚本：`experiments/05-cpt/t2_cross_config_soup.py`（prepare_traj/select/soup/barrier/interp/eval）
