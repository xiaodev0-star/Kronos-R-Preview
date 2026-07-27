# Exp 02 · Tokenizer 结构调参：embedding_dim × hidden_dim

> **状态**：已完成（2026-07-24~25 重跑 seed=42；2026-07-26 修复收尾元数据、复核重析并清洗目录）
> **结论**：在 bits=9+7（继承 Exp 01 选型）下，**embedding_dim=64、hidden_dim=192** 仍是最优结构，已写入 [`rerun_seed42/selection.json`](rerun_seed42/selection.json) 供 Exp 03 读取。64x192 在成熟期拿下 Tok MAE / MAPE / P90 日坍缩 / 幅度校准四项第一，DA 第二（仅落后 96x256 0.27pp），RankIC 是 96x256 的 2.4 倍。
> **限定**：健康门 0/300 通过、holdout 未开、单 seed；且旧结论的招牌数字 **"MAE −14.5%" 在 9+7 码本下已不复存在**（−0.12%）——64x192 的胜出理由从"重建更好"换成了"下游全面更好"。

---

## 1. 实验问题

BSQ tokenizer 的结构参数（`embedding_dim` × `hidden_dim`）决定 latent 表达容量。本实验在固定码本 bits 之后扫描 3×2 结构网格（48/64/96 × 192/256），回答：**结构参数对重建质量与 GPT 下游表现的真实影响是什么？**

与 Exp 01 相同，本实验存在两代版本：

| | 历史版（2026-07-11，已退役） | 重跑版（2026-07-24，当前有效） |
|---|---|---|
| 码本 | bits=8+6（16K，旧 Exp 01 结论） | **bits=9+7**（65,536，Exp 01 重跑选型） |
| 方法 | 两阶段：重建 MAE 排名 → 仅对赢家做 GPT"不退化"门槛验证 | 6 组**全部**训练 GPT 50 epochs，逐 epoch 多窗评估 |
| 结论 | 64x192（MAE −14.5%，GPT 门槛通过） | 64x192（下游质量/行为最优，重建优势基本消失） |

## 2. 历史实验存档与退役原因

### 2.1 历史结论（原始数据保留于 [`tok_sweep_results.json`](tok_sweep_results.json) / [`tok_sweep_gpt_validation.json`](tok_sweep_gpt_validation.json)）

阶段一（bits=8+6，重建 MAE）：64x192 **0.1947（−14.5%）** > 48x256 0.2114 > 96x192 0.2125 > 96x256 0.2219 > 48x192 0.2277（基线）> 64x256 0.2321。
阶段二（GPT 门槛验证，仅 64x192 vs 48x192）：DA 49.29% vs 49.15%（+0.14pp，通过 ±1pp 门槛）、Collapse 27.2% vs 24.9%、RankIC 0.0343 vs 0.0208（+65%）。

历史报告还记录了"tokenizer 级与 GPT 级 Unique/Collapse 方向反转"现象（tokenizer 更健康、GPT 反而更集中）。

### 2.2 退役原因

1. **码本前提变了**：8+6 已被 Exp 01 重跑取代为 9+7，结构结论必须在新码本下复验；
2. 阶段二只验证了赢家一组，其余 4 组从未接受下游检验——"MAE 排名 = 结构排名"是未经检验的假设（重跑证明该假设不成立，见 §4.3）；
3. 与 Exp 01 历史版共享全部协议缺陷（终点单评、无 holdout、pooled 口径等，见 Exp 01 报告 §2.2）。

历史脚本（`sweep_tokenizer.py`、`sweep_tokenizer_validate.py`、`gen_plots.py`）与历史报告（`Exp-TokenizerTuning.md`）已移除（git 可找回）。**注意**：历史产物 [`tok_sweep_emb64_hid192.pt`](tok_sweep_emb64_hid192.pt)（bits=8+6 的旧生产 tokenizer）**仍被 `run_eval.py` 与 Exp 04 全部三个脚本（loss 消融 / optimizer 消融 / HPO）引用**，为在役权重，连同 `tok_sweep_emb48_hid192.pt`（旧基线）一并保留。

## 3. 重跑协议（rerun_seed42）

| 项目 | 固定值 |
|---|---|
| bits | **9+7**（读取 Exp 01 `selection.json`，sha256 锁定） |
| 网格 | embedding_dim ∈ {48, 64, 96} × hidden_dim ∈ {192, 256}，共 6 组 |
| Tokenizer | 100 epochs、warmup+cosine、取验证损失最优 |
| GPT / 评估 / 判读 / 健康门 | 与 Exp 01 重跑完全一致（CE+AdamW 50 epochs 全量；每 epoch × 4 窗 × 20 日；无加权总分；日级健康门） |
| Holdout | offset 400 起 80 日封存未启用 |

共 6 × 50 = 300 个 GPT checkpoint 全部评估。

运行审计：首轮 2026-07-25 07:07 正常跑完；08:03 二次启动做幂等复核时，最后一个配置的 `run.json` 原子改名遭遇 Windows 文件锁 `PermissionError`，manifest 被误标 `failed`。2026-07-26 核验数据零缺失（6 配置轨迹文件逐一比对、汇总 300 行齐全）后，仅修复状态字段并重跑分析，全过程记录于 `study_manifest.json` 的 `note` 字段（原 error 保留作历史）。

## 4. 重跑结果

### 4.1 成熟期总表（后 10 epoch 中位数）

| Config | Tok MAE | 成熟 DA | 峰值 DA (ep) | 成熟 RankIC | 成熟 MAPE | 成熟 P90 坍缩 | 成熟 unique | 成熟 Amp | joint-Pareto epochs |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48x192 | 0.1944 | 51.18% | 52.06% (3) | **0.0623** | 3.621 | 85.3% | 22 | 1.238 | 25 |
| 48x256 | 0.1986 | 51.56% | 53.44% (3) | 0.0252 | 3.676 | 59.9% | 33 | 1.319 | 17 |
| **64x192** | **0.1942** | 53.34% | 53.93% (3) | 0.0455 | **3.362** | **56.0%** | 32 | **1.170** | 13 |
| 64x256 | 0.1994 | 49.46% | 51.00% (3) | 0.0481 | 3.784 | 60.2% | 32 | 1.366 | 14 |
| 96x192 | 0.2014 | 52.22% | 54.27% (3) | 0.0315 | 3.448 | 66.4% | 22 | 1.176 | 3 |
| 96x256 | 0.2147 | **53.61%** | 53.94% (19) | 0.0193 | 3.395 | 61.2% | 33 | 1.168 | 19 |

统计口径：300 点；joint Pareto 91、quality 26、behaviour 19；**健康门通过 0/300**。

### 4.2 逐组 epoch 动力学

- **共性**：6 组中 5 组峰值 DA 在 epoch 3（早峰→回落→缓慢恢复），后 10 epoch DA 标准差 ≤0.04pp，末期极稳；坍缩从 e1 的 ~100% 单调降至 55–67%。
- **64x192**：早峰 53.93%@3 → 谷底 51.4%@10 → 爬回 53.37%@50，末期几乎复现早峰；P90 坍缩全场最低且仍在缓降（最佳 54.5%@43）；best RankIC 0.0654@3、best MAPE 2.519@1。
- **96x256**：**唯一晚峰型**（best 53.94%@19），成熟期 DA 全场第一、末期平稳——大 latent 需要 GPT 更长时间消化；但 RankIC 0.0193 全场最差（横截面选股信号折半再折半）、重建 MAE 最差（比 64x192 差 10.6%）。
- **48x192**：病态样本。P90 坍缩卡死 85%+（全程最佳也只有 85.0%）、unique 仅 22，末期 DA 方差近乎冻结——其全场第一的 RankIC（0.0623）建立在行为塌缩之上，可信度存疑。
- **64x256**：全场最差（成熟 DA 49.46%，从未过 51%），复现历史"hidden=256 对 emb=64 有害"结论。
- **48x256 / 96x192**：一个行为面尚可但质量面弱，一个质量面尚可但多样性差（unique 22、joint-Pareto 仅 3 epoch），均为中间态。

图表：[tokenizer_grid](rerun_seed42/plots/tokenizer_grid.png) ·
[quality_trajectories](rerun_seed42/plots/quality_trajectories.png) ·
[behaviour_trajectories](rerun_seed42/plots/behaviour_trajectories.png) ·
[late_metric_dashboard](rerun_seed42/plots/late_metric_dashboard.png) ·
[tokenizer_vs_downstream](rerun_seed42/plots/tokenizer_vs_downstream.png) ·
[da_vs_collapse_pareto](rerun_seed42/plots/da_vs_collapse_pareto.png)

### 4.3 重建质量与下游：历史"反转"只部分留存

- val_loss ↔ DA：ρ = **+0.127**（p=0.028，300 点）——"损失更低 DA 反而略低"的弱反转仍在；val_loss ↔ RankIC −0.243、↔ P90 坍缩 +0.311 为正常方向。
- Tok MAE ↔ 成熟 DA：ρ = +0.257（n=6，p=0.62，不显著）；MAE ↔ 成熟 RankIC：ρ = −0.60（不显著且方向"正常"）。
- **最直接的反例：96x256 重建全场最差却拿下成熟 DA 第一**。

结论：**tokenizer 重建 MAE 不是下游 DA 的可靠代理**（这正是历史两阶段法的隐含假设）；但也不能反着用——"反转"本身在 9+7 协议下不构成稳健规律。结构选择必须直接看下游，全网格下游评估是必要开销。

## 5. 选型：64x192（Exp 03 依赖）

完整记录于 [`rerun_seed42/selection.json`](rerun_seed42/selection.json)：

- 64x192 成熟期四项第一（Tok MAE 0.1942、MAPE 3.362、P90 坍缩 56.0%、幅度校准 1.170）+ DA 第二（53.34%，−0.27pp）+ 未塌缩组中 RankIC 第一（0.0455）；
- **对 96x256**（唯一 DA 更高者）：RankIC 翻 2.4 倍（0.0455 vs 0.0193）、坍缩 −5.3pp、重建 MAE −10.6%；接受的代价是 −0.27pp 成熟 DA 与更少的 joint-Pareto epoch（13 vs 19）。对金融任务，牺牲横截面排序能力换 0.27pp DA 不划算；
- **跨实验可比性红利**：64x192@9+7 与 Exp 01 的 9+7 配置**同架构同 seed**，tokenizer MAE 逐位相同、GPT 成熟 DA 差异 ≈0.0004pp——Exp 03 的 scaling 结论可与 Exp 01/02 全部历史直接对比；
- 若 Exp 03 之后 GPT 容量显著扩大，96x256 的"晚熟 + 大 latent"特征值得重新入场（它是唯一晚峰型，可能更受益于更强的消化能力）。

## 6. 与历史结论的关系

1. **"MAE −14.5%"消失了**：9+7 码本下 64x192 与 48x192 的 MAE 仅差 −0.12%（0.1942 vs 0.1944）。旧的重建优势主要是 8+6 码本 + early-stop 制度的产物——9 bit 的 coarse 层把小 embedding 的重建也"救"了回来；
2. **赢家不变，理由换了**：64x192 的优势从重建转移到下游（DA +2.16pp、坍缩 −29.5pp vs 48x192）。48x192 的行为塌缩病态旧协议完全没看出来（它只做了 MAE 排名）；
3. "hidden=256 对 emb=64 有害"**复现**（64x256 全场最差）；"embedding_dim 是真杠杆"**弱化**（96x256 证明大 emb 需配大 hidden 且晚熟才能兑现，杠杆非单调）；
4. 历史"tokenizer 级 vs GPT 级指标反转"降级为弱现象（见 §4.3）。

## 7. 合理性与可靠性评估

### 7.1 设计合理性

- 全网格 6 组都做完整下游评估，修复了历史两阶段法"用 MAE 代理下游"的结构性缺陷（§4.3 证明该修复是必要的，不是过度工程）；
- bits 依赖通过 `selection.json` + sha256 锁定，与 Exp 01 形成可审计的依赖链；
- 与 Exp 01 共享的遗留问题同样适用：CE 配方条件性、评估窗 regime 覆盖有限（见 Exp 01 报告 §7.1）。

### 7.2 结论可信度分级

| 级别 | 结论 | 依据 |
|---|---|---|
| 高 | 64x256 有害、48x192 行为塌缩、MAE 不是下游代理 | 效应量大（−3.9pp DA / +29pp 坍缩），且与历史独立证据一致 |
| 高 | 管线确定性（同配置跨实验逐位复现） | 见 §5 第三条 |
| 中 | 64x192 vs 96x256 的选型 | DA 差距仅 0.27pp（单 seed 噪声量级内），选型实质由 RankIC/坍缩/重建的**多指标权衡**支撑而非 DA 单指标；权衡本身稳健，但若换 seed 出现 DA 反超 1pp 级别，应重议 |
| 低 | DA 绝对水平 | 健康门 0/300，同 Exp 01 §7.2 |

### 7.3 升级可靠性的动作

1. 3-seed 复验 64x192 vs 96x256（其余 4 组可淘汰，不必复验）；
2. holdout 揭盲与 Exp 01 合并执行（同一批窗口，一次性）；
3. Exp 03（GPT scaling）完成后回看 96x256：若更大 GPT 显著受益于大 latent，结构结论需要与容量联合调参。

## 8. 目录结构与复现

2026-07-26 清洗：移除历史脚本/报告/图表（git 可找回）与可再生缓存（`cache/`、`shared/`）；保留历史原始数据 JSON 与在役旧权重（见 §2.2 注意事项）、重跑全部权重/日志/评估数据/分析产物。

```text
02-tokenizer-tuning/
├── README.md                        ← 本报告（唯一实验报告）
├── rerun_tokenizer_epochwise.py     ← 重跑主脚本（依赖 Exp 01 selection.json）
├── analyze_tokenizer_epochwise.py   ← 非合成分数分析（可独立重跑）
├── tok_sweep_emb64_hid192.pt        ← 旧生产 tokenizer（bits=8+6）：run_eval.py 与 Exp 04 在役依赖，勿删
├── tok_sweep_emb48_hid192.pt        ← 旧基线 tokenizer（历史存档）
├── tok_sweep_results.json           ← 历史阶段一原始数据（6 组 MAE）
├── tok_sweep_gpt_validation.json    ← 历史阶段二原始数据（GPT 门槛验证）
└── rerun_seed42/
    ├── study_manifest.json          ← 参数/依赖/审计链（含 2026-07-26 修复记录）
    ├── selection.json               ← 64x192 选型（Exp 03 依赖入口，勿手改）
    ├── combined_epoch_summary.{csv,json}  ← 300 config-epoch 评估数据
    ├── analysis.json / *.csv        ← Pareto、包络
    ├── plots/                       ← 报告引用的全部图表
    ├── unattended.*.log             ← 无人值守运行日志
    └── configs/emb_XXX_hid_XXX/     ← 每配置：tokenizer.pt、model_ep1..50.pt、
                                        epoch_trajectory/、logs/、run.json
```

```powershell
# 正式重跑（自动断点续跑；数据齐全时不会重复训练）
& 'D:\conda_envs\llm-t\Scripts\python.exe' .\experiments\02-tokenizer-tuning\rerun_tokenizer_epochwise.py

# 仅重建分析与图表
& 'D:\conda_envs\llm-t\Scripts\python.exe' .\experiments\02-tokenizer-tuning\analyze_tokenizer_epochwise.py
```
