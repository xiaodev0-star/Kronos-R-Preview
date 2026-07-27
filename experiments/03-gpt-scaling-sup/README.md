# Exp 03-Sup：受控 GPT 容量补充实验

> **状态**：脚本与评估协议已就绪，正式 50-epoch 实验尚未启动。  
> **目的**：补齐 Exp 03 中 `deep`（4.48M）到 `xlarge`（16.96M）之间的容量空档，分离深度、宽度和 KV heads 的影响；用 target-relative coarse 指标选出单-seed 入围尺寸，并用完整 fine/joint 真实—预测分布审计结论边界。  
> **Python**：`D:\conda_envs\llm-t\Scripts\python.exe`（Python 3.12.10）  
> **Holdout**：offset 400 继续封存。

## 1. 为什么需要这个补充实验

Exp 03 的逐 epoch 结果显示：在五个受测架构中，`xlarge` 的 coarse target-relative 码本指标最好，`deep` 是明显的参数效率拐点；但二者之间同时改变了 `dim/depth/heads/kv_heads`，参数量又从 4.48M 跳到 16.96M。因此原实验只能给出“当时网格中的领先点”，不能定位容量曲线的拐点，也不能判断收益来自加深、加宽还是增加 KV heads。

准备 Sup 时进一步审计出一个影响公平性的旧协议问题：token-budget microbatch 可以跨过梯度累积边界，训练循环随后把整个 microbatch 一次计入并清零；同时 loader shuffle 依赖模型初始化后的全局 RNG。结果，旧 Exp 03 虽然 scheduler 都按 4160 steps 构造，五个架构实际只执行了：

| Config | 实际 optimizer steps |
|---|---:|
| `baseline` | 3934 |
| `wide` | 3981 |
| `deep` | 3984 |
| `large` | 4039 |
| `xlarge` | 4104 |

因此旧 `deep/xlarge` 不能作为严格受控的冻结锚点。Exp 03-Sup 会在修正后的统一 loader 下**重训表中的全部六个架构**；旧 checkpoint 只保留为历史证据，不与 Sup 结果混排。

## 2. 受控架构网格

| 角色 | Config | dim | depth | heads | kv_heads | 参数量 | 相对 `deep` 唯一结构变化 |
|---|---|---:|---:|---:|---:|---:|---|
| 共同原点 | `deep` | 256 | 4 | 4 | 1 | 4.48M | — |
| 深度点 1 | `depth6` | 256 | 6 | 4 | 1 | 6.38M | depth |
| 深度点 2 | `depth8` | 256 | 8 | 4 | 1 | 8.29M | depth |
| 宽度点 1 | `width384_d4` | 384 | 4 | 6 | 1 | 9.62M | dim（heads 随宽度调整） |
| 宽度点 2 | `width512_d4_kv1` | 512 | 4 | 8 | 1 | 16.70M | dim（heads 随宽度调整） |
| 上界对照 | `xlarge` | 512 | 4 | 8 | 2 | 16.96M | 相对上一行仅 kv_heads |

宽度轴上的 `heads` 不是独立容量变量，而是随 `dim` 同步调整以固定 `head_dim=64`。因此：

- 深度轴为 `deep → depth6 → depth8`，`dim/heads/kv_heads` 全部固定；
- 宽度轴为 `deep → width384_d4 → width512_d4_kv1`，`depth/kv_heads` 固定；
- `width512_d4_kv1 → xlarge` 只改变 `kv_heads: 1 → 2`。

四个新增点全部位于 `deep` 与 `xlarge` 的参数区间内，且每条轴至少有两个中间点。

## 3. 固定实验协议

| 项目 | 固定值 |
|---|---|
| Seed | 42（第一阶段只做单 seed） |
| Tokenizer | Exp 02 选型：embedding 64、hidden 192、bits 9+7 |
| 数据 | 全量股票、每只股票完整序列 |
| 训练 | CE + AdamW、lr=3e-4、weight decay=0.01、dropout=0.1 |
| Epoch | 50；每个 epoch 保存 checkpoint |
| 累积策略 | 保留旧实验的名义策略：ep1–15 每 32 个序列更新，ep16 起每 64 个序列更新；但修正其边界实现 |
| Optimizer steps | ep1–15 各 128 steps、ep16–50 各 64 steps，共 **4160**；每个 epoch 与累计值都硬校验 |
| Loader | 独立于模型 RNG 的本地 seed=42；六个架构逐 epoch 使用同一序列顺序 |
| Microbatch | 六个架构统一 `batch_tokens=6144`；每个 adaptive microbatch 块严格结束在 optimizer 边界，不允许跨界 |
| 评估 | offset 0/100/200/300，各 20 日；所有配置 eval batch size=1 |
| Holdout | offset 400 起封存，不参与容量选择 |

每个 epoch 先用架构无关的本地 RNG 产生序列排列，随机轮换丢弃不足一个完整累积块的尾部，再在每个精确累积块内部按长度打包 microbatch。这样 `batch_tokens` 只影响一次 forward 的装箱，不再改变被看到的序列、optimizer step 数或 LR 轨迹；controlled 模式若 OOM 会直接终止，不会跳过 batch 后继续生成不可比结果。

脚本还会校验父 Exp 03 的 tokenizer hash、seed、epoch 数、完整序列标志和评估窗口，以锁定数据与评估口径；它不会复用父实验权重。全部六个配置统一使用完整序列、gradient checkpointing、训练 batch token budget 和 eval batch size。

## 4. 完整层级码本评估

新版共享 evaluator 对每条预测同时记录：

- `coarse_id / true_coarse_id`；
- `fine_id / true_fine_id`；
- `joint_id = coarse_id × 128 + fine_id` 及对应真实 ID。

对 coarse、fine、joint 三层分别计算：

- token accuracy；
- prediction support precision、target support recall、support F1；
- raw unique、entropy 与有效 token 数 `2^H`；
- prediction/target collapse 及其对齐度；
- Jensen–Shannon divergence；
- target-relative codebook balance。

三层都计算同一定义的诊断指标：

```text
CodebookBalance = (
    support_F1
    × (1 - JSD)
    × effective_token_alignment
    × collapse_alignment
) ^ (1/4)
```

它是码本行为诊断，不混入 DA、RankIC、MAPE 或 AmpRatio。几何平均可避免“偶尔撒出大量 token”用高 Unique 掩盖 support、频率形状或 collapse 的失败。

本轮**容量主排名只使用 coarse 层**。代码审查发现，继承自父实验的 fine 训练存在既有条件错位：位置 `t` 的 fine target 在训练时由 `coarse_(t-1)` 作 teacher condition，而推理时由当前预测的 `coarse_t` 作 condition。为了让六个容量点只改变架构，本实验不会在中途同时修改该语义；fine/joint 指标仍完整记录，但只作审计，不把其差异单独归因于容量。修复 fine conditioning 必须另立质量实验，并在统一修复后重训完整网格。

每个 epoch 除 JSON 摘要外，还保存压缩 NPZ sidecar，包含四个验证窗口的完整计数向量：

```text
coarse_pred_counts / coarse_true_counts   [4, 512]
fine_pred_counts   / fine_true_counts     [4, 128]
joint_pred_counts  / joint_true_counts    [4, 65536]
```

因此任何关于 65,536 joint codebook 的结论都能回到真实/预测分布复核，而不是从 coarse Unique 外推。

## 5. 成熟 5-epoch 窗口与选型规则

对每个配置：

1. 成熟起点定义为首次满足 `val_loss <= 1.01 × 该配置最小 val_loss`；
2. 只枚举连续 5 epoch 且都在该 loss basin 内的窗口；
3. 窗口主排名依次看 coarse CodebookBalance、P10 coarse balance、coarse support F1 与 coarse JSD；
4. DA、逐日 RankIC、MAPE、`|log(AmpRatio)|` 不进入码本得分，只作退化 guardrail；
5. guardrail 以 `deep` 成熟窗口为参照，容忍区间为 `max(预设最小实际差异, 2 × deep 窗口内 epoch SD)`；
6. 在 guardrail 通过的“参数量—coarse balance”Pareto 前沿上，保留距离“guardrail 通过者中的领先点”不超过其窗口内 1 个 SD 的配置，再选参数最少者作为**单-seed 临时选型**；
7. 同时单独报告原始 coarse 指标领先者、参数效率拐点，以及 fine/joint 审计轨道，不把这些角色混写成一个结论。

这里的 5 个 epoch 不是五次独立实验，窗口 SD 只能抑制 checkpoint 偶然峰值，不能替代跨 seed 置信区间。正式冻结尺寸前，应对临时选型、原始领先者及相邻边界点补多 seed。

## 6. 运行

```powershell
# 完整实验：六个配置按统一受控协议重训、逐 epoch 评估并自动分析
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\03-gpt-scaling-sup\run_gpt_capacity_sup.py

# 只运行部分配置（输出会保持 partial，不会提前生成正式 selection）
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\03-gpt-scaling-sup\run_gpt_capacity_sup.py `
  --configs depth6,depth8

# 快速验证训练/评估链路；使用临时目录，不污染正式结果
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\03-gpt-scaling-sup\run_gpt_capacity_sup.py --smoke

# 所有轨迹完成后单独重建报告
& 'D:\conda_envs\llm-t\Scripts\python.exe' `
  .\experiments\03-gpt-scaling-sup\analyze_gpt_capacity_sup.py `
  --root .\experiments\03-gpt-scaling-sup\run_seed42
```

正式产物将写入：

```text
experiments/03-gpt-scaling-sup/run_seed42/
├── study_manifest.json
├── combined_epoch_summary.{json,csv}
├── selection.json
├── analysis.json
├── ANALYSIS.md
├── capacity_window_ranking.csv
├── axis_comparisons.csv
├── parameter_codebook_pareto.csv
├── representative_distribution_summary.json
├── capacity_frontier.png
└── configs/<deep|depth6|depth8|width384_d4|width512_d4_kv1|xlarge>/
    ├── model_ep1.pt ... model_ep50.pt
    └── epoch_trajectory/
```

在 Exp 03-Sup 的单-seed 结果完成并审阅前，不应据此启动依赖容量结论的 Exp 04 正式架构线。旧 Exp 03 的 `xlarge@18` 仍是旧网格的 coarse 指标领先点，但由于 optimizer-step 与数据顺序混杂，只能视为**待 Sup 复核的历史候选**，不再视为已冻结尺寸。
