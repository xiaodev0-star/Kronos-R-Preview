# Experiment 03 Plan: GPT Model Scaling

## 1. 背景

实验 01（BitSweep）确定 bits=[8,6]，实验 02（Tokenizer 调参）确定 embedding_dim=64, hidden_dim=192。
两轮实验共同结论：**tokenizer 配置对 GPT 下游性能影响极小**（DA 变化 < 0.5pp）。
当前 2.7M GPT 的 DA ≈ 49%，全部跑输基线 ~21pp——瓶颈在 GPT 模型本身。

## 2. 实验目标

回答：**GPT 模型扩容后 DA 能否突破 49% 平台？**

具体：
1. 扩容后的 DA / Collapse / RankIC 是否显著改善？
2. 深度 vs 宽度哪个更有效？
3. 扩容后 tokenizer 的 bits=[8,6] 是否仍然充分（Unique/Collapse 是否变化）？

## 3. 实验设计

### 3.1 架构扫描

固定：bits=[8,6], embedding_dim=64, hidden_dim=192, focal γ=4, het=ON, dropout=0.1,
全量 4695 stocks, early_stop_patience=5

| 配置 | dim | depth | heads | kv_heads | ~Params | gradient ckpt | 理由 |
|------|-----|-------|-------|----------|---------|---------------|------|
| **baseline** | 256 | 2 | 4 | 1 | 2.7M | 否 | 当前最优，校准基准 |
| **wide** | 384 | 2 | 6 | 1 | ~6M | 否 | 纯宽度扩展 |
| **deep** | 256 | 4 | 4 | 1 | ~5M | 建议 | 纯深度扩展 |
| **large** | 384 | 3 | 6 | 1 | ~10M | 是 | 宽度+深度 |
| **xlarge** | 512 | 4 | 8 | 2 | ~25M | 是 + 降 seq_len | 极限测试 |

### 3.2 评估方式

每个架构训练完成后立即评估：
- `eval.py windowed --n_stocks 0 --n_days 999`（全量 test stocks，全部可用日期）
- 评估跑在独立子进程（内存隔离）

### 3.3 评估指标

| 指标 | 决策角色 |
|------|----------|
| DA (per-date) | 主指标 |
| Collapse% | 健康指标（≤30%） |
| Unique | 健康指标（≥64） |
| RankIC | 排序质量 |
| AmpRatio | 幅度校准 |

## 4. GPU 预算

GPU: RTX 4060 Laptop, 8.2GB VRAM
每 epoch 约 8.5 min（2.7M baseline，全量数据），early stop patience=5 预计 20-25 epochs 收敛。

| 配置 | ~Params | ~每 epoch | ~总耗时（20ep） | 需 gradient ckpt |
|------|---------|-----------|-----------------|-----------------|
| baseline 256/2/4 | 2.7M | ~9 min | ~3 h | 否 |
| wide 384/2/6 | 6M | ~18 min | ~6 h | 否 |
| deep 256/4/4 | 5M | ~25 min | ~8 h | 建议 |
| large 384/3/6 | 10M | ~35 min | ~12 h | 是 |
| xlarge 512/4/8 | 25M | ~60 min | ~20 h | 是 + 降 seq_len |

**总计约 49 小时**（5 个架构串行）。xlarge 可能需 `--max_seq_len 4096` 防 OOM。

## 5. 关键假设与预期

| 假设 | 检验方式 | 若成立 | 若不成立 |
|------|----------|--------|----------|
| 扩容提升 DA | 对比 baseline vs wide/deep/large | GPT 容量是瓶颈 | 瓶颈在数据或训练策略 |
| 深度比宽度更重要 | 对比 wide vs deep | Transformer 深度是关键 | 宽度更重要 |
| 25M 过大 | 检查 xlarge 的 val_loss 是否过拟合 | 10M 是甜点 | 可以继续扩 |
| bits=[8,6] 对大模型仍充分 | 检查大模型的 Unique/Collapse | Tokenizer 配置通用 | 大模型需要更大码本 |

## 6. 脚本需求

- 复用 `train_base.py`（已支持 `--dim`, `--depth`, `--heads`, `--num_kv_heads`, `--gradient_checkpointing`）
- 复用 `eval.py windowed`（子进程隔离）
- 新增 `sweep_gpt_arch.py`：编排 5 个架构的训练+评估，结果保存到 JSON

## 7. 产出

| 产出 | 路径 |
|------|------|
| 架构扫描结果 | `checkpoints/gpt_arch_sweep_results.json` |
| 最优 GPT 权重 | `checkpoints/gpt_best_arch.pt` |
| 实验报告 | `experiments/03-gpt-scaling/Exp-GPTScaling.md` |
| 图表 | `experiments/03-gpt-scaling/plt/` |
