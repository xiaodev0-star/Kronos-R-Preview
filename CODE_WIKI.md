# Kronos-R-Preview — Code Wiki

## 项目概述

基于 LLM 因果预测范式的 A 股时序模型。每只股票从头读到尾，next-token prediction。
**当前架构为 GPT + BERT 双模型协同**（GPT 提案 + BERT 验证）：

- **训练脚本**：`train_base.py` (GPT) + `train_bert.py` (BERT 校准器) + `train_tokenizer.py`
- **评估脚本**：`eval_batch_1step.py` (GPT-only) / `eval_cross_loss.py` / `eval_bert_calibration.py` (V1) / `eval_bert_calibration_v2.py` (V2, 推荐)

完整实验报告见 `TEMP/EXP_2026_06_17_BERT_CALIBRATION/REPORT.md`。

## 数据流

```
CSV → load_stocks() → split_stocks(train/val/test, 按股票切分, cutoff隔离)
                         ↓
              pack_stocks_v2() → historical Z-Score 归一化 → Tokenizer(4D OHLC) → Token 缓存
                         ↓                                     ↘ VA 连续值 (vol/amt)
              PackedDatasetV2 → make_dataloader_v2(bs=1, causal mask)
                         ↓
                     train_base.py (GPT 训练)
                         ↓
                  checkpoints/expA_v2.pt
                         ↓
              ─── 评估路径 ───
                  ├── eval_batch_1step.py / eval_cross_loss.py    (纯 GPT 评估)
                  └── eval_bert_calibration_v2.py                   (GPT+BERT 推荐)
                         ↑
              train_bert.py (BERT MLM 训练) → checkpoints/kronos_bert_big_v1.pt
```

关键设计决策：
- **Historical Z-Score 归一化**: 统计量 (mean/std) 仅来自该股票截止日前的训练数据，对序列内每个位置统一应用
- **VA 连续注入**: Volume/Amount 不经 Tokenizer 量化，以连续 embedding (MLP) 注入 Transformer
- **单股票独立序列**: 每支股票一个序列，纯因果 mask，无需 segment 隔离
- **Token 缓存**: NPZ 格式 + MD5 校验，tokenizer 权重变更自动失效

## 归一化详解

### Historical Z-Score (价格: OHLC)

```
stats = mean/std of features_raw[:cutoff_idx]   # 仅 train 数据
price_normed[t] = (price_raw[t] - stats.mean) / stats.std   # 对所有 t 统一
```

- 每支股票一组 stats，序列内所有位置共享
- cutoff 前数据计算 stats → 应用于全序列（含 test，用 train 的 stats）
- **不会泄露 test 数据**: stats 仅从 train 部分计算
- **不会泄露序列全局信息**: causal attention mask 保证 position t 只能 attend 到 [0..t]，position 0 的 hidden state 在训练和推理时计算完全相同

### 首日基线 + Z-Score (VA: Volume/Amount)

```
va_rel[t] = log1p(vol[t]) - log1p(vol[0])   # 相对首日变化
va_normed[t] = (va_rel[t] - mean(va_rel[:ci])) / std(va_rel[:ci])
```

### 训推一致性保证

| 关注点 | 答案 |
|--------|------|
| 模型能否在 position 0 看到 position 3000 的 token? | 不能，causal mask 严格屏蔽未来 |
| 归一化统计量是否对模型透明? | 是，模型只看到 token IDs (lossy 量化)，无法反推 mean/std |
| 训练和推理的 position 0 输入是否相同? | 完全相同：同一个 token、同一个 time_id、同一个 va_value |
| 梯度回传是否会泄露未来信息? | 只影响参数更新，不影响 forward 计算；参数固定后训推一致 |

## 模型架构

### KronosPreview (2.7M)

```
token_emb(ids) + time_emb(day/month/year) + va_proj(vol,amt)
    → RoPE → 2×TransformerBlock(GQA SDPA, SiLU-gated FFN) → head
```

- `token_emb`: Embedding(1026, 256) — 1024 vocab + BOS + EOS
- `time_emb`: 3 个独立 Embedding (day/month/year) 各 256 维，additive
- `va_proj`: Linear(2→64) → GELU → Linear(64→256)，additive
- GQA: 4 query heads, 1 KV head, head_dim=64
- RoPE: base=10000, 位置 ID 外部传入
- **Causal attention** (tril mask)

### KronosBert (2.5M ~ 16M) — 校准器

```
token_emb(ids) + time_emb(day/month/year) + va_proj(vol,amt)
    → RoPE → N×TransformerBlock(GQA SDPA, SiLU-gated FFN) → head_coarse
```

- `token_emb`: Embedding(1027, dim) — 1024 vocab + BOS + EOS + **MASK**
- 架构与 KronosPreview 类似
- **Bidirectional attention** (无 causal mask) — 训练时 MLM 目标要求
- 训练目标：MLM 15% 随机 mask + CE loss
- 不针对 next-token 预测优化
- 推理时作为"中间 token 可解释性"的一致性检查器

完整实现见 `TEMP/EXP_2026_06_17_BERT_CALIBRATION/model/kronos_bert.py`

### KronosPreviewWithReasoning

```
KronosPreview → CausalReasoningBlock(cross-attn to N learnable memory tokens, tanh gate + FFN) → head
```

- Memory tokens 学习全局模式，gate 控制注入量
- `frozen`: 仅训练 reasoning block
- 不加 KV-cache，自回归时全序列重算

### BERT 校准推理 (V2) — 2026-06-17 新增

```
For each test position p (predicting tok_p):
    1. GPT 给出 top-K 候选 {y_1, ..., y_K} 及概率
    2. 对每个 y_k 构造 BERT 输入: [BOS, tok_0, ..., MASK_at_p, y_k]
       (MASK 在最后一个 history token 位置, y_k 作为"未来")
    3. BERT 预测 at MASK: score_k = P_BERT(tok_{p-1} | context_with_y_k)
    4. argmax_k of score_k 选出最终 token
```

详见 `TEMP/EXP_2026_06_17_BERT_CALIBRATION/REPORT.md`

## 训练 (`train_base.py`)

### 关键 CLI

| 参数 | 说明 |
|------|------|
| `--loss ce/focal --gamma 6.0` | Loss 函数选择 |
| `--label_smoothing 0.05` | 标签平滑 |
| `--dropout 0.05` | 覆盖 ModelConfig.dropout |
| `--weight_decay 0.001` | AdamW 权重衰减 |
| `--reasoning` | 启用推理模块 |
| `--reasoning_frozen` | 冻结 Transformer，仅训 reasoning |
| `--base_checkpoint PATH` | 从预训练权重初始化 |
| `--epochs 10 --tag NAME` | 基础参数 |

### 断点续算
- 每 epoch 保存 `save_path.ckpt` (model+optimizer+scheduler)
- 最佳 checkpoint → `save_path` (含 `completed=True` 标记)
- 重新运行相同的 `--save_path` 自动恢复

## HPO 编排器模式

JSON 状态机 (`hpo_vX_state.json`: `{"current_idx": N, "log": [...]}`):
- 串行调用 `train_base.py` via subprocess
- 跳过已完成的 checkpoint (`completed=True`)
- 中断后重跑自动从断点继续

## 关键修复
- SDPA mask: 3D `[B,N,N]` → 4D `[B,1,N,N]`，解决 batch_size>1 时的维度错误
- focal_loss: `targets.clamp(min=0)` 避免 -100 padding 导致 gather 越界
- Padding mask: `p_mask[j, L:, 0] = True` 防止 padding 位置 softmax(NaN)
- Token 缓存校验: 新增 MD5 hash，旧缓存自动失效
