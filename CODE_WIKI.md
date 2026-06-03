# Kronos-R-Preview — Code Wiki

## 项目概述

基于 LLM 因果预测范式的 A 股时序模型。每只股票从头读到尾，next-token prediction。

## 数据流

```
CSV → load_stocks() → split_stocks(train/val/test, 按股票切分, cutoff隔离)
                         ↓
              pack_stocks() → 滚动归一化 → Tokenizer编码 → Token缓存
                         ↓
              PackedDataset → make_dataloader(bs=1, segment-isolated mask)
                         ↓
                     train_base.py
```

关键设计决策：
- **滚动归一化** (W=252): cumsum 向量化实现，只用历史数据
- **多股票打包**: 填充至 context_len=8192，per-stock RoPE 重置
- **Segment隔离**: block-diagonal causal mask，BOS 对全段可见
- **Token 缓存**: NPZ 格式 + MD5 校验，tokenizer 权重变更自动失效

## 模型架构

### KronosPreview (2.7M)
```
token_emb + time_emb(day/month/year) → RoPE → 2×TransformerBlock(SDPA, SiLU-gated FFN) → head
```

### KronosPreviewWithReasoning
```
KronosPreview → CausalReasoningBlock(cross-attn to N learnable memory tokens, tanh gate + FFN) → head
```

- Memory tokens 学习全局模式，gate 控制注入量
- `frozen`: 仅训练 reasoning block
- 不加 KV-cache，自回归时全序列重算

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
- Segment mask: `pass` → `_build_segment_mask()` (2026-06-02)
- Token 缓存校验: 新增 MD5 hash，旧缓存自动失效
- Tokenizer val split: 随机特征向量 → 按股票 ID
- CLI 扩展: `--label_smoothing`, `--dropout`, `--entropy_alpha`
