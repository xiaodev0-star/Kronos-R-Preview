# Kronos-R-Preview 架构图

## 整体数据 / 训练流程

```mermaid
flowchart LR
    A[CSV 行情数据<br/>dataset/] --> B[data_processor.py<br/>Z-Score 归一化 + 打包]
    B --> C[train_tokenizer.py<br/>Stage A: BSQ Tokenizer]
    C --> D[train_base.py<br/>Stage B: GPT 训练]
    D --> E[checkpoints/<br/>expA_v2.pt]
    E --> F[eval_batch_1step.py<br/>AR 评估 / 方向准确率]
    G[run_hpo_*.py] -. 超参搜索 .-> D

    E2[tokenized 序列] --> H[train_bert.py<br/>BERT MLM 训练]
    H --> I[checkpoints/kronos_bert_big_v1.pt<br/>BERT 校准器]
    I --> J[eval_bert_calibration_v2.py<br/>GPT 提案 + BERT 验证]
    E --> J
```

## 模型内部结构 (2.7M 参数)

```mermaid
flowchart TB
    subgraph Input["输入 (单只股票的一段时序)"]
        P["OHLC Token<br/>(BSQ vocab=1024)"]
        V["VA 连续特征<br/>(Volume, Amount)"]
    end

    P --> ET["Token Embedding<br/>(1024 → 256)"]
    V --> VA["VA MLP<br/>Linear(2,64) → GELU → Linear(64,256)"]
    ET --> ADD(("+"))
    VA --> ADD
    ADD --> TR["Transformer × 2 层<br/>dim=256, heads=4, GQA<br/>RoPE + SDPA + RMSNorm<br/>SiLU-gated FFN<br/>Causal Mask (GPT)"]
    TR --> HEAD["LM Head<br/>(1024 类)"]
    TR -. 可选 .-> CR["CausalReasoningBlock<br/>(cross-attn memory)"]
```

## 双模型架构 (2026-06-17 新增) ★

```mermaid
flowchart LR
    subgraph GPT["GPT (KronosPreview) - 提案者"]
        H1[history tokens] --> GE[Token Emb] --> GT[2 层 Causal Attn] --> GHEAD[LM Head]
        GHEAD --> GK["top-K 候选 {y_1, ..., y_K}"]
    end

    subgraph BERT["BERT (KronosBert) - 验证者"]
        H2[history + MASK_at_p + y_k] --> BE[Token Emb] --> BT["2-4 层 Full Attn<br/>(无 Causal)"] --> BHEAD[LM Head]
        BHEAD --> BS["P_BERT(tok_{p-1} | context_with_y_k)"]
    end

    GK --> CAL[argmax over K of P_BERT]
    BS --> CAL
    CAL --> OUT[最终 token]
```

## Loss / 训练选项

```mermaid
flowchart LR
    HEAD[LM Head 输出] --> L1[CE Loss]
    HEAD --> L2[Focal Loss<br/>γ = 3~10]
    L2 --> L3[+ Label Smoothing<br/>0.05]
    L2 --> L4[+ Entropy Reg]

    subgraph BERT_Training["BERT Training"]
        BHEAD2[BERT LM Head] --> MLM[MLM Loss<br/>15% 随机 mask + CE]
    end
```

## 文件模块一览

| 层级 | 文件 | 职责 |
|------|------|------|
| 配置 | [config.py](file:///d:/Kronos-R-Preview/config.py) | NormConfig / DataConfig / TokenizerConfig / ModelConfig / TrainingConfig |
| 数据 | [data_processor.py](file:///d:/Kronos-R-Preview/data_processor.py) | 归一化 + 文档打包 |
| Tokenizer | [model/tokenizer.py](file:///d:/Kronos-R-Preview/model/tokenizer.py) | BSQ 2-level hierarchical |
| 模型 (GPT) | [model/kronos_preview.py](file:///d:/Kronos-R-Preview/model/kronos_preview.py) | GPT Transformer + VA Embedding (causal) |
| 模型 (BERT) | [model/kronos_bert.py](file:///d:/Kronos-R-Preview/model/kronos_bert.py) | BERT Transformer (bidirectional + MLM) |
| 训练 | [train_tokenizer.py](file:///d:/Kronos-R-Preview/train_tokenizer.py) | Stage A：Tokenizer 训练 |
| 训练 | [train_base.py](file:///d:/Kronos-R-Preview/train_base.py) | Stage B：GPT 训练 |
| 训练 | [train_bert.py](file:///d:/Kronos-R-Preview/train_bert.py) | Stage C：BERT 校准器训练（MLM 15% mask） |
| 评估 | [eval_batch_1step.py](file:///d:/Kronos-R-Preview/eval_batch_1step.py) | GPT-only 1-step 评估 |
| 评估 | [eval_cross_loss.py](file:///d:/Kronos-R-Preview/eval_cross_loss.py) | 跨 loss 公平评估 |
| 评估 | [eval_bert_calibration.py](file:///d:/Kronos-R-Preview/eval_bert_calibration.py) | V1 联合打分 P_GPT^α·P_BERT^(1-α) |
| 评估 | [eval_bert_calibration_v2.py](file:///d:/Kronos-R-Preview/eval_bert_calibration_v2.py) | ★ V2 提案-审议（推荐方案） |
| HPO | [run_hpo_v3_het.py](file:///d:/Kronos-R-Preview/run_hpo_v3_het.py) / [run_hpo_v4_het_refined.py](file:///d:/Kronos-R-Preview/run_hpo_v4_het_refined.py) | 异方差 HPO 搜索 |
| 流水线 | [run_full_pipeline.py](file:///d:/Kronos-R-Preview/run_full_pipeline.py) | 端到端串联 |

## 推理流程

```mermaid
flowchart LR
    A[用户输入<br/>股票历史] --> B[Tokenizer 编码]
    B --> C[GPT 提案<br/>top-K 候选]
    C --> D[对每个 y_k<br/>构造 BERT 输入]
    D --> E[BERT 验证<br/>P_BERT score]
    E --> F[argmax over K<br/>选出最终 token]
    F --> G[Decoder<br/>token → log return]
    G --> H[反归一化<br/>+ price space]
    H --> I[输出预测价格]
```

## 完整技术栈

- **数据**: 4695 只 A 股，cutoff 2024-02-01
- **Tokenizer**: BSQ Hierarchical (4D OHLC → vocab 1024)
- **GPT**: dim=256, depth=2, heads=4, GQA (kv_heads=1), causal
- **BERT**: dim=256~512, depth=2~4, heads=4~8, full bidirectional
- **训练**: AdamW, bfloat16, gradient checkpointing (可选)
- **推理**: bfloat16, KV cache 未用 (AR 全序列重算)
