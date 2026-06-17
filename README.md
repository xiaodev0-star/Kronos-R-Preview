# Kronos-R-Preview

基于 LLM 范式的金融时序因果预测项目。每只股票 = 一篇文档，从头读到尾，标准因果 next-token prediction。
**当前架构：GPT + BERT 双模型协同范式**（GPT 提案 + BERT 验证）。

## 核心结果

### 1-Step AR 评估（30 stocks × 14735 预测）

| Method | DA | MAPE | AmpRatio | Collapse | Unique |
|:-------|:---:|:----:|:--------:|:--------:|:------:|
| **Best baseline** (refine_k2_1000stocks, 14h HPO) | 48.85% | 3.12% | 0.86x | 40.7% | 44 |
| **GPT + BERT v2** (big BERT 16M, K=5) | **48.69%** | 3.46% | **1.021x** | **14.5%** | **65** |
| GPT-only (expA_v2 argmax) | 46.97% | 3.42% | 0.999x | 46.0% | 43 |

**GPT+BERT 方案**：
- DA 追平最优 baseline（-0.16pp 在统计噪声内）
- **AmpRatio 1.02x**（向 1.0 大幅靠近，幅度坍塌修复）
- **Collapse 14.5%**（vs baseline 40.7%，断崖式下降）
- **Unique 65 tokens**（vs baseline 44，多样性 +48%）

完整历史结果（含 v1, v2, 小/大 BERT, 1000/4695 stocks）见 `TEMP/EXP_2026_06_17_BERT_CALIBRATION/REPORT.md`。

## 快速开始

```bash
# 1. 训练 Tokenizer (4D OHLC)
python train_tokenizer.py

# 2. 训练 GPT 模型（提案者）
python train_base.py --loss focal --gamma 6.0 --label_smoothing 0.05

# 3. 训练 BERT 校准器（验证者）— 推荐配置: 16M, 全部 4695 stocks
python train_bert.py \
  --dim 512 --depth 4 --heads 8 --num_kv_heads 2 \
  --epochs 6 --max_stocks 0 --max_seq_len 1024 \
  --save_path checkpoints/kronos_bert_big_v1.pt

# 4a. GPT-only 评估（baseline）
python eval_batch_1step.py

# 4b. GPT + BERT 校准评估（推荐 V2 方案）
python eval_bert_calibration_v2.py \
  --bert_ckpt checkpoints/kronos_bert_big_v1.pt \
  --K 5 --mask_strategy boundary --combine bert_only

# 4c. 联合打分 V1 评估（备选，P_GPT^α · P_BERT^(1-α)）
python eval_bert_calibration.py \
  --bert_ckpt checkpoints/kronos_bert_big_v1.pt \
  --alphas 0 0.3 0.5 0.7 1.0
```

## 模型架构

### GPT (KronosPreview) — 提案者
- **2.7M 参数**: dim=256, depth=2, heads=4, GQA (kv_heads=1)
- Causal attention（tril mask）
- BSQ Tokenizer: 2-level hierarchical, vocab=1024
- VA Embedding: Volume/Amount 连续 MLP 注入
- Loss: CE / Focal (γ=3-10) / +Label Smoothing / +Entropy Reg
- 归一化: 价格 historical Z-Score (per-stock train-history); VA 首日基线 + Z-Score

### BERT (KronosBert) — 验证者
- **2.5M ~ 16M 参数**（dim=256 ~ 512, depth=2 ~ 4）
- **Bidirectional** attention（无 causal mask）
- 训练目标：**MLM**（15% 随机 mask + CE loss）
- **不针对 next-token 预测优化**——是序列自然性的判别器
- 在推理时用作"一致性检查器"：把 y_hat 作为"未来"信息，检查中间 token 是否仍可被还原

### 双模型推理流程（V2 推荐）
```
For each test position p:
  1. GPT 给出 top-K 候选 {y_1, ..., y_K} 及概率
  2. 对每个 y_k 构造 BERT 输入: [BOS, tok_0, ..., MASK_at_p, y_k]
     (MASK 在最后一个 history token 位置, y_k 作为"未来")
  3. BERT 预测 at MASK: score_k = P_BERT(tok_{p-1} | context_with_y_k)
  4. argmax_k of score_k 选出最终 token
```

## 关键超参 (CLI)

### GPT 训练

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| `--loss` | `focal` | Focal loss 抑制零坍塌 |
| `--gamma` | `6.0` | 1-Step 最优; 8.0 为 10-Step AR 最优 |
| `--label_smoothing` | `0.05` | 配合 Focal 进一步改善 |
| `--weight_decay` | `0.001` | CE 模型推荐; Focal 可用默认 0.01 |

### BERT 训练

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| `--dim` | `512` | 隐藏维度（small=256, big=512） |
| `--depth` | `4` | Transformer 层数（small=2, big=4） |
| `--heads` | `8` | 注意力头数（small=4, big=8） |
| `--num_kv_heads` | `2` | KV 头数（GQA, small=1, big=2） |
| `--max_stocks` | `0` | 0=全部 4695 stocks |
| `--mlm_prob` | `0.15` | MLM mask 概率 |
| `--lr` | `3e-4` | 学习率 |

### V2 评估

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| `--K` | `5` | GPT 候选数 |
| `--mask_strategy` | `boundary` | mask 边界 token |
| `--combine` | `bert_only` | 直接用 BERT 分数（优于 product） |

## 文档

| 文档 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 数据 / 训练流程图 + 模型内部结构 + Loss 选项 |
| `CODE_WIKI.md` | 数据流、模型架构、训练/HPO 关键修复 |
| `IMPROVE.md` | 历史改进计划 + 待做实验 |
| `TEST_REPORT.md` | 跨模型测试结果对比 |
| `TEMP/EXP_2026_06_17_BERT_CALIBRATION/REPORT.md` | **BERT 校准 session 完整报告** ★ |
| `TEMP/EXP_2026_06_17_BERT_CALIBRATION/docs/KRONOS_R_DESIGN.md` | 设计哲学详细版 |
| `TEMP/README.md` | 所有历史实验汇总 |

## 项目结构

```
Kronos-R-Preview/
├── config.py                    # 全局配置
├── data_processor.py            # 数据管道 (归一化 + 打包)
├── train_tokenizer.py           # Stage A: Tokenizer 训练
├── train_base.py                # Stage B: GPT 训练
├── train_bert.py                # Stage C: BERT 校准器训练（MLM）★
├── eval_batch_1step.py          # GPT-only 1-step 评估
├── eval_bert_calibration.py     # V1 联合打分评估
├── eval_bert_calibration_v2.py  # ★ V2 提案-审议评估（推荐方案）
├── eval_cross_loss.py           # 跨 loss 公平评估
├── model/
│   ├── kronos_preview.py        # GPT 模型定义 + VA embedding
│   ├── kronos_bert.py           # BERT 模型定义（双向 + MLM 头）★
│   ├── tokenizer.py             # BSQ Tokenizer
│   └── tokenizer_config.py
├── checkpoints/
│   ├── expA_v2.pt               # 预训练 GPT (生产 baseline)
│   ├── kronos_bert_big_v1.pt    # 预训练 BERT big (16M, 4695 stocks) ★
│   ├── kronos_bert_v1.pt        # 预训练 BERT small (2.5M, 1000 stocks)
│   └── tokenizer_v2_ohlc.pt     # 预训练 Tokenizer
├── dataset/                     # 4695 只 A 股 CSV
├── TEMP/                        # 历史实验归档
│   ├── EXP_2026_06_17_BERT_CALIBRATION/  # 本 session 详细记录
│   ├── EXP_2026_06_THINKING_V4/           # 之前 thinking 最佳
│   └── ...
└── README.md                    # 本文件
```

## 设计哲学摘要

> **不要让一个模型做所有事。让 GPT 做生成，让 BERT 做审视。
> 候选由生成者提出，验证由审视者盖章。**

详细设计哲学见 `TEMP/EXP_2026_06_17_BERT_CALIBRATION/docs/KRONOS_R_DESIGN.md`。
