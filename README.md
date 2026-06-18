# Kronos-R-Preview

基于 LLM 范式的金融时序因果预测项目。每只股票 = 一篇文档，从头读到尾，标准因果 next-token prediction。
**当前架构：GPT + BERT 双模型协同范式**（GPT 提案 + BERT 验证）。

## 核心结果

### 1-Step AR 评估（30 stocks × 14735 预测）

| Method | DA | MAPE | AmpRatio | Collapse | Unique |
|:-------|:---:|:----:|:--------:|:--------:|:------:|
| **HPO 2026-06-18 best** (phase3_t000 + big BERT, K=5 bert_only) | **48.12%** | 3.48% | **1.062x** | **19.4%** | **63** |
| HPO 2026-06-18 trading variant (phase4_t003 + big BERT, K=5 bert_only) | 47.93% | 3.45% | 1.036x | **16.8%** | **69** |
| GPT-only (expA_v2_hpo argmax, HPO config) | 47.29% | 3.22% | 0.927x | 41.4% | 45 |
| Pre-HPO baseline (expA_v2 γ=6+ls=0.05 + big BERT, K=5 bert_only) | 48.69% | 3.46% | 1.021x | 14.5% | 65 |

**HPO 2026-06-18 关键发现**：
- **focal γ=4（不是 γ=6）** 是新最优，label_smoothing 应为 0（不是 0.05）
- **heteroscedastic head 默认开**（het_weight=0.1），相比 CE-only 显著提升 DA
- 训练默认参数已对齐：直接 `python train_base.py` 即可复现 phase3_t000
- V2 校准依然贡献最大：DA +0.83pp，坍塌率减半

完整历史结果见 `TEMP/hpo_final_report_2026_06_18.md` + `TEMP/EXP_2026_06_17_BERT_CALIBRATION/REPORT.md`。

## 快速开始

```bash
# 1. 训练 Tokenizer (4D OHLC)
python train_tokenizer.py

# 2. 训练 GPT 模型（提案者）— HPO 2026-06-18 best (phase3_t000)
#    默认：focal γ=4 + heteroscedastic=ON + dropout=0.1 + 10 epochs
python train_base.py
#    → 保存到 checkpoints/expA_v2_hpo.pt

#    备选（低坍塌 + 高AR）：phase4_t003 配置 — DA 47.93%, Coll 16.8%
#    python train_base.py --epochs 15 --dropout 0.05

# 3. 训练 BERT 校准器（验证者）— HPO 2026-06-18 best: 16M, 全部 4695 stocks
#    下面所有参数已是 train_bert.py 的默认，可直接：
python train_bert.py
#    → 保存到 checkpoints/kronos_bert_big_v1.pt

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

### GPT 训练（HPO 2026-06-18 best = phase3_t000）

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| `--loss` | `focal` | 默认 focal；CE 用 `--loss ce` |
| `--gamma` | `4.0` | HPO 2026-06-18 best（γ=6 是旧 best，已被 γ=4 超过） |
| `--label_smoothing` | `0.0` | HPO 2026-06-18 best（γ=4 不需要 LS） |
| `--weight_decay` | `0.01` | HPO 2026-06-18 best (focal) |
| `--heteroscedastic` | `ON` | HPO 2026-06-18 best，默认开（用 `--no-heteroscedastic` 关掉） |
| `--het_weight` | `0.1` | HPO 2026-06-18 best（异方差 NLL 损失权重） |
| `--dropout` | `0.1` | HPO 2026-06-18 best；trading 变体用 `--dropout 0.05` |
| `--epochs` | `10` | HPO 2026-06-18 best（phase3）；trading 变体用 `--epochs 15` |

### BERT 训练（HPO 2026-06-18 best = big BERT, 16M）

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| `--dim` | `512` | 隐藏维度（small=256, big=512） |
| `--depth` | `4` | Transformer 层数（small=2, big=4） |
| `--heads` | `8` | 注意力头数（small=4, big=8） |
| `--num_kv_heads` | `2` | KV 头数（GQA, small=1, big=2） |
| `--max_stocks` | `0` | 0=全部 4695 stocks（big BERT 解锁全量数据价值） |
| `--max_seq_len` | `1024` | 推理时序列长度 |
| `--epochs` | `6` | big BERT 训练轮数 |
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
| `TEMP/hpo_final_report_2026_06_18.md` | **HPO 2026-06-18 完整报告** ★（推荐先看） |
| `TEMP/session_report_2026_06_18.md` | 本 session 的 screening + HPO 总结 |
| `TEMP/EXP_2026_06_17_BERT_CALIBRATION/REPORT.md` | BERT 校准 session 完整报告 |
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
│   ├── expA_v2_hpo.pt           # ★ HPO 2026-06-18 best GPT (focal γ=4 + het, 10ep)
│   ├── expA_v2.pt               # 旧 baseline (γ=6+ls=0.05) — 保留兼容
│   ├── kronos_bert_big_v1.pt    # ★ 预训练 BERT big (16M, 4695 stocks)
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
