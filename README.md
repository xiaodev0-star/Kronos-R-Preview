# Kronos-R-Preview

基于 LLM 范式的金融时序因果预测项目。每只股票 = 一篇文档，从头读到尾，标准因果 next-token prediction。
**当前架构：GPT + BERT 双模型协同范式**（GPT 提案 + BERT 验证）。

## 快速开始

```bash
# 1. 训练 Tokenizer (4D OHLC)
python train_tokenizer.py

# 2. 训练 GPT — HPO Regime best (d=0.05, γ=6.0, lr=5e-4)
python train_base.py --gamma 6.0 --dropout 0.05 --lr 5e-4 --epochs 8
#    → checkpoints/best_gpt_regime.pt

# 3. 训练 BERT 校准器 — big BERT (16M, 全量 4695 stocks)
python train_bert.py
#    → checkpoints/kronos_bert_big_v1.pt

# 4. GPT + BERT V2 评估
python eval_bert_calibration_v2.py \
  --bert_ckpt checkpoints/kronos_bert_big_v1.pt \
  --K 5 --mask_strategy boundary --combine bert_only
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

### 双模型推理流程（V2）
```
For each test position p:
  1. GPT 给出 top-K 候选 {y_1, ..., y_K} 及概率
  2. 对每个 y_k 构造 BERT 输入: [BOS, tok_0, ..., MASK_at_p, y_k]
     (MASK 在最后一个 history token 位置, y_k 作为"未来")
  3. BERT 预测 at MASK: score_k = P_BERT(tok_{p-1} | context_with_y_k)
  4. argmax_k of score_k 选出最终 token
```

## 项目结构

```
Kronos-R-Preview/
├── config.py                    # 全局配置
├── data_processor.py            # 数据管道 (归一化 + 打包)
├── train_tokenizer.py           # Stage A: Tokenizer 训练
├── train_base.py                # Stage B: GPT 训练
├── train_bert.py                # Stage C: BERT 校准器训练
├── eval_bert_calibration_v2.py  # V2 GPT+BERT 评估
├── eval_helpers.py              # 共享评估工具
├── model/
│   ├── kronos_preview.py        # GPT 模型
│   ├── kronos_bert.py           # BERT 模型
│   ├── tokenizer.py             # BSQ Tokenizer
│   └── tokenizer_config.py
├── checkpoints/
│   ├── best_gpt_regime.pt       # HPO Regime best (d=0.05 γ=6 lr=5e-4)
│   ├── kronos_bert_big_v1.pt    # BERT big (16M, 4695 stocks)
│   └── tokenizer_v2_ohlc.pt     # Tokenizer
├── dataset/                     # 4695 只 A 股 CSV
└── README.md
```
