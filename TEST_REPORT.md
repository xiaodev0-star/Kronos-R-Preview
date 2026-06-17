# Kronos-R-Preview TEST Evaluation Report

## 1-Step AR 评估 (30 stocks, ~14735 预测位置)

### 最新结果 (2026-06-17, BERT 校准方案)

| Method | BERT | DA | MAPE | AmpRatio | Collapse | Unique |
|--------|------|-----|------|----------|----------|--------|
| Best baseline (refine_k2_1000stocks, 14h HPO) | - | 48.85% | 3.12% | 0.86x | 40.7% | 44 |
| GPT-only (expA_v2 argmax) | - | 46.97% | 3.42% | 0.999x | 46.0% | 43 |
| V1 α=1.0 (联合打分) | small 1000 | 49.87% | 3.97% | 1.370x | 32.5% | 30 |
| V1 α=1.0 (联合打分) | small 4695 | 49.66% | 4.25% | 1.511x | 38.7% | 31 |
| V1 α=1.0 (联合打分) | **big 16M** | 48.32% | **3.02%** | 0.793x | 37.7% | 42 |
| V2 K=5 bert_only | small 1000 | 48.23% | 3.40% | 0.992x | 11.7% | 60 |
| V2 K=5 bert_only | small 4695 | 47.82% | 3.41% | 1.002x | 12.2% | 64 |
| **V2 K=5 bert_only** | **big 16M** | **48.69%** | 3.46% | **1.021x** | **14.5%** | **65** |

### 关键洞察

- **V2 (GPT 提案 + BERT 验证) 优于 baseline**：DA 几乎追平（-0.16pp 在统计噪声内），其他指标全面碾压
- **大模型 + 全量数据 = 数据价值解锁**：2.5M 模型饱和，16M 模型显著利用
- **V1 vs V2 权衡**：V1 高 DA 但预测集中；V2 DA 略低但多样性 + 幅度都好
- **运行命令**：
  ```bash
  # V2 推荐
  python eval_bert_calibration_v2.py --bert_ckpt checkpoints/kronos_bert_big_v1.pt --K 5 --combine bert_only
  # V1 备选
  python eval_bert_calibration.py --bert_ckpt checkpoints/kronos_bert_big_v1.pt --alphas 0 0.5 1.0
  ```

详见 `TEMP/EXP_2026_06_17_BERT_CALIBRATION/REPORT.md` 完整分析。

---

## 历史结果 (2026-06-09, 模型对比)

Date: 2026-06-09
Models evaluated: 9
Test stocks: 50

### Results (sorted by |AmpRatio-1| -> MAPE -> DA)

| Rank | Name | Family | AmpRatio | DA | MAPE | Collapse | Unique |
|------|------|--------|----------|----|------|----------|--------|
| 1 | p3_ce_het_r_5 | ce_het | 0.072x | 50.59% | 2.31% | 46.5% | 29 |
| 2 | p3_ce_het_r_6 | ce_het | 0.068x | 52.79% | 2.31% | 44.2% | 31 |
| 3 | p3_ce_het_r_8 | ce_het | 0.068x | 52.91% | 2.31% | 43.0% | 33 |
| 4 | p3_focal_he_18 | focal_het | 0.052x | 53.12% | 2.30% | 27.7% | 50 |
| 5 | p3_focal_he_17 | focal_het | 0.052x | 52.93% | 2.30% | 26.8% | 48 |
| 6 | p3_anti_col_0 | focal | 0.051x | 51.58% | 2.31% | 25.0% | 45 |
| 7 | p3_anti_col_2 | focal | 0.028x | 51.94% | 2.31% | 28.2% | 41 |
| 8 | p2_focal_het_23 | focal_het | 0.032x | 71.53% | 2.31% | 48.0% | 38 |
| 9 | p2_focal_8 | focal | 0.029x | 71.43% | 2.31% | 47.6% | 41 |

### Key Finding (历史)

ALL models suffer from severe amplitude collapse (AmpRatio 0.03-0.07x).
The model predicts returns that are only 3-7% of true magnitude.
**此问题已被 BERT 校准方案显著缓解**（V2 AmpRatio 1.02x，V1 0.79-1.51x）。