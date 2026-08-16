# Exp 05：继续预训练（CPT）

## 目标

Exp 04-B 的 HPO 定出两个最优配置。因未下载服务器权重，改为从零重训并延长至
100 epoch，在 400 窗验证协议下逐 epoch 评估，考察两个问题：更长的训练能否继续
改善 token 质量；下游信号是否随之改善。

## 设定

两配置仅 `lr_muon` 不同（0.005 / 0.01），其余一致：Muon+AdamW、lr=3e-4、
dropout 0.1、wd 0.01、fine_w 0.3、het_w 0.1、5% warmup + cosine to 0、
accumulation 32、seed 42。模型/词表沿用 `config.py` 默认（dim 256 / depth 6 /
heads 4 / GQA-1、7+7 位 BSQ 码本 128+128）。400 窗单日预测，4543 只股票、
约 180 万次预测/epoch，逐 epoch 全量推理。

## 结果

token 质量全程单调改善：balance 0.19→0.58、collapse 88%→42%、unique 11→63。

ep100 两配置几乎打平：

| 指标 | lr_muon=0.005 | lr_muon=0.01 |
|---|---|---|
| coarse balance | 0.576 | 0.581 |
| JSD | 0.319 | 0.308 |
| DA | 48.8% | 49.6% |
| MAPE | 3.358% | 3.366% |
| RankIC | 0.0311 | 0.0287 |
| collapse（med/p90） | 42.2/59.3% | 42.8/61.9% |
| unique tokens | 63 | 61 |

- 最佳 balance：0.005@ep93、0.01@ep97；ep85–100 已饱和（仅 +0.004）。
- val_loss 谷底：0.005@ep13=2.884、0.01@ep20=2.861，之后回升，但 token 质量持续改善。

## 分析

1. **val_loss 被坍塌污染**：触底后回升与 token 质量单调改善背离，再次确认 raw
   val_loss 不可作健康指标。
2. **续训预算已耗尽**：ep85 后 balance 饱和，该 recipe 下 100 epoch 即为上限。
3. **两配置无实质差异**：balance 差 0.005、JSD 差 0.011，lr_muon 0.005→0.01 处于
   平坦区；下游 DA/RankIC 为单 seed 观测，不足以区分二者。
4. **核心缺口依旧**：collapse ~42%（远高于 ~11% 目标）、unique 63（目标 97）、
   DA ~49%（不高于多数类基线）、RankIC ~0.03。CPT 把 token 分布从极坍缩拉到中等
   坍缩，但下游读出信号仍弱。

## 对 06 的铺垫

1. **停止 pre-train 尺度的工作**：token 质量已饱和，继续续训无增益。
2. **瓶颈疑在读出一侧**：token 分布已含结构（unique 63、support F1 ~0.79），但
   greedy 解码的 DA/RankIC 仍弱——待 06 检验：弱在下游，是表示不足，还是 greedy
   readout 丢信息。
3. **06 的切入点**：冻结 CPT backbone，直接测其隐藏表示是否已含可提取的横截面排序
   信号；若含，则问题在 readout，用 exact posterior / 冻结 rank head 即可提取。
4. **collapse 与排序分开治理**：降 collapse（token 分布）与提 RankIC（排序）未必
   同源，06 不应假设前者自动带来后者。

> 注：06 的上游是分布匹配分支（基于 lr_muon=0.01 的 ep100，进一步把 collapse 压到
> ~25%、balance 提到 ~0.70）；本文的两个权重是它的上游。
