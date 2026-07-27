# Experiment 04：Loss、Optimizer 与 HPO 刷新

> **依赖门**：Exp 03-Sup 正在补做受控深度/宽度容量实验。04-A/B/C 默认读取 `experiments/03-gpt-scaling-sup/run_seed42/selection.json`；该文件尚未产生时会主动停止，不会静默回退到带 optimizer-step 混杂的旧 `xlarge@18`。

旧 Exp 03 的 target-relative 重分析把 `xlarge@18` 识别为历史 coarse 指标候选，但 Sup 审计发现五个架构实际 optimizer steps 不同。04 的旧结果仍是历史证据，不与新协议的绝对数值混用；刷新后的顺序是：

```text
Exp 02 tokenizer selection (64x192 @ 9+7)
                    +
Exp 03-Sup controlled capacity selection (pending)
                    |
                    v
04-A loss: Focal vs CE
                    |
                    v
04-B optimizer: AdamW vs Muon
                    |
                    v
04-C deterministic full-data HPO
                    |
                    v
sealed holdout (only if a healthy winner exists)
```

所有正式研究固定 seed=42、全量股票、完整序列、30 epochs、accumulation=32，并使用架构无关的 loader seed 与精确 accumulation blocks，得到每 epoch 128、总计 3840 个 optimizer steps。每个 epoch 都在四个 validation windows 上评估。分析不再默认最后 10 epoch，而是在 near-best-loss 区间内选择连续 5 epoch 成熟窗；DA/RankIC/MAPE/AmpRatio 与 target-relative codebook 指标分组报告，不用不透明的加权总分。

## 推荐执行顺序

```powershell
$py = 'D:\conda_envs\llm-t\Scripts\python.exe'

& $py .\experiments\04\a-loss-ablation\sweep_loss.py --smoke
& $py .\experiments\04\b-optimizer-ablation\sweep_optimizer.py --smoke
& $py .\experiments\04\c-hpo\sweep_hpo.py --smoke

& $py .\experiments\04\a-loss-ablation\sweep_loss.py
& $py .\experiments\04\b-optimizer-ablation\sweep_optimizer.py
& $py .\experiments\04\c-hpo\sweep_hpo.py
```

这些命令应在 Sup 选型审阅完成后执行。04-B 会校验 04-A 的选择；04-C 会同时校验 04-A/04-B。任何上游 selection、源码、数据签名或正式协议变化都会改变 fingerprint，防止新旧结果混用。
