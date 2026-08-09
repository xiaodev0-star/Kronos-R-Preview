#!/usr/bin/env bash
# BERT-Critic 续跑编排：calib 候选/打分 -> F-INT -> B0-B5 arms。
# 前置：全量 eval 打分已完成（scores_eval_K8_w512_stride1.npz）。
set -e
cd "$(dirname "$0")/../.."

ROOTS="server_runs/weights/07-bert-critic/seed42"

echo "== [1/5] calib 候选（light：top-K，跳过后验；F-STK 时再补全） =="
python experiments/07-bert-critic/build_gpt_candidates.py --region calib --device cuda --light
echo "== [2/5] calib BERT 打分 =="
python experiments/07-bert-critic/score_bert.py --region calib --device cuda
echo "== [3/5] F-INT（calib 切片选 λ） =="
python experiments/07-bert-critic/fuse_scores.py --arm fint
echo "== [4/5] B0-B5 arms 评估 =="
python experiments/07-bert-critic/eval_critic.py --mode arms
echo "== [5/5] done =="
