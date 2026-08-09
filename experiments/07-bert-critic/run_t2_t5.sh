#!/usr/bin/env bash
# T-series execution after the T1 eval scoring completes:
#   T2 proposals cache (ep100) -> T2 fine-tune -> T5 fit hidden (T1 weights) -> T5 head.
set -e
cd "$(dirname "$0")/../.."

echo "== [1/4] T2 GPT proposals cache (ep100) =="
python -u experiments/07-bert-critic/cache_gpt_proposals.py --which ep100 2>&1 | tail -3

echo "== [2/4] T2 GPT-noise denoising fine-tune (from T1) =="
python -u experiments/07-bert-critic/train_bert_t2.py \
    --init_from checkpoints/bert_critic_mlm_t1_w512.pt \
    --proposals1 checkpoints/gpt_proposals_ep100.pt \
    --save_path checkpoints/bert_critic_mlm_t2.pt \
    --tag t2 --epochs 3 --lr 3e-5 2>&1 | tail -6

echo "== [3/4] T5 fit-region BERT hidden cache (T1 weights) =="
python -u experiments/07-bert-critic/cache_bert_hidden.py \
    --region fit --bert checkpoints/bert_critic_mlm_t1_w512.pt \
    --window 512 --suffix t1_w512 2>&1 | tail -4

echo "== [4/4] T5 BERT-hidden rank head =="
python -u experiments/07-bert-critic/train_bert_head.py \
    --suffix t1_w512 --seed 42 2>&1 | tail -8

echo "== done =="
