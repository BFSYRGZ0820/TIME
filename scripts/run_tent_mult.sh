#!/bin/bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=6,7
export HF_DATASETS_OFFLINE=1

python tent_moe_mult.py --model_name="./models/Qwen1.5-MoE-A2.7B-Chat" --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" --eval_batch_size=8 --lora_r=16 --lora_alpha=16 \
  --lr=1e-5 --reg_lambda=0.0 --time_random_reset=False --time_reset_prob=0.01 --result_path="results/tent_mult_results_qwen_chat.txt"
python tent_moe_mult.py --model_name="./models/Qwen1.5-MoE-A2.7B" --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" --eval_batch_size=8 --lora_r=16 --lora_alpha=16 \
  --lr=1e-5 --reg_lambda=0.0 --time_random_reset=False --time_reset_prob=0.01 --result_path="results/tent_mult_results_qwen.txt"
python tent_moe_mult.py --model_name="./models/DeepSeek-V2-Lite" --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" --eval_batch_size=4 --lora_r=16 --lora_alpha=16 \
  --lr=1e-5 --reg_lambda=0.0 --time_random_reset=False --time_reset_prob=0.01 --result_path="results/tent_mult_results_deepseek.txt"

echo "All multi-GPU Tent experiments completed!"
