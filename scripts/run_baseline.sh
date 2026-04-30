#!/bin/bash
set -euo pipefail

# 自动选择最空闲的 GPU（可用 GPU_ID 环境变量覆盖）
pick_gpu() {
	if command -v nvidia-smi >/dev/null 2>&1; then
		nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
		| awk 'BEGIN{min=1e12; id=0; i=0} {v=$1+0; if (v<min){min=v; id=i} i++} END{print id}'
	else
		echo 0
	fi
}
GPU_ID="${GPU_ID:-$(pick_gpu)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1

accelerate launch --config_file static/evaluation_config.yaml baseline.py \
  --model_name="./models/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" \
  --eval_batch_size=8 \
  --result_path="results/baseline_qwen_chat.txt"

accelerate launch --config_file static/evaluation_config.yaml baseline.py \
  --model_name="./models/Qwen1.5-MoE-A2.7B" \
  --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" \
  --eval_batch_size=8 \
  --result_path="results/baseline_qwen.txt"

accelerate launch --config_file static/evaluation_config.yaml baseline.py \
  --model_name="./models/DeepSeek-V2-Lite" \
  --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" \
  --eval_batch_size=4 \
  --result_path="results/baseline_deepseek.txt"

echo "All experiments completed!"

python ../send_email.py  --body "<p>baseline</p>"
