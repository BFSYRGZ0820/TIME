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

python train_ssa_new.py

echo "All experiments completed!"

python ../send_email.py  --body "<p>train</p>"
