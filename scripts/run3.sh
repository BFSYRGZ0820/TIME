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
#export HF_DATASETS_OFFLINE=1
#python moe-qwen.py --model_name="./Qwen1.5-MoE-A2.7B-Chat" --task="winogrande" --train_batch_size=8 --eval_batch_size=8 --epsilon=0.01 \
#  --ot_strength=0.0 --lr=1e-5 --result_path="results/results_qwen_test.txt" --output_path="results/qwen/test"

python moe-qwen_oral.py --model_name="./Qwen1.5-MoE-A2.7B-Chat" --task="winogrande,arc_challenge,arc_easy,openbookqa,rte,xquad_zh,xquad_es,boolq" --train_batch_size=8 --eval_batch_size=8 --epsilon=0.01 \
  --ot_strength=5.0 --lr=1e-5 --result_path="results/results_qwen_test1.txt" --output_path="results/qwen/test1"

#python baseline_qwen.py

echo "All experiments completed!"

python ../send_email.py  --body "<p>moe-qwen_oral</p>"
