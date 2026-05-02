#!/bin/bash
set -euo pipefail

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file static/finetune_config.yaml --main_process_port 0 hera_moe_opt.py \
  --model_name="./models/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" \
  --eval_batch_size=8 --lora_r=16 --lora_alpha=16 --gradient_checkpointing=True \
  --lr=1e-5 --target_layer=0 --warmup_batches=10 \
  --tau_d=0.2 --tau_i=0.2 --tau_low=0.35 --tau_high=0.65 \
  --lambda_r=2.0 --lambda_bal=0.05 --lambda_anc=1e-4 \
  --lambda_q=1.0 --lambda_c=1.0 \
  --alpha_d=0.5 --alpha_i=1.0 --alpha_h=0.5 \
  --kappa=0.3 --eta_e=0.9 --eta_q=0.95 --eta_c=0.95 \
  --safe_top_b=4 --restoration_rho=0.1 \
  --a_n=0.5 --a_q=0.25 --a_c=0.25 --m_min=0.1 --m_max=1.0 \
  --result_path="results/hera_results_qwen_chat.txt"

accelerate launch --config_file static/finetune_config.yaml --main_process_port 0 hera_moe_opt.py \
  --model_name="./models/Qwen1.5-MoE-A2.7B" \
  --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" \
  --eval_batch_size=8 --lora_r=16 --lora_alpha=16 --gradient_checkpointing=True \
  --lr=1e-5 --target_layer=0 --warmup_batches=10 \
  --tau_d=0.2 --tau_i=0.2 --tau_low=0.35 --tau_high=0.65 \
  --lambda_r=2.0 --lambda_bal=0.05 --lambda_anc=1e-4 \
  --lambda_q=1.0 --lambda_c=1.0 \
  --alpha_d=0.5 --alpha_i=1.0 --alpha_h=0.5 \
  --kappa=0.3 --eta_e=0.9 --eta_q=0.95 --eta_c=0.95 \
  --safe_top_b=4 --restoration_rho=0.1 \
  --a_n=0.5 --a_q=0.25 --a_c=0.25 --m_min=0.1 --m_max=1.0 \
  --result_path="results/hera_results_qwen.txt"

accelerate launch --config_file static/finetune_config.yaml --main_process_port 0 hera_moe_opt.py \
  --model_name="./models/DeepSeek-V2-Lite" \
  --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" \
  --eval_batch_size=4 --lora_r=16 --lora_alpha=16 --gradient_checkpointing=True \
  --lr=1e-5 --target_layer=0 --warmup_batches=10 \
  --tau_d=0.2 --tau_i=0.2 --tau_low=0.35 --tau_high=0.65 \
  --lambda_r=2.0 --lambda_bal=0.05 --lambda_anc=1e-4 \
  --lambda_q=1.0 --lambda_c=1.0 \
  --alpha_d=0.5 --alpha_i=1.0 --alpha_h=0.5 \
  --kappa=0.3 --eta_e=0.9 --eta_q=0.95 --eta_c=0.95 \
  --safe_top_b=4 --restoration_rho=0.1 \
  --a_n=0.5 --a_q=0.25 --a_c=0.25 --m_min=0.1 --m_max=1.0 \
  --result_path="results/hera_results_deepseek.txt"

echo "All HERA experiments completed!"
