#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=moe
#SBATCH --gres=gpu:1

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 0 moe-qwen.py \
  --model_name="./Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --epsilon=0.01 \
  --ot_strength=5.0 \
  --lr=1e-5 \
  --result_path="results/results_qwen_test.txt" \
  --output_path="results/qwen/merge-45e/test" |& tee results/log_45e_test
