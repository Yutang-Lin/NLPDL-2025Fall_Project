#!/usr/bin/env bash

#SBATCH --partition=h100
#SBATCH --job-name=qwen
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time 72:00:00

export HF_ENDPOINT=https://hf-mirror.com
source ~/proxy/set_proxy.sh
uv run python qwen_experiment.py \
  --capability-file data/example/dataset_capability_english_final.json \
  --layer-index 20 \
  --max-samples 64 \
  --alpha 3.0 \
  --question "Explain quantum entanglement in simple language but with deep insights."