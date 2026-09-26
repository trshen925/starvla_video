#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export STARVLA_DISABLE_DEEPSPEED="${STARVLA_DISABLE_DEEPSPEED:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CONFIG="examples/realRobots/DROID/train_files/starvla_minicpm_gr00t_gen2act.yaml"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/MiniCPM-V-4.6}"
RUN_ID="${RUN_ID:-minicpm_gr00t_gen2act_droid_h15}"
MAX_STEPS="${MAX_STEPS:-50000}"
MAX_SAMPLES="${MAX_SAMPLES:-null}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"

accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.max_samples "${MAX_SAMPLES}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --run_id "${RUN_ID}"
