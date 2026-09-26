#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export STARVLA_DISABLE_DEEPSPEED="${STARVLA_DISABLE_DEEPSPEED:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CONFIG="examples/realRobots/DROID/train_files/starvla_minicpm_gr00t_gen2act_finetune.yaml"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/MiniCPM-V-4.6}"
RUN_ID="${RUN_ID:-minicpm_gr00t_gen2act_droid_h15_vlmft}"
MAX_STEPS="${MAX_STEPS:-150000}"
MAX_SAMPLES="${MAX_SAMPLES:-null}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"

${ACCELERATE_BIN} --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.max_samples "${MAX_SAMPLES}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --run_id "${RUN_ID}"
