#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
LOG="${PROJECT_DIR}/logs/minicpm_gr00t_gen2act_droid_h15_vlmft_resume150k_8gpu.log"
while true; do
  step=$(rg -o 'Step [0-9]+' "$LOG" | tail -1 | awk '{print $2}')
  if [[ "${step:-0}" -ge 60000 ]]; then break; fi
  sleep 30
done
sleep 30
for pid in $(pgrep -f 'starVLA/training/train_starvla.py' || true); do kill -TERM "$pid" 2>/dev/null || true; done
sleep 20
cd "$PROJECT_DIR"
export PYTHON_BIN=/usr/bin/python3
export PYTHONPATH="${PROJECT_DIR}:/usr/local/lib/python3.12/dist-packages"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8 MAX_STEPS=150000 WANDB_MODE=online
export WANDB_API_KEY='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'
exec bash examples/realRobots/DROID/train_files/run_minicpm_gr00t_gen2act_finetune.sh
