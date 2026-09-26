#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
LOCK_FILE="/tmp/starvla_minicpm_gr00t_gen2act_vlmft_150k_8gpu.lock"

# Keep only one waiter/training launcher alive on this host.
exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "Another VLMFT 150k waiter or training launcher is already active."
  exit 1
}

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') launching VLMFT to step 150000 on all 8 GPUs"
if false; then
  mapfile -t GPU_STATS < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  if [[ "${#GPU_STATS[@]}" -eq 8 ]]; then
    ALL_IDLE=1
    for STAT in "${GPU_STATS[@]}"; do
      IFS=',' read -r USED UTIL <<<"${STAT}"
      USED="${USED//[[:space:]]/}"
      UTIL="${UTIL//[[:space:]]/}"
      if (( USED >= 1000 || UTIL >= 10 )); then
        ALL_IDLE=0
        break
      fi
    done
    if (( ALL_IDLE == 1 )); then
      break
    fi
  fi
  sleep 30
done
fi

cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export MAX_STEPS=150000
exec bash examples/realRobots/DROID/train_files/run_minicpm_gr00t_gen2act_finetune.sh
