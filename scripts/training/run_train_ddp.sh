#!/usr/bin/env bash
# Train the head + right-wrist RGB policy with one or more GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash scripts/training/run_train_ddp.sh ZARR[,ZARR2] [EXP] [BATCH_PER_GPU] [EPOCHS] [GPUS] [PORT]'
  echo 'Environment: WANDB_MODE WANDB_ENTITY WANDB_PROJECT CKPT CKPT_EVERY LOG_DIR'
  exit 0
fi
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
ZARR="${1:-${ZARRS:-}}"
EXP="${2:-${EXP:-arrange_flowers}}"
BATCH="${3:-${PER_GPU_BATCH:-64}}"
EPOCHS="${4:-${EPOCHS:-1000}}"
GPUS="${5:-${GPUS:-0}}"
PORT="${6:-${PORT:-29520}}"
[[ -n "$ZARR" ]] || { echo 'Supply a Zarr dataset path as the first argument.' >&2; exit 1; }
IFS=',' read -ra DATASETS <<< "$ZARR"
DATASET_PATH='['
for path in "${DATASETS[@]}"; do
  [[ -d "$path" ]] || { echo "Dataset directory does not exist: $path" >&2; exit 1; }
  DATASET_PATH+="\"$path\","
done
DATASET_PATH="${DATASET_PATH%,}]"
NAME="$(basename "${DATASETS[0]}" .zarr)"
export CUDA_VISIBLE_DEVICES="$GPUS" PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
LOGGING=("logging.mode=$WANDB_MODE" "logging.project=${WANDB_PROJECT:-petaldex}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then LOGGING+=("logging.entity=$WANDB_ENTITY"); fi
CKPT_ARGS=()
if [[ -n "${CKPT:-}" ]]; then
  [[ -f "$CKPT" ]] || { echo "Checkpoint does not exist: $CKPT" >&2; exit 1; }
  CKPT_ARGS=("optimizer.start_ckpt_path=$CKPT")
fi
IFS=',' read -ra GPU_IDS <<< "$GPUS"
LAUNCH=(accelerate launch --config_file "$ROOT/accelerate_config.yaml" --num_processes "${#GPU_IDS[@]}" --main_process_port "$PORT")
if (( ${#GPU_IDS[@]} > 1 )); then LAUNCH+=(--multi_gpu); fi
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
"${LAUNCH[@]}" train.py \
  "task.dataset_path=$DATASET_PATH" "task.dataset_name=$NAME" \
  "exp_name=$EXP" "dataloader.batch_size=$BATCH" "val_dataloader.batch_size=$BATCH" \
  "training.num_epochs=$EPOCHS" "training.checkpoint_every=${CKPT_EVERY:-20}" \
  "${LOGGING[@]}" "${CKPT_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train_${EXP}.log"
