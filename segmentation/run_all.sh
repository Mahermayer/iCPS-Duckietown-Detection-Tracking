#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEG_DIR="${REPO_ROOT}/segmentation"
TRAIN_CONFIG="${1:-${SEG_DIR}/configs/train/focal_dice.yaml}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_ARGS=()
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ARGS=(--dataset-root "${DATASET_ROOT}")
fi

VARIANTS=(
  vanilla
  depthwise
  se_decoder
  aspp
  depthwise_se
  depthwise_aspp
  se_aspp
  full
)

for variant in "${VARIANTS[@]}"; do
  echo "Starting variant: ${variant}"
  OUTPUT_DIR="${REPO_ROOT}/outputs/${variant}"
  mkdir -p "${OUTPUT_DIR}"
  python3 "${SEG_DIR}/train.py" \
    --model-config "${SEG_DIR}/configs/model/${variant}.yaml" \
    --train-config "${TRAIN_CONFIG}" \
    "${DATASET_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
done
