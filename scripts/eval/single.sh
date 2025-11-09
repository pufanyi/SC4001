#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=$(pwd)

CONFIG_NAME="${CONFIG_NAME:-resnet}"
CHECKPOINT_FORMAT="${CHECKPOINT_FORMAT:-auto}"

. .venv/bin/activate

MODEL_PATHS=()
EXTRA_ARGS=()

while (($#)); do
  case "$1" in
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      MODEL_PATHS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  MODEL_PATHS+=("")
fi

for CHECKPOINT_PATH in "${MODEL_PATHS[@]}"; do
  echo "[single-eval] Evaluating checkpoint: ${CHECKPOINT_PATH:-<pretrained>}"
  overrides=("--config-name" "${CONFIG_NAME}")
  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    overrides+=("evaluation.checkpoint_path=${CHECKPOINT_PATH}")
    overrides+=("evaluation.checkpoint_format=${CHECKPOINT_FORMAT}")
  fi
  python -m eval.pipeline.run \
    "${overrides[@]}" \
    "${EXTRA_ARGS[@]}"
done
