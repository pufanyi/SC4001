#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONFIG_NAME="${CONFIG_NAME:-resnet}"
CHECKPOINT_FORMAT="${CHECKPOINT_FORMAT:-fsdp}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi

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
  echo "Error: provide at least one checkpoint path for distributed evaluation." >&2
  exit 1
fi

for CHECKPOINT_PATH in "${MODEL_PATHS[@]}"; do
  echo "[distributed-eval] Evaluating checkpoint: ${CHECKPOINT_PATH}"
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m eval.pipeline.run \
    --config-name "${CONFIG_NAME}" \
    evaluation.checkpoint_path="${CHECKPOINT_PATH}" \
    evaluation.checkpoint_format="${CHECKPOINT_FORMAT}" \
    "${EXTRA_ARGS[@]}"
done
