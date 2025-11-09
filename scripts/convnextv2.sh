#!/usr/bin/env bash

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

source "${REPO_ROOT}/.venv/bin/activate"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m train.pipeline.run \
  --config-name default_config \
  "$@"
