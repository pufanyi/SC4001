export PYTHONPATH=$(pwd)

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

. .venv/bin/activate

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m train.pipeline.run \
  --config-name convnextv2 \
  "$@"
