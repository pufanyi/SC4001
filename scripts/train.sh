NPROC_PER_NODE=2


PYTHONPATH=$PYTHONPATH:$(pwd)
. .venv/bin/activate
torchrun --nproc_per_node=$NPROC_PER_NODE -m train.pipeline.run