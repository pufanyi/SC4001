NPROC_PER_NODE=4


PYTHONPATH=$PYTHONPATH:$(pwd)
. .venv/bin/activate
torchrun --nproc_per_node=$NPROC_PER_NODE -m train.pipeline.run