# Hydra Configuration Reference

Hydra lets you compose experiment settings at runtime. This project stores structured defaults under `train/conf/`. Each YAML file mirrors the dataclasses defined in `train/config.py`, giving strongly typed access throughout the code.

## Defaults Tree

`train/conf/config.yaml` declares the base:

```yaml
defaults:
  - dataset: flowers
  - model: qwen3_vl
  - trainer: default
  - _self_
seed: 42
hydra:
  run:
    dir: .
  output_subdir: null
```

When you run `python -m train`, Hydra merges:

1. `dataset/flowers.yaml`
2. `model/qwen3_vl.yaml`
3. `trainer/default.yaml`
4. Inline CLI overrides

The `_self_` entry ensures values in `config.yaml` override earlier defaults.

## Available Fields

### dataset.*

- `dataset_id`: Hugging Face dataset repo (e.g. `pufanyi/flowers102`).
- `instruction`: Text prompt appended to each conversation.
- `max_train_samples` / `max_eval_samples`: Optional limits (useful for debugging).

### model.*

- `model_id`: Any checkpoint compatible with `Qwen3VLForConditionalGeneration`.
- `gradient_checkpointing`: Toggle memory saving; disables `use_cache`.

### trainer.*

- `output_dir`: Saved checkpoints + processor.
- `num_train_epochs`: Float allowed to support partial epochs.
- `per_device_train_batch_size`: Number of conversational examples per GPU.
- `gradient_accumulation_steps`: Combine steps to simulate larger global batch.
- `learning_rate`, `weight_decay`, `warmup_ratio`: Optimiser settings (AdamW defaults from HF Trainer).
- `logging_steps`, `save_steps`, `save_total_limit`: Logging/checkpoint cadence.
- `dataloader_num_workers`: Increase >0 for faster image loading.
- `push_to_hub`, `hub_model_id`: Publish checkpoints automatically.
- `precision.bf16` / `precision.fp16`: Choose one mixed-precision mode.
- `evaluation_strategy`, `report_to`: Standard Trainer controls.

## Common Overrides

```bash
# Fast sanity check with fewer examples and lighter precision
python -m train \
  dataset.max_train_samples=256 \
  dataset.max_eval_samples=128 \
  trainer.precision.fp16=true \
  trainer.output_dir=runs/debug-fp16

# Enable gradient checkpointing and change base model
python -m train \
  model.model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.gradient_checkpointing=true
```

Hydra also accepts overrides from config files:

```bash
python -m train --config-name=config --config-path=train/conf +trainer.num_train_epochs=3
```

## Adding New Variants

1. Drop a new YAML into `train/conf/dataset/` or `train/conf/trainer/`.
2. Reference it via `python -m train dataset=<name>` (Hydra picks `dataset/<name>.yaml`).
3. If additional fields are required, extend the dataclasses in `train/config.py`.

This keeps experiments reproducible and explicit without editing Python code for every run.
