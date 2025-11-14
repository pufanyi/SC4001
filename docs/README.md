# Qwen3-VL Flowers Training Guide

This project fine-tunes the multimodal **Qwen3-VL** family on the **Oxford 102 Flowers** dataset. The repository is intentionally organised around two top-level packages:

- `data/` – utilities for preparing and uploading datasets (e.g. converting the classic Oxford Flowers corpus into a modern Hugging Face Hub dataset).
- `train/` – Hydra-driven training entry point that composes configs, preprocessing, model loading, and Hugging Face `Trainer` orchestration.

The following sections walk you through environment setup, dataset preparation, configuration management, and training execution.

---

## 1. Prerequisites

- Python 3.10+ (tested on 3.13)
- GPU with ≥24 GB VRAM recommended for full fine-tuning (adjust hyperparameters otherwise)
- Hugging Face account and write token (`huggingface-cli login`)
- [uv](https://docs.astral.sh/uv/) for dependency + virtual environment management:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync
  ```
  All subsequent commands assume `uv run …` to ensure they execute inside the synced environment.

---

## 2. Prepare the Dataset

The Oxford Flowers data is pulled via `torchvision.datasets.Flowers102`, then re-packed as a Hugging Face dataset with custom train/validation/test splits.

```bash
uv run python -m data.upload_flowers102 \
  --repo-id pufanyi/flowers102 \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --data-root ~/.cache/flowers102
```

Key flags:

- `--repo-id`: target dataset repo (`username/dataset-name`). Defaults to `pufanyi/flowers102`.
- `--token`: optional; otherwise the script uses `HUGGINGFACE_HUB_TOKEN` or cached credentials from `huggingface-cli login`.
- `--train-ratio`, `--val-ratio`, `--test-ratio`: any positive numbers; they are normalised automatically.
- `--branch`: push to a non-default branch if required.
- `--private`: create/update the Hub repo as private.

After the script finishes, verify the dataset exists at `https://huggingface.co/datasets/<repo-id>`.

---

## 3. Configuration Layout (Hydra)

Hydra keeps configuration modular in `train/conf/`:

- `config.yaml`: root experiment defaults (`seed`, Hydra runtime options, defaults list).
- `dataset/flowers.yaml`: dataset ID, instruction prompt, optional sample limits.
- `model/qwen3_vl.yaml`: base checkpoint and gradient checkpointing toggle.
- `trainer/default.yaml`: training hyperparameters and precision flags.

Override values inline when launching:

```bash
uv run python -m train trainer.output_dir=runs/qwen3-fp16 trainer.precision.fp16=true dataset.max_train_samples=512
```

Hydra writes outputs in-place (`hydra.run.dir=.`) so checkpoints stay under the specified `output_dir`.

---

## 4. Run Training

To fine-tune `Qwen/Qwen3-VL` on your prepared dataset:

```bash
uv run python -m train
```

Common overrides:

- Change base model: `model.model_id=Qwen/Qwen3-VL-4B-Instruct`
- Enable gradient checkpointing: `model.gradient_checkpointing=true`
- Switch to bfloat16: `trainer.precision.bf16=true`
- Reduce dataset size for smoke tests: `dataset.max_train_samples=256 dataset.max_eval_samples=128`

Trainer metrics and checkpoints are stored under `trainer.output_dir`. The script also saves the processor (`AutoProcessor`) alongside the model weights.

> Tip: use `uv run accelerate launch ... python -m train ...` if you need multi-GPU support via Hugging Face Accelerate.

---

## 5. Directory Overview

```
data/
  upload_flowers102.py     # Hugging Face dataset creation script
train/
  main.py                  # Hydra entry point
  config.py                # Structured dataclasses for hydra configs
  data.py                  # Dataset loading + preprocessing + collator
  modeling.py              # Qwen3-VL model/processor loading
  training.py              # Trainer + TrainingArguments helpers
  conf/
    config.yaml            # Hydra base config
    dataset/flowers.yaml   # Dataset settings
    model/qwen3_vl.yaml    # Model defaults
    trainer/default.yaml   # Training hyperparameters
```

---

## 6. Extending the Pipeline

The modular layout simplifies future experiments:

- **Alternate datasets** – add new YAMLs under `train/conf/dataset/` and adjust `data.py` to branch on `config.dataset`.
- **Chain-of-thought supervision** – extend `train/data.py` to include additional conversational turns, or point `dataset_id` toward a CoT-annotated dataset on the Hub.
- **Custom training loop** – reuse `prepare_datasets`, `load_model`, and `QwenVLDataCollator` inside bespoke training scripts if you need specialised objectives.

Pull requests and external configs can drop into the `conf/` hierarchy without touching code.

---

## 7. Troubleshooting

- **Missing token**: ensure `huggingface-cli login` has been run or provide `--token` explicitly.
- **Import errors (`torchvision`, `scipy`)**: install the prerequisites listed in §1.
- **Out-of-memory**: lower `per_device_train_batch_size`, increase `gradient_accumulation_steps`, or enable `model.gradient_checkpointing=true`.
- **Dataset upload fails with 409**: confirm you have permissions to create/update the target repo and that the name is unique.

---

Questions or improvements? Open an issue or tweak the configs—Hydra makes experimentation cheap.
