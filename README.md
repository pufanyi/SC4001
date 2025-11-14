# SC4001 – Qwen3-VL Flowers Pipeline

Fine-tuning, evaluating, and visualising the Qwen3-VL family on the Oxford 102 Flowers dataset. The repo includes dataset tooling, Hydra-driven training, evaluation scripts, and publication-ready visualisations.

## Highlights
- **Dataset tooling** – convert and upload Flowers102 to the Hugging Face Hub.
- **Hydra configs** – reproducible training/eval overrides without editing code.
- **Evaluation scripts** – batch inference for Hub or local checkpoints.
- **Visualisations** – generate clean augmentation diagrams for papers or slides.

## Requirements & Setup
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync dependencies (creates a `.venv` managed by uv):
   ```bash
   uv sync
   ```
3. Run project commands through `uv run …` so they execute inside the managed environment.

## Prepare the Dataset
Upload the Oxford Flowers dataset to your Hugging Face namespace:

```bash
uv run python -m data.upload_flowers102 \
  --repo-id <username>/flowers102 \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --data-root ~/.cache/flowers102
```

See `docs/dataset.md` for all arguments and troubleshooting tips.

## Train Qwen3-VL
Launch training with Hydra overrides inline:

```bash
uv run python -m train \
  trainer.output_dir=outputs/qwen3-vl \
  trainer.precision.bf16=true \
  dataset.max_train_samples=512
```

Common overrides:
- Switch base model: `model.model_id=Qwen/Qwen3-VL-4B-Instruct`
- Enable grad checkpointing: `model.gradient_checkpointing=true`
- Smoke test: reduce `dataset.max_*` counts

Hydra writes checkpoints/processors under `trainer.output_dir`. Configuration details live in `docs/configuration.md`.

## Evaluate & Export
- **Batch evaluation** (e.g., flowers test split):
  ```bash
  uv run python eval/pipeline/run.py \
    --model outputs/convnextv2-huge-22k-384/step_200/hf_export \
    --dataset pufanyi/flowers102 \
    --split test \
    --batch-size 64 \
    --metrics-output results.json
  ```
- **Export to Hugging Face format** (for FSDP checkpoints):
  ```bash
  uv run python tools/export_fsdp2_to_hf.py \
    --checkpoint-dir outputs/convnextv2-huge-22k-384/step_200 \
    --repo-id pufanyi/SC4001-convnextv2-huge-22k-384-wsd-adamw
  ```

## Visualisations
Generate the redesigned augmentation diagram (PNG + PDF):

```bash
uv run python visualisation/visualize_vit_transforms_paper.py
```

Artifacts land in `visualisation/imgs/`.

## Repository Layout
```
data/              # Dataset upload helper
docs/              # Extended guides (dataset, Hydra, training walkthrough)
eval/              # Evaluation pipeline scripts
train/             # Hydra entry point + configs
visualisation/     # Matplotlib scripts used in the paper
tools/             # Export and utility scripts
```

## Documentation
- `docs/README.md` – full training guide with tips
- `docs/dataset.md` – dataset upload reference
- `docs/configuration.md` – Hydra configuration schema

Contributions: open issues/PRs with proposed configs or fixes. Use `uv run ruff check .` before submitting.
