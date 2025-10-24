# Dataset Preparation Reference

The Oxford 102 Flowers dataset is distributed as MATLAB archives through Oxford's VGG group. For reproducible training we materialise it as a Hugging Face dataset with the expected train/validation/test splits.

## Script Overview

`data/upload_flowers102.py` performs the following steps:

1. Downloads the raw data using `torchvision.datasets.Flowers102` (requires `torchvision` and `scipy`).
2. Converts torchvision's internal files into a `datasets.DatasetDict` with image paths and integer labels.
3. Shuffles and re-splits the corpus according to the requested ratios (default 80/10/10).
4. Pushes the resulting dataset to the Hugging Face Hub, updating/creating the target repo and README card.

## Usage

```bash
python -m data.upload_flowers102 \
  --repo-id <username>/flowers102 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --data-root ~/.cache/flowers102
```

Important arguments:

- `--repo-id`: destination dataset repository on the Hub. Defaults to `pufanyi/flowers102`.
- `--token`: optional; if omitted the script reads `HUGGINGFACE_HUB_TOKEN` or cached credentials.
- `--private`: push the dataset as private.
- `--branch`: push to a specific git revision.
- `--max-shard-size`: control dataset shard size during upload (useful when tweaking Hub storage).

The script writes a README with the chosen split ratios and references the torchvision source (`pytorch/oxford-flowers`).

## Tips

- Delete `~/.cache/flowers102` if you want to force a re-download (careful with bandwidth).
- To verify the dataset locally without pushing, omit `split_dataset.push_to_hub` and inspect `DatasetDict` in a Python shell.
- Consider pinning the dataset version in Hydra configs with `dataset.dataset_id=<repo-id>@<revision>` once published.
