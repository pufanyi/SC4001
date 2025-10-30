#!/usr/bin/env python3
"""Split the Oxford 102 Flowers dataset and upload it to the Hugging Face Hub.

Dependencies
------------
- datasets
- huggingface_hub
- torch
- torchvision
- scipy (required by torchvision.datasets.Flowers102)

Example
-------
    python -m data.upload_flowers102
    python -m data.upload_flowers102 --repo-id your-name/flowers102-split
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image,
    concatenate_datasets,
)
from huggingface_hub import HfApi, HfFolder


@dataclass(frozen=True)
class SplitRatios:
    """Container for normalized split ratios."""

    train: float
    validation: float
    test: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Oxford 102 Flowers dataset, create train/validation/test "
            "splits, and push the processed dataset to the Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--repo-id",
        default="pufanyi/flowers102",
        help=(
            "Destination dataset repo on the Hugging Face Hub. Default: "
            "pufanyi/flowers102."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Hugging Face token with write permissions. Defaults to "
            "HUGGINGFACE_HUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the target repository as private.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio before normalization. Default: 0.8.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio before normalization. Default: 0.1.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio before normalization. Default: 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits. Default: 42.",
    )
    parser.add_argument(
        "--data-root",
        default="~/.cache/flowers102",
        help="Directory to cache the raw Oxford 102 data (downloaded via torchvision).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help=(
            "Optional git branch on the Hub to push to. Defaults to the repo's "
            "main branch."
        ),
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help=(
            "Maximum size for dataset shards when uploading. Default mirrors "
            "datasets default."
        ),
    )
    parser.add_argument(
        "--commit-message",
        default="Add Oxford 102 Flowers custom split",
        help="Commit message used when pushing to the Hub.",
    )
    return parser.parse_args()


def normalize_ratios(train: float, val: float, test: float) -> SplitRatios:
    if any(r < 0 for r in (train, val, test)):
        raise ValueError("Split ratios must be non-negative.")

    total = train + val + test
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    train_norm = train / total
    val_norm = val / total
    test_norm = test / total

    if not 0.0 < train_norm < 1.0:
        raise ValueError("Train ratio must be between 0 and 1 once normalized.")
    if val_norm <= 0.0:
        raise ValueError("Validation ratio must be greater than zero once normalized.")
    if test_norm <= 0.0:
        raise ValueError("Test ratio must be greater than zero once normalized.")

    return SplitRatios(train=train_norm, validation=val_norm, test=test_norm)


def create_splits(dataset: DatasetDict, ratios: SplitRatios, seed: int) -> DatasetDict:
    combined = concatenate_datasets(list(dataset.values()))

    shuffled = combined.shuffle(seed=seed)

    first_split = shuffled.train_test_split(train_size=ratios.train, seed=seed)
    train_split = first_split["train"]
    holdout = first_split["test"]

    test_fraction = ratios.test / (ratios.validation + ratios.test)
    val_test = holdout.train_test_split(test_size=test_fraction, seed=seed)
    validation_split = val_test["train"]
    test_split = val_test["test"]

    train_split = train_split.cast_column("image", Image())
    validation_split = validation_split.cast_column("image", Image())
    test_split = test_split.cast_column("image", Image())

    return DatasetDict(train=train_split, validation=validation_split, test=test_split)


def ensure_repo(api: HfApi, repo_id: str, token: str | None, private: bool) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )


def build_dataset_card(ratios: SplitRatios, source_repo: str) -> str:
    return textwrap.dedent(
        f"""\
        ---
        pretty_name: Oxford 102 Flowers (Custom Split)
        task_categories:
        - image-classification
        task_ids:
        - image-classification
        license: cc-by-4.0
        source_datasets:
        - {source_repo}
        tags:
        - flowers
        - oxford-102
        - computer-vision
        - train-validation-test-split
        ---

        # Oxford 102 Flowers (Custom Split)

        This dataset re-packages the Oxford 102 Flowers dataset with a custom
        train/validation/test split produced by `data.upload_flowers102`.

        ## Split Ratios

        - Train: {ratios.train:.2%}
        - Validation: {ratios.validation:.2%}
        - Test: {ratios.test:.2%}

        ## Source

        Original images and annotations come from the Oxford 102 Flowers dataset,
        as distributed on the Hugging Face Hub under `{source_repo}`.
        """
    )


def push_readme(
    api: HfApi,
    repo_id: str,
    token: str | None,
    readme: str,
    branch: str | None,
    commit_message: str,
) -> None:
    api.upload_file(
        path_or_fileobj=BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        revision=branch,
        commit_message=commit_message,
    )


def load_flowers102_via_torchvision(data_root: Path) -> DatasetDict:
    try:
        from torchvision.datasets import Flowers102
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torchvision is required to download the Oxford 102 dataset. "
            "Install it via `pip install torch torchvision scipy`."
        ) from exc

    try:
        import scipy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "scipy is required by torchvision.datasets.Flowers102. Install it via "
            "`pip install scipy`."
        ) from exc

    data_root = data_root.expanduser()
    split_map = {"train": "train", "val": "validation", "test": "test"}

    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(num_classes=102, names=list(Flowers102.classes)),
        }
    )

    datasets_by_split = {}
    for tv_split, hf_split in split_map.items():
        flowers = Flowers102(root=str(data_root), split=tv_split, download=True)
        image_paths = [str(path) for path in flowers._image_files]
        labels = [int(label) for label in flowers._labels]
        datasets_by_split[hf_split] = Dataset.from_dict(
            {"image": image_paths, "label": labels},
            features=features,
        )
    return DatasetDict(datasets_by_split)


def main() -> None:
    args = parse_args()

    token = (
        args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or HfFolder.get_token()
    )
    if not token:
        print(
            "A Hugging Face token is required. Run `huggingface-cli login`, set "
            "HUGGINGFACE_HUB_TOKEN, or pass --token.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        ratios = normalize_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    except ValueError as exc:
        print(f"Invalid split ratios: {exc}", file=sys.stderr)
        sys.exit(1)

    data_root = Path(args.data_root)
    print(
        f"Downloading Oxford 102 Flowers dataset via torchvision into "
        f"{data_root.expanduser()}...",
        flush=True,
    )
    try:
        source_dataset = load_flowers102_via_torchvision(data_root)
    except RuntimeError as exc:
        print(f"Failed to prepare dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Creating train/validation/test splits...", flush=True)
    split_dataset = create_splits(source_dataset, ratios, seed=args.seed)

    api = HfApi(token=token)
    print(f"Ensuring repository {args.repo_id!r} exists on the Hub...", flush=True)
    ensure_repo(api, repo_id=args.repo_id, token=token, private=args.private)

    print("Uploading dataset shards to the Hub (this can take a while)...", flush=True)
    push_kwargs = {
        "repo_id": args.repo_id,
        "token": token,
        "private": args.private,
        "max_shard_size": args.max_shard_size,
        "commit_message": args.commit_message,
    }
    if args.branch:
        push_kwargs["revision"] = args.branch

    split_dataset.push_to_hub(**push_kwargs)

    print("Uploading dataset card...", flush=True)
    card = build_dataset_card(
        ratios=ratios,
        source_repo="pytorch/oxford-flowers",
    )
    push_readme(
        api=api,
        repo_id=args.repo_id,
        token=token,
        readme=card,
        branch=args.branch,
        commit_message="Update dataset card",
    )

    print(
        "Done! Dataset is available at https://huggingface.co/datasets/" + args.repo_id
    )


if __name__ == "__main__":
    main()
