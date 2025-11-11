#!/usr/bin/env python3
"""
Minimal evaluation script for Hugging Face image classification models.

Given a model (local path or Hub repo) and a dataset hosted on the Hugging Face Hub,
this script computes top-k accuracy metrics without relying on Hydra configs or
FSDP-specific logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier.data.collator.collator import DataCollator
from classifier.data.dataset.dataset import HFDataset
from classifier.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face model on a dataset split.")
    parser.add_argument(
        "--model",
        required=True,
        help="Local path or repo id of the model to evaluate.",
    )
    parser.add_argument(
        "--processor",
        default=None,
        help="Optional processor name/id. Defaults to the same as --model.",
    )
    parser.add_argument(
        "--dataset",
        default="pufanyi/flowers102",
        help="Dataset repo id on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to evaluate on (e.g. validation, test).",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column name that stores images inside the dataset.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column name that stores labels inside the dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to evaluate (useful for smoke tests).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[1, 5],
        help="List of top-k accuracies to compute.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g. cuda, cuda:1, cpu). Defaults to cuda if available else cpu.",
    )
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        help="Enable automatic mixed precision (only valid on CUDA).",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable automatic mixed precision.",
    )
    parser.set_defaults(use_amp=True)
    parser.add_argument(
        "--amp-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type to use inside autocast when AMP is enabled.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON file to dump metrics.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bar.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoModel/AutoProcessor (required for some repos).",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_topk(topk_values: Sequence[int]) -> list[int]:
    unique = sorted({int(k) for k in topk_values if int(k) > 0})
    return unique or [1]


def _build_dataloader(
    dataset_name: str,
    split: str,
    processor,
    image_column: str,
    label_column: str,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> DataLoader:
    dataset = HFDataset(
        dataset_name=dataset_name,
        split=split,
        processor=processor,
        transform=None,
        image_column=image_column,
        label_column=label_column,
    )
    if max_samples is not None:
        limit = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(limit)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollator(),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    topk_values: Sequence[int],
    use_amp: bool,
    amp_dtype: torch.dtype,
    disable_tqdm: bool,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    iterator = dataloader
    if not disable_tqdm:
        iterator = tqdm(iterator, desc="Evaluating", total=len(dataloader))

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    topk_correct = torch.zeros(len(topk_values), dtype=torch.float64)
    max_requested_k = max(topk_values)
    use_autocast = use_amp and device.type == "cuda"

    with torch.no_grad():
        for batch in iterator:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if use_autocast
                else nullcontext()
            )
            with context:
                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits
                loss = outputs.loss if outputs.loss is not None else F.cross_entropy(logits, labels)
            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()

            available_k = min(max_requested_k, logits.size(-1))
            topk_preds = logits.topk(k=available_k, dim=-1).indices
            matches = topk_preds.eq(labels.unsqueeze(1))
            for idx, k in enumerate(topk_values):
                effective_k = min(k, available_k)
                correct = matches[:, :effective_k].any(dim=1).sum().item()
                topk_correct[idx] += correct

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1) * 100,
    }
    for idx, k in enumerate(topk_values):
        metrics[f"accuracy@{k}"] = topk_correct[idx].item() / max(total_samples, 1) * 100
    return metrics


def _dump_metrics(metrics: dict[str, float], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    logger.info(f"Saved metrics to {path}")


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    topk_values = _normalize_topk(args.topk)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    processor_name = args.processor or args.model
    logger.info(f"Loading processor from {processor_name}")
    processor = AutoImageProcessor.from_pretrained(
        processor_name,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info(f"Loading model from {args.model}")
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    dataloader = _build_dataloader(
        dataset_name=args.dataset,
        split=args.split,
        processor=processor,
        image_column=args.image_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    logger.info(
        f"Evaluating {args.model} on {args.dataset}[{args.split}] ({len(dataloader.dataset) if hasattr(dataloader.dataset, '__len__') else -1} samples)...",
    )
    metrics = _evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        topk_values=topk_values,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
        disable_tqdm=args.disable_tqdm,
    )
    formatted = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    logger.info(f"Results - {formatted}")
    _dump_metrics(metrics, args.metrics_output)


if __name__ == "__main__":
    main()
