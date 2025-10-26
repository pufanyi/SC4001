"""Dataset preparation utilities for Qwen3-VL flower fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Sequence

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

from .config import DatasetConfig
from .constants import FLOWER_CLASSES

LOGGER = logging.getLogger(__name__)


def _ensure_list_of_tensors(
    items: Sequence[torch.Tensor],
    padding_value: int,
) -> torch.Tensor:
    tensors = [torch.as_tensor(x) for x in items]
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


@dataclass
class QwenVLDataCollator:
    processor: AutoProcessor

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": _ensure_list_of_tensors(
                [f["input_ids"] for f in features],
                padding_value=self.processor.tokenizer.pad_token_id,
            ),
            "attention_mask": _ensure_list_of_tensors(
                [f["attention_mask"] for f in features],
                padding_value=0,
            ),
            "labels": _ensure_list_of_tensors(
                [f["labels"] for f in features],
                padding_value=-100,
            ),
            "pixel_values": torch.stack([torch.as_tensor(f["pixel_values"]) for f in features]),
            "image_grid_thw": torch.stack([torch.as_tensor(f["image_grid_thw"]) for f in features]),
        }

        if "pixel_values_videos" in features[0]:
            batch["pixel_values_videos"] = torch.stack(
                [torch.as_tensor(f["pixel_values_videos"]) for f in features]
            )
        if "video_grid_thw" in features[0]:
            batch["video_grid_thw"] = torch.stack(
                [torch.as_tensor(f["video_grid_thw"]) for f in features]
            )
        return batch


def build_messages(label_idx: int, instruction: str) -> list[Dict[str, Any]]:
    label_text = FLOWER_CLASSES[label_idx]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": label_text}],
        },
    ]


def preprocess_example(
    processor: AutoProcessor,
    instruction: str,
    example: Dict[str, Any],
) -> Dict[str, Any]:
    messages = build_messages(example["label"], instruction)

    prompt_text = processor.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    processed = processor(
        text=[full_text],
        images=[example["image"]],
        return_tensors="pt",
    )

    prompt_inputs = processor(
        text=[prompt_text],
        images=[example["image"]],
        return_tensors="pt",
    )

    input_ids = processed["input_ids"][0]
    attention_mask = processed["attention_mask"][0]
    labels = input_ids.clone()
    prompt_length = prompt_inputs["input_ids"].shape[-1]
    labels[:prompt_length] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": processed["pixel_values"][0],
        "image_grid_thw": processed["image_grid_thw"][0],
    }


def _maybe_select(ds: Dataset, max_samples: int | None) -> Dataset:
    if max_samples:
        return ds.select(range(min(len(ds), max_samples)))
    return ds


def prepare_datasets(
    config: DatasetConfig,
    processor: AutoProcessor,
) -> DatasetDict:
    LOGGER.info("Loading dataset %s", config.dataset_id)
    dataset = load_dataset(config.dataset_id)

    transform = partial(preprocess_example, processor, config.instruction)

    LOGGER.info("Preprocessing training split...")
    train_dataset = dataset["train"].map(
        transform,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train split",
    )
    train_dataset = _maybe_select(train_dataset, config.max_train_samples)

    LOGGER.info("Preprocessing validation split...")
    validation_dataset = dataset["validation"].map(
        transform,
        remove_columns=dataset["validation"].column_names,
        desc="Tokenizing validation split",
    )
    validation_dataset = _maybe_select(validation_dataset, config.max_eval_samples)

    return DatasetDict(train=train_dataset, validation=validation_dataset)
