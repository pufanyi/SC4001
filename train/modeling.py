"""Model and processor loading utilities."""

from __future__ import annotations

import logging

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .config import ModelConfig

LOGGER = logging.getLogger(__name__)


def load_model(config: ModelConfig) -> Qwen3VLForConditionalGeneration:
    LOGGER.info("Loading model %s", config.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(config.model_id)
    if config.gradient_checkpointing:
        LOGGER.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = False
    return model


def load_processor(model_id: str) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor
