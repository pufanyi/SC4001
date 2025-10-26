"""Hydra entrypoint for fine-tuning Qwen3-VL on Oxford Flowers."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from .config import ExperimentConfig
from .data import QwenVLDataCollator, prepare_datasets
from .modeling import load_model, load_processor
from .training import build_trainer, build_training_arguments

LOGGER = logging.getLogger(__name__)


def _resolve_config(cfg: DictConfig) -> ExperimentConfig:
    base = OmegaConf.structured(ExperimentConfig())
    merged = OmegaConf.merge(base, cfg)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = _resolve_config(cfg)

    set_seed(config.seed)

    model = load_model(config.model)
    processor = load_processor(config.model.model_id)

    datasets = prepare_datasets(config.dataset, processor)
    data_collator = QwenVLDataCollator(processor=processor)

    training_args = build_training_arguments(config.trainer)
    trainer = build_trainer(
        training_args=training_args,
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    LOGGER.info("Saving model and processor to %s", config.trainer.output_dir)
    trainer.save_model(config.trainer.output_dir)
    processor.save_pretrained(config.trainer.output_dir)

    if config.trainer.push_to_hub:
        LOGGER.info("Pushing trained weights to the Hugging Face Hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
