"""Configuration dataclasses for Qwen3-VL flower fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import DEFAULT_INSTRUCTION


@dataclass
class DatasetConfig:
    dataset_id: str = "pufanyi/flowers102"
    instruction: str = DEFAULT_INSTRUCTION
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3-VL"
    gradient_checkpointing: bool = False


@dataclass
class PrecisionConfig:
    bf16: bool = False
    fp16: bool = False


@dataclass
class TrainerConfig:
    output_dir: str = "runs/qwen3-vl-flowers"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    push_to_hub: bool = False
    hub_model_id: str | None = None
    evaluation_strategy: str = "steps"
    report_to: str = "none"
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)


@dataclass
class ExperimentConfig:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
