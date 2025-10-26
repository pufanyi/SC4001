"""Trainer construction helpers."""

from __future__ import annotations

from transformers import Trainer, TrainingArguments

from .config import TrainerConfig


def build_training_arguments(config: TrainerConfig) -> TrainingArguments:
    precision = config.precision
    if precision.bf16 and precision.fp16:
        raise ValueError("Only one of bf16 or fp16 can be enabled.")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=config.dataloader_num_workers,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.save_steps,
        report_to=config.report_to,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        bf16=precision.bf16,
        fp16=precision.fp16,
    )

    return training_args


def build_trainer(
    *,
    training_args: TrainingArguments,
    model,
    train_dataset,
    eval_dataset,
    data_collator,
) -> Trainer:
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
