import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_cosine_schedule_with_warmup, get_wsd_schedule


def get_lr_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer) -> LambdaLR:
    num_warmup_steps = int(config.trainer.num_steps * config.trainer.lr_scheduler.num_warmup_rate)
    if config.trainer.lr_scheduler.name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=config.trainer.num_steps,
        )
    elif config.trainer.lr_scheduler.name == "wsd":
        return get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_decay_steps=config.trainer.lr_scheduler.num_decay_steps,
            num_training_steps=config.trainer.num_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.trainer.lr_scheduler.name}")
