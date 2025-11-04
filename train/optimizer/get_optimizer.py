from omegaconf import DictConfig
import torch
from transformers.optimization import get_wsd_schedule


def get_optimizer(config: DictConfig, model: torch.nn.Module):
    if config.optimizer.name == "wsd":
        return get_wsd_schedule(optimizer=optimizer, num_warmup_steps=config.optimizer.num_warmup_steps, num_training_steps=config.optimizer.num_training_steps)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")