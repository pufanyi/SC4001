from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import PreTrainedModel

from .muon import Muon


def get_optimizer(config: DictConfig, model: PreTrainedModel):
    optimizer_cfg = config.trainer.optimizer
    if optimizer_cfg.name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=optimizer_cfg.weight_decay,
        )
    elif optimizer_cfg.name == "muon":
        return Muon(
            model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=optimizer_cfg.weight_decay,
            momentum=optimizer_cfg.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_cfg.name}")
