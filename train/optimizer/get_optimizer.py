from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import PreTrainedModel
from .muon import Muon


def get_optimizer(config: DictConfig, model: PreTrainedModel):
    if config.trainer.optimizer.name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
        )
    elif config.trainer.optimizer.name == "muon":
        return Muon(
            model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
            momentum=config.trainer.momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.trainer.name}")
