from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import PreTrainedModel


def get_optimizer(config: DictConfig, model: PreTrainedModel):
    if config.trainer.optimizer.name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.trainer.name}")
