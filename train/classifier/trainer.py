import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedModel

from classifier.data.dataset import HFDataset
from train.optimizer.get_lr_scheduler import get_lr_scheduler
from train.optimizer.get_optimizer import get_optimizer


class ClassifierTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_dataset: HFDataset,
        val_dataset: HFDataset,
        config: DictConfig,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

    def create_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(self.config, self.model)

    def create_lr_scheduler(self) -> LambdaLR:
        return get_lr_scheduler(self.config, self.optimizer)

    def get_cast_type(self) -> torch.dtype:
        if self.config.trainer.cast_type == "bf16":
            return torch.bfloat16
        elif self.config.trainer.cast_type == "fp16":
            return torch.float16
        elif self.config.trainer.cast_type == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unknown cast type: {self.config.trainer.cast_type}")

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cast_type = self.get_cast_type()
        with torch.autocast(device_type="cuda", dtype=cast_type):
            outputs = self.model(**batch)
            loss = outputs["loss"]
        return loss

    def train(self):
        optimizer = self.create_optimizer()
        lr_scheduler = self.create_lr_scheduler()
