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

    def train(self):
        pass
