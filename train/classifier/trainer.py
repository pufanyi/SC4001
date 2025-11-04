from omegaconf import DictConfig

from classifier.data.dataset import HFDataset
from classifier.models import Model


class ClassifierTrainer:
    def __init__(
        self,
        model: Model,
        train_dataset: HFDataset,
        val_dataset: HFDataset,
        config: DictConfig,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

    def create_optimizer(self):
        pass

    def train(self):
        pass
