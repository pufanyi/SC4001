from classifier.models import Model
from classifier.data.dataset import HFDataset
from omegaconf import DictConfig

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
