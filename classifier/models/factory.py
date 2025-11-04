from omegaconf import DictConfig
from transformers import AutoModelForImageClassification, PreTrainedModel

from ..utils.constants import ID2LABEL, LABEL2ID, NUM_CLASSES


class ModelFactory:
    @staticmethod
    def get_model(config: DictConfig) -> PreTrainedModel:
        model_name = config.model.name
        return AutoModelForImageClassification.from_pretrained(
            model_name,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            num_labels=NUM_CLASSES,
        )
