import torch
from transformers import ConvNextV2ForImageClassification

from ...utils.constants import NUM_CLASSES
from ..model import Model


class ConvNeXtV2(Model):
    def __init__(
        self,
        model_name: str = "facebook/convnextv2-huge-22k-384",
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__("convnextv2", num_classes)
        self.model = ConvNextV2ForImageClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)
