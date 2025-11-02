from abc import ABC, abstractmethod

import torch


class Model(ABC, torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Batched tensor of shape (B, C, H, W)

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        raise NotImplementedError("Subclasses must implement this method")
