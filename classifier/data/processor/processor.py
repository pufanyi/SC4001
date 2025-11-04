from abc import ABC, abstractmethod

from PIL import Image


class Processor(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, images: list[Image.Image]) -> dict:
        """
        Process a list of images.

        Args:
            images: List of PIL Images in RGB format

        Returns:
            tensor: Preprocessed images tensor of shape (B, C, H, W)
        """
        raise NotImplementedError("Subclasses must implement this method")
