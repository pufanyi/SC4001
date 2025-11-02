import torch
from abc import ABC, abstractmethod
from PIL import Image
from typing import Any


class Processor(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Process a single image.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            tensor: Preprocessed image tensor of shape (C, H, W)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Get processor configuration for logging and reproducibility.
        
        Returns:
            config: Dictionary containing processor parameters
        """
        raise NotImplementedError("Subclasses must implement this method")

