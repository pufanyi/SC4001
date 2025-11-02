import torch
from abc import ABC, abstractmethod
from typing import Any


class Collator(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.
        
        Args:
            batch: List of samples, each sample is a dict with keys:
                - 'pixel_values': torch.Tensor of shape (C, H, W)
                - 'labels': int or torch.Tensor
                - (optional) other fields
        
        Returns:
            batched_data: Dictionary containing:
                - 'pixel_values': torch.Tensor of shape (B, C, H, W)
                - 'labels': torch.Tensor of shape (B,)
                - (optional) other batched fields
        """
        raise NotImplementedError("Subclasses must implement this method")


class DefaultCollator(Collator):
    def __init__(self):
        super().__init__(name="default")
    
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Stack images and labels into batched tensors.
        
        Args:
            batch: List of dicts with 'pixel_values' and 'labels'
            
        Returns:
            Dictionary with batched 'pixel_values' and 'labels'
        """
        # Stack all pixel_values: (B, C, H, W)
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        
        # Stack or collect labels: (B,)
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }

