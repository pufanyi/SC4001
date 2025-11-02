import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Union
from datasets import load_dataset

from ..processor import Processor


class HFDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        processor: Processor,
        transform: Optional[callable] = None,
        image_column: str = "image",
        label_column: str = "label",
    ):
        """
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            split: Dataset split ('train', 'validation', 'test')
            processor: Processor instance for preprocessing images
            transform: Optional additional transforms
            image_column: Name of the column containing images
            label_column: Name of the column containing labels
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.processor = processor
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - 'pixel_values': torch.Tensor of shape (C, H, W)
                - 'labels': int
        """
        sample = self.dataset[idx]
        
        # Get image (already PIL Image from HF datasets)
        image = sample[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        # Apply additional transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Process image
        pixel_values = self.processor(image)
        
        # Get label
        label = sample[self.label_column]
        
        return {
            'pixel_values': pixel_values,
            'labels': label
        }

