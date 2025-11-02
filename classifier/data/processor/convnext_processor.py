from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from .processor import Processor


class ConvNeXtProcessor(Processor):
    """
    Processor for ConvNeXt models.

    Uses standard ImageNet preprocessing:
    - Resize to 256
    - Center crop to 224
    - Convert to tensor
    - Normalize with ImageNet statistics
    """

    def __init__(
        self,
        image_size: int = 224,
        resize_size: int = 256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            image_size: Target size for center crop
            resize_size: Size to resize the shorter edge to
            mean: Normalization mean (RGB order)
            std: Normalization std (RGB order)
        """
        super().__init__(name="convnext_processor")

        self.image_size = image_size
        self.resize_size = resize_size
        self.mean = mean
        self.std = std

        # Build transform pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Process a single image.

        Args:
            image: PIL Image in RGB format

        Returns:
            tensor: Preprocessed image tensor of shape (C, H, W)
        """
        return self.transform(image)

    def get_config(self) -> dict[str, Any]:
        """
        Get processor configuration.

        Returns:
            config: Dictionary containing processor parameters
        """
        return {
            "name": self.name,
            "image_size": self.image_size,
            "resize_size": self.resize_size,
            "mean": self.mean,
            "std": self.std,
        }


class ConvNeXtTrainProcessor(Processor):
    """
    Processor for ConvNeXt models with training-time augmentation.

    Adds data augmentation:
    - Random resized crop
    - Random horizontal flip
    - (Optional) color jitter, rotation, etc.
    """

    def __init__(
        self,
        image_size: int = 224,
        scale: tuple[float, float] = (0.8, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        hflip_prob: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        color_jitter: bool = False,
    ):
        """
        Args:
            image_size: Target size for random crop
            scale: Range of size of the origin size cropped
            ratio: Range of aspect ratio of the origin aspect ratio cropped
            hflip_prob: Probability of horizontal flip
            mean: Normalization mean (RGB order)
            std: Normalization std (RGB order)
            color_jitter: Whether to apply color jitter
        """
        super().__init__(name="convnext_train_processor")

        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std
        self.color_jitter = color_jitter

        # Build augmentation pipeline
        transform_list = [
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=ratio,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=hflip_prob),
        ]

        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                )
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.transform = transforms.Compose(transform_list)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Process a single image with augmentation.

        Args:
            image: PIL Image in RGB format

        Returns:
            tensor: Augmented and preprocessed image tensor of shape (C, H, W)
        """
        return self.transform(image)

    def get_config(self) -> dict[str, Any]:
        """
        Get processor configuration.

        Returns:
            config: Dictionary containing processor parameters
        """
        return {
            "name": self.name,
            "image_size": self.image_size,
            "scale": self.scale,
            "ratio": self.ratio,
            "hflip_prob": self.hflip_prob,
            "mean": self.mean,
            "std": self.std,
            "color_jitter": self.color_jitter,
        }
