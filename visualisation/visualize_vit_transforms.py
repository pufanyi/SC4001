"""
Visualize ViT training transformations step by step.
This script demonstrates the data augmentation pipeline used for ViT training.
"""

from pathlib import Path

# Set matplotlib backend to avoid display issues
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")


def load_config():
    """Load ViT configuration."""
    config_path = Path(__file__).parent.parent / "config" / "vit.yaml"
    return OmegaConf.load(config_path)


def get_sample_image():
    """Load a sample image from flowers102 dataset."""
    dataset = load_dataset("pufanyi/flowers102", split="train")
    # Get a nice looking sample
    sample = dataset[42]  # Use a fixed index for reproducibility
    return sample["image"], sample["label"]


def denormalize_tensor(tensor, mean, std):
    """Denormalize a tensor for visualization."""
    if tensor.dim() == 3:
        for t, m, s in zip(tensor, mean, std, strict=False):
            t.mul_(s).add_(m)
    return tensor


def tensor_to_image(tensor, mean=None, std=None):
    """Convert a tensor to a displayable image."""
    if mean is not None and std is not None:
        tensor = denormalize_tensor(tensor.clone(), mean, std)

    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy and transpose
    img = tensor.numpy()
    if img.shape[0] == 3:  # C, H, W -> H, W, C
        img = np.transpose(img, (1, 2, 0))

    return img


def create_transform_steps(config):
    """
    Create individual transformation steps based on config.
    Returns a list of (name, transform, needs_tensor) tuples.
    """
    image_size = 224  # Default ViT size
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    steps = []

    # Step 1: Random Resized Crop
    steps.append(
        (
            "1. RandomResizedCrop\n(scale=[0.5, 1.0], ratio=[0.75, 1.33])",
            transforms.RandomResizedCrop(
                size=image_size,
                scale=tuple(config.processor.train.scale),
                ratio=tuple(config.processor.train.ratio),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            False,
        )
    )

    # Step 2: Random Horizontal Flip
    if config.processor.train.hflip_prob > 0:
        steps.append(
            (
                f"2. RandomHorizontalFlip\n(p={config.processor.train.hflip_prob})",
                transforms.RandomHorizontalFlip(p=1.0),  # Force to show effect
                False,
            )
        )

    # Step 3: Random Rotation
    if config.processor.train.rotation_degrees > 0:
        steps.append(
            (
                f"3. RandomRotation\n(degrees=±{config.processor.train.rotation_degrees}°)",
                transforms.RandomRotation(
                    degrees=config.processor.train.rotation_degrees,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                False,
            )
        )

    # Step 4: RandAugment
    if config.processor.train.use_randaugment:
        steps.append(
            (
                f"4. RandAugment\n(n={config.processor.train.randaugment_n}, m={config.processor.train.randaugment_m})",
                transforms.RandAugment(
                    num_ops=config.processor.train.randaugment_n,
                    magnitude=config.processor.train.randaugment_m,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                False,
            )
        )

    # Step 5: Color Jitter
    if config.processor.train.use_color_jitter:
        steps.append(
            (
                f"5. ColorJitter\n(brightness={config.processor.train.brightness}, contrast={config.processor.train.contrast})",
                transforms.ColorJitter(
                    brightness=config.processor.train.brightness,
                    contrast=config.processor.train.contrast,
                    saturation=config.processor.train.saturation,
                    hue=config.processor.train.hue,
                ),
                False,
            )
        )

    # Step 6: Random Grayscale
    if config.processor.train.grayscale_prob > 0:
        steps.append(
            (
                f"6. RandomGrayscale\n(p={config.processor.train.grayscale_prob})",
                transforms.RandomGrayscale(p=1.0),  # Force to show effect
                False,
            )
        )

    # Step 7: Gaussian Blur
    if config.processor.train.blur_prob > 0:
        steps.append(
            (
                f"7. GaussianBlur\n(p={config.processor.train.blur_prob}, kernel=23)",
                transforms.GaussianBlur(
                    kernel_size=23, sigma=tuple(config.processor.train.blur_sigma)
                ),
                False,
            )
        )

    # Step 8: ToTensor + Normalize
    steps.append(
        (
            "8. ToTensor + Normalize\n(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])",
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            ),
            False,
        )
    )

    # Step 9: Random Erasing
    if config.processor.train.use_random_erasing:
        steps.append(
            (
                f"9. RandomErasing\n(p={config.processor.train.erasing_prob}, scale=[0.02, 0.2])",
                transforms.RandomErasing(
                    p=1.0,  # Force to show effect
                    scale=tuple(config.processor.train.erasing_scale),
                    ratio=tuple(config.processor.train.erasing_ratio),
                    value="random",
                ),
                True,  # Needs tensor input
            )
        )

    return steps, mean, std


def visualize_transforms(output_path="visualisation/imgs/sample.png"):
    """Create a visualization of all transformation steps."""
    print("Loading configuration...")
    config = load_config()

    print("Loading sample image...")
    original_image, label = get_sample_image()

    print("Creating transformation steps...")
    transform_steps, mean, std = create_transform_steps(config)

    # Determine grid layout
    num_steps = len(transform_steps) + 1  # +1 for original image
    cols = 3
    rows = (num_steps + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Set a nice style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig.patch.set_facecolor("white")

    # Show original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold", pad=10)
    axes[0].axis("off")

    # Apply transformations step by step
    current_image = original_image
    current_tensor = None

    for idx, (name, transform, needs_tensor) in enumerate(transform_steps, start=1):
        print(f"Applying step {idx}: {name.split(chr(10))[0]}...")

        # Set random seed for reproducibility
        torch.manual_seed(42 + idx)
        np.random.seed(42 + idx)

        try:
            if needs_tensor:
                # Apply to tensor
                if current_tensor is None:
                    raise ValueError("Expected tensor input but got PIL Image")
                result = transform(current_tensor)
                current_tensor = result
                # Convert tensor to image for display
                display_image = tensor_to_image(result, mean, std)
            else:
                # Apply to PIL Image
                if isinstance(current_image, np.ndarray):
                    current_image = Image.fromarray(
                        (current_image * 255).astype(np.uint8)
                    )

                result = transform(current_image)

                # Check if result is tensor (from ToTensor + Normalize)
                if isinstance(result, torch.Tensor):
                    current_tensor = result
                    display_image = tensor_to_image(result, mean, std)
                else:
                    current_image = result
                    display_image = np.array(result)

            # Display
            axes[idx].imshow(display_image)
            axes[idx].set_title(name, fontsize=12, fontweight="bold", pad=10)
            axes[idx].axis("off")

        except Exception as e:
            print(f"Error in step {idx}: {e}")
            axes[idx].text(
                0.5,
                0.5,
                f"Error:\n{str(e)[:50]}",
                ha="center",
                va="center",
                fontsize=10,
            )
            axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(num_steps, len(axes)):
        axes[idx].axis("off")

    # Add title
    fig.suptitle(
        "ViT Training Data Transformation Pipeline",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Visualization saved successfully!")

    plt.close()


def main():
    """Main function."""
    output_path = "visualisation/imgs/sample.png"
    visualize_transforms(output_path)


if __name__ == "__main__":
    main()
