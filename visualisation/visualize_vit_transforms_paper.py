"""
Visualize ViT training transformations - Academic Paper Version
Creates a clean, publication-ready figure showing the data augmentation pipeline.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image

# Set matplotlib backend and style
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11


def load_config():
    """Load ViT configuration."""
    config_path = Path(__file__).parent.parent / "config" / "vit.yaml"
    return OmegaConf.load(config_path)


def get_sample_image():
    """Load a sample image from flowers102 dataset."""
    dataset = load_dataset("pufanyi/flowers102", split="train")
    sample = dataset[100]  # Use a different sample
    return sample["image"], sample["label"]


def denormalize_tensor(tensor, mean, std):
    """Denormalize a tensor for visualization."""
    if tensor.dim() == 3:
        tensor_copy = tensor.clone()
        for t, m, s in zip(tensor_copy, mean, std):
            t.mul_(s).add_(m)
        return tensor_copy
    return tensor


def tensor_to_image(tensor, mean=None, std=None):
    """Convert a tensor to a displayable image."""
    if mean is not None and std is not None:
        tensor = denormalize_tensor(tensor, mean, std)
    
    tensor = torch.clamp(tensor, 0, 1)
    img = tensor.numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    return img


def create_pipeline_transforms(config):
    """
    Create the complete transformation pipeline with key steps highlighted.
    Returns selected steps for visualization.
    """
    image_size = 224
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # Define the key transformation steps to visualize
    steps = [
        ("Original", None, False, "Input image"),
        
        ("Random Crop", 
         transforms.RandomResizedCrop(
             size=image_size,
             scale=tuple(config.processor.train.scale),
             ratio=tuple(config.processor.train.ratio),
             interpolation=transforms.InterpolationMode.BICUBIC,
         ), False, f"scale∈{config.processor.train.scale}"),
        
        ("Horizontal Flip", 
         transforms.RandomHorizontalFlip(p=1.0), 
         False, f"p={config.processor.train.hflip_prob}"),
        
        ("RandAugment", 
         transforms.RandAugment(
             num_ops=config.processor.train.randaugment_n,
             magnitude=config.processor.train.randaugment_m,
             interpolation=transforms.InterpolationMode.BILINEAR,
         ), False, f"n={config.processor.train.randaugment_n}, m={config.processor.train.randaugment_m}"),
        
        ("Color Jitter", 
         transforms.ColorJitter(
             brightness=config.processor.train.brightness,
             contrast=config.processor.train.contrast,
             saturation=config.processor.train.saturation,
             hue=config.processor.train.hue,
         ), False, "brightness, contrast, saturation, hue"),
        
        ("Normalize", 
         transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=mean, std=std)
         ]), False, "μ=[0.5, 0.5, 0.5], σ=[0.5, 0.5, 0.5]"),
        
        ("Random Erasing", 
         transforms.RandomErasing(
             p=1.0,
             scale=tuple(config.processor.train.erasing_scale),
             ratio=tuple(config.processor.train.erasing_ratio),
             value="random",
         ), True, f"p={config.processor.train.erasing_prob}"),
    ]
    
    return steps, mean, std


def add_arrow(ax, start_pos, end_pos):
    """Add an arrow between subplots."""
    ax.annotate('', xy=end_pos, xytext=start_pos,
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))


def visualize_transforms_paper(output_path="visualisation/imgs/sample.png"):
    """Create a publication-ready visualization."""
    print("Loading configuration...")
    config = load_config()
    
    print("Loading sample image...")
    original_image, label = get_sample_image()
    
    print("Creating transformation pipeline...")
    transform_steps, mean, std = create_pipeline_transforms(config)
    
    # Create figure with custom layout: Original on top center, 2x3 grid for transformations below
    fig = plt.figure(figsize=(13, 7.5))
    
    # Create grid: 3 rows x 3 columns
    # Row 0: Original (centered)
    # Rows 1-2: 2x3 grid of transformations
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.15,
                         left=0.05, right=0.95, top=0.94, bottom=0.02,
                         height_ratios=[1.1, 1, 1])
    
    # Set white background
    fig.patch.set_facecolor('white')
    
    current_image = original_image
    current_tensor = None
    
    # Process all steps
    for idx, (name, transform, needs_tensor, params) in enumerate(transform_steps):
        print(f"Processing step {idx + 1}/{len(transform_steps)}: {name}...")
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + idx)
        np.random.seed(42 + idx)
        
        # Create subplot with custom positioning
        if idx == 0:
            # Original image - centered in top row, spanning all 3 columns
            ax = fig.add_subplot(gs[0, 1])
        else:
            # Transformation steps - arranged in 2x3 grid (rows 1-2)
            transform_idx = idx - 1  # 0-5 for the 6 transformations
            row = 1 + (transform_idx // 3)  # Row 1 or 2
            col = transform_idx % 3  # Column 0, 1, or 2
            ax = fig.add_subplot(gs[row, col])
        
        try:
            if idx == 0:
                # Original image
                display_image = np.array(original_image)
            else:
                if needs_tensor:
                    if current_tensor is None:
                        raise ValueError("Expected tensor input")
                    result = transform(current_tensor)
                    current_tensor = result
                    display_image = tensor_to_image(result, mean, std)
                else:
                    if isinstance(current_image, np.ndarray):
                        current_image = Image.fromarray((current_image * 255).astype(np.uint8))
                    
                    result = transform(current_image)
                    
                    if isinstance(result, torch.Tensor):
                        current_tensor = result
                        display_image = tensor_to_image(result, mean, std)
                    else:
                        current_image = result
                        display_image = np.array(result)
            
            # Display image
            ax.imshow(display_image)
            
            # Different styling for original vs transformations
            if idx == 0:
                ax.set_title(f"{name}\n{params}", fontsize=11, pad=6, fontweight='bold')
                ax.text(0.05, 0.95, f"Step {idx}", transform=ax.transAxes,
                       fontsize=9, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax.set_title(f"{name}\n{params}", fontsize=9.5, pad=5)
                ax.text(0.05, 0.95, f"Step {idx}", transform=ax.transAxes,
                       fontsize=8.5, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.axis('off')
            
        except Exception as e:
            print(f"Error in step {idx}: {e}")
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}", 
                   ha='center', va='center', fontsize=9)
            ax.axis('off')
    
    # Add main title
    fig.suptitle('Vision Transformer (ViT) Training Data Augmentation Pipeline', 
                 fontsize=15, fontweight='bold')
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✓ Publication-ready visualization saved!")
    
    plt.close()


def main():
    """Main function."""
    # Generate both versions
    print("=" * 60)
    print("Generating publication-ready visualization...")
    print("=" * 60)
    visualize_transforms_paper("visualisation/imgs/vit_transforms_paper.png")
    
    print("\n" + "=" * 60)
    print("Generating detailed version...")
    print("=" * 60)
    # Import the original function
    from visualize_vit_transforms import visualize_transforms
    visualize_transforms("visualisation/imgs/vit_transforms_detailed.png")
    
    print("\n✓ All visualizations complete!")
    print(f"  - Paper version: visualisation/imgs/vit_transforms_paper.png")
    print(f"  - Detailed version: visualisation/imgs/vit_transforms_detailed.png")


if __name__ == "__main__":
    main()

