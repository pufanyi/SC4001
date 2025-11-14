# Visualization Scripts

This directory contains scripts for visualizing the ViT training data augmentation pipeline.

## Scripts

### 1. `visualize_vit_transforms.py`
Generates a detailed visualization of all ViT training transformations step by step.

**Usage:**
```bash
uv run visualisation/visualize_vit_transforms.py
```

**Output:** `imgs/sample.png` (detailed version with all augmentation steps)

**Features:**
- Shows all 9+ transformation steps
- Includes detailed parameters for each step
- Displays effects of each augmentation technique

### 2. `visualize_vit_transforms_paper.py`
Generates publication-ready visualizations suitable for academic papers.

**Usage:**
```bash
uv run visualisation/visualize_vit_transforms_paper.py
```

**Outputs:**
- `imgs/vit_transforms_paper.png` - Clean, academic-style visualization (7 key steps)
- `imgs/vit_transforms_detailed.png` - Detailed version with all steps

**Features:**
- Professional typography (Times New Roman font)
- Optimized layout for papers (2 rows × 4 columns)
- Clean, minimalist design
- High resolution (300 DPI)
- Step numbers and parameter annotations

## Generated Images

The scripts generate visualizations showing the complete data augmentation pipeline:

1. **Original Image** - Input from flowers102 dataset
2. **Random Crop** - Random resized crop with scale ∈ [0.5, 1.0]
3. **Horizontal Flip** - Random horizontal flip (p=0.5)
4. **Rotation** - Random rotation (±10°)
5. **RandAugment** - Automated augmentation (n=2, m=9)
6. **Color Jitter** - Brightness, contrast, saturation, hue adjustments
7. **Grayscale** - Random grayscale conversion (p=0.05)
8. **Gaussian Blur** - Random blur (p=0.1)
9. **Normalize** - Normalization with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
10. **Random Erasing** - Random region erasing (p=0.2)

## Configuration

The transformations are configured in `config/vit.yaml` under the `processor.train` section:

```yaml
processor:
  train:
    use_randaugment: true
    randaugment_n: 2
    randaugment_m: 9
    scale: [0.5, 1.0]
    ratio: [0.75, 1.33]
    hflip_prob: 0.5
    vflip_prob: 0.0
    rotation_degrees: 10
    use_color_jitter: true
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    grayscale_prob: 0.05
    blur_prob: 0.1
    blur_sigma: [0.1, 2.0]
    use_random_erasing: true
    erasing_prob: 0.2
    erasing_scale: [0.02, 0.2]
    erasing_ratio: [0.3, 3.3]
```

## Requirements

All dependencies are managed via `uv` and specified in `pyproject.toml`:
- matplotlib >= 3.10.7
- torch >= 2.9.0
- torchvision >= 0.24.0
- transformers >= 4.57.1
- datasets >= 4.3.0
- omegaconf (via hydra-core)

## Examples

### For Paper Publication
The paper version provides the cleanest output:
```bash
uv run visualisation/visualize_vit_transforms_paper.py
```

Use `imgs/vit_transforms_paper.png` in your LaTeX document:
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\textwidth]{visualisation/imgs/vit_transforms_paper.png}
  \caption{ViT training data augmentation pipeline showing key transformation steps.}
  \label{fig:vit-augmentation}
\end{figure}
```

### For Detailed Analysis
The detailed version shows all augmentations:
```bash
uv run visualisation/visualize_vit_transforms.py
```

## Notes

- All scripts use a fixed random seed (42) for reproducibility
- Images are sampled from the flowers102 dataset
- Transformations are applied sequentially, showing cumulative effects
- Some augmentations (like flip, grayscale) are forced to p=1.0 for visualization purposes
- The actual training pipeline may apply these transformations probabilistically

