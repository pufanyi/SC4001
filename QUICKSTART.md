# Quick Start Guide

## 📋 Prerequisites

```bash
# Installed dependencies (see pyproject.toml)
pip install torch torchvision datasets hydra-core loguru rich
```

## 🚀 5-Minute Quick Start

### 1. View Example Code

```bash
# Run the example script to understand how each component works
python -m classifier.example_usage
```

This will demonstrate:
- ✅ How the Processor handles a single image
- ✅ How the Collator batches data
- ✅ How the Model performs inference
- ✅ How the components work together

### 2. Start Training

#### Training with Default Configuration

```bash
python -m classifier.train
```

This will use the default configuration:
- Dataset: Flowers102
- Model: ConvNeXt Tiny
- Batch size: 32
- Epochs: 100

#### Quick Debugging (with a small amount of data)

```bash
python -m classifier.train \
    dataset.max_train_samples=100 \
    dataset.max_val_samples=50 \
    trainer.epochs=3 \
    trainer.log_interval=1
```

#### Custom Configuration

```bash
# Use a larger model
python -m classifier.train model=convnext_small

# Use the ViT Base preset (the script will automatically activate .venv)
./scripts/train/vit.sh \
    trainer.train_batch_size=32 \
    trainer.learning_rate=3e-5

# Adjust hyperparameters
python -m classifier.train \
    trainer.batch_size=64 \
    trainer.optimizer.lr=5e-4 \
    trainer.epochs=50

# Modify data augmentation
python -m classifier.train \
    model.train_processor.hflip_prob=0.8 \
    model.train_processor.color_jitter=false
```

### 3. View Results

Training outputs are saved in `outputs/YYYY-MM-DD/HH-MM-SS/`:

```
outputs/2024-01-01/12-00-00/
├── config.yaml          # Full configuration
├── best_model.pth      # Best model
└── last_model.pth      # Last model
```

Load a model:

```python
import torch
from classifier.models import ConvNeXtModel

# Create the model
model = ConvNeXtModel("convnext_tiny", num_classes=102)

# Load weights
checkpoint = torch.load("outputs/.../best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 4. Run Evaluation

```bash
# Single-GPU evaluation of multiple checkpoints (supports any number, use -- to separate extra Hydra parameters)
./scripts/eval/single.sh \
    outputs/resnet152_lr1e-5/step_300 \
    outputs/resnet152_lr1e-5/step_400 \
    -- evaluation.split=test evaluation.metrics_output_path=outputs/test_metrics.json

# Distributed evaluation (FSDP checkpoints require the same world size as training)
NPROC_PER_NODE=4 ./scripts/eval/distributed.sh \
    outputs/resnet152_lr1e-5/step_500 \
    outputs/resnet152_lr1e-5/step_450 \
    -- evaluation.topk=[1,3,5]

# You can still call the Hydra entry point directly (for non-scripted scenarios)
python -m eval.pipeline.run evaluation.checkpoint_path=outputs/resnet152_lr1e-5/step_500
```

- `evaluation.split`: `train` / `validation` / `test` or any HF split; defaults to `validation`.
- `evaluation.max_samples`: Limits the number of samples for quick checks.
- `evaluation.metrics_output_path`: Provide a JSON path to automatically write out metrics.
- The `CHECKPOINT_FORMAT` environment variable controls how the script loads checkpoints (defaults to `auto` for single-GPU, `fsdp` for multi-GPU).
- When using FSDP sharded weights, the `NPROC_PER_NODE` for `torchrun` evaluation must match the training.

## 📚 Configuration Details

### Dataset Configuration (`dataset=...`)

Defined in `classifier/conf/dataset/`:

```yaml
# flowers102.yaml
dataset_id: pufanyi/flowers102
num_classes: 102
max_train_samples: null  # Limit the number of training samples (for debugging)
```

### Model Configuration (`model=...`)

Defined in `classifier/conf/model/`:

```yaml
# convnext_tiny.yaml
name: convnext_tiny
pretrained: true
dropout: 0.1

processor:
  image_size: 224
  resize_size: 256

train_processor:
  image_size: 224
  hflip_prob: 0.5
  color_jitter: true
```

### Trainer Configuration (`trainer.*`)

Defined in `classifier/conf/trainer/default.yaml`:

```yaml
epochs: 100
batch_size: 32
num_workers: 4

optimizer:
  name: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4

scheduler:
  name: cosine
  warmup_epochs: 5
```

## 🎯 Common Tasks

### Task 1: Add a New Dataset

1. Create a configuration file `classifier/conf/dataset/my_dataset.yaml`:

```yaml
name: my_dataset
dataset_id: username/my_dataset  # HuggingFace dataset
num_classes: 10

train_split: train
val_split: validation
test_split: test

image_column: image
label_column: label
```

2. Run training:

```bash
python -m classifier.train dataset=my_dataset
```

### Task 2: Adjust Data Augmentation

```bash
# More aggressive augmentation
python -m classifier.train \
    model.train_processor.scale=[0.5,1.0] \
    model.train_processor.hflip_prob=0.8 \
    model.train_processor.color_jitter=true

# Disable augmentation
python -m classifier.train \
    model.train_processor.hflip_prob=0.0 \
    model.train_processor.color_jitter=false
```

### Task 3: Modify Optimizer Settings

```bash
# Use a larger learning rate
python -m classifier.train trainer.optimizer.lr=1e-3

# Modify the learning rate scheduler
python -m classifier.train \
    trainer.scheduler.warmup_epochs=10 \
    trainer.scheduler.min_lr=1e-7
```

### Task 4: Use a Different Model Size

```bash
# ConvNeXt Small
python -m classifier.train model=convnext_small

# ConvNeXt Base (requires more GPU memory)
python -m classifier.train model=convnext_base trainer.batch_size=16
```

### Task 5: Resume Training

```python
# In a Python script
checkpoint = torch.load("outputs/.../checkpoint_epoch_50.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
best_acc = checkpoint['best_acc']

# Then continue training...
```

## 🔧 Development and Debugging

### Quick Pipeline Test

```bash
# Quick test with 10 samples
python -m classifier.train \
    dataset.max_train_samples=10 \
    dataset.max_val_samples=5 \
    trainer.epochs=2 \
    trainer.batch_size=2 \
    trainer.num_workers=0
```

### View Configuration

```bash
# Hydra will print the full, merged configuration
python -m classifier.train --cfg job
```

### Override Output Directory

```bash
python -m classifier.train output_dir=my_experiment
```

## 📖 More Resources

- **Detailed Documentation**: `classifier/README.md`
- **Design Document**: `CLASSIFIER_DESIGN.md`
- **Example Code**: `classifier/example_usage.py`
- **Configuration Files**: `classifier/conf/`

## ❓ Frequently Asked Questions

### Q: CUDA out of memory

```bash
# Reduce the batch size
python -m classifier.train trainer.batch_size=16

# Or use gradient accumulation
python -m classifier.train trainer.batch_size=8 trainer.gradient_accumulation_steps=4
```

### Q: Data loading is too slow

```bash
# Increase the number of workers
python -m classifier.train trainer.num_workers=8
```

### Q: How to use my own images?

```python
from classifier.data import ImageClassificationDataset
from classifier.data.convnext_processor import ConvNeXtProcessor

processor = ConvNeXtProcessor()
dataset = ImageClassificationDataset(
    image_paths=["path/to/img1.jpg", "path/to/img2.jpg", ...],
    labels=[0, 1, 2, ...],
    processor=processor,
)
```

## 🎉 Start Exploring

Now you're all set! Start training your model:

```bash
python -m classifier.train
```

Happy training! 🚀
