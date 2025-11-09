# Classifier Training Framework Design Document

## Design Overview

This framework adopts the **separation of responsibilities** design principle, clearly assigning different responsibilities in the training process to different components.

## Core Design Principles

### 1. Single Responsibility Principle

Each component is responsible for only one thing:

```
┌─────────────┐
│  Processor  │  Responsibility: Preprocessing a single sample (transform)
└─────────────┘  ❌ Not responsible for: batching, model inference

┌─────────────┐
│  Collator   │  Responsibility: Batching
└─────────────┘  ❌ Not responsible for: single sample preprocessing, model inference

┌─────────────┐
│    Model    │  Responsibility: Forward pass
└─────────────┘  ❌ Not responsible for: data preprocessing, batching

┌─────────────┐
│   Dataset   │  Responsibility: Data indexing and loading
└─────────────┘  ❌ Not responsible for: batching (done by DataLoader + Collator)
```

### 2. Conformance with the PyTorch Ecosystem

Follows the standard PyTorch pattern:

```python
# Standard PyTorch training loop
for batch in dataloader:  # DataLoader uses collate_fn for batching
    images = batch['pixel_values']  # Already a batched tensor
    outputs = model(images)  # forward accepts a tensor, not a list
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## Component Detailed Design

### 1. Processor

**Location**: `classifier/data/processor.py`

**Responsibility**:
- Preprocessing a single sample
- Image resize, crop, normalize
- Convert to tensor

**Not responsible for**:
- Batching
- Data loading

**Interface Design**:
```python
class Processor(ABC):
    @abstractmethod
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Args:
            image: PIL Image (C, H, W)
        Returns:
            tensor: Preprocessed tensor (C, H, W)
        """
        pass
```

**Implementation Example**:
```python
class ConvNeXtProcessor(Processor):
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)
```

**Design Rationale**:
- ✅ Can preprocess data without loading the model
- ✅ Supports multiprocessing in DataLoader (no need to serialize the model)
- ✅ Easy to test and reuse
- ✅ Can use different processors for training and validation (augmentation)

### 2. Collator

**Location**: `classifier/data/collator.py`

**Responsibility**:
- Combine multiple samples into a batch
- Stack or pad tensors

**Not responsible for**:
- Single sample preprocessing
- Model inference

**Interface Design**:
```python
class Collator(ABC):
    @abstractmethod
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: List of samples, each is a dict with:
                - 'pixel_values': torch.Tensor (C, H, W)
                - 'labels': int
        Returns:
            batched: Dict with:
                - 'pixel_values': torch.Tensor (B, C, H, W)
                - 'labels': torch.Tensor (B,)
        """
        pass
```

**Implementation Example**:
```python
class DefaultCollator(Collator):
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        return {'pixel_values': pixel_values, 'labels': labels}
```

**Design Rationale**:
- ✅ Works perfectly with PyTorch DataLoader's `collate_fn`
- ✅ Can handle images of different sizes (via padding)
- ✅ Easy to customize batching logic
- ✅ Does not affect the implementation of Dataset and Model

### 3. Model

**Location**: `classifier/models/model.py`

**Responsibility**:
- Only responsible for the forward pass

**Not responsible for**:
- Data preprocessing
- Batching
- Loss calculation
- Optimization

**Interface Design**:
```python
class Model(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batched tensor (B, C, H, W)
        Returns:
            logits: Tensor (B, num_classes)
        """
        pass
```

**Implementation Example**:
```python
class ConvNeXtModel(Model):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__(model_name, num_classes)
        self.backbone = models.convnext_tiny(pretrained=pretrained)
        # Replace the classifier head
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
```

**Design Rationale**:
- ✅ Conforms to the PyTorch nn.Module standard
- ✅ Can be used directly with DataLoader
- ✅ Easy to export (ONNX, TorchScript)
- ✅ Single responsibility, easy to test

### 4. Dataset

**Location**: `classifier/data/dataset.py`

**Responsibility**:
- Manage data indexing and loading
- Call the Processor to process samples

**Not responsible for**:
- Batching (done by DataLoader + Collator)

**Interface Design**:
```python
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        pixel_values = self.processor(image)
        return {
            'pixel_values': pixel_values,
            'labels': self.labels[idx]
        }
```

**Design Rationale**:
- ✅ Conforms to the PyTorch Dataset standard
- ✅ Can use different Processors (for training/validation)
- ✅ Supports multiprocessing in DataLoader
- ✅ Returns a consistent dict format

## Data Flow

```
1. Image Loading
   ↓
   Dataset.__getitem__(idx)
   │
   ├─ Load PIL Image
   └─ Call Processor
      ↓
2. Single Sample Processing
   ↓
   Processor.__call__(image)
   │
   ├─ Resize
   ├─ Crop
   ├─ ToTensor
   └─ Normalize
      ↓
   Returns: torch.Tensor (C, H, W)
   ↓
3. Batching (in DataLoader)
   ↓
   Collator.__call__(batch)
   │
   ├─ Stack pixel_values: (B, C, H, W)
   └─ Stack labels: (B,)
      ↓
4. Model Forward
   ↓
   Model.forward(pixel_values)
   │
   └─ Returns: logits (B, num_classes)
```

## Complete Usage Flow

```python
# 1. Create Processor
train_processor = ConvNeXtTrainProcessor(image_size=224, augment=True)
val_processor = ConvNeXtProcessor(image_size=224)

# 2. Create Dataset
train_dataset = HuggingFaceDataset(
    dataset_name="pufanyi/flowers102",
    split="train",
    processor=train_processor,  # Use training processor (with augmentation)
)
val_dataset = HuggingFaceDataset(
    dataset_name="pufanyi/flowers102",
    split="validation",
    processor=val_processor,  # Use validation processor (no augmentation)
)

# 3. Create Collator
collator = DefaultCollator()

# 4. Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator,  # Use collator for batching
)

# 5. Create Model
model = ConvNeXtModel(
    model_name="convnext_tiny",
    num_classes=102,
    pretrained=True,
)

# 6. Training Loop
for batch in train_loader:
    images = batch['pixel_values']  # (B, C, H, W) - already batched
    labels = batch['labels']        # (B,)

    outputs = model(images)         # forward accepts a tensor, not a list
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
```

## Comparison with Old Design

### Old Design ❌

```python
class Model(ABC, torch.nn.Module):
    @abstractmethod
    def process_image(self, image: Image.Image) -> torch.Tensor:
        pass  # Problem: Model should not be responsible for data preprocessing

    @abstractmethod
    def collate_images(self, images: list[torch.Tensor]) -> torch.Tensor:
        pass  # Problem: Model should not be responsible for batching

    @abstractmethod
    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        pass  # Problem: forward should accept a tensor, not a list
```

**Problems**:
1. **Mixed Responsibilities**: Model is responsible for preprocessing, batching, and inference.
2. **Doesn't Follow PyTorch Pattern**: `forward` accepts a list instead of a tensor.
3. **Difficult to Integrate with DataLoader**: `collate_images` duplicates DataLoader's `collate_fn`.
4. **Multiprocessing Issues**: DataLoader workers need to serialize the model, which includes preprocessing logic.
5. **Difficult to Extend**: Adding a new model requires implementing all methods, even if the preprocessing logic is the same.

### New Design ✅

```python
class Processor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        pass  # ✅ Only responsible for single sample preprocessing

class Collator:
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        pass  # ✅ Only responsible for batching

class Model(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # ✅ Only responsible for the forward pass
```

**Advantages**:
1. ✅ **Clear Responsibilities**: Each component does one thing.
2. ✅ **Follows PyTorch Ecosystem**: Adheres to standard patterns.
3. ✅ **Easy to Extend**: Independently add new models, processors, collators.
4. ✅ **Supports Multiprocessing**: Processor does not depend on the model.
5. ✅ **Easy to Test**: Components can be tested independently.
6. ✅ **Flexibility**: Can arbitrarily combine different components.

## Configuration Management (Hydra)

Use Hydra to manage configurations, supporting modularity and command-line overrides:

```yaml
# config.yaml
defaults:
  - dataset: flowers102
  - model: convnext_tiny
  - trainer: default

# dataset/flowers102.yaml
dataset_id: pufanyi/flowers102
num_classes: 102

# model/convnext_tiny.yaml
name: convnext_tiny
pretrained: true
processor:
  image_size: 224

# trainer/default.yaml
epochs: 100
batch_size: 32
optimizer:
  lr: 1e-4
```

Command-line override:
```bash
python -m classifier.train model=convnext_small trainer.batch_size=64
```

## Extension Guide

### Adding a New Model

1. Implement the Model:
```python
class YourModel(Model):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
```

2. Implement the Processor:
```python
class YourProcessor(Processor):
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)
```

3. Create a configuration file:
```yaml
# model/your_model.yaml
name: your_model
pretrained: true
processor:
  image_size: 224
```

4. Update `create_model()` and `create_processors()` in the training script.

## Summary

This design follows these principles:

1. **Single Responsibility Principle**: Each component does one thing.
2. **Open/Closed Principle**: Open for extension, closed for modification.
3. **Dependency Inversion**: Depend on abstract interfaces, not concrete implementations.
4. **Conformance to Standards**: Follows best practices of the PyTorch ecosystem.

This makes the code:
- Easy to understand and maintain
- Easy to test
- Easy to extend
- Compliant with industry standards