# Classifier 训练框架设计文档

## 设计概述

本框架采用**职责分离**的设计原则，将训练流程中的不同职责清晰地分配到不同的组件中。

## 核心设计原则

### 1. 单一职责原则（Single Responsibility Principle）

每个组件只负责一件事情：

```
┌─────────────┐
│  Processor  │  职责：单个样本的预处理（transform）
└─────────────┘  ❌ 不负责：批处理、模型推理

┌─────────────┐
│  Collator   │  职责：批处理（batching）
└─────────────┘  ❌ 不负责：单样本预处理、模型推理

┌─────────────┐
│    Model    │  职责：前向传播（forward pass）
└─────────────┘  ❌ 不负责：数据预处理、批处理

┌─────────────┐
│   Dataset   │  职责：数据索引和加载
└─────────────┘  ❌ 不负责：批处理（由 DataLoader + Collator 完成）
```

### 2. 符合 PyTorch 生态

遵循 PyTorch 的标准模式：

```python
# 标准 PyTorch 训练流程
for batch in dataloader:  # DataLoader 使用 collate_fn 批处理
    images = batch['pixel_values']  # 已经是 batched tensor
    outputs = model(images)  # forward 接收 tensor，不是 list
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## 组件详细设计

### 1. Processor（处理器）

**位置**: `classifier/data/processor.py`

**职责**:
- 单个样本的预处理
- 图像 resize, crop, normalize
- 转换为 tensor

**不负责**:
- 批处理（batching）
- 数据加载

**接口设计**:
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

**实现示例**:
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

**为什么这样设计**:
- ✅ 可以在不加载模型的情况下预处理数据
- ✅ 支持多进程 DataLoader（不需要序列化模型）
- ✅ 易于测试和复用
- ✅ 训练和验证可以使用不同的 processor（augmentation）

### 2. Collator（批处理器）

**位置**: `classifier/data/collator.py`

**职责**:
- 将多个样本组合成批次
- 堆叠（stack）或填充（pad）张量

**不负责**:
- 单样本预处理
- 模型推理

**接口设计**:
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

**实现示例**:
```python
class DefaultCollator(Collator):
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        return {'pixel_values': pixel_values, 'labels': labels}
```

**为什么这样设计**:
- ✅ 与 PyTorch DataLoader 的 `collate_fn` 完美配合
- ✅ 可以处理不同大小的图像（通过 padding）
- ✅ 易于自定义批处理逻辑
- ✅ 不影响 Dataset 和 Model 的实现

### 3. Model（模型）

**位置**: `classifier/models/model.py`

**职责**:
- 只负责前向传播（forward pass）

**不负责**:
- 数据预处理
- 批处理
- 损失计算
- 优化

**接口设计**:
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

**实现示例**:
```python
class ConvNeXtModel(Model):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__(model_name, num_classes)
        self.backbone = models.convnext_tiny(pretrained=pretrained)
        # 替换分类头
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
```

**为什么这样设计**:
- ✅ 符合 PyTorch nn.Module 标准
- ✅ 可以直接与 DataLoader 配合使用
- ✅ 易于导出（ONNX, TorchScript）
- ✅ 职责单一，易于测试

### 4. Dataset（数据集）

**位置**: `classifier/data/dataset.py`

**职责**:
- 管理数据索引和加载
- 调用 Processor 处理样本

**不负责**:
- 批处理（由 DataLoader + Collator 完成）

**接口设计**:
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

**为什么这样设计**:
- ✅ 符合 PyTorch Dataset 标准
- ✅ 可以使用不同的 Processor（训练/验证）
- ✅ 支持多进程 DataLoader
- ✅ 返回统一的 dict 格式

## 数据流

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

## 完整使用流程

```python
# 1. 创建 Processor
train_processor = ConvNeXtTrainProcessor(image_size=224, augment=True)
val_processor = ConvNeXtProcessor(image_size=224)

# 2. 创建 Dataset
train_dataset = HuggingFaceDataset(
    dataset_name="pufanyi/flowers102",
    split="train",
    processor=train_processor,  # 使用训练 processor（有 augmentation）
)
val_dataset = HuggingFaceDataset(
    dataset_name="pufanyi/flowers102",
    split="validation",
    processor=val_processor,  # 使用验证 processor（无 augmentation）
)

# 3. 创建 Collator
collator = DefaultCollator()

# 4. 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator,  # 使用 collator 进行批处理
)

# 5. 创建 Model
model = ConvNeXtModel(
    model_name="convnext_tiny",
    num_classes=102,
    pretrained=True,
)

# 6. 训练循环
for batch in train_loader:
    images = batch['pixel_values']  # (B, C, H, W) - 已经批处理好了
    labels = batch['labels']        # (B,)
    
    outputs = model(images)         # forward 接收 tensor，不是 list
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()
```

## 与旧设计的对比

### 旧设计 ❌

```python
class Model(ABC, torch.nn.Module):
    @abstractmethod
    def process_image(self, image: Image.Image) -> torch.Tensor:
        pass  # 问题：Model 不应该负责数据预处理
    
    @abstractmethod
    def collate_images(self, images: list[torch.Tensor]) -> torch.Tensor:
        pass  # 问题：Model 不应该负责批处理
    
    @abstractmethod
    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        pass  # 问题：forward 应该接收 tensor，不是 list
```

**问题**:
1. **职责混乱**: Model 同时负责预处理、批处理和推理
2. **不符合 PyTorch 模式**: forward 接收 list 而不是 tensor
3. **难以与 DataLoader 集成**: collate_images 和 DataLoader 的 collate_fn 重复
4. **多进程问题**: DataLoader workers 需要序列化包含预处理逻辑的 model
5. **难以扩展**: 添加新模型需要实现所有方法，即使预处理逻辑相同

### 新设计 ✅

```python
class Processor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        pass  # ✅ 只负责单样本预处理

class Collator:
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        pass  # ✅ 只负责批处理

class Model(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # ✅ 只负责前向传播
```

**优势**:
1. ✅ **职责清晰**: 每个组件只负责一件事
2. ✅ **符合 PyTorch 生态**: 遵循标准模式
3. ✅ **易于扩展**: 独立添加新模型、processor、collator
4. ✅ **支持多进程**: Processor 不依赖 model
5. ✅ **易于测试**: 各组件可以独立测试
6. ✅ **灵活性**: 可以任意组合不同的组件

## 配置管理（Hydra）

使用 Hydra 管理配置，支持模块化和命令行覆盖：

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

命令行覆盖:
```bash
python -m classifier.train model=convnext_small trainer.batch_size=64
```

## 扩展指南

### 添加新模型

1. 实现 Model:
```python
class YourModel(Model):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
```

2. 实现 Processor:
```python
class YourProcessor(Processor):
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)
```

3. 创建配置文件:
```yaml
# model/your_model.yaml
name: your_model
pretrained: true
processor:
  image_size: 224
```

4. 更新训练脚本中的 `create_model()` 和 `create_processors()`

## 总结

这个设计遵循了以下原则：

1. **单一职责原则**: 每个组件只负责一件事
2. **开闭原则**: 对扩展开放，对修改封闭
3. **依赖倒置**: 依赖抽象接口，不依赖具体实现
4. **符合标准**: 遵循 PyTorch 生态的最佳实践

这使得代码：
- 易于理解和维护
- 易于测试
- 易于扩展
- 符合行业标准

