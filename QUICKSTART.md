# 快速开始指南

## 📋 前置条件

```bash
# 已安装的依赖（参考 pyproject.toml）
pip install torch torchvision datasets hydra-core loguru rich
```

## 🚀 5 分钟快速开始

### 1. 查看示例代码

```bash
# 运行示例脚本，了解各个组件如何工作
python -m classifier.example_usage
```

这会展示：
- ✅ Processor 如何处理单个图像
- ✅ Collator 如何批处理
- ✅ Model 如何进行推理
- ✅ 组件如何协同工作

### 2. 开始训练

#### 默认配置训练

```bash
python -m classifier.train
```

这会使用默认配置：
- 数据集: Flowers102
- 模型: ConvNeXt Tiny
- Batch size: 32
- Epochs: 100

#### 快速调试（使用少量数据）

```bash
python -m classifier.train \
    dataset.max_train_samples=100 \
    dataset.max_val_samples=50 \
    trainer.epochs=3 \
    trainer.log_interval=1
```

#### 自定义配置

```bash
# 使用更大的模型
python -m classifier.train model=convnext_small

# 调整超参数
python -m classifier.train \
    trainer.batch_size=64 \
    trainer.optimizer.lr=5e-4 \
    trainer.epochs=50

# 修改数据增强
python -m classifier.train \
    model.train_processor.hflip_prob=0.8 \
    model.train_processor.color_jitter=false
```

### 3. 查看结果

训练输出保存在 `outputs/YYYY-MM-DD/HH-MM-SS/`:

```
outputs/2024-01-01/12-00-00/
├── config.yaml          # 完整配置
├── best_model.pth      # 最佳模型
└── last_model.pth      # 最后的模型
```

加载模型:

```python
import torch
from classifier.models import ConvNeXtModel

# 创建模型
model = ConvNeXtModel("convnext_tiny", num_classes=102)

# 加载权重
checkpoint = torch.load("outputs/.../best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 4. 进行评测

```bash
# 单卡评测（使用默认的 validation split）
python -m eval.pipeline.run

# 指定检查点（如分布式训练得到的 step_500）
python -m eval.pipeline.run \
    evaluation.checkpoint_path=outputs/resnet152_lr1e-5/step_500 \
    evaluation.checkpoint_format=fsdp

# 多卡评测需保持与训练相同的 world size
torchrun --nproc_per_node=4 -m eval.pipeline.run \
    evaluation.checkpoint_path=outputs/resnet152_lr1e-5/step_500
```

- `evaluation.split`：选择 `train` / `validation` / `test` 或自定义 HF split。
- `evaluation.max_samples`：限制样本数量，便于快速抽查。
- `evaluation.metrics_output_path`：指定 JSON 文件路径，可自动落盘评测指标。
- 如果使用 FSDP 切分的权重，评测时需用 `torchrun` 并保持 world size 一致。

## 📚 配置说明

### 数据集配置 (`dataset=...`)

在 `classifier/conf/dataset/` 中定义：

```yaml
# flowers102.yaml
dataset_id: pufanyi/flowers102
num_classes: 102
max_train_samples: null  # 限制训练样本数（调试用）
```

### 模型配置 (`model=...`)

在 `classifier/conf/model/` 中定义：

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

### 训练配置 (`trainer.*`)

在 `classifier/conf/trainer/default.yaml` 中定义：

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

## 🎯 常见任务

### 任务 1：添加新的数据集

1. 创建配置文件 `classifier/conf/dataset/my_dataset.yaml`:

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

2. 运行训练:

```bash
python -m classifier.train dataset=my_dataset
```

### 任务 2：调整数据增强

```bash
# 更激进的增强
python -m classifier.train \
    model.train_processor.scale=[0.5,1.0] \
    model.train_processor.hflip_prob=0.8 \
    model.train_processor.color_jitter=true

# 关闭增强
python -m classifier.train \
    model.train_processor.hflip_prob=0.0 \
    model.train_processor.color_jitter=false
```

### 任务 3：修改优化器设置

```bash
# 使用更大的学习率
python -m classifier.train trainer.optimizer.lr=1e-3

# 修改学习率调度
python -m classifier.train \
    trainer.scheduler.warmup_epochs=10 \
    trainer.scheduler.min_lr=1e-7
```

### 任务 4：使用不同的模型大小

```bash
# ConvNeXt Small
python -m classifier.train model=convnext_small

# ConvNeXt Base (需要更多显存)
python -m classifier.train model=convnext_base trainer.batch_size=16
```

### 任务 5：恢复训练

```python
# 在 Python 脚本中
checkpoint = torch.load("outputs/.../checkpoint_epoch_50.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
best_acc = checkpoint['best_acc']

# 然后继续训练...
```

## 🔧 开发和调试

### 快速测试流水线

```bash
# 使用 10 个样本快速测试
python -m classifier.train \
    dataset.max_train_samples=10 \
    dataset.max_val_samples=5 \
    trainer.epochs=2 \
    trainer.batch_size=2 \
    trainer.num_workers=0
```

### 查看配置

```bash
# Hydra 会打印完整的合并后的配置
python -m classifier.train --cfg job
```

### 覆盖输出目录

```bash
python -m classifier.train output_dir=my_experiment
```

## 📖 更多资源

- **详细文档**: `classifier/README.md`
- **设计文档**: `CLASSIFIER_DESIGN.md`
- **示例代码**: `classifier/example_usage.py`
- **配置文件**: `classifier/conf/`

## ❓ 常见问题

### Q: CUDA out of memory

```bash
# 减小 batch size
python -m classifier.train trainer.batch_size=16

# 或使用梯度累积
python -m classifier.train trainer.batch_size=8 trainer.gradient_accumulation_steps=4
```

### Q: 数据加载太慢

```bash
# 增加 workers
python -m classifier.train trainer.num_workers=8
```

### Q: 想要使用自己的图像

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

## 🎉 开始探索

现在你已经准备好了！开始训练你的模型吧：

```bash
python -m classifier.train
```

祝训练愉快！ 🚀
