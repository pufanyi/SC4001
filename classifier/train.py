#!/usr/bin/env python3
"""
训练脚本

使用方法:
    python -m classifier.train

    # 使用自定义配置
    python -m classifier.train model=convnext_small trainer.batch_size=64
"""

import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier.data.collator import DefaultCollator
from classifier.data.dataset import HuggingFaceDataset
from classifier.data.processor import ConvNeXtProcessor, ConvNeXtTrainProcessor
from classifier.models.convnext import ConvNeXtModel
from classifier.utils.logger import logger


def set_seed(seed: int):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保确定性 (可能会降低性能)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def create_model(cfg: DictConfig) -> nn.Module:
    """创建模型"""
    logger.info(f"Creating model: {cfg.model.name}")

    if cfg.model.name.startswith("convnext"):
        model = ConvNeXtModel(
            model_name=cfg.model.model_name,
            num_classes=cfg.dataset.num_classes,
            pretrained=cfg.model.pretrained,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    return model


def create_processors(cfg: DictConfig):
    """创建训练和验证的 processor"""
    logger.info("Creating processors")

    if cfg.model.name.startswith("convnext"):
        train_processor = ConvNeXtTrainProcessor(
            image_size=cfg.model.train_processor.image_size,
            scale=tuple(cfg.model.train_processor.scale),
            ratio=tuple(cfg.model.train_processor.ratio),
            hflip_prob=cfg.model.train_processor.hflip_prob,
            mean=tuple(cfg.model.train_processor.mean),
            std=tuple(cfg.model.train_processor.std),
            color_jitter=cfg.model.train_processor.color_jitter,
        )

        val_processor = ConvNeXtProcessor(
            image_size=cfg.model.processor.image_size,
            resize_size=cfg.model.processor.resize_size,
            mean=tuple(cfg.model.processor.mean),
            std=tuple(cfg.model.processor.std),
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    return train_processor, val_processor


def create_datasets(cfg: DictConfig, train_processor, val_processor):
    """创建训练和验证数据集"""
    logger.info(f"Loading dataset: {cfg.dataset.dataset_id}")

    # 训练集
    train_dataset = HuggingFaceDataset(
        dataset_name=cfg.dataset.dataset_id,
        split=cfg.dataset.train_split,
        processor=train_processor,
        image_column=cfg.dataset.image_column,
        label_column=cfg.dataset.label_column,
    )

    # 验证集
    val_dataset = HuggingFaceDataset(
        dataset_name=cfg.dataset.dataset_id,
        split=cfg.dataset.val_split,
        processor=val_processor,
        image_column=cfg.dataset.image_column,
        label_column=cfg.dataset.label_column,
    )

    # 限制样本数量 (用于调试)
    if cfg.dataset.max_train_samples is not None:
        train_dataset.dataset = train_dataset.dataset.select(
            range(cfg.dataset.max_train_samples)
        )

    if cfg.dataset.max_val_samples is not None:
        val_dataset.dataset = val_dataset.dataset.select(
            range(cfg.dataset.max_val_samples)
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_dataloaders(cfg: DictConfig, train_dataset, val_dataset):
    """创建 DataLoader"""
    collator = DefaultCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_optimizer(cfg: DictConfig, model: nn.Module):
    """创建优化器"""
    if cfg.trainer.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.trainer.optimizer.lr,
            weight_decay=cfg.trainer.optimizer.weight_decay,
            betas=tuple(cfg.trainer.optimizer.betas),
        )
    elif cfg.trainer.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.trainer.optimizer.lr,
            weight_decay=cfg.trainer.optimizer.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.trainer.optimizer.name}")

    return optimizer


def create_scheduler(cfg: DictConfig, optimizer, steps_per_epoch: int):
    """创建学习率调度器"""
    total_steps = cfg.trainer.epochs * steps_per_epoch
    warmup_steps = cfg.trainer.scheduler.warmup_epochs * steps_per_epoch

    if cfg.trainer.scheduler.name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=cfg.trainer.scheduler.min_lr,
        )

        # Combine warmup + cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    elif cfg.trainer.scheduler.name == "step":
        from torch.optim.lr_scheduler import StepLR

        scheduler = StepLR(optimizer, step_size=30 * steps_per_epoch, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.trainer.scheduler.name}")

    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    cfg: DictConfig,
) -> tuple[float, float]:
    """训练一个 epoch"""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 将数据移到设备
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪 (可选)
        if cfg.trainer.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.trainer.grad_clip.max_norm
            )

        optimizer.step()
        scheduler.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        if batch_idx % cfg.trainer.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                    "lr": f"{current_lr:.6f}",
                }
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """验证"""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="Validating")
    for batch in pbar:
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
        )

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    output_dir: Path,
    filename: str = "checkpoint.pth",
):
    """保存 checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
    }

    checkpoint_path = output_dir / filename
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """主训练函数"""
    # 打印配置
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # 设置随机种子
    set_seed(cfg.seed)

    # 创建输出目录
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Saved config to {config_path}")

    # 设置设备
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 创建模型
    model = create_model(cfg)
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 创建 processors 和数据集
    train_processor, val_processor = create_processors(cfg)
    train_dataset, val_dataset = create_datasets(cfg, train_processor, val_processor)
    train_loader, val_loader = create_dataloaders(cfg, train_dataset, val_dataset)

    # 创建优化器和调度器
    optimizer = create_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer, len(train_loader))

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0.0
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(1, cfg.trainer.epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch, cfg
        )

        logger.info(f"Epoch {epoch}/{cfg.trainer.epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 验证
        if epoch % cfg.trainer.eval_interval == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                if cfg.trainer.save_best:
                    save_checkpoint(
                        model, optimizer, epoch, best_acc, output_dir, "best_model.pth"
                    )
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if cfg.trainer.early_stopping.enabled:
                if patience_counter >= cfg.trainer.early_stopping.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

        # 定期保存 checkpoint
        if epoch % cfg.trainer.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_acc,
                output_dir,
                f"checkpoint_epoch_{epoch}.pth",
            )

    # 保存最后的模型
    if cfg.trainer.save_last:
        save_checkpoint(
            model, optimizer, cfg.trainer.epochs, best_acc, output_dir, "last_model.pth"
        )

    logger.info(f"Training completed! Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
