import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from classifier.data.collator.collator import DataCollator
from classifier.data.dataset.dataset import HFDataset
from classifier.data.processor.factory import ProcessorFactory
from classifier.models.factory import ModelFactory
from classifier.utils.logger import logger
from train.classifier.trainer import ClassifierTrainer


def setup_distributed():
    """Sets up the distributed environment."""
    if dist.is_available() and dist.is_initialized():
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


@hydra.main(config_path="../../config", config_name="default_config", version_base=None)
def main(cfg: DictConfig):
    """Main training pipeline."""
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info(f"Starting training with config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Running on {world_size} GPUs.")

    # Load dataset configuration
    # Hydra automatically merges the chosen dataset config
    dataset_cfg = cfg.dataset

    # --- Create Datasets and Dataloaders ---
    inference_processor = ProcessorFactory.get_processor(cfg)
    train_transforms = ProcessorFactory.get_train_processor(cfg)

    train_dataset = HFDataset(
        dataset_name=dataset_cfg.dataset_id,
        split=dataset_cfg.train_split,
        processor=inference_processor,
        transform=train_transforms,
        image_column=dataset_cfg.image_column,
        label_column=dataset_cfg.label_column,
    )

    val_dataset = HFDataset(
        dataset_name=dataset_cfg.dataset_id,
        split=dataset_cfg.val_split,
        processor=inference_processor,
        transform=None,  # No augmentation for validation
        image_column=dataset_cfg.image_column,
        label_column=dataset_cfg.label_column,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    collator = DataCollator()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.train_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=cfg.trainer.dataloader_num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.eval_batch_size,
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=cfg.trainer.dataloader_num_workers,
        pin_memory=True,
    )

    # --- Create Model ---
    # The constants NUM_CLASSES, ID2LABEL, LABEL2ID are loaded when ModelFactory is imported
    model = ModelFactory.get_model(cfg)

    # --- Create Trainer and Run ---
    trainer = ClassifierTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=cfg,
    )

    trainer.train()

    logger.info("Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
