import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
from torch.distributed import ReduceOp
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from classifier.data.collator.collator import DataCollator
from classifier.data.dataset.dataset import HFDataset
from classifier.data.processor.factory import ProcessorFactory
from classifier.models.factory import ModelFactory
from classifier.utils.logger import logger
from train.classifier.trainer import apply_fsdp2, fsdp2_load_full_state_dict


def _setup_distributed() -> tuple[int, int]:
    """Initialise torch.distributed if launched via torchrun."""
    if not dist.is_available():
        return 0, 1
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" not in os.environ:
        return 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return dist.get_rank(), dist.get_world_size()


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _canonical_split_key(split: str) -> str | None:
    key = split.lower()
    if key == "train":
        return "train"
    if key in {"val", "validation"}:
        return "val"
    if key == "test":
        return "test"
    return None


def _resolve_split_name(dataset_cfg: DictConfig, target_split: str) -> str:
    split_key = target_split.lower()
    mapping = {
        "train": dataset_cfg.get("train_split"),
        "val": dataset_cfg.get("val_split"),
        "validation": dataset_cfg.get("val_split"),
        "test": dataset_cfg.get("test_split"),
    }
    resolved = mapping.get(split_key, target_split)
    if resolved is None:
        raise ValueError(f"Split '{target_split}' is not defined in the dataset config.")
    return resolved


def _resolve_max_samples(
    dataset_cfg: DictConfig, canonical_split: str | None, override: int | None
) -> int | None:
    if override is not None:
        return int(override)
    split_to_key = {
        "train": "max_train_samples",
        "val": "max_val_samples",
        "test": "max_test_samples",
    }
    if canonical_split is None:
        return None
    cfg_key = split_to_key.get(canonical_split)
    if cfg_key is None:
        return None
    max_value = dataset_cfg.get(cfg_key)
    if max_value is None:
        return None
    return int(max_value)


def _limit_dataset(dataset: HFDataset, max_samples: int | None) -> HFDataset | Subset[HFDataset]:
    if max_samples is None:
        return dataset
    limit = min(max_samples, len(dataset))
    indices = list(range(limit))
    return Subset(dataset, indices)


def _build_dataloader(
    cfg: DictConfig, processor, rank: int, world_size: int
) -> tuple[DataLoader, str]:
    eval_cfg = cfg.evaluation
    dataset_cfg = cfg.dataset
    resolved_split = _resolve_split_name(dataset_cfg, eval_cfg.split)
    canonical_key = _canonical_split_key(eval_cfg.split)
    max_samples = _resolve_max_samples(dataset_cfg, canonical_key, eval_cfg.max_samples)
    dataset = HFDataset(
        dataset_name=dataset_cfg.dataset_id,
        split=resolved_split,
        processor=processor,
        transform=None,
        image_column=dataset_cfg.image_column,
        label_column=dataset_cfg.label_column,
    )
    dataset = _limit_dataset(dataset, max_samples)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    batch_size = eval_cfg.batch_size or cfg.trainer.eval_batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=DataCollator(),
        num_workers=cfg.trainer.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if rank == 0:
        logger.info(
            f"Evaluating split '{resolved_split}' with {len(dataset)} samples "
            f"and batch size {batch_size}."
        )
    return dataloader, resolved_split


def _prepare_model_for_eval(model: torch.nn.Module, cfg: DictConfig, use_fsdp: bool) -> torch.nn.Module:
    if not use_fsdp:
        return model
    param_dtype = getattr(torch, cfg.trainer.precision.param_type)
    reduce_dtype = getattr(torch, cfg.trainer.precision.reduction_type)
    output_dtype = getattr(torch, cfg.trainer.precision.output_type)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype, output_dtype=output_dtype
    )
    fsdp_config = {
        "mp_policy": mp_policy,
        "reshard_after_forward": cfg.trainer.reshard_after_forward,
    }
    full_state = model.state_dict()
    transformer_cls_to_wrap = cfg.trainer.transformer_cls_names_to_wrap
    apply_fsdp2(model, fsdp_config, transformer_cls_to_wrap)
    fsdp2_load_full_state_dict(model, full_state)
    del full_state
    torch.cuda.empty_cache()
    return model


def _parse_world_size_from_shard(shard_path: Path) -> int:
    # Expected pattern: ws_{world_size}_rank_{rank}.pt
    parts = shard_path.stem.split("_")
    try:
        ws_index = parts.index("ws")
        world_size = int(parts[ws_index + 1])
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse world size from shard name {shard_path.name}") from None
    return world_size


def _load_checkpoint_if_needed(
    model: torch.nn.Module,
    eval_cfg: DictConfig,
    world_size: int,
    rank: int,
) -> None:
    checkpoint_path = eval_cfg.checkpoint_path
    if checkpoint_path is None:
        logger.info("No checkpoint path provided; evaluating pretrained weights.")
        return
    checkpoint_path = Path(checkpoint_path)
    checkpoint_format = eval_cfg.checkpoint_format
    if checkpoint_format not in {"auto", "fsdp", "state_dict"}:
        raise ValueError(f"Unknown checkpoint_format '{checkpoint_format}'")

    if checkpoint_path.is_file() and checkpoint_format in {"auto", "state_dict"}:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=eval_cfg.strict_load)
        if missing and rank == 0:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected and rank == 0:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")
        logger.info(f"Loaded checkpoint from file {checkpoint_path}")
        return

    if checkpoint_path.is_dir():
        model_dir = checkpoint_path / "model"
        if model_dir.exists() and checkpoint_format in {"auto", "fsdp"}:
            sample_shards = sorted(model_dir.glob("ws_*_rank_*.pt"))
            if not sample_shards:
                raise FileNotFoundError(
                    f"No FSDP shards found under {model_dir}. "
                    "Ensure the checkpoint directory is correct."
                )
            expected_world_size = _parse_world_size_from_shard(sample_shards[0])
            if expected_world_size != world_size:
                raise ValueError(
                    f"Checkpoint expects world_size={expected_world_size} "
                    f"but evaluation is running with world_size={world_size}."
                )
            shard_path = model_dir / f"ws_{world_size}_rank_{rank}.pt"
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Shard {shard_path} not found for rank {rank}. "
                    "Make sure evaluation uses the same number of processes as training."
                )
            state_dict = torch.load(shard_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(
                state_dict, strict=eval_cfg.strict_load
            )
            if missing and rank == 0:
                logger.warning(f"Missing keys when loading shard: {missing}")
            if unexpected and rank == 0:
                logger.warning(f"Unexpected keys when loading shard: {unexpected}")
            logger.info(f"Loaded sharded checkpoint from {shard_path}")
            return

    raise FileNotFoundError(
        f"Could not interpret checkpoint path '{checkpoint_path}'. "
        "Set evaluation.checkpoint_format to 'fsdp' or 'state_dict' if detection fails."
    )


def _reduce_tensor(value: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=op)
    return value


def _evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    eval_cfg = cfg.evaluation
    topk_values = sorted({int(k) for k in eval_cfg.topk if int(k) > 0})
    if not topk_values:
        topk_values = [1]
    max_requested_k = max(topk_values)
    disable_amp = not (eval_cfg.use_amp and torch.cuda.is_available())

    iterator: Iterable = dataloader
    if rank == 0 and not eval_cfg.disable_tqdm:
        iterator = tqdm(iterator, desc="Evaluating", total=len(dataloader))

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    topk_correct = torch.zeros(len(topk_values), dtype=torch.float64, device=device)
    warned_topk = False

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            batch = send_to_device(batch, device=device)
            context = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if not disable_amp
                else nullcontext()
            )
            with context:
                outputs = model(**batch)
                logits = outputs["logits"]
                if "loss" in outputs and outputs["loss"] is not None:
                    loss = outputs["loss"]
                else:
                    loss = F.cross_entropy(logits, batch["labels"])
            batch_size = batch["labels"].size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch["labels"]).sum().item()
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            available_k = min(max_requested_k, logits.size(-1))
            if max_requested_k > logits.size(-1) and not warned_topk and rank == 0:
                logger.warning(
                    "Requested top-k accuracy larger than number of classes; "
                    f"clipping to {available_k}."
                )
                warned_topk = True
            if available_k > 0:
                topk_preds = logits.topk(k=available_k, dim=-1).indices
                matches = topk_preds.eq(batch["labels"].unsqueeze(1))
                for idx, k in enumerate(topk_values):
                    effective_k = min(k, available_k)
                    correct = matches[:, :effective_k].any(dim=1).sum().item()
                    topk_correct[idx] += correct

    device_for_reduce = device if device.type == "cuda" else torch.device("cpu")
    tensors = {
        "loss": torch.tensor(total_loss, device=device_for_reduce, dtype=torch.float64),
        "correct": torch.tensor(total_correct, device=device_for_reduce, dtype=torch.float64),
        "samples": torch.tensor(total_samples, device=device_for_reduce, dtype=torch.float64),
        "topk": topk_correct.to(device_for_reduce),
    }
    for key, tensor in tensors.items():
        tensors[key] = _reduce_tensor(tensor, ReduceOp.SUM)

    total_samples = max(tensors["samples"].item(), 1.0)
    metrics = {
        "loss": tensors["loss"].item() / total_samples,
        "accuracy": tensors["correct"].item() / total_samples,
    }
    for idx, k in enumerate(topk_values):
        metrics[f"accuracy@{k}"] = tensors["topk"][idx].item() / total_samples

    return metrics


def _dump_metrics(metrics: dict[str, float], path: str | None, rank: int) -> None:
    if path is None or rank != 0:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    logger.info(f"Saved evaluation metrics to {output_path}")


@hydra.main(config_path="../../config", config_name="default_config", version_base=None)
def main(cfg: DictConfig) -> None:
    rank, world_size = _setup_distributed()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    if rank == 0:
        logger.info("Starting evaluation with configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

    processor = ProcessorFactory.get_processor(cfg)
    dataloader, split_name = _build_dataloader(cfg, processor, rank, world_size)
    model = ModelFactory.get_model(cfg)
    use_fsdp = world_size > 1
    model = _prepare_model_for_eval(model, cfg, use_fsdp)
    if not use_fsdp:
        model.to(device)

    _load_checkpoint_if_needed(model, cfg.evaluation, world_size, rank)

    metrics = _evaluate(model, dataloader, device, cfg, rank, world_size)
    if rank == 0:
        formatted = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"[split={split_name}] {formatted}")

    _dump_metrics(metrics, cfg.evaluation.metrics_output_path, rank)
    _cleanup_distributed()


if __name__ == "__main__":
    main()
