import gc
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import all_reduce
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed import ReduceOp
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from train.optimizer.get_lr_scheduler import get_lr_scheduler
from train.optimizer.get_optimizer import get_optimizer


def _get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _distributed_barrier():
    if not dist.is_available() or not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


# Adapted from https://github.com/EvolvingLMMs-Lab/lmms-engine/blob/main/src/lmms_engine/utils/fsdp2_utils.py
def apply_fsdp2(
    model: PreTrainedModel,
    fsdp_kwargs: dict,
    fsdp_transformer_layer_cls_to_wrap: str | list[str] | None = None,
):
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = (
        default_transformer_cls_names_to_wrap
        if fsdp_transformer_layer_cls_to_wrap is None
        else fsdp_transformer_layer_cls_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert (
        len(fsdp_transformer_layer_cls_to_wrap) > 0
        and fsdp_transformer_layer_cls_to_wrap[0] is not None
    )

    modules = []
    for module in model.modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, torch.nn.Embedding)
            and not (
                hasattr(model.config, "tie_word_embeddings")
                and model.config.tie_word_embeddings
            )
        ):
            modules.append(module)

    for module in modules:
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)


# Adapted from https://github.com/EvolvingLMMs-Lab/lmms-engine/blob/main/src/lmms_engine/utils/fsdp2_utils.py
def fsdp2_load_full_state_dict(
    model: PreTrainedModel,
    full_state: dict,
    device_mesh: DeviceMesh | None = None,
    cpu_offload: bool | None = None,
):
    # To broadcast, it needs to be instantiated in the GPU.
    if _get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        model = model.to_empty(device=torch.cuda.current_device())

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True
    )
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for _name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(torch.cuda.current_device())


class ClassifierTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: DictConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.global_step = 0
        self.wandb_run = None

    def create_optimizer_and_lr_scheduler(self):
        self.optimizer = get_optimizer(self.config, self.model)
        self.lr_scheduler = get_lr_scheduler(self.config, self.optimizer)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cast_type = getattr(torch, self.config.trainer.precision.param_type)
        with torch.autocast(device_type="cuda", dtype=cast_type):
            outputs = self.fsdp2_model(**batch)
            loss = outputs["loss"]
        return outputs, loss

    def prepare_model(self):
        param_dtype = getattr(torch, self.config.trainer.precision.param_type)
        reduce_dtype = getattr(torch, self.config.trainer.precision.reduction_type)
        output_dtype = getattr(torch, self.config.trainer.precision.output_type)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
        )
        reshard_after_forward = self.config.trainer.reshard_after_forward
        fsdp_config = {
            "mp_policy": mp_policy,
            "reshard_after_forward": reshard_after_forward,
        }
        full_state = self.model.state_dict()
        logger.info("Applying FSDP2 to model")
        transformer_cls_names_to_wrap = (
            self.config.trainer.transformer_cls_names_to_wrap
        )
        apply_fsdp2(self.model, fsdp_config, transformer_cls_names_to_wrap)
        logger.info("Loading full state dict into model")
        fsdp2_load_full_state_dict(self.model, full_state)
        logger.info("FSDP2 applied to model")
        self.fsdp2_model = self.model
        del full_state
        gc.collect()
        torch.cuda.empty_cache()

    def should_stop(self):
        return self.global_step >= self.config.trainer.num_steps

    def eval(self):
        rank = _get_rank()
        world_size = _get_world_size()
        self.fsdp2_model.eval()
        total_loss = 0.0
        total_batches = 0
        num_correct = 0
        num_samples = 0
        
        for batch in self.val_dataloader:
            batch = send_to_device(batch, device=torch.cuda.current_device())
            batch_size = batch["labels"].size(0)
            num_samples += batch_size
            
            with torch.no_grad():
                outputs, loss = self.compute_loss(batch)
                total_loss += loss.item() * batch_size  # weight by batch size
                total_batches += 1
                preds = outputs["logits"].argmax(dim=-1)
                num_correct += (preds == batch["labels"]).sum().item()
        
        # Synchronize metrics across all ranks in distributed training
        if world_size > 1:
            # Convert to tensors for all_reduce
            metrics_tensor = torch.tensor(
                [total_loss, num_correct, num_samples], 
                dtype=torch.float32,
                device=torch.cuda.current_device()
            )
            all_reduce(metrics_tensor, op=ReduceOp.SUM)
            total_loss, num_correct, num_samples = metrics_tensor.tolist()
        
        # Calculate global metrics
        final_loss = total_loss / max(num_samples, 1)
        accuracy = num_correct / max(num_samples, 1)
        
        if rank == 0:
            logger.info(f"Step {self.global_step} evaluation loss: {final_loss:.4f}, accuracy: {accuracy:.4f}")
            self._log({"eval/loss": final_loss, "eval/accuracy": accuracy})
        
        return final_loss

    def save_model(self):
        rank = _get_rank()
        world_size = _get_world_size()
        output_dir = (
            Path(self.config.trainer.get("output_dir", "outputs"))
            / f"step_{self.global_step}"
        )
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model" / f"ws_{world_size}_rank_{rank}.pt"
        optimizer_path = output_dir / "optimizer" / f"ws_{world_size}_rank_{rank}.pt"
        extra_state_path = (
            output_dir / "extra_state" / f"ws_{world_size}_rank_{rank}.pt"
        )
        dataloader_state_path = (
            output_dir / "dataloader_state" / f"ws_{world_size}_rank_{rank}.pt"
        )
        if rank == 0:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            optimizer_path.parent.mkdir(parents=True, exist_ok=True)
            extra_state_path.parent.mkdir(parents=True, exist_ok=True)
            dataloader_state_path.parent.mkdir(parents=True, exist_ok=True)
        _distributed_barrier()
        torch.save(self.fsdp2_model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        extra_state = {
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "rng": self.get_rng_state(),
            "global_step": self.global_step,
        }
        torch.save(extra_state, extra_state_path)
        if hasattr(self.train_dataloader, "state_dict"):
            dataloader_state = self.train_dataloader.state_dict()
            torch.save(dataloader_state, dataloader_state_path)
        else:
            logger.warning(
                "Train dataloader does not implement state_dict; skipping save"
            )
        if rank == 0:
            logger.info(f"State saved to {output_dir}")
            self._log({"checkpoint/step": self.global_step})
            if (
                self.config.trainer.save_total_limit is not None
                and self.config.trainer.save_total_limit > 0
            ):
                self._prune_checkpoints(output_dir.parent)

    def get_rng_state(self):
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
        return rng_state

    def set_rng_state(self, rng_state: dict):
        if not rng_state:
            return
        if "cpu" in rng_state:
            torch.set_rng_state(rng_state["cpu"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state(rng_state["cuda"], torch.cuda.current_device())
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "random" in rng_state:
            random.setstate(rng_state["random"])

    def setup_logging(self):
        report_to = self.config.trainer.get("report_to", None)
        if report_to is None:
            self.report_to = []
            return
        if isinstance(report_to, str):
            report_to = [report_to]
        self.report_to = report_to
        if "wandb" in report_to and _get_rank() == 0:
            try:
                import wandb

                wandb_config = OmegaConf.to_container(self.config, resolve=True)
                kwargs = {
                    "project": self.config.trainer.get("wandb_project", "sc4001"),
                    "config": wandb_config,
                }
                if self.config.trainer.get("wandb_run_name"):
                    kwargs["name"] = self.config.trainer.wandb_run_name
                if self.config.trainer.get("wandb_entity"):
                    kwargs["entity"] = self.config.trainer.wandb_entity
                if self.config.trainer.get("wandb_run_id"):
                    kwargs["id"] = self.config.trainer.wandb_run_id
                    kwargs["resume"] = "allow"
                tags = self.config.trainer.get("wandb_tags")
                if tags:
                    kwargs["tags"] = list(tags)
                self.wandb_run = wandb.init(**kwargs)
            except ImportError:
                logger.warning("wandb is not installed; skipping wandb initialization")
            except Exception as exc:
                logger.warning(f"Failed to initialize wandb: {exc}")

    def close_logging(self):
        if self.wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception as exc:
                logger.warning(f"Failed to close wandb run cleanly: {exc}")
            finally:
                self.wandb_run = None

    def _log(self, metrics: dict[str, float]):
        if not metrics:
            return
        if not hasattr(self, "report_to"):
            self.report_to = []
        if "wandb" in self.report_to and self.wandb_run is not None:
            if _get_rank() == 0:
                try:
                    import wandb

                    wandb.log(metrics, step=self.global_step)
                except Exception as exc:
                    logger.warning(f"Failed to log metrics to wandb: {exc}")

    def _prune_checkpoints(self, parent_dir: Path):
        checkpoints_with_steps = []
        for path in parent_dir.glob("step_*"):
            if not path.is_dir():
                continue
            step = self._extract_step_from_path(path)
            if step is not None:
                checkpoints_with_steps.append((step, path))
        checkpoints_with_steps.sort(key=lambda item: item[0])
        checkpoints = [path for _, path in checkpoints_with_steps]
        limit = self.config.trainer.save_total_limit
        if len(checkpoints) <= limit:
            return
        to_remove = checkpoints[: len(checkpoints) - limit]
        for ckpt in to_remove:
            logger.info(f"Removing old checkpoint {ckpt}")
            shutil.rmtree(ckpt, ignore_errors=True)

    def maybe_resume(self):
        checkpoint_dir = self._resolve_checkpoint_path()
        if checkpoint_dir is None:
            return
        rank = _get_rank()
        world_size = _get_world_size()
        logger.info(f"Resuming training from checkpoint {checkpoint_dir}")
        model_path = checkpoint_dir / "model" / f"ws_{world_size}_rank_{rank}.pt"
        optimizer_path = (
            checkpoint_dir / "optimizer" / f"ws_{world_size}_rank_{rank}.pt"
        )
        extra_state_path = (
            checkpoint_dir / "extra_state" / f"ws_{world_size}_rank_{rank}.pt"
        )
        dataloader_state_path = (
            checkpoint_dir / "dataloader_state" / f"ws_{world_size}_rank_{rank}.pt"
        )

        map_location = "cpu"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=map_location)
            self.fsdp2_model.load_state_dict(state_dict)
        else:
            logger.warning(f"Model checkpoint not found at {model_path}")
        if optimizer_path.exists():
            optim_state = torch.load(optimizer_path, map_location=map_location)
            self.optimizer.load_state_dict(optim_state)
        else:
            logger.warning(f"Optimizer checkpoint not found at {optimizer_path}")
        if extra_state_path.exists():
            extra_state = torch.load(extra_state_path, map_location=map_location)
            lr_state = extra_state.get("lr_scheduler_state_dict")
            if lr_state is not None:
                self.lr_scheduler.load_state_dict(lr_state)
            self.set_rng_state(extra_state.get("rng", {}))
            self.global_step = extra_state.get("global_step", self.global_step)
        else:
            logger.warning(f"Extra state checkpoint not found at {extra_state_path}")
        if dataloader_state_path.exists() and hasattr(
            self.train_dataloader, "load_state_dict"
        ):
            dataloader_state = torch.load(
                dataloader_state_path, map_location=map_location
            )
            self.train_dataloader.load_state_dict(dataloader_state)
        elif dataloader_state_path.exists():
            logger.warning(
                "Train dataloader does not support load_state_dict; skipping resume"
            )

    def _resolve_checkpoint_path(self) -> Path | None:
        resume_from = self.config.trainer.get("resume_from_checkpoint")
        if resume_from is None:
            return None
        if isinstance(resume_from, str) and resume_from.lower() == "latest":
            base_dir = Path(self.config.trainer.output_dir)
            candidates = []
            for path in base_dir.glob("step_*"):
                if not path.is_dir():
                    continue
                step = self._extract_step_from_path(path)
                if step is not None:
                    candidates.append((step, path))
            candidates.sort(key=lambda item: item[0])
            if not candidates:
                logger.warning(
                    "resume_from_checkpoint was 'latest' but no checkpoints were found"
                )
                return None
            return candidates[-1][1]
        resume_path = Path(resume_from)
        if not resume_path.exists():
            logger.warning(f"Checkpoint path {resume_path} does not exist")
            return None
        return resume_path

    @staticmethod
    def _extract_step_from_path(path: Path) -> int | None:
        try:
            return int(path.name.split("_")[-1])
        except (ValueError, IndexError):
            return None

    def train(self):
        self.prepare_model()
        self.create_optimizer_and_lr_scheduler()
        self.setup_logging()
        try:
            self.maybe_resume()
        except Exception as exc:
            logger.warning(f"Failed to resume from checkpoint: {exc}")
        num_steps = self.config.trainer.num_steps
        rank = _get_rank()
        world_size = _get_world_size()
        pbar = tqdm(total=num_steps, desc="Training", disable=rank != 0)
        if self.global_step > 0:
            pbar.update(self.global_step)
        try:
            while not self.should_stop():
                for batch in self.train_dataloader:
                    if self.should_stop():
                        break
                    batch = send_to_device(batch, device=torch.cuda.current_device())
                    self.fsdp2_model.train()
                    self.optimizer.zero_grad()
                    outputs, loss = self.compute_loss(batch)
                    if world_size > 1:
                        loss = loss.mean()
                    loss_item = loss.item()
                    loss.backward()
                    grad_norm = clip_grad_norm_(
                        self.fsdp2_model.parameters(),
                        self.config.trainer.grad_norm_clip,
                    )
                    grad_norm_tensor = (
                        grad_norm
                        if isinstance(grad_norm, torch.Tensor)
                        else torch.tensor(grad_norm)
                    )
                    grad_norm_value = (
                        grad_norm_tensor.item()
                        if grad_norm_tensor.numel() == 1
                        else float(grad_norm_tensor.mean().item())
                    )
                    if torch.isfinite(grad_norm_tensor).all():
                        self.optimizer.step()
                    else:
                        logger.warning(
                            f"Step {self.global_step} gradient norm is not finite, skipping optimizer step"
                        )
                        self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    loss_item = torch.tensor(
                        loss_item, device=torch.cuda.current_device()
                    )
                    all_reduce(loss_item, op=ReduceOp.AVG)
                    self.global_step += 1
                    if rank == 0:
                        logger.info(
                            f"Step {self.global_step} training loss: {loss_item.item()}"
                        )
                    metrics = {
                        "train/loss": loss_item.item(),
                        "train/grad_norm": grad_norm_value,
                        "train/lr": current_lr,
                        "train/epoch": self.global_step / len(self.train_dataloader),
                    }
                    if rank == 0 and self.global_step % self.config.trainer.logging_steps == 0:
                        self._log(metrics)
                    if self.global_step % self.config.trainer.eval_steps == 0:
                        _distributed_barrier()
                        self.eval()
                        _distributed_barrier()
                        self.fsdp2_model.train()  # Set back to train mode
                    if self.global_step % self.config.trainer.save_steps == 0:
                        self.save_model()
                    pbar.update(1)
        finally:
            pbar.close()
            self.close_logging()
