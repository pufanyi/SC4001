import gc

import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from loguru import logger
from omegaconf import DictConfig
from torch.distributed import all_reduce
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.fsdp import DeviceMesh, MixedPrecisionPolicy, fully_shard
from torch.distributed.reduce_op import ReduceOp
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import PreTrainedModel

from classifier.data.dataset import HFDataset
from train.optimizer.get_lr_scheduler import get_lr_scheduler
from train.optimizer.get_optimizer import get_optimizer


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
        fully_shard(module, fsdp_kwargs)
    fully_shard(model, fsdp_kwargs)


# Adapted from https://github.com/EvolvingLMMs-Lab/lmms-engine/blob/main/src/lmms_engine/utils/fsdp2_utils.py
def fsdp2_load_full_state_dict(
    model: PreTrainedModel,
    full_state: dict,
    device_mesh: DeviceMesh | None = None,
    cpu_offload: bool | None = None,
):
    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:
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
        train_dataset: HFDataset,
        val_dataset: HFDataset,
        config: DictConfig,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.global_step = 0

    def create_optimizer_and_lr_scheduler(self):
        self.optimizer = get_optimizer(self.config, self.model)
        self.lr_scheduler = get_lr_scheduler(self.config, self.optimizer)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cast_type = getattr(torch, self.config.trainer.precision.param_type)
        with torch.autocast(device_type="cuda", dtype=cast_type):
            outputs = self.fsdp2_model(**batch)
            loss = outputs["loss"]
        return loss

    def prepare_model(self):
        param_type = getattr(torch, self.config.trainer.precision.param_type)
        reduct_type = getattr(torch, self.config.trainer.precision.reduction_type)
        output_type = getattr(torch, self.config.trainer.precision.output_type)
        mp_policy = MixedPrecisionPolicy(
            param_type=param_type, reduction_type=reduct_type, output_type=output_type
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
        self.fsdp2_model.eval()
        total_loss = 0
        for batch in self.val_dataset:
            batch = send_to_device(batch, device=torch.cuda.current_device())
            with torch.no_grad():
                loss = self.compute_loss(batch)
                total_loss += loss.item()
        final_loss = total_loss / len(self.val_dataset)
        logger.info(f"Step {self.global_step} evaluation loss: {final_loss}")
        return final_loss

    def save_model(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        output_dir = Path(self.config.trainer.output_dir) / f"step_{self.global_step}"
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
        dist.barrier()
        torch.save(self.fsdp2_model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        extra_state = {
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "rng": self.get_rng_state(),
        }
        torch.save(extra_state, extra_state_path)
        torch.save(self.train_dataset.state_dict(), dataloader_state_path)
        logger.info(f"State saved to {output_dir}")

    def get_rng_state(self):
        return {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

    def train(self):
        self.prepare_model()
        self.create_optimizer_and_lr_scheduler()
        num_steps = self.config.trainer.num_steps
        rank = dist.get_rank()
        dist.get_world_size()
        self.global_step = 0
        pbar = tqdm(total=num_steps, desc="Training", disable=rank != 0)
        while not self.should_stop():
            for batch in self.train_dataset:
                if self.should_stop():
                    break
                batch = send_to_device(batch, device=torch.cuda.current_device())
                self.fsdp2_model.train()
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                if dist.get_world_size() > 1:
                    loss = loss.mean()
                loss_item = loss.item()
                loss.backward()
                grad_norm = clip_grad_norm_(
                    self.fsdp2_model.parameters(), self.config.trainer.grad_norm_clip
                )
                if torch.isfinite(grad_norm):
                    self.optimizer.step()
                else:
                    logger.warning(
                        f"Step {self.global_step} gradient norm is not finite, skipping optimizer step"
                    )
                    self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.lr_scheduler.get_last_lr()[0]
                loss_item = torch.tensor(loss_item, device=torch.cuda.current_device())
                all_reduce(loss_item, op=ReduceOp.AVG)
                self.global_step += 1
                logger.info(
                    f"Step {self.global_step} training loss: {loss_item.item()}"
                )
                if self.global_step % self.config.trainer.eval_steps == 0:
                    self.eval()
                if self.global_step % self.config.trainer.save_steps == 0:
                    self.save_model()
                pbar.update(1)

        pbar.close()
