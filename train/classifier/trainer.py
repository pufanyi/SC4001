import gc

import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.fsdp import DeviceMesh, MixedPrecisionPolicy, fully_shard
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
    for name, buf in model.named_buffers():
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

    def create_optimizer_and_lr_scheduler(self):
        self.optimizer = get_optimizer(self.config, self.model)
        self.lr_scheduler = get_lr_scheduler(self.config, self.optimizer)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cast_type = getattr(torch, self.config.trainer.precision.param_type)
        with torch.autocast(device_type="cuda", dtype=cast_type):
            outputs = self.model(**batch)
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
        fsdp_config = dict(
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
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

    def train(self):
        self.create_optimizer_and_lr_scheduler()
