#!/usr/bin/env python3
"""
Convert an FSDP2 checkpoint produced by the training loop into a Hugging Face
compatible format and upload it to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed._tensor  # noqa: F401 - registers DTensor pickling
from huggingface_hub import HfApi, HfFolder
from torch.distributed._tensor import DTensor
from torch.distributed._tensor import _utils as dt_utils
from transformers import AutoImageProcessor, AutoModelForImageClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier.utils.constants import ID2LABEL, LABEL2ID, NUM_CLASSES

logger = logging.getLogger("export_fsdp2_to_hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an FSDP2 checkpoint to Hugging Face format and push it to the Hub."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to the step_* directory that contains model/optimizer/etc. sub-folders.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face Hub repo id, e.g. username/convnextv2-flowers102.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/resnet-152",
        help="Base model name to instantiate before loading the checkpoint.",
    )
    parser.add_argument(
        "--processor-name",
        default=None,
        help="Optional image processor name. Defaults to --model-name when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store the Hugging Face artifacts before uploading. "
        "Defaults to <checkpoint-dir>/hf_export.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Falls back to HUGGINGFACE_HUB_TOKEN or cached login.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional branch/revision name to push to.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add model export",
        help="Commit message used when uploading to the Hub.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="2GB",
        help="Max shard size passed to save_pretrained (supports suffixes like 500MB, 2GB).",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save checkpoints using safetensors (recommended).",
    )
    parser.add_argument(
        "--no-safe-serialization",
        dest="safe_serialization",
        action="store_false",
        help="Disable safetensors when saving.",
    )
    parser.set_defaults(safe_serialization=True)
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Enable strict state_dict loading. Defaults to non-strict.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the Hub repo as private.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to the Hub (useful for local dry-runs).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Use weights_only=True when loading shards (PyTorch 2.6 default).",
    )
    return parser.parse_args()


def _extract_rank_info(path: Path) -> tuple[int, int]:
    """Parse world size and rank from a shard filename."""
    parts = path.stem.split("_")
    if len(parts) != 4 or parts[0] != "ws" or parts[2] != "rank":
        raise ValueError(
            f"Shard {path.name} does not follow the ws_<world>_rank_<rank>.pt pattern."
        )
    try:
        world_size = int(parts[1])
        rank = int(parts[3])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse shard metadata from {path.name}") from exc
    return world_size, rank


def find_model_shards(checkpoint_dir: Path) -> list[Path]:
    model_dir = checkpoint_dir / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{model_dir} does not exist or is not a directory.")
    shards = sorted(model_dir.glob("ws_*_rank_*.pt"))
    if not shards:
        raise FileNotFoundError(
            f"No shards found under {model_dir}. Expected files named ws_*_rank_*.pt."
        )
    world_size, _ = _extract_rank_info(shards[0])
    if len(shards) != world_size:
        logger.warning(
            "Expected %d shards based on filenames but found %d. Proceeding anyway.",
            world_size,
            len(shards),
        )
    shards_with_rank = ((_extract_rank_info(path)[1], path) for path in shards)
    sorted_shards = [path for _, path in sorted(shards_with_rank, key=lambda x: x[0])]
    return sorted_shards


def load_shard(path: Path, weights_only: bool) -> OrderedDict[str, object]:
    logger.info("Loading shard %s", path)
    state = torch.load(path, map_location="cpu", weights_only=weights_only)
    if not isinstance(state, OrderedDict):
        raise TypeError(f"Shard {path} did not contain an OrderedDict state_dict.")
    return state


def _materialize_dtensor(
    dtensor: DTensor,
) -> tuple[tuple[slice, ...], torch.Tensor] | None:
    """Return the global slice indices and CPU tensor for a DTensor shard."""
    local_tensor = dtensor.to_local()
    if local_tensor.numel() == 0:
        return None
    local_tensor = local_tensor.to("cpu")
    _, global_offset = dt_utils.compute_local_shape_and_global_offset(
        dtensor.shape, dtensor.device_mesh, dtensor.placements
    )
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(global_offset, local_tensor.shape, strict=False)
    )
    return slices, local_tensor


def assemble_full_state(
    shard_states: Sequence[OrderedDict[str, object]],
) -> OrderedDict[str, torch.Tensor]:
    logger.info("Reconstructing full state dict from %d shards", len(shard_states))
    full_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    keys = shard_states[0].keys()
    for key in keys:
        sample_value = shard_states[0][key]
        if isinstance(sample_value, DTensor):
            dtype = sample_value.to_local().dtype
            tensor = torch.empty(sample_value.shape, dtype=dtype, device="cpu")
            for shard_state in shard_states:
                dt_value = shard_state[key]
                materialized = _materialize_dtensor(dt_value)
                if materialized is None:
                    continue
                slices, local_tensor = materialized
                tensor[slices] = local_tensor
            full_state[key] = tensor
        elif torch.is_tensor(sample_value):
            full_state[key] = sample_value.to("cpu")
        else:
            raise TypeError(
                f"Unsupported state type for key '{key}': {type(sample_value)}"
            )
    logger.info("Full state dict contains %d tensors", len(full_state))
    return full_state


def ensure_token(token_arg: str | None) -> str:
    token = token_arg or os.environ.get("HUGGINGFACE_HUB_TOKEN") or HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. "
            "Pass --token, set HUGGINGFACE_HUB_TOKEN, or run `huggingface-cli login`."
        )
    return token


def save_hf_artifacts(
    model,
    processor,
    output_dir: Path,
    safe_serialization: bool,
    max_shard_size: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s", output_dir)
    model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )
    logger.info("Saving processor to %s", output_dir)
    processor.save_pretrained(output_dir)


def upload_to_hub(
    folder: Path,
    repo_id: str,
    token: str,
    private: bool,
    commit_message: str,
    branch: str | None,
) -> None:
    api = HfApi(token=token)
    logger.info("Ensuring Hub repo %s exists (private=%s)", repo_id, private)
    api.create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    logger.info("Uploading %s to the Hub...", folder)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=commit_message,
        token=token,
        revision=branch,
    )
    logger.info("Upload complete. View the model at https://huggingface.co/%s", repo_id)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=getattr(logging, args.log_level),
    )
    checkpoint_dir = args.checkpoint_dir
    processor_name = args.processor_name or args.model_name

    shard_paths = find_model_shards(checkpoint_dir)
    shard_states = [load_shard(path, args.weights_only) for path in shard_paths]
    full_state = assemble_full_state(shard_states)
    del shard_states

    logger.info("Instantiating base model %s", args.model_name)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    missing, unexpected = model.load_state_dict(full_state, strict=args.strict_load)
    if missing:
        logger.warning("Missing keys when loading state dict: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading state dict: %s", unexpected)

    processor = AutoImageProcessor.from_pretrained(processor_name)

    output_dir = args.output_dir or (checkpoint_dir / "hf_export")
    save_hf_artifacts(
        model=model,
        processor=processor,
        output_dir=output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )

    if args.skip_upload:
        logger.info("Skipping upload as requested. Artifacts saved to %s", output_dir)
        return

    token = ensure_token(args.token)
    upload_to_hub(
        folder=output_dir,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
        branch=args.branch,
    )


if __name__ == "__main__":
    main()
