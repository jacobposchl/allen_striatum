"""NuCLR construction and trusted local checkpoint compatibility."""
import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.nuclr import NuCLR


def checkpoint_run_dir(checkpoint):
    """Support both run/checkpoint.pt and run/ckpt/epoch_NNNN.pt layouts."""
    parent = Path(checkpoint).expanduser().parent
    return parent.parent if parent.name == "ckpt" else parent


def model_settings(config):
    cfg = OmegaConf.create(config) if not OmegaConf.is_config(config) else config
    model_cfg = dict(OmegaConf.to_container(cfg.model, resolve=True))
    if "backbone" in cfg:
        if not str(model_cfg.get("_target_", "")).endswith(".VanillaNuCLR"):
            raise ValueError("only Bottleneck VanillaNuCLR checkpoints are supported")
        model_cfg = dict(OmegaConf.to_container(cfg.backbone, resolve=True))
        model_cfg.pop("_target_", None)
    time_bins = float(cfg.sampling.duration) / float(cfg.sampling.bin_size)
    if abs(time_bins - round(time_bins)) > 1e-6:
        raise ValueError("sampling.duration must be divisible by sampling.bin_size")
    model_cfg["time_bins"] = int(round(time_bins))
    return {"bins_per_latent": 10, "dim": 48, "heads": 4, "dropout": 0.25, **model_cfg}


def build_model(config):
    cfg = OmegaConf.create(config) if not OmegaConf.is_config(config) else config
    model = NuCLR(**model_settings(cfg))
    # Runtime layout option; old checkpoints retain their original padded path.
    model.compact_temporal = bool(cfg.get("compact_temporal", False))
    return model


def load_checkpoint(checkpoint):
    # Load only trusted local checkpoints: optimizer/config state requires pickle.
    saved = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not {"model", "config"} <= saved.keys():
        raise ValueError("Expected a NuCLR checkpoint containing model weights and config")
    return saved


def model_state(saved):
    """Strip the wrapper used by the original Bottleneck VanillaNuCLR runs."""
    return {key.removeprefix("backbone."): value for key, value in saved["model"].items()}


def validate_initialization(config, pretrained_config):
    """Fine-tuning changes the data/optimizer, not the meaning of input bins."""
    target, source = model_settings(config), model_settings(pretrained_config)
    for key in ("time_bins", "bins_per_latent", "dim", "heads"):
        if target[key] != source[key]:
            raise ValueError(f"Fine-tuning model.{key}={target[key]} differs from checkpoint {source[key]}; "
                             "choose the checkpoint's model configuration")
    for key in ("duration", "bin_size"):
        if float(config["sampling"][key]) != float(pretrained_config["sampling"][key]):
            raise ValueError(f"Fine-tuning sampling.{key} must match the pretrained checkpoint")


def checkpoint_signature(checkpoint):
    """Identify an exact checkpoint even after it has been relocated."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(checkpoint):
    return {"encoder_checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "encoder_signature": checkpoint_signature(checkpoint)}


def load_model(checkpoint, device="cpu"):
    saved = load_checkpoint(checkpoint)
    config = saved["config"]
    model = build_model(config)
    model.load_state_dict(model_state(saved), strict=True)
    model.attention_chunk_size = config.get("attention_chunk_size", 8)
    return model.to(device).eval(), config, saved
