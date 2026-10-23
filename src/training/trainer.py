"""Shared NuCLR engine for fresh pretraining, fine-tuning, and exact resumption."""
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import random
from time import perf_counter

from src.data.io import ROOT
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from src.data.dataset import BrainsetsDataset
from src.data.loading import PairDataset, make_loader
from src.training.contrastive import NuCLRLoss
from src.embeddings.export import export_embeddings
from src.common.checkpoints import (
    build_model, checkpoint_identity, checkpoint_run_dir, load_checkpoint,
    model_state, validate_initialization,
)
from src.common.logging import RunLog, StageTimer


class NuCLRTrainer:
    def __init__(self, cfg):
        self.cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        self.root = ROOT
        self.history = []
        self.start_epoch = 0
        self.saved = None
        self.initialization = None
        stage = self.cfg.get("stage", "pretrain")
        if stage not in ("pretrain", "finetune"):
            raise ValueError("stage must be pretrain or finetune")
        if self.cfg.get("resume") and self.cfg.get("checkpoint"):
            raise ValueError("resume and checkpoint are mutually exclusive: resume continues a run; checkpoint initializes a new one")
        runtime_changes = {}
        if self.cfg.get("resume"):
            self.resume_path = self.path(self.cfg.resume)
            self.saved = load_checkpoint(self.resume_path)
            saved_cfg = self.saved["config"]
            if "optimizer" not in self.saved:
                raise ValueError("This checkpoint supports inference only; resuming needs optimizer state")
            # Restore training semantics; runtime options and the desired epoch count remain configurable.
            for key in ("precision", "activation_checkpointing", "attention_chunk_size", "compact_temporal"):
                current = self.cfg.get(key, False)
                if saved_cfg.get(key, False) != current:
                    runtime_changes[key] = {"saved": saved_cfg.get(key, False), "current": current}
            for key in ("data", "model", "sampling", "loss", "optimizer", "seed"):
                self.cfg[key] = saved_cfg[key]
            if "backbone" in saved_cfg:
                self.cfg["backbone"] = saved_cfg["backbone"]
            self.cfg.cohort = saved_cfg.get("cohort")
            self.cfg.stage = saved_cfg.get("stage", "pretrain")
            # A moved HDF5 directory changes location, not selection semantics.
            if self.cfg.get("data_root"):
                self.cfg.data.root = str(self.path(self.cfg.data_root))
        elif stage == "finetune":
            if not self.cfg.get("checkpoint"):
                raise ValueError("Fine-tuning requires checkpoint=/path/to/pretrained/checkpoint.pt")
            self.cfg.checkpoint = str(self.path(self.cfg.checkpoint).resolve())
            self.initialization = load_checkpoint(self.cfg.checkpoint)
            validate_initialization(self.cfg, self.initialization["config"])
        elif self.cfg.get("checkpoint"):
            raise ValueError("Use stage=finetune to initialize from a checkpoint")
        self.validate_config()
        torch.set_num_threads(int(self.cfg.cpu_threads))
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        name = self.cfg.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(name)
        if self.cfg.precision == "float16" and self.device.type != "cuda":
            raise ValueError("float16 training requires CUDA; use float32 or bfloat16 on CPU")
        self.dtype = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.cfg.precision]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and self.dtype == torch.float16)
        self.model = build_model(self.cfg).to(self.device)
        if self.initialization is not None:
            self.model.load_state_dict(model_state(self.initialization), strict=True)
            # Retain no optimizer or RNG state from the source experiment.
            self.initialization = None
        self.model.activation_checkpointing = bool(self.cfg.activation_checkpointing)
        self.model.attention_chunk_size = self.cfg.attention_chunk_size
        self.objective = NuCLRLoss(**self.cfg.loss)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.optimizer.learning_rate,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        self.dataset = BrainsetsDataset(self.cfg.data, cohort=self.cfg.get("cohort"))
        # Absolute data paths make saved checkpoints usable from other working directories.
        self.cfg.data.root = str(self.dataset.root.resolve())
        self.keys = self.dataset.keys()
        if not self.keys:
            raise ValueError("The configured cohort selects no eligible Brainsets recordings")
        self.train_data = PairDataset(self.dataset, self.cfg.sampling, True,
                                      max_open_sessions=self.cfg.get("max_open_sessions", 1))
        if self.saved is not None:
            self.output_dir = checkpoint_run_dir(self.resume_path)
            self.restore()
        else:
            stamp = datetime.now().strftime("%Y-%m-%d/%H%M%S_%f")
            run_name = self.cfg.run_name or f"nuclr_seed{self.cfg.seed}"
            self.output_dir = self.path(self.cfg.output_root) / stamp / run_name
            self.output_dir.mkdir(parents=True, exist_ok=False)
        self.dataset.manifest.to_csv(self.output_dir / "data_manifest.csv", index=False)
        (self.output_dir / "cohort.json").write_text(json.dumps({
            "recording_keys": [list(key) for key in self.keys],
            "data_fingerprint": self.dataset.fingerprint,
        }, indent=2) + "\n")
        OmegaConf.save(self.cfg, self.output_dir / "config.yaml", resolve=True)
        self.log = RunLog(self.output_dir)
        self.log.info(f"attempt={self.log.attempt} device={self.device} results_dir={self.output_dir}")
        self.log.info(f"host={platform.node()} torch={torch.__version__} numpy={np.__version__} "
                      f"cpu_affinity={len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.log.info(f"gpu={props.name} total_vram={props.total_memory / 1024**3:.2f}GiB "
                          f"cuda={torch.version.cuda}")
        self.log.info(f"stage={self.cfg.get('stage', 'pretrain')} recordings={len(self.keys)} batch_size={self.cfg.batch_size} "
                      f"workers={self.cfg.num_workers} cpu_threads={self.cfg.cpu_threads} "
                      f"precision={self.cfg.precision} attention_chunk_size={self.cfg.attention_chunk_size} "
                      f"compact_temporal={self.model.compact_temporal} "
                      f"activation_checkpointing={self.cfg.activation_checkpointing}")
        if runtime_changes:
            self.log.info(f"resumed with changed runtime settings (not bitwise continuation): {runtime_changes}")

    def path(self, value):
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else self.root / path

    def validate_config(self):
        cfg = self.cfg
        if cfg.data.get("format") != "brainsets":
            raise ValueError("Training requires data.format=brainsets and a processed HDF5 data.root")
        for key in ("epochs", "batch_size", "cpu_threads", "export.batch_size", "attention_chunk_size"):
            value = OmegaConf.select(cfg, key)
            if value < 1 or int(value) != value:
                raise ValueError(f"{key} must be a positive integer")
        if cfg.num_workers < 0 or int(cfg.num_workers) != cfg.num_workers:
            raise ValueError("num_workers must be nonnegative")
        for key in ("prefetch_factor", "max_open_sessions"):
            value = cfg.get(key, 1)
            if value < 1 or int(value) != value:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("max_steps_per_epoch", "export.max_windows"):
            value = OmegaConf.select(cfg, key)
            if value is not None and (value < 1 or int(value) != value):
                raise ValueError(f"{key} must be null or a positive integer")
        interval = cfg.get("checkpoint_every", 1)
        if isinstance(interval, bool) or interval < 1 or int(interval) != interval:
            raise ValueError("checkpoint_every must be a positive integer")
        if cfg.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("precision must be float32, float16, or bfloat16")
        sampling = cfg.sampling
        if sampling.duration <= 0 or sampling.bin_size <= 0:
            raise ValueError("sampling duration and bin_size must be positive")
        if not 0 <= sampling.min_distance <= sampling.max_distance:
            raise ValueError("sampling distances must satisfy 0 <= min_distance <= max_distance")
        if not 0 < sampling.min_unit_fraction <= 1:
            raise ValueError("min_unit_fraction must be in (0, 1]")
        if cfg.optimizer.learning_rate <= 0 or cfg.optimizer.weight_decay < 0 or cfg.optimizer.grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")

    def autocast(self):
        return nullcontext() if self.dtype is None else torch.autocast(self.device.type, dtype=self.dtype)

    def memory_stats(self):
        if not self.cfg.get("log_vram", False) or self.device.type != "cuda":
            return {}
        gib = 1024**3
        free, total = torch.cuda.mem_get_info(self.device)
        return dict(vram_gib=torch.cuda.memory_allocated(self.device) / gib,
                    peak_vram_gib=torch.cuda.max_memory_allocated(self.device) / gib,
                    reserved_vram_gib=torch.cuda.memory_reserved(self.device) / gib,
                    free_vram_gib=free / gib, total_vram_gib=total / gib)

    def run_epoch(self, dataset, indices, training=False, epoch=0):
        if not indices:
            raise ValueError("no usable paired windows; check duration and unit counts")
        self.model.train(training)
        loader = make_loader(
            dataset, indices, self.cfg.batch_size, self.cfg.num_workers,
            self.cfg.pin_memory and self.device.type == "cuda", self.cfg.seed + epoch,
            prefetch_factor=self.cfg.get("prefetch_factor", 1),
        )
        totals, weight, steps = {}, 0, 0
        step_count = min(len(loader), self.cfg.max_steps_per_epoch or len(loader)) if training else len(loader)
        timing_enabled = self.cfg.get("log_timing", True)
        synchronize = (lambda: torch.cuda.synchronize(self.device)) if self.device.type == "cuda" else None
        if timing_enabled and synchronize:
            synchronize()
        epoch_start = perf_counter()
        timer = StageTimer(synchronize) if timing_enabled else None
        timing_totals = dict(load_s=0.0, transfer_s=0.0, forward_loss_s=0.0, backward_update_s=0.0)
        pairs_seen = 0
        self.log.info(f"epoch={epoch} starting loader; steps={step_count} (first load includes worker startup)")
        for step, batch in enumerate(loader, 1):
            load_s = timer.mark() if timer else None
            batch_pairs = batch["window_a"].shape[0]
            units_a, units_b = batch["window_a"].shape[1], batch["window_b"].shape[1]
            real_units = int(batch["unit_mask_a"].sum()) + int(batch["unit_mask_b"].sum())
            padding_fraction = 1 - real_units / (batch_pairs * (units_a + units_b))
            if training and self.cfg.get("log_vram", False) and self.device.type == "cuda":
                # Include transfer, forward, backward, and the optimizer in this step's peak.
                torch.cuda.reset_peak_memory_stats(self.device)
            batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
            transfer_s = timer.mark() if timer else None
            with torch.set_grad_enabled(training):
                with self.autocast():
                    first = self.model(batch["window_a"], batch["unit_mask_a"])
                    second = self.model(batch["window_b"], batch["unit_mask_b"])
                loss, metrics = self.objective(first, second, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                forward_loss_s = timer.mark() if timer else None
                if training and metrics["_weight"]:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            backward_update_s = timer.mark() if timer else None
            count = metrics.pop("_weight")
            weight += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
            steps += 1
            pairs_seen += batch_pairs
            timings = dict(load_s=load_s, transfer_s=transfer_s, forward_loss_s=forward_loss_s,
                           backward_update_s=backward_update_s)
            memory = self.memory_stats()
            elapsed = perf_counter() - epoch_start
            record = dict(epoch=epoch, step=step, steps=step_count, training=training,
                          batch_pairs=batch_pairs, anchors=count, units_a=units_a, units_b=units_b,
                          padding_fraction=padding_fraction, **metrics, **memory)
            compact_temporal = bool(self.cfg.get("compact_temporal", False))
            record.update(compact_temporal=compact_temporal,
                          temporal_neuron_sequences=real_units if compact_temporal else batch_pairs * (units_a + units_b))
            timing_text = ""
            if timer:
                step_s = sum(timings.values())
                for key, value in timings.items():
                    timing_totals[key] += value
                record.update(timings, step_s=step_s, pairs_per_s=batch_pairs / max(step_s, 1e-9),
                              epoch_elapsed_s=elapsed, epoch_eta_s=elapsed / steps * (step_count - steps))
                timing_text = (f" load={load_s:.2f}s transfer={transfer_s:.2f}s "
                               f"forward_loss={forward_loss_s:.2f}s backward_update={backward_update_s:.2f}s "
                               f"step={step_s:.2f}s pairs/s={record['pairs_per_s']:.2f} "
                               f"eta={record['epoch_eta_s'] / 60:.1f}min")
            self.log.step(record)
            if training:
                memory_text = (f" peak={memory['peak_vram_gib']:.2f}GiB "
                               f"reserved={memory['reserved_vram_gib']:.2f}GiB "
                               f"free={memory['free_vram_gib']:.2f}GiB") if memory else ""
                self.log.info(f"epoch={epoch} step={step}/{step_count} loss={metrics['loss']:.4f}"
                              f"{timing_text} padding={padding_fraction:.1%}{memory_text}")
            # Release the completed batch before measuring/allocating the next one.
            del first, second, loss, batch
            if training and self.cfg.max_steps_per_epoch and step >= self.cfg.max_steps_per_epoch:
                break
            timer = StageTimer(synchronize) if timing_enabled else None
        if weight == 0:
            raise ValueError("no valid contrastive anchors with negatives in this epoch")
        performance = ({**timing_totals, "epoch_s": perf_counter() - epoch_start,
                        "pairs": pairs_seen} if timing_enabled else {})
        return {**{key: value / weight for key, value in totals.items()}, "anchors": weight,
                "steps": steps, **performance}

    def train(self):
        try:
            return self._train()
        except BaseException:
            self.log.exception("Training interrupted or failed")
            raise
        finally:
            self.train_data.close()
            self.log.close()

    def _train(self):
        if self.cfg.epochs <= self.start_epoch:
            raise ValueError(f"epochs must exceed the completed epoch count ({self.start_epoch})")
        for epoch in range(self.start_epoch + 1, self.cfg.epochs + 1):
            if self.checkpoint_path(epoch).exists():
                raise FileExistsError(f"Checkpoint already exists: {self.checkpoint_path(epoch)}. "
                                      "Resume from the latest saved epoch or use a separate run directory.")
        for epoch in range(self.start_epoch + 1, self.cfg.epochs + 1):
            started = perf_counter()
            indices = self.train_data.indices(self.keys, self.cfg.seed + 1000 + epoch)
            self.log.info(f"epoch={epoch} paired_windows={len(indices)} indexing_s={perf_counter() - started:.2f}")
            train = self.run_epoch(self.train_data, indices, training=True, epoch=epoch)
            self.history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}})
            if epoch % int(self.cfg.get("checkpoint_every", 1)) == 0 or epoch == self.cfg.epochs:
                started = perf_counter()
                saved_path = self.save(epoch)
                self.log.info(f"epoch={epoch} checkpoint={saved_path} saved in {perf_counter() - started:.2f}s")
            pd.DataFrame(self.history).to_csv(self.output_dir / "training_history.csv", index=False)
            self.log.info(f"epoch={epoch} train_loss={train['loss']:.4f}")
        self.train_data.close()
        return self.finish()

    def checkpoint_path(self, epoch):
        return self.output_dir / "ckpt" / f"epoch_{epoch:04d}.pt"

    def save(self, epoch):
        path = self.checkpoint_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite saved checkpoint: {path}")
        payload = {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(), "config": OmegaConf.to_container(self.cfg, resolve=True),
            "epoch": epoch, "history": self.history, "data_fingerprint": self.dataset.fingerprint,
            "recording_keys": [list(key) for key in self.keys],
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }
        temporary = path.with_suffix(".pt.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        # Keep the established latest-checkpoint path without duplicating weights.
        latest = self.output_dir / "checkpoint.pt"
        temporary_link = latest.with_suffix(".pt.link.tmp")
        temporary_link.unlink(missing_ok=True)
        try:
            temporary_link.symlink_to(path.relative_to(self.output_dir))
            temporary_link.replace(latest)
        finally:
            temporary_link.unlink(missing_ok=True)
        return path

    def restore(self):
        saved = self.saved
        if saved["data_fingerprint"] != self.dataset.fingerprint:
            raise ValueError("Brainsets recording contents or selection settings changed since checkpoint creation")
        expected_keys = saved.get("recording_keys")
        if expected_keys is None:
            raise ValueError("Resume requires explicit recording identities saved in the checkpoint")
        if {tuple(key) for key in expected_keys} != set(self.keys):
            raise ValueError("Brainsets cohort differs from the checkpoint's selected recordings")
        self.model.load_state_dict(model_state(saved), strict=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.scaler.load_state_dict(saved["scaler"])
        self.start_epoch, self.history = saved["epoch"], saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if saved["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        self.saved = None

    def finish(self):
        selected = self.checkpoint_path(self.history[-1]["epoch"])
        exports = {}
        if self.cfg.export.enabled:
            self.log.info("Starting embedding export")
            exports = export_embeddings(
                self.model, self.dataset, self.keys, self.device,
                self.cfg.sampling.duration, self.cfg.sampling.bin_size,
                self.output_dir / "embeddings", batch_size=self.cfg.export.batch_size,
                save_latents=self.cfg.export.save_latents, max_windows=self.cfg.export.max_windows,
                provenance={
                    **checkpoint_identity(selected),
                    "data_fingerprint": self.dataset.fingerprint,
                    "data": OmegaConf.to_container(self.cfg.data, resolve=True),
                    "cohort": OmegaConf.to_container(self.cfg.cohort, resolve=True) if self.cfg.get("cohort") else None,
                    "recording_keys": [list(key) for key in self.keys],
                    "sampling": OmegaConf.to_container(self.cfg.sampling, resolve=True),
                },
            )
        summary = {
            "device": str(self.device), "completed_epochs": self.history[-1]["epoch"],
            "selected_checkpoint": str(selected), "recordings": len(self.keys), "embeddings": exports,
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        self.log.info(f"finished: {self.output_dir / 'summary.json'}")
        return summary
