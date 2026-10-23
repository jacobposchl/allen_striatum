"""Export per-window NuCLR representations with their data and encoder provenance."""
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.io import resolve
from src.data.features import recording_features
from src.training.sampling import dense_window, session_starts


def _write(path, name, values, frame, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {}
    for key in frame:
        value = frame[key].to_numpy()
        columns[key] = value.astype(str) if value.dtype.kind == "O" else value
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **{name: values.astype(np.float32, copy=False)}, **columns, **metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


@torch.inference_mode()
def export_embeddings(model, dataset, keys, device, duration, bin_size, output,
                      batch_size=1, save_latents=False, max_windows=None, provenance=None):
    """Stream per-window trained embeddings and optional raw temporal latents."""
    if batch_size < 1 or (max_windows is not None and max_windows < 1):
        raise ValueError("export batch_size and max_windows must be positive")
    model.eval()
    output = Path(output)
    pooled_paths, latent_paths = [], []
    recording_windows = {}
    if duration <= 0 or bin_size <= 0 or abs(duration / bin_size - round(duration / bin_size)) > 1e-6:
        raise ValueError("Export duration must be a positive integer multiple of bin_size")
    n_bins = int(round(duration / bin_size))
    if n_bins != model.time_bins:
        raise ValueError("Export window bin count differs from the checkpoint model")
    provenance = dict(provenance or {})
    for animal, recording in keys:
        session = dataset.load_session(animal, recording)
        try:
            starts = session_starts(session, duration)
            if max_windows is not None:
                starts = starts[:max_windows]
            session_offset = session.metadata.get("session_offset_s")
            recording_windows[f"{animal}/{recording}"] = {
                "start_source_s": (starts + session.origin_s).tolist(),
                "start_session_s": ((starts + session_offset).tolist() if session_offset is not None else [None] * len(starts)),
                "duration_s": float(duration),
                "source_identity": session.recording.identity,
            }
            units = np.arange(session.n_units)
            frame = recording_features(session.recording, session.units, session.source_intervals).reset_index(drop=True)
            frame["unit_index"] = units
            frame["cutoff_session_s"] = session_offset + session.t_end if session_offset is not None else np.nan
            observations = np.zeros(session.n_units, dtype=np.int64)
            for offset in range(0, len(starts), batch_size):
                batch_starts = starts[offset:offset + batch_size]
                windows = []
                present = np.zeros((len(batch_starts), session.n_units), dtype=bool)
                spike_counts = np.zeros((len(batch_starts), session.n_units), dtype=np.int64)
                for index, start in enumerate(batch_starts):
                    active, values = session.make_view(float(start), duration, bin_size)
                    present[index, active] = True
                    spike_counts[index, active] = values.sum(axis=1, dtype=np.float64).astype(np.int64)
                    windows.append(dense_window(active, values, units, n_bins))
                if not present.any():
                    continue
                out = model(torch.from_numpy(np.stack(windows)).to(device))
                pooled = out.embedding.float().cpu().numpy()
                observations += present.sum(axis=0)
                window_rows, unit_rows = np.nonzero(present)
                window_frame = frame.iloc[unit_rows].reset_index(drop=True)
                window_frame["view_index"] = offset + window_rows
                window_frame["view_start_s"] = batch_starts[window_rows]
                window_frame["view_start_session_s"] = batch_starts[window_rows] + (session_offset if session_offset is not None else np.nan)
                window_frame["view_start_source_s"] = batch_starts[window_rows] + session.origin_s
                window_frame["view_duration_s"] = duration
                pooled_frame = window_frame.copy()
                pooled_frame["window_spike_count"] = spike_counts[present]
                pooled_frame["window_firing_rate_hz"] = spike_counts[present] / duration
                pooled_paths.append(_write(
                    output / "pooled" / animal / recording / f"batch_{offset:06d}.npz",
                    "pooled", pooled[present], pooled_frame,
                    pooled_representation="trained_temporal_pool_per_window",
                    **{k: provenance[k] for k in ("encoder_checkpoint", "encoder_signature") if k in provenance},
                ))
                if save_latents:
                    keep = torch.from_numpy(present).to(out.latents.device)
                    latent_paths.append(_write(
                        output / "latent" / animal / recording / f"batch_{offset:06d}.npz",
                        "latent", out.latents[keep].float().cpu().numpy(), window_frame,
                        latent_representation="raw_temporal_after",
                    ))
                del out, pooled, windows
            observed = observations > 0
            print(f"exported {animal}/{recording}: {observed.sum()} units, "
                  f"{observations.sum()} neuron-window embeddings from {len(starts)} windows", flush=True)
        finally:
            session.close()
    # List only this export's batches: reruns can leave older batches on disk.
    pooled_root = output / "pooled"
    pooled_root.mkdir(parents=True, exist_ok=True)
    manifest = pooled_root / "manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps({
            "files": [str(Path(path).relative_to(pooled_root)) for path in pooled_paths],
            "pooled_representation": "trained_temporal_pool_per_window",
            "recording_windows": recording_windows,
            **provenance,
        }, indent=2) + "\n")
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    exports = {"pooled": pooled_paths}
    if latent_paths:
        exports["latent"] = latent_paths
    return exports


def run(cfg):
    """Apply a checkpoint to its training data or an explicitly configured target cohort.

    Target data may intentionally differ (e.g. a fine-tuned encoder on passive
    labeled neurons). Only the checkpoint controls temporal input semantics.
    """
    from src.common.checkpoints import checkpoint_identity, load_model
    from src.data.loading import build_dataset

    if not cfg.get("checkpoint"):
        raise ValueError("export requires checkpoint=/path/to/checkpoint.pt")
    if not cfg.get("output"):
        raise ValueError("export requires output=/path/to/a/new/export/directory")
    threads = int(cfg.get("cpu_threads", 1))
    if threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(threads)
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = resolve(cfg.checkpoint)
    model, saved_cfg, saved = load_model(checkpoint, device=device)
    model.attention_chunk_size = cfg.get("attention_chunk_size", saved_cfg.get("attention_chunk_size", 8))
    data_cfg = cfg.get("data")
    using_training_data = data_cfg is None
    data_cfg = OmegaConf.create(saved_cfg["data"]) if using_training_data else data_cfg
    if cfg.get("data_root"):
        data_cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(data_cfg), resolve=True))
        data_cfg.root = str(resolve(cfg.data_root))
    cohort = cfg.get("cohort")
    if cohort is None and using_training_data:
        cohort = saved_cfg.get("cohort")
    dataset = build_dataset(data_cfg, cohort=cohort)
    if using_training_data and saved.get("data_fingerprint") != dataset.fingerprint:
        raise ValueError("Checkpoint training data changed; provide an explicit target data config to export a new cohort")
    keys = dataset.keys()
    if not keys:
        raise ValueError("No recordings selected for export")
    output = resolve(cfg.output)
    if (output / "pooled" / "manifest.json").exists():
        raise FileExistsError(f"Export already exists at {output}; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    data = OmegaConf.to_container(OmegaConf.create(data_cfg), resolve=True)
    data["root"] = str(dataset.root.resolve())
    cohort_dict = OmegaConf.to_container(OmegaConf.create(cohort), resolve=True) if cohort is not None else None
    provenance = {
        **checkpoint_identity(checkpoint),
        "data_fingerprint": dataset.fingerprint, "data": data, "cohort": cohort_dict,
        "recording_keys": [list(key) for key in keys],
        "sampling": dict(saved_cfg["sampling"]),
    }
    # Consumers need the target data configuration, not just the encoder's training data.
    OmegaConf.save(OmegaConf.create({**OmegaConf.to_container(cfg, resolve=True),
                                    **provenance}), output / "config.yaml", resolve=True)
    dataset.manifest.to_csv(output / "data_manifest.csv", index=False)
    exports = export_embeddings(
        model, dataset, keys, device,
        float(saved_cfg["sampling"]["duration"]), float(saved_cfg["sampling"]["bin_size"]), output,
        batch_size=int(cfg.get("batch_size", 1)), save_latents=bool(cfg.get("save_latents", False)),
        max_windows=cfg.get("max_windows"), provenance=provenance,
    )
    summary = {"output": str(output), "recordings": len(keys),
               "pooled_batches": len(exports["pooled"]), **provenance}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
