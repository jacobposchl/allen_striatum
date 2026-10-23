"""Read exported embeddings independently of the source HDF5 recordings.

One mean vector is accumulated per original neuron. Only exported active
windows contribute, matching the historical NuCLR export convention.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import resolve
from src.data.cohort import filter_frame


IDENTITY = ("global_unit_id", "animal", "session", "recording")


def filter_cohort(frame, cohort=None):
    return filter_frame(frame, cohort, require_all=False, require_nonempty=False)


def read_table(path):
    return pd.read_csv(resolve(path), dtype={name: str for name in (*IDENTITY, "unit_id", "source_unit_row")})


def export_manifest(path):
    path = resolve(path)
    if path.is_dir():
        path = path / "manifest.json" if (path / "manifest.json").exists() else path / "pooled/manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No embedding manifest: {path}. Export embeddings first.")
    data = json.loads(path.read_text())
    if data.get("pooled_representation") != "trained_temporal_pool_per_window":
        raise ValueError("Expected trained per-window pooled NuCLR embeddings")
    return path, data


def provenance(path, manifest):
    result = {key: manifest.get(key) for key in ("encoder_signature", "encoder_checkpoint", "data_fingerprint", "data", "cohort")}
    result.update(export_manifest=str(path), export_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  pooling="mean of available active-window embeddings")
    config_path = next((p / "config.yaml" for p in path.parents if (p / "config.yaml").is_file()), None)
    if config_path:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        if result["data"] is None and cfg.get("data") is not None:
            result["data"] = OmegaConf.to_container(cfg.data, resolve=True)
        result["encoder_signature"] = result["encoder_signature"] or cfg.get("encoder_signature")
        result["encoder_checkpoint"] = result["encoder_checkpoint"] or cfg.get("encoder_checkpoint")
        result["data_fingerprint"] = result["data_fingerprint"] or cfg.get("data_fingerprint")
        summary = config_path.parent / "summary.json"
        if not result["encoder_checkpoint"] and summary.exists():
            result["encoder_checkpoint"] = json.loads(summary.read_text()).get("selected_checkpoint")
    if not result["encoder_signature"] and result["encoder_checkpoint"]:
        checkpoint = resolve(result["encoder_checkpoint"])
        if checkpoint.is_file():
            # This only hashes the file; it does not instantiate a model or load weights.
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result["encoder_signature"] = digest.hexdigest()
    return result


def load_means(path, cohort=None):
    path, manifest = export_manifest(path)
    sums, counts, metadata, spike_totals, durations = {}, {}, {}, {}, {}
    seen = set()
    dimension = None
    batch_signature = manifest.get("encoder_signature")
    signed_batches = None
    for relative in manifest.get("files", []):
        batch_path = (path.parent / relative).resolve()
        if not batch_path.is_relative_to(path.parent.resolve()) or relative in seen:
            raise ValueError("Invalid or duplicate embedding batch path in manifest")
        seen.add(relative)
        with np.load(batch_path, allow_pickle=False) as batch:
            signed = "encoder_signature" in batch
            if signed_batches is not None and signed != signed_batches:
                raise ValueError("Export mixes batches with and without encoder identities; re-export the complete dataset")
            signed_batches = signed
            if signed:
                signature = str(batch["encoder_signature"].item())
                if batch_signature and signature != batch_signature:
                    raise ValueError(f"Mixed encoder checkpoints in export: {batch_path}")
                batch_signature = signature
            elif manifest.get("encoder_signature"):
                raise ValueError(f"Export batch lacks its manifest's encoder identity: {batch_path}")
            vectors = batch["pooled"]
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError(f"Nonfinite or invalid embedding matrix: {batch_path}")
            if dimension is not None and vectors.shape[1] != dimension:
                raise ValueError("Embedding dimension varies across batches")
            dimension = vectors.shape[1]
            columns = {}
            for key in batch.files:
                if key == "pooled":
                    continue
                array = batch[key]
                if array.ndim == 1 and len(array) == len(vectors):
                    columns[key] = array
            frame = pd.DataFrame(columns)
            if not set(IDENTITY).issubset(frame.columns):
                raise ValueError(f"Missing neuron identity metadata in {batch_path}")
            frame = filter_cohort(frame, cohort)
            for position, row in frame.iterrows():
                key = str(row.global_unit_id)
                if key in metadata:
                    for field in (*IDENTITY[1:], "label", "unit_id", "source_unit_row", "decoder_label"):
                        if field in row and str(row[field]) != str(metadata[key].get(field)):
                            raise ValueError(f"Conflicting {field} for neuron {key}")
                    sums[key] += vectors[position]
                    counts[key] += 1
                else:
                    sums[key] = vectors[position].astype(np.float64).copy()
                    counts[key] = 1
                    metadata[key] = {k: v for k, v in row.items()
                                     if not k.startswith("view_") and k not in {"window_spike_count", "window_firing_rate_hz"}}
                    spike_totals[key], durations[key] = 0., 0.
                if "window_spike_count" in row and "view_duration_s" in row:
                    spike_totals[key] += float(row.window_spike_count)
                    durations[key] += float(row.view_duration_s)
    if not sums:
        raise ValueError("No exported neurons match the selected cohort")
    keys = sorted(sums)
    frame = pd.DataFrame([metadata[key] for key in keys])
    frame["n_windows"] = [counts[key] for key in keys]
    frame["embedding_window_rate_hz"] = [spike_totals[key] / durations[key] if durations[key] else np.nan for key in keys]
    values = np.stack([sums[key] / counts[key] for key in keys]).astype(np.float32)
    information = provenance(path, manifest)
    if batch_signature:
        if information.get("encoder_signature") and information["encoder_signature"] != batch_signature:
            raise ValueError("Export batch encoder differs from its saved configuration")
        information["encoder_signature"] = batch_signature
    return values, frame.reset_index(drop=True), information


def attach_table(frame, table, *, columns=None):
    """Join a neuron-level table while rejecting ambiguous IDs and identities."""
    if "global_unit_id" not in table or table.global_unit_id.isna().any() or table.global_unit_id.duplicated().any():
        raise ValueError("Joined table requires unique, nonmissing global_unit_id values")
    left, right = frame.set_index("global_unit_id"), table.set_index("global_unit_id")
    shared = left.index.intersection(right.index)
    for key in IDENTITY[1:]:
        if key in left and key in right:
            if not (left.loc[shared, key].astype(str) == right.loc[shared, key].astype(str)).all():
                raise ValueError(f"Neuron identity conflict in {key}")
    selected = columns if columns is not None else [k for k in right if k not in left]
    for name in selected:
        if name not in right:
            raise ValueError(f"Missing joined feature: {name}")
        left[name] = right[name].reindex(left.index)
    return left.reset_index()
