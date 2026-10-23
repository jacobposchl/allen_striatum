"""Same-window labeled-neighbor agreement on original per-window embeddings."""
from datetime import datetime, timezone
import json
import os
import socket

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from src.embeddings.io import provenance
from src.embeddings.windows import RecordingExport
from src.data.io import resolve


def nearest_labeled(vectors, labels, k):
    """Input rows are labeled units in ID order. Exclude self, not other units.

    Zero-length/nonfinite vectors cannot define a cosine neighbor and are omitted.
    A query must have exactly k available other candidates to receive a score.
    """
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > 0)
    positions = np.flatnonzero(valid)
    fraction = np.full(len(labels), np.nan)
    neighbors = np.full((len(labels), k), -1, dtype=np.int32)
    if len(positions) <= k:
        return fraction, neighbors, valid
    normalized = vectors[positions] / norms[positions, None]
    distance = 1. - normalized @ normalized.T
    np.fill_diagonal(distance, np.inf)
    # Stable sorting resolves exact distance ties by the caller's unit-ID order.
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :k]
    neighbors[positions] = positions[nearest]
    fraction[positions] = (labels[positions, None] == labels[positions[nearest]]).mean(axis=1)
    return fraction, neighbors, valid


def run(cfg):
    if int(cfg.k) < 1 or cfg.metric != "cosine":
        raise ValueError("Require k >= 1 and metric=cosine")
    classes = list(map(str, cfg.classes))
    if len(classes) != 2 or len(set(classes)) != 2:
        raise ValueError("The red/green comparison requires two distinct configured classes")
    output = resolve(cfg.output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    exported = RecordingExport(cfg.embeddings, f"{cfg.animal}/{cfg.session}/{cfg.recording}")
    path, manifest = exported.path, exported.manifest
    units, windows = {}, {}
    dimension = None
    print(f"Reading {len(exported.files)} batches for {exported.key}", flush=True)
    for view, batch in exported.windows():
        vectors = np.asarray(batch["pooled"], dtype=np.float64)
        if vectors.ndim != 2 or (dimension is not None and vectors.shape[1] != dimension):
            raise ValueError("Invalid or inconsistent embedding dimension")
        dimension = vectors.shape[1]
        names = ("global_unit_id", "unit_id", "source_unit_row", "label", "decoder_label",
                 "animal", "session", "recording")
        frame = pd.DataFrame({key: batch[key] for key in names})
        for key in ("animal", "session", "recording"):
            if not frame[key].astype(str).eq(str(cfg[key])).all():
                raise ValueError(f"Selected export has inconsistent {key}")
        labeled = frame.loc[frame.label.isin(classes)].sort_values("global_unit_id")
        for row in labeled.itertuples():
            identity = dict(global_unit_id=str(row.global_unit_id), unit_id=str(row.unit_id),
                            source_unit_row=int(row.source_unit_row), label=str(row.label),
                            decoder_label=str(row.decoder_label), animal=str(row.animal),
                            session=str(row.session), recording=str(row.recording))
            old = units.setdefault(identity["global_unit_id"], identity)
            if old != identity:
                raise ValueError("A unit's identity or class changes across windows")
        windows[view] = dict(ids=labeled.global_unit_id.astype(str).to_numpy(),
                             labels=labeled.label.astype(str).to_numpy(),
                             vectors=vectors[labeled.index.to_numpy()])
    if not units:
        raise ValueError("No labeled units in the selected recording")
    order = {label: index for index, label in enumerate(classes)}
    table = pd.DataFrame(sorted(units.values(), key=lambda row: (order[row["label"]], row["source_unit_row"])))
    row_index = {key: index for index, key in enumerate(table.global_unit_id)}
    view_indices = np.arange(len(exported.starts_source))
    duration = exported.duration
    starts = exported.starts_session
    if not np.isfinite(starts).all():
        raise ValueError("Session-clock alignment is unavailable for this recording")
    agreement = np.full((len(table), len(view_indices)), np.nan)
    present = np.zeros_like(agreement, dtype=bool)
    vector_valid = np.zeros_like(present)
    neighbor_rows = np.full((*agreement.shape, int(cfg.k)), -1, dtype=np.int32)
    eligible_counts = np.zeros(len(view_indices), dtype=int)
    with threadpool_limits(limits=int(cfg.cpu_threads)):
        for column, view in enumerate(view_indices):
            if int(view) not in windows:
                continue
            window = windows[int(view)]
            rows = np.array([row_index[key] for key in window["ids"]], dtype=int)
            score, nearest, valid = nearest_labeled(window["vectors"], window["labels"], int(cfg.k))
            agreement[rows, column] = score
            present[rows, column] = True
            vector_valid[rows, column] = valid
            eligible_counts[column] = int(valid.sum())
            computed = np.isfinite(score)
            neighbor_rows[rows[computed], column] = rows[nearest[computed]]
    information = dict(**provenance(path, manifest), animal=str(cfg.animal), session=str(cfg.session),
                       recording=str(cfg.recording), recording_source_identity=exported.source_identity,
                       window_schedule="All scheduled windows; excluded gaps keep their actual time coordinates", created_utc=datetime.now(timezone.utc).isoformat(),
                       hostname=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                       dimensions=dimension, k=int(cfg.k), metric="cosine", classes=classes,
                       n_labeled_units=len(table), n_windows=len(view_indices), window_duration_s=duration,
                       class_counts=table.label.value_counts().to_dict(),
                       sorting_counts=table.decoder_label.value_counts().to_dict(),
                       neighbor_pool="Other labeled units in the same recording and same 10-second window; self excluded",
                       score="Number of k nearest neighbors with the query's class, divided by k; no class-balance correction",
                       missing="Query absent or invalid, or fewer than k valid other labeled units",
                       ties="Exact distance ties resolved by sorted original global unit ID",
                       row_order="Class, then original source-unit row; fixed for all windows",
                       finite_cells=int(np.isfinite(agreement).sum()),
                       absent_cells=int((~present).sum()), invalid_vector_cells=int((present & ~vector_valid).sum()),
                       insufficient_neighbor_cells=int((vector_valid & ~np.isfinite(agreement)).sum()))
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml")
    table.to_csv(output / "units.csv", index=False)
    pd.DataFrame(dict(view_index=view_indices, start_session_s=starts, duration_s=duration,
                      n_valid_labeled_units=eligible_counts)).to_csv(output / "windows.csv", index=False)
    pd.DataFrame(agreement, index=pd.Index(table.global_unit_id, name="global_unit_id"),
                 columns=starts).to_csv(output / "same_class_fraction.csv")
    np.savez_compressed(output / "neighbors.npz", same_class_fraction=agreement,
                        global_unit_id=table.global_unit_id.to_numpy(str), labels=table.label.to_numpy(str),
                        start_session_s=starts, duration_s=duration, view_index=view_indices,
                        present=present, vector_valid=vector_valid, neighbor_row_indices=neighbor_rows,
                        eligible_labeled_counts=eligible_counts)
    (output / "metadata.json").write_text(json.dumps(information, indent=2, allow_nan=False) + "\n")
    from src.experiments.neighbors.plots.temporal_neighbors import plot
    plot(agreement, table, starts, duration, information, output / "labeled_neighbors_over_time.png")
    print(json.dumps({key: information[key] for key in ("n_labeled_units", "n_windows", "class_counts", "sorting_counts", "finite_cells", "absent_cells", "invalid_vector_cells", "insufficient_neighbor_cells")}), flush=True)
    print(output / "labeled_neighbors_over_time.png", flush=True)
    return output
