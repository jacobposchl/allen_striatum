# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["brainsets @ git+https://github.com/neuro-galaxy/brainsets@d8e4b87bfd26425f805fbb23a4df22b47b99cece", "temporaldata==0.1.6", "numpy>=1.26,<3", "pandas>=2.2,<3", "h5py>=3.10,<4", "zarr>=2.18,<3", "hdmf-zarr==0.12.0", "pynwb==3.1.3", "hdmf==4.3.1"]
# ///
"""Brainsets pipeline for the active Allen dynamic-foraging recordings.

Retains all source data. Runtime readers use canonical Brainsets fields only.
The Python entry point also supports inventory, export, and validate commands.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainsets.pipeline import BrainsetPipeline
import h5py
import numpy as np
import pandas as pd

from src.data.schema import SCHEMA_VERSION
from src.data.io import require_compute
from preprocess.allen_striatum_active.source_io import SourceArray, SourceGroup, open_source
from preprocess.allen_striatum_active.utils import (
    blocks, copy_tree, data_group, discover, hash_update, sidecars, source_identity,
)


class Pipeline(BrainsetPipeline):
    brainset_id = "allen_striatum_active"

    @classmethod
    def get_manifest(cls, raw_dir, args):
        return pd.DataFrame(discover(raw_dir)).set_index("id", drop=False)

    def download(self, manifest_item):
        return {k: v for k, v in manifest_item._asdict().items() if k != "Index"}

    def process(self, download_output):
        summary = export_recording(download_output, self.processed_dir)
        validate_export(summary["output"])
        return summary


def implementation_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for name in ("src/data/schema.py", "src/data/features.py"):
        digest.update((Path(__file__).resolve().parents[2] / name).read_bytes())
    return digest.hexdigest()


def clock_metadata(item, source, metadata):
    from preprocess.allen_striatum_active.source_timing import recording_clock, recording_numbers, place_recording, session_timing
    path, asset = Path(item["source"]), Path(item["asset"])
    info = dict(item, numbers=recording_numbers(path))
    try:
        info["clock"] = recording_clock(asset, path, source)
    except (ValueError, KeyError, FileNotFoundError) as error:
        info["clock"] = {"unavailable_reason": str(error)}
    timing = session_timing(metadata.get("session.json", {}), 0) if metadata.get("session.json", {}).get("session_start_time") else None
    info["session_timing"] = timing
    info["session_offset_s"] = None
    info["placement_method"] = "unavailable"
    if "origin_s" in info["clock"]:
        if info["numbers"] == (1, 1):
            info.update(session_offset_s=0.0, placement_method="session_start_anchor")
        elif timing:
            candidates = [p for p in [*asset.glob("*.nwb"), *(asset / "nwb").glob("*.nwb")]
                          if recording_numbers(p) == (1, 1)]
            if len(candidates) == 1:
                try:
                    primary = recording_clock(asset, candidates[0], open_source(candidates[0]))
                    offset, method = place_recording(info["clock"], primary, info["numbers"], timing)
                    info.update(session_offset_s=offset, placement_method=method)
                except (ValueError, KeyError, FileNotFoundError) as error:
                    info["placement_method"] = str(error)
    return info


def availability(source):
    paths = {
        "units": "units/id", "trials": "intervals/trials/id",
        "cue": "intervals/trials/goCue_start_time", "source_default_qc": "units/default_qc",
        "anatomy": "units/ccf_acronym", "cell_type": "units/predicted_cell_type",
        "waveforms": "units/waveform_mean", "opto": "stimulus/presentation/Opto Stim/start_time",
        "behavior": "processing/behavior", "lfp": "processing/ecephys/LFP",
    }
    result = {}
    for name, path in paths.items():
        present = path in source
        if present and hasattr(source[path], "shape"):
            present = bool(np.prod(source[path].shape))
        result[name] = present
    return result


def export_recording(item, output):
    require_compute()
    from preprocess.allen_striatum_active.session_extractor import build
    from preprocess.allen_striatum_active.session_units import annotate
    from temporaldata import Data
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / (item["id"] + ".h5")
    identity, implementation = source_identity(item), implementation_digest()
    if target.exists():
        with h5py.File(target, "r") as file:
            matches = (file.attrs.get("complete", False)
                       and file.attrs.get("source_identity") == identity
                       and file.attrs.get("implementation_sha256") == implementation
                       and file.attrs.get("active_schema_version") == SCHEMA_VERSION)
            if not matches:
                raise ValueError(f"Existing export differs; choose a new output directory: {target}")
            summary = json.loads(file["provenance"].attrs["summary_json"])
        return dict(summary, status="reused", output=str(target), bytes=target.stat().st_size)
    # An exclusive per-recording lock prevents two runners replacing each other.
    lock = target.with_suffix(".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = target.with_suffix(f".h5.partial.{os.getpid()}")
    try:
        os.write(lock_fd, f"{socket.gethostname()} pid={os.getpid()}\n".encode())
        source = open_source(item["source"])
        sidecar_data = sidecars(item)
        info = clock_metadata(item, source, sidecar_data)
        with h5py.File(temporary, "w", libver="latest") as file:
            file.attrs.update(active_schema_version=SCHEMA_VERSION, complete=False,
                              source_identity=identity, implementation_sha256=implementation,
                              created_utc=datetime.now(timezone.utc).isoformat(), compression="none")
            checks = copy_tree(source, data_group(file, "source_nwb"))
            preserved = SourceGroup(file["source_nwb"])
            warnings = build(file, preserved, info, sidecar_data)
            qc = annotate(preserved, file, info)
            provenance = data_group(file, "provenance")
            provenance.attrs.update(recording_json=json.dumps(info), sidecars_json=json.dumps(sidecar_data),
                                    array_checksums_json=json.dumps(checks), warnings_json=json.dumps(warnings),
                                    availability_json=json.dumps(availability(preserved)),
                                    packages_json=json.dumps({k: version(k) for k in ("brainsets", "temporaldata", "zarr", "h5py", "numpy")}))
            summary = {k: item[k] for k in ("id", "animal", "session", "recording")}
            summary.update(n_units=len(preserved["units/id"]) if "units/id" in preserved else 0,
                           n_spikes=len(preserved["units/spike_times"]) if "units/spike_times" in preserved else 0,
                           n_trials=len(preserved["intervals/trials/id"]) if "intervals/trials/id" in preserved else 0,
                           source_arrays_verified=len(checks), **availability(preserved), **qc,
                           clock_available="origin_s" in info["clock"], warnings=warnings)
            provenance.attrs["summary_json"] = json.dumps(summary)
            file.flush()
        # Confirm normal Brainsets/temporaldata lazy loading before publishing.
        with h5py.File(temporary, "r") as file:
            data = Data.from_hdf5(file, lazy=True)
            if summary["n_units"] and len(data.units) != summary["n_units"]:
                raise ValueError("Canonical unit count mismatch")
            if summary["n_spikes"] and len(data.spikes) != summary["n_spikes"]:
                raise ValueError("Canonical spike count mismatch")
        if source_identity(item) != identity:
            raise ValueError("Source changed during conversion")
        with h5py.File(temporary, "r+") as file:
            file.attrs["complete"] = True
            file.flush()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target)
        return dict(summary, status="exported", output=str(target), bytes=target.stat().st_size)
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def validate_export(path):
    """Independently compare canonical fields and archive values on a CPU node."""
    require_compute()
    with h5py.File(path, "r") as file:
        if file.attrs.get("active_schema_version") != SCHEMA_VERSION or not file.attrs.get("complete", False):
            raise ValueError(f"Incomplete or unsupported active Brainsets export: {path}")
        checks = json.loads(file["provenance"].attrs["array_checksums_json"])
        for check in checks:
            group = file["source_nwb/" + check["path"]]
            array = SourceArray(group)
            digest = hashlib.sha256()
            values = group["values"].asstr() if array.dtype.kind in "OU" else group["values"]
            for selection in blocks(array):
                hash_update(digest, values[selection if array.ndim else 0])
            if digest.hexdigest() != check["sha256"] or digest.hexdigest() != group.attrs["sha256"]:
                raise ValueError(f"Array checksum mismatch: {path}: {check['path']}")
            if list(array.shape) != check["shape"] or array.dtype.str != check["dtype"]:
                raise ValueError(f"Array metadata mismatch: {check['path']}")
        source = SourceGroup(file["source_nwb"])
        if "units/id" in source:
            ids = source["units/id"][:]
            if not np.array_equal(ids, file["units/source_id"][:]):
                raise ValueError("Canonical/source unit ID mismatch")
            if "spikes" in file:
                times = file["spikes/timestamps"][:]
                indices = file["spikes/unit_index"][:]
                if np.any(np.diff(times) < 0) or not np.isfinite(times).all() or np.any(indices >= len(ids)):
                    raise ValueError("Invalid canonical spike timestamps/indexes")
                # Regroup once, then compare every spike of every source unit.
                order = np.argsort(indices, kind="stable")
                ordered_times = times[order]
                counts = np.bincount(indices.astype(np.int64), minlength=len(ids))
                ends = source["units/spike_times_index"][:].astype(np.int64)
                if not np.array_equal(counts, np.diff(np.r_[0, ends])):
                    raise ValueError("Canonical/source spike count mismatch")
                for row, (low, high) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
                    expected = np.sort(source["units/spike_times"][int(low):int(high)], kind="stable")
                    if not np.array_equal(ordered_times[low:high], expected):
                        raise ValueError(f"Canonical/source spike mismatch for unit row {row}")
                    a, b = file["spike_index/offsets"][row:row + 2]
                    if not np.array_equal(file["spike_index/timestamps"][int(a):int(b)], expected):
                        raise ValueError(f"Per-unit spike index mismatch for unit row {row}")
        if "intervals/trials/id" in source:
            if not np.array_equal(file["trials/id"][:], source["intervals/trials/id"][:]):
                raise ValueError("Trial ID mismatch")
        from preprocess.allen_striatum_active.session_extractor import EVENT_PATHS
        for name, source_path in EVENT_PATHS.items():
            if source_path in source:
                if not np.array_equal(file[f"events/{name}/timestamps"][:], source[source_path][:], equal_nan=True):
                    raise ValueError(f"Canonical/source event mismatch: {name}")
        result = json.loads(file["provenance"].attrs["summary_json"])
        from preprocess.allen_striatum_active.signal_domains import validate_signal_domains
        result["signal_domain_views_verified"] = validate_signal_domains(file)
        result.update(path=str(path), validation="passed", arrays_verified=len(checks), bytes=Path(path).stat().st_size)
        return result


def main():
    parser = argparse.ArgumentParser(description="Lossless active Brainsets preprocessing")
    parser.add_argument("command", choices=("inventory", "export", "validate"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/active"))
    parser.add_argument("--recording", action="append", help="Exact recording ID; repeat to select several")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "validate":
        paths = sorted(args.output.glob("*.h5"))
        if args.recording:
            paths = [p for p in paths if p.stem in args.recording]
        if not paths:
            parser.error("No matching HDF5 exports found")
        results = []
        for path in paths:
            results.append(validate_export(path))
            print(f"validated: {path.name}", flush=True)
        (args.output / "validation_report.json").write_text(json.dumps(results, indent=2) + "\n")
        return
    if args.raw_root is None:
        parser.error("--raw-root is required for inventory and export")
    records = discover(args.raw_root)
    if args.recording:
        missing = set(args.recording) - {r["id"] for r in records}
        if missing:
            parser.error(f"Unknown recording IDs: {sorted(missing)}")
        records = [r for r in records if r["id"] in args.recording]
    if args.command == "inventory":
        print(json.dumps(records, indent=2))
        return
    require_compute()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        tasks = {executor.submit(export_recording, record, args.output): record for record in records}
        for task in as_completed(tasks):
            record = tasks[task]
            try:
                row = task.result()
            except Exception:
                row = {k: record[k] for k in ("id", "animal", "session", "recording")}
                row.update(status="failed", error=traceback.format_exc())
            results.append(row)
            print(f"{row['status']}: {row['id']}" + ("\n" + row["error"] if "error" in row else ""), flush=True)
            # A crash still leaves a readable report of finished records.
            temporary = args.output / "export_report.json.tmp"
            temporary.write_text(json.dumps(sorted(results, key=lambda r: r["id"]), indent=2) + "\n")
            temporary.replace(args.output / "export_report.json")
    keys = sorted({k for r in results for k in r})
    with (args.output / "export_report.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["id"]))
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
