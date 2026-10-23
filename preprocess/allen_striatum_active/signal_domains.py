"""Portable domain views and explicit timestamp defects; source samples stay intact."""
import json

import h5py
import numpy as np

from preprocess.allen_striatum_active.session_intervals import domain_intervals
from preprocess.allen_striatum_active.utils import data_group


DOMAIN_NAMES = ("baseline", "behavior", "optotagging", "pre_opto")


def selection(times, bounds, aligned=True, timestamp_source="explicit"):
    """Keep source order; a reversal contaminates its entire timestamp span.

    Nonfinite timestamps cannot be assigned reliably to a domain, so all
    domains remain unavailable in that case. No samples are silently repaired.
    """
    structural = times.ndim != 1 or not aligned or timestamp_source == "unavailable"
    nonfinite = np.flatnonzero(~np.isfinite(times)).tolist() if times.ndim == 1 else []
    backward = np.flatnonzero(np.diff(times) < 0) if times.ndim == 1 else np.empty(0, int)
    defects = [dict(index_before=int(i), index_after=int(i + 1),
                    before_s=float(times[i]), after_s=float(times[i + 1]),
                    jump_s=float(times[i + 1] - times[i])) for i in backward]
    affected = [d for d in defects if any(d["after_s"] < b and d["before_s"] >= a for a, b in bounds)]
    mask = np.zeros(len(times) if times.ndim else 0, bool)
    if times.ndim == 1:
        for low, high in bounds:
            mask |= (times >= low) & (times < high)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    status = ("missing_domain" if not len(bounds) else "invalid" if structural or nonfinite or affected
              else "present" if mask.any() else "empty")
    quality = dict(status=status, available=status in {"present", "empty"},
                   nonfinite_indices=nonfinite, backward_jumps=affected,
                   structural_error=bool(structural), sample_count=int(mask.sum()))
    return edges, times[mask] if times.ndim == 1 else np.empty(0), quality, defects


def virtual_rows(target, name, dataset, ranges):
    """Same-file virtual datasets survive renaming and workstation transfers."""
    count = int(sum(b - a for a, b in ranges))
    shape = (count, *dataset.shape[1:])
    if not count:
        target.create_dataset(name, shape=shape, dtype=dataset.dtype)
    else:
        layout = h5py.VirtualLayout(shape=shape, dtype=dataset.dtype)
        source = h5py.VirtualSource(".", dataset.name, shape=dataset.shape)
        offset = 0
        for low, high in ranges:
            layout[offset:offset + high - low] = source[low:high]
            offset += high - low
        target.create_virtual_dataset(name, layout)
    target[name].attrs.update(dict(dataset.attrs))


def add_signal_domains(file, source, info):
    from preprocess.allen_striatum_active.session_extractor import SIGNAL_PATHS, add_series_domain, interval

    root = data_group(file, "signal_domains")
    interval(np.column_stack((file["domain/start"][:], file["domain/end"][:]))).to_hdf5(root.create_group("domain"))
    views = {}
    for name in DOMAIN_NAMES:
        try:
            bounds = domain_intervals(source, info, name)
        except ValueError:
            bounds = np.empty((0, 2))
        group = data_group(root, name)
        group.attrs.update(available=bool(len(bounds)), time_reference="source_clock")
        interval(bounds).to_hdf5(group.create_group("domain"))
        signals = data_group(group, "signals")
        interval(bounds).to_hdf5(signals.create_group("domain"))
        views[name] = (signals, bounds)

    leaves = []
    file["signals"].visititems(lambda name, node: leaves.append(node) if isinstance(node, h5py.Group)
                              and "timestamps" in node and "source_path" in node.attrs else None)
    for node in leaves:
        times = np.asarray(node["timestamps"][:], float)
        count = len(times) if times.ndim else 0
        aligned = "data" not in node or (node["data"].ndim > 0 and len(node["data"]) == count)
        timestamp_source = node.attrs.get("timestamp_source", "explicit")
        qualities = {}
        source_path = str(node.attrs["source_path"])
        for name, (signals, bounds) in views.items():
            ranges, selected, quality, defects = selection(times, bounds, aligned, timestamp_source)
            qualities[name] = quality
            parent = signals
            parts = source_path.split("/")
            for part in parts[:-1]:
                if part not in parent:
                    nested = data_group(parent, part)
                    interval(bounds).to_hdf5(nested.create_group("domain"))
                parent = parent[part]
            target = data_group(parent, parts[-1])
            target.attrs.update(dict(node.attrs))
            target.attrs.update(object="Data", absolute_start=0.0, status=quality["status"],
                                available=quality["available"], selection_domain=name,
                                source_signal_path="/signals/" + source_path,
                                source_ranges_json=json.dumps(ranges.tolist()),
                                quality_json=json.dumps(quality))
            for key, dataset in node.items():
                if isinstance(dataset, h5py.Dataset) and dataset.ndim and len(dataset) == count:
                    if key != "timestamp_indices_1s":
                        virtual_rows(target, key, dataset, ranges)
            if quality["available"]:
                target.attrs["object"] = "ArrayDict"
                if len(selected):
                    add_series_domain(target, selected)
        node.attrs["quality_json"] = json.dumps(dict(
            status=str(node.attrs["status"]), backward_jumps=defects,
            nonfinite_indices=np.flatnonzero(~np.isfinite(times)).tolist(),
            structural_error=bool(times.ndim != 1 or not aligned or timestamp_source == "unavailable"),
            domains=qualities))

    # Preserve the same convenient names as the full-recording signal tree.
    for signals, _ in views.values():
        for name, path in SIGNAL_PATHS.items():
            if path not in signals:
                missing = data_group(signals, name)
                missing.attrs.update(status="missing", available=False)
                continue
            candidate = signals[path]
            children = [g for g in candidate.values() if isinstance(g, h5py.Group) and "timestamps" in g]
            if "timestamps" not in candidate and len(children) == 1:
                candidate = children[0]
            signals[name] = candidate


def validate_signal_domains(file):
    """Check domain membership, quality, source row mappings, and boundary values."""
    verified = 0
    for name in DOMAIN_NAMES:
        root = file[f"signal_domains/{name}"]
        bounds = np.column_stack((root["domain/start"][:], root["domain/end"][:]))
        leaves = []
        root["signals"].visititems(lambda key, node: leaves.append(node) if isinstance(node, h5py.Group)
                                  and "source_signal_path" in node.attrs else None)
        for node in leaves:
            source = file[node.attrs["source_signal_path"]]
            times = source["timestamps"][:]
            aligned = "data" not in source or (source["data"].ndim > 0 and len(source["data"]) == len(times))
            ranges, expected, quality, _ = selection(times, bounds, aligned, source.attrs.get("timestamp_source", "explicit"))
            if quality != json.loads(node.attrs["quality_json"]) or ranges.tolist() != json.loads(node.attrs["source_ranges_json"]):
                raise ValueError(f"Domain quality/index mismatch: {node.name}")
            if not np.array_equal(node["timestamps"][:], expected, equal_nan=True):
                raise ValueError(f"Domain timestamp mismatch: {node.name}")
            for key, dataset in node.items():
                if not isinstance(dataset, h5py.Dataset) or key == "timestamp_indices_1s":
                    continue
                if dataset.is_virtual and any(v.file_name != "." for v in dataset.virtual_sources()):
                    raise ValueError(f"Nonportable virtual dataset: {dataset.name}")
                offset = 0
                for low, high in ranges:
                    for a, b in ((offset, low), (offset + high - low - 1, high - 1)):
                        if not np.array_equal(dataset[int(a)], source[key][int(b)]):
                            # NaNs are retained measurements, not conversion errors.
                            if dataset.dtype.kind not in "fc" or not np.array_equal(dataset[int(a)], source[key][int(b)], equal_nan=True):
                                raise ValueError(f"Domain measurement mismatch: {dataset.name}")
                    offset += high - low
            verified += 1
    return verified
