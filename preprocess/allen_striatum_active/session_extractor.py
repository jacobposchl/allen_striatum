"""Canonical Brainsets recording views with lossless links to source arrays."""
from datetime import datetime
import json
import re

import h5py
import numpy as np
from brainsets import serialize_fn_map
from brainsets.descriptions import BrainsetDescription, SessionDescription, SubjectDescription
from temporaldata import Data, Interval, IrregularTimeSeries

from src.data.schema import SCHEMA_VERSION
from preprocess.allen_striatum_active.utils import data_group
from preprocess.allen_striatum_active.session_intervals import epochs, selection_opto_start


EVENT_PATHS = {
    "left_lick": "acquisition/left_lick_time/timestamps",
    "right_lick": "acquisition/right_lick_time/timestamps",
    "left_reward": "acquisition/left_reward_delivery_time/timestamps",
    "right_reward": "acquisition/right_reward_delivery_time/timestamps",
    "opto_start": "stimulus/presentation/Opto Stim/start_time",
}
SIGNAL_PATHS = {
    "speed": "processing/behavior/Linear velocity/linear velocity",
    "pupil_diameter": "processing/behavior/Pupil diameter",
    "pupil_position": "processing/behavior/Pupil position",
    "body_keypoints": "processing/behavior/Body keypoint tracking",
    "face_keypoints": "processing/behavior/Face keypoint tracking",
    "tongue_keypoints": "processing/behavior/Tongue keypoint tracking",
}


def interval(values):
    values = np.asarray(values, float).reshape(-1, 2)
    return Interval(start=values[:, 0], end=values[:, 1])


def add_series_domain(node, times):
    """Write the temporaldata slicing index for a nonempty ordered series."""
    bounds = [float(times[0]), float(np.nextafter(times[-1], np.inf))]
    node.attrs.update(object="IrregularTimeSeries", timekeys=np.asarray(["timestamps"], dtype="S"))
    interval([bounds]).to_hdf5(node.create_group("domain"))
    grid = np.arange(bounds[0], bounds[1] + 1.0, 1.0)
    node.create_dataset("timestamp_indices_1s", data=np.searchsorted(times, grid))
    return bounds


def link_columns(source, target, count, exclude=()):
    unicode_keys = list(np.asarray(target.attrs.get("_unicode_keys", []), dtype=str))
    for name, child in source.items():
        if name in exclude or name in target or "source_shape_json" not in child.attrs:
            continue
        if child.attrs.get("encoding") == "object_json":
            continue  # Arbitrary NWB references remain in the private archive.
        shape = json.loads(child.attrs["source_shape_json"])
        if not shape or shape[0] != count:
            continue
        target[name] = child["values"]
        target[name].attrs["source_metadata_json"] = child.attrs.get("source_attrs_json", "{}")
        for key, value in json.loads(child.attrs.get("source_attrs_json", "{}")).items():
            if isinstance(value, (str, int, float, bool)):
                target[name].attrs[key] = value
        if np.dtype(child.attrs["source_dtype"]).kind in "OU":
            unicode_keys.append(name)
    target.attrs["_unicode_keys"] = np.asarray(unicode_keys, dtype="S")


def copy_signal_metadata(source, target):
    """Keep calibration next to the canonical signal, including dataset attrs."""
    source_attrs = json.loads(source.attrs.get("source_attrs_json", "{}"))
    data_attrs = (json.loads(source["data"].attrs.get("source_attrs_json", "{}"))
                  if "data" in source else {})
    target.attrs["source_path"] = source.name.removeprefix("/source_nwb/")
    target.attrs["source_metadata_json"] = json.dumps(source_attrs)
    target.attrs["data_metadata_json"] = json.dumps(data_attrs)
    for name in ("unit", "units", "conversion", "offset", "resolution", "description"):
        value = data_attrs.get(name, source_attrs.get(name))
        if isinstance(value, (str, int, float, bool)):
            target.attrs[name] = value
            if "data" in target:
                target["data"].attrs[name] = value
    if "conversion" not in target.attrs:
        target.attrs["conversion"] = 1.0
    if "offset" not in target.attrs:
        target.attrs["offset"] = 0.0


def add_timeseries_views(source, target, warnings):
    """Retain every sampled series and expose its timestamp validity explicitly."""
    domains = []
    for name, child in source.items():
        if "source_shape_json" in child.attrs:
            continue
        is_series = "timestamps" in child or "starting_time" in child
        if not is_series:
            nested = data_group(target, name)
            domains.extend(add_timeseries_views(child, nested, warnings))
            if not len(nested):
                del target[name]
            continue
        node = target.create_group(name)
        node.attrs.update(object="ArrayDict", _unicode_keys=np.array([], dtype="S"))
        if "timestamps" in child:
            times = np.asarray(child["timestamps/values"][:], float)
            node["timestamps"] = child["timestamps/values"]
            timestamp_source = "explicit"
        else:
            rate = json.loads(child["starting_time"].attrs.get("source_attrs_json", "{}")).get("rate")
            start = np.asarray(child["starting_time/values"][()]).reshape(-1)
            shape = json.loads(child["data"].attrs["source_shape_json"]) if "data" in child else []
            if (rate is not None and np.isfinite(rate) and rate > 0 and len(start) == 1
                    and np.isfinite(start[0]) and len(shape)):
                times = start[0] + np.arange(shape[0], dtype=float) / rate
                node.create_dataset("timestamps", data=times)
                timestamp_source = "starting_time_and_rate"
                node.attrs.update(starting_time=float(start[0]), rate=float(rate))
            else:
                times = np.empty(0)
                timestamp_source = "unavailable"
                node.create_dataset("timestamps", data=times)
        count = len(times) if times.ndim else 0
        link_columns(child, node, count, exclude={"timestamps", "starting_time"})
        # Preserve misaligned data too; consumers must reject its explicit status.
        if "data" in child and "data" not in node and child["data"].attrs.get("encoding") != "object_json":
            node["data"] = child["data/values"]
        aligned = "data" not in node or (node["data"].ndim and node["data"].shape[0] == count)
        valid = times.ndim == 1 and np.isfinite(times).all() and np.all(np.diff(times) >= 0) and aligned
        status = ("missing_timestamps" if timestamp_source == "unavailable" else
                  "invalid" if not valid else "present" if count else "empty")
        node.attrs.update(status=status, available=status in {"present", "empty"}, timestamp_source=timestamp_source)
        copy_signal_metadata(child, node)
        if valid and timestamp_source != "unavailable" and count:
            domains.append(add_series_domain(node, times))
        elif not valid or timestamp_source == "unavailable":
            # Invalid/misaligned arrays cannot satisfy ArrayDict's shared row
            # length. Preserve them in a generic Data node, marked unavailable.
            node.attrs["object"] = "Data"
            warnings.append(f"Canonical signal {node.name}: {status}; source values preserved")
    if domains:
        interval([[min(a for a, _ in domains), max(b for _, b in domains)]]).to_hdf5(target.create_group("domain"))
    return domains


def add_events(file, archive):
    target = data_group(file, "events")
    for name, path in EVENT_PATHS.items():
        node = target.create_group(name)
        node.attrs.update(object="ArrayDict", _unicode_keys=np.array([], dtype="S"), source_path=path)
        if path not in archive:
            node.create_dataset("timestamps", data=np.empty(0))
            node.attrs.update(status="missing", available=False)
            continue
        node["timestamps"] = archive[path + "/values"]
        times = np.asarray(node["timestamps"][:], float)
        valid = times.ndim == 1 and np.isfinite(times).all() and np.all(np.diff(times) >= 0)
        node.attrs.update(status="invalid" if not valid else "present" if len(times) else "empty", available=bool(valid))
        if valid and len(times):
            add_series_domain(node, times)
        elif not valid:
            node.attrs["object"] = "Data"


def add_electrodes(file, archive, n_units):
    path = "general/extracellular_ephys/electrodes"
    table = file.create_group("electrodes")
    table.attrs.update(object="ArrayDict", _unicode_keys=np.array([], dtype="S"))
    if path in archive and "id" in archive[path]:
        source = archive[path]
        table["source_id"] = source["id/values"]
        link_columns(source, table, len(table["source_id"]), exclude={"id"})
        table.attrs["status"] = "present"
        if "group_name" in table:
            names = table["group_name"].asstr()[:]
            probe_names = []
            for name in names:
                group_path = "general/extracellular_ephys/" + name
                attrs = json.loads(archive[group_path].attrs.get("source_attrs_json", "{}")) if group_path in archive else {}
                links = [link["path"].rsplit("/", 1)[-1] for link in attrs.get("zarr_link", []) if link.get("name") == "device"]
                probe_names.append(links[0] if len(links) == 1 else "")
            if "device_name" not in table:
                table.create_dataset("device_name", data=np.asarray(probe_names, dtype="S"))
                table.attrs["_unicode_keys"] = np.asarray([*table.attrs["_unicode_keys"].astype(str), "device_name"], dtype="S")
    else:
        table.attrs["status"] = "missing"
    add_waveform_electrodes(file, archive, n_units)
    probes = data_group(file, "probes")
    if "general/devices" in archive:
        for name, device in archive["general/devices"].items():
            node = data_group(probes, name)
            node.attrs["source_metadata_json"] = device.attrs.get("source_attrs_json", "{}")
            for field, value in device.items():
                if "source_shape_json" in value.attrs and value.attrs.get("encoding") != "object_json":
                    node[field] = value["values"]


def add_waveform_electrodes(file, archive, n_units):
    """Resolve each waveform channel to an electrode-table row exactly once.

    Source ragged references take precedence and retain their AP-channel order.
    With no source references, the device's AP order is usable only when its
    length exactly matches the stored waveform channel axis.
    """
    target = data_group(file, "unit_electrodes")
    table = file["electrodes"]
    offsets, row_parts, statuses = [0], [], []
    shapes = json.loads(archive["units/waveform_mean"].attrs["source_shape_json"]) if "units/waveform_mean" in archive else []
    channel_count = shapes[2] if len(shapes) == 3 and shapes[0] == n_units else None
    names = table["channel_name"].asstr()[:] if "channel_name" in table else np.asarray([], dtype=str)
    ap = np.asarray([re.fullmatch(r"(?:AP|CH)\d+", name) is not None for name in names], dtype=bool)
    devices = table["device_name"].asstr()[:] if "device_name" in table else np.full(len(names), "")
    unit_devices = archive["units/device_name/values"].asstr()[:] if "units/device_name" in archive else np.full(n_units, "")
    has_references = "units/electrodes" in archive and "units/electrodes_index" in archive
    raw_offsets, raw_rows = None, None
    valid_references = False
    if has_references:
        try:
            raw_offsets = np.r_[0, archive["units/electrodes_index/values"][:]].astype(np.int64)
            raw_rows = np.asarray(archive["units/electrodes/values"][:], dtype=np.int64)
            valid_references = (len(raw_offsets) == n_units + 1 and np.all(np.diff(raw_offsets) >= 0)
                                and raw_offsets[-1] == len(raw_rows)
                                and np.all((raw_rows >= 0) & (raw_rows < len(names))))
        except (ValueError, TypeError):
            pass
    for row in range(n_units):
        rows = np.empty(0, dtype=np.int64)
        status = "missing"
        if channel_count is not None and len(names):
            if has_references and valid_references:
                candidates = raw_rows[raw_offsets[row]:raw_offsets[row + 1]]
                rows = candidates[ap[candidates]]
                status = "source_references"
            elif has_references:
                status = "invalid_source_references"
            elif unit_devices[row]:
                rows = np.flatnonzero(ap & (devices == unit_devices[row]))
                status = "device_ap_order"
            if (len(rows) != channel_count or len(np.unique(rows)) != len(rows)
                    or (unit_devices[row] and not np.all(devices[rows] == unit_devices[row]))):
                rows = np.empty(0, dtype=np.int64)
                status = "invalid_channel_axis" if status != "missing" else status
        statuses.append(status)
        row_parts.append(rows)
        offsets.append(offsets[-1] + len(rows))
    target.create_dataset("electrode_row", data=np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64))
    target.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
    target.create_dataset("mapping_status", data=np.asarray(statuses, dtype="S"))
    target.attrs.update(status="present" if n_units and all(s in {"source_references", "device_ap_order"} for s in statuses) else "partial" if any(s in {"source_references", "device_ap_order"} for s in statuses) else "missing",
                        axis="units/waveform_mean final axis", reference="electrodes table row")


def add_waveform_metadata(units, source):
    attrs = dict(source["units"].attrs) if "units" in source else {}
    # NWB writers may place the same declared scalar on the table or in a
    # scalar dataset. Retain either representation without guessing a rate.
    for key in ("waveform_rate", "waveform_sampling_rate_hz", "sampling_rate_hz", "sampling_rate", "waveform_unit"):
        path = f"units/{key}"
        if key not in attrs and path in source and source[path].shape in {(), (1,)}:
            value = np.asarray(source[path][()]).reshape(-1)[0]
            attrs[key] = value.item() if isinstance(value, np.generic) else value
            if isinstance(attrs[key], bytes):
                attrs[key] = attrs[key].decode("utf-8")
    units.attrs["source_metadata_json"] = json.dumps(attrs)
    waveform = source.get("units/waveform_mean")
    waveform_attrs = dict(waveform.attrs) if waveform is not None else {}
    for metadata in (attrs, waveform_attrs):
        for key in ("waveform_rate", "waveform_sampling_rate_hz", "sampling_rate_hz", "sampling_rate"):
            value = metadata.get(key)
            if value is not None and np.ndim(value) == 0:
                try:
                    rate = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(rate) and rate > 0:
                    units.attrs["waveform_sampling_rate_hz"] = rate
                    units.attrs["waveform_sampling_rate_source"] = key
                    break
        if "waveform_sampling_rate_hz" in units.attrs:
            break
    unit = attrs.get("waveform_unit", waveform_attrs.get("unit"))
    if unit is not None:
        units.attrs["waveform_unit"] = str(unit)
        if "waveform_mean" in units:
            units["waveform_mean"].attrs["unit"] = str(unit)
    units.attrs["waveform_sampling_rate_available"] = "waveform_sampling_rate_hz" in units.attrs


def add_units(file, source, archive, info):
    n = len(source["units/id"]) if "units/id" in source else 0
    units = file.create_group("units")
    units.attrs.update(object="ArrayDict", _unicode_keys=np.array([], dtype="S"))
    if not n:
        add_waveform_metadata(units, source)
        units.create_dataset("id", data=np.asarray([], dtype="S1"))
        units.create_dataset("source_id", data=np.asarray([], dtype=np.int64))
        units.create_dataset("source_unit_row", data=np.asarray([], dtype=np.int64))
        units.attrs["status"] = "missing" if "units/id" not in source else "empty"
        add_electrodes(file, archive, n)
        return []
    units.attrs["status"] = "present"
    link_columns(archive["units"], units, n,
                 exclude={"id", "spike_times", "spike_times_index", "electrodes", "electrodes_index"})
    add_waveform_metadata(units, source)
    units["source_id"] = archive["units/id/values"]
    ids = source["units/id"][:]
    if len(np.unique(ids)) != n:
        raise ValueError("Duplicate source unit IDs")
    global_ids = [f"{info['animal']}/{info['session']}/{info['recording']}/{i}" for i in ids]
    units.create_dataset("id", data=np.asarray(global_ids, dtype="S"))
    units["global_unit_id"] = units["id"]
    units.create_dataset("source_unit_row", data=np.arange(n))
    units.attrs["_unicode_keys"] = np.asarray([*units.attrs["_unicode_keys"].astype(str), "id", "global_unit_id"], dtype="S")
    add_electrodes(file, archive, n)
    ends = np.asarray(source["units/spike_times_index"][:], np.int64)
    times = np.asarray(source["units/spike_times"][:])
    offsets = np.r_[0, ends]
    if (len(ends) != n or np.any(np.diff(offsets) < 0)
            or int(ends[-1]) != len(times) or not np.isfinite(times).all()):
        raise ValueError("Malformed source spike indexing or nonfinite timestamps")
    index = data_group(file, "spike_index")
    index.create_dataset("offsets", data=offsets)
    sorted_units = all(np.all(np.diff(times[a:b]) >= 0) for a, b in zip(offsets[:-1], offsets[1:]))
    if sorted_units:
        index["timestamps"] = archive["units/spike_times/values"]
    else:
        ordered = times.copy()
        for a, b in zip(offsets[:-1], offsets[1:]):
            ordered[a:b].sort(kind="stable")
        index.create_dataset("timestamps", data=ordered, chunks=True)
    index.attrs.update(order="unit_major", time_reference="source_clock")
    indices = np.repeat(np.arange(n, dtype=np.uint32), np.diff(offsets))
    order = np.argsort(times, kind="stable")
    limits = [[float(times[order[0]]), float(np.nextafter(times[order[-1]], np.inf))]] if len(times) else []
    if len(times):
        IrregularTimeSeries(timestamps=times[order], unit_index=indices[order],
                            domain=interval(limits)).to_hdf5(file.create_group("spikes"))
    else:
        spikes = file.create_group("spikes")
        spikes.attrs.update(object="ArrayDict", _unicode_keys=np.array([], dtype="S"), status="empty")
        spikes.create_dataset("timestamps", data=times)
        spikes.create_dataset("unit_index", data=indices)
    return limits


def add_trials(file, source, archive, warnings):
    target = file.create_group("trials")
    target.attrs.update(object="ArrayDict", _unicode_keys=np.array([], dtype="S"))
    path = "intervals/trials"
    if path not in source:
        target.attrs.update(status="missing", available=False)
        return []
    trials = source[path]
    count = len(trials["id"]) if "id" in trials else len(trials["start_time"])
    starts = np.asarray(trials["start_time"][:], float) if "start_time" in trials else np.full(count, np.nan)
    ends = np.asarray(trials["stop_time"][:], float) if "stop_time" in trials else np.full(count, np.nan)
    same_shape = starts.shape == ends.shape == (count,)
    valid = (np.isfinite(starts) & np.isfinite(ends) & (ends > starts)) if same_shape else np.zeros(count, bool)
    temporal = bool(same_shape and valid.all() and np.all(np.diff(starts) >= 0) and np.all(starts[1:] >= ends[:-1]))
    if temporal:
        Interval(start=starts, end=ends).to_hdf5(target)
    else:
        target.create_dataset("start", data=starts)
        target.create_dataset("end", data=ends)
        warnings.append("Trial boundaries are invalid/overlapping; canonical table retained with valid_bounds flags")
    link_columns(archive[path], target, count)
    target.create_dataset("source_trial_row", data=np.arange(count))
    target.create_dataset("valid_bounds", data=valid)
    target.attrs.update(status="present" if temporal and count else "empty" if not count else "invalid_bounds", available=True)
    if temporal:
        target.attrs["timekeys"] = np.asarray(["start", "end"] + [k for k in target if k.endswith("_time")], dtype="S")
    return [[float(starts[valid].min()), float(ends[valid].max())]] if valid.any() else []


def build(file, source, info, sidecars):
    warnings = []
    subject = sidecars.get("subject.json", {})
    session = sidecars.get("session.json", {})
    date = session.get("session_start_time")
    date = datetime.fromisoformat(date) if date else datetime.strptime(info["session"], "%Y-%m-%d_%H-%M-%S")
    Data(
        brainset=BrainsetDescription(id="allen_striatum_active", origin_version="source NWB-Zarr",
                                    derived_version=SCHEMA_VERSION, source="Allen Institute dynamic foraging recordings",
                                    description="All source observations and explicit availability; downstream selection is optional."),
        subject=SubjectDescription(id=info["animal"], species="mus_musculus", sex=subject.get("sex")),
        session=SessionDescription(id=info["id"], recording_date=date),
    ).to_hdf5(file, serialize_fn_map=serialize_fn_map)
    file.attrs.update(time_reference="source_clock", schema_version=SCHEMA_VERSION)
    archive = file["source_nwb"]
    bounds = add_units(file, source, archive, info)
    domains = data_group(file, "intervals")
    for name, values in epochs(source).items():
        if len(values):
            interval(values).to_hdf5(domains.create_group(name))
            bounds.extend(values.tolist())
    onset = selection_opto_start(source, info)
    domains.attrs["opto_onset_s"] = np.nan if onset is None else onset
    domains.attrs["opto_onset_available"] = onset is not None
    if bounds:
        interval([[min(a for a, _ in bounds), max(b for _, b in bounds)]]).to_hdf5(domains.create_group("domain"))
    bounds.extend(add_trials(file, source, archive, warnings))
    add_events(file, archive)
    signals = data_group(file, "signals")
    signal_bounds = []
    for section in ("processing", "acquisition", "stimulus"):
        if section in archive:
            signal_bounds.extend(add_timeseries_views(archive[section], data_group(signals, section), warnings))
    for name, path in SIGNAL_PATHS.items():
        if path in signals:
            candidate = signals[path]
            children = [node for node in candidate.values() if isinstance(node, h5py.Group) and "timestamps" in node]
            if "timestamps" not in candidate and len(children) == 1:
                candidate = children[0]
            signals[name] = candidate
            if "status" not in candidate.attrs:
                candidate.attrs.update(status="present", available=True)
        else:
            missing = data_group(signals, name)
            missing.attrs.update(status="missing", available=False, source_path=path)
    if signal_bounds:
        interval([[min(a for a, _ in signal_bounds), max(b for _, b in signal_bounds)]]).to_hdf5(signals.create_group("domain"))
        bounds.extend(signal_bounds)
    clock = info.get("clock", {})
    if clock.get("origin_s") is not None and clock.get("duration_s") is not None:
        bounds.append([clock["origin_s"], clock["origin_s"] + clock["duration_s"]])
    if bounds:
        interval([[min(a for a, _ in bounds), max(b for _, b in bounds)]]).to_hdf5(file.create_group("domain"))
    return warnings
