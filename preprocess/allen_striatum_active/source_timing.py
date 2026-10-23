from datetime import datetime
import json
import re

import numpy as np

from preprocess.allen_striatum_active.source_io import discard_cache


def session_timing(metadata, buffer_s=300.0):
    start = datetime.fromisoformat(metadata["session_start_time"])
    if start.utcoffset() is None:
        raise ValueError("Session start must include a timezone")
    events = [datetime.fromisoformat(e["stimulus_start_time"])
              for e in metadata.get("stimulus_epochs", [])
              if "Optogenetics" in e.get("stimulus_modalities", []) and e.get("stimulus_start_time")]
    if not events:
        return None
    elapsed = (min(events) - start).total_seconds()
    if not np.isfinite([elapsed, buffer_s]).all() or elapsed <= 0 or buffer_s < 0:
        raise ValueError("Invalid optotagging onset or buffer")
    streams = []
    for stream in metadata.get("data_streams", []):
        a, b = (datetime.fromisoformat(stream[k]) for k in ("stream_start_time", "stream_end_time"))
        streams.append({"offset_s": (a - start).total_seconds(), "duration_s": (b - a).total_seconds()})
    return dict(session_start_time=start.isoformat(), first_opto_elapsed_s=elapsed,
                cutoff_session_s=elapsed - buffer_s, buffer_s=buffer_s, streams=streams)


def recording_numbers(path):
    match = re.search(r"experiment(\d+)_recording(\d+)", path.name)
    if not match:
        raise ValueError(f"Recording identity missing: {path}")
    return tuple(map(int, match.groups()))


def edge_records(path, count=16000):
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        # Use public readers: newer NumPy versions hide _read_array_header.
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f"Unsupported source sorting NPY version {version}: {path}")
        shape, fortran, dtype = readers[version](handle)
        if len(shape) != 1 or fortran or not shape[0]:
            raise ValueError(f"Unexpected source sorting array: {path}")
        offset = handle.tell()
        count = min(count, shape[0])
        first = np.frombuffer(handle.read(count * dtype.itemsize), dtype=dtype)
        handle.seek(offset + (shape[0] - count) * dtype.itemsize)
        last = np.frombuffer(handle.read(count * dtype.itemsize), dtype=dtype)
        discard_cache(handle.fileno())
    if np.any(first["segment_index"] != 0) or np.any(last["segment_index"] != 0):
        raise ValueError("Multiple source segments are not supported")
    return first, last


def boundary_spikes(nwb):
    ends = nwb["units/spike_times_index"][:].astype(np.int64)
    starts = np.r_[0, ends[:-1]]
    nonempty = ends > starts
    if not nonempty.any():
        raise ValueError("Recording has no spikes to establish timing")
    positions = np.r_[starts[nonempty], ends[nonempty] - 1]
    spikes = nwb["units/spike_times"]
    values = np.empty(len(positions), dtype=np.float64)
    chunks = positions // spikes.chunks[0]
    for chunk in np.unique(chunks):
        lo = int(chunk) * spikes.chunks[0]
        block = spikes[lo:min(lo + spikes.chunks[0], len(spikes))]
        selected = chunks == chunk
        values[selected] = block[positions[selected] - lo]
    n = int(nonempty.sum())
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite spike timestamps")
    return float(values[:n].min()), float(values[n:].max())


def recording_clock(asset, path, nwb):
    exp, rec = recording_numbers(path)
    sources = [p for p in (asset / "curated").glob(f"experiment{exp}_*_recording{rec}")
               if (p / "numpysorting_info.json").exists()]
    if not sources:
        if (exp, rec) != (1, 1) or not asset.name.endswith("_nwb"):
            raise ValueError("No source sample clock; cannot establish recording origin")
        origin, last = boundary_spikes(nwb)
        return dict(origin_s=origin, duration_s=last - origin, clock_scale=1.0,
                    fit_error_s=None, clock_method="earliest_spike", clock_source=str(path))
    if len(sources) != 1:
        raise ValueError("Ambiguous source sorting")
    source = sources[0]
    info = json.loads((source / "numpysorting_info.json").read_text())
    if info.get("num_segments", 1) != 1:
        raise ValueError("Multiple source segments are not supported")
    rate = float(info["sampling_frequency"])
    ids = np.asarray(info["unit_ids"])
    first, last = edge_records(source / "spikes.npy")
    unit_ids = nwb["units/ks_unit_id"][:]
    offsets = np.r_[0, nwb["units/spike_times_index"][:].astype(np.int64)]
    spikes = nwb["units/spike_times"]
    pairs = []
    for row in np.argsort(np.diff(offsets))[::-1]:
        found = np.flatnonzero(ids == unit_ids[row])
        if len(found) != 1 or offsets[row + 1] == offsets[row]:
            continue
        a = first["sample_index"][first["unit_index"] == found[0]][:8]
        b = last["sample_index"][last["unit_index"] == found[0]][-8:]
        if not len(a) or not len(b):
            continue
        y = np.r_[spikes[offsets[row]:offsets[row] + len(a)],
                  spikes[offsets[row + 1] - len(b):offsets[row + 1]]].astype(np.float64)
        x = np.r_[a, b].astype(np.float64) / rate
        if x[-1] > x[0]:
            pairs.append((x, y))
        if len(pairs) == 3:
            break
    if not pairs:
        raise ValueError("Cannot match NWB spikes to source sample indices")
    x, y = pairs[0]
    scale = float((y[-1] - y[0]) / (x[-1] - x[0]))
    origin = float(y[0] - scale * x[0])
    error = max(float(np.max(np.abs(yy - (origin + scale * xx)))) for xx, yy in pairs)
    if not np.isfinite([origin, scale, error]).all() or abs(scale - 1) > 0.001 or error > 0.002:
        raise ValueError(f"Source clock fit failed: scale={scale}, residual={error}s")
    attrs = json.loads((asset / "postprocessed" / (source.name + ".zarr") /
                       "recording_info/.zattrs").read_text())["recording_attributes"]
    duration = float(attrs["num_samples"][0]) / rate * scale
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid source recording duration")
    return dict(origin_s=origin, duration_s=duration, clock_scale=scale,
                fit_error_s=error, clock_method="source_samples", clock_source=str(source))


def place_recording(clock, primary, numbers, timing):
    if numbers == (1, 1):
        return 0.0, "session_start_anchor"
    delta = clock["origin_s"] - primary["origin_s"]
    if delta >= primary["duration_s"] - 1.0:
        # Require a later, nonoverlapping interval on the same numerical clock.
        stream_end = max((s["offset_s"] + s["duration_s"] for s in timing["streams"]), default=0)
        if delta + clock["duration_s"] <= stream_end + 2.0:
            return delta, "shared_sample_clock"
    matches = [s for s in timing["streams"]
               if s["offset_s"] >= primary["duration_s"] - 1.0
               and abs(s["duration_s"] - clock["duration_s"]) <= 1.0]
    if len(matches) == 1:
        return matches[0]["offset_s"], "unique_stream_duration"
    return None, "unresolved_clock"
