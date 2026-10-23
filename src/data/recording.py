"""Canonical Brainsets recording access on the original source clock.

Runtime readers never inspect source_nwb, raw sorting files, or CSV caches.
Active exports and processed passive Brainsets recordings share this interface.
"""
from functools import cached_property
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.data.features import waveform_columns
from src.data.intervals import intersect, merge, subtract
from src.data.io import resolve
from src.data.schema import SCHEMA_VERSION


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def array(dataset):
    return dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:]


def table(group, count=None):
    if count is None:
        count = len(group["id"]) if "id" in group else len(group["start"])
    return pd.DataFrame({name: array(value) for name, value in group.items()
                         if isinstance(value, h5py.Dataset) and value.shape == (count,)})


def lower_bound(dataset, value):
    low, high = 0, len(dataset)
    while low < high:
        middle = (low + high) // 2
        if float(dataset[middle]) < value:
            low = middle + 1
        else:
            high = middle
    return low


def resolve_recording(value, root=None):
    """Resolve an explicit HDF5 path or a unique recording ID in a data root."""
    requested = Path(str(value)).expanduser()
    if requested.suffix and requested.suffix != ".h5":
        raise ValueError("Recording inputs must be Brainsets .h5 files or recording IDs")
    if requested.is_absolute() or (requested.parent != Path(".") and requested.suffix == ".h5"):
        path = resolve(requested)
        if path.suffix != ".h5" or not path.is_file():
            raise FileNotFoundError(path)
        return path
    if requested.suffix == ".h5" and resolve(requested).is_file():
        return resolve(requested)
    passive = len(requested.parts) == 3 and requested.parts[-1] == "passive"
    default_root = os.environ.get("NUCLR_PASSIVE_H5_ROOT", "data/processed/passive") if passive else os.environ.get("NUCLR_ACTIVE_H5_ROOT", "data/processed/active")
    directory = resolve(root or default_root)
    recording_id = requested.parts[1] if passive else "_".join(requested.parts) if requested.parent != Path(".") else requested.stem
    path = directory / (recording_id + ".h5")
    if not path.is_file():
        raise FileNotFoundError(f"Brainsets recording not found: {path}")
    return path


def discover_recordings(root):
    records = []
    for path in sorted(resolve(root).glob("*.h5")):
        with Recording(path) as recording:
            info = recording.info
            records.append(dict(id=info["id"], animal=info["animal"], session=info["session"],
                                recording=info["recording"], source=str(path.resolve()),
                                path=path.resolve(), source_identity=recording.identity,
                                n_units=len(recording.units)))
    return records


class Recording:
    """One open HDF5 recording. All public timestamps retain the source clock."""

    def __init__(self, path):
        self.path = resolve_recording(path)
        self.file = h5py.File(self.path, "r")
        try:
            active = self.file.attrs.get("active_schema_version")
            if active is not None:
                if text(active) != SCHEMA_VERSION or not self.file.attrs.get("complete", False):
                    raise ValueError(f"Reprocess this recording with active schema {SCHEMA_VERSION}: {self.path}")
            if not {"subject", "session", "units"}.issubset(self.file):
                raise ValueError(f"Missing canonical Brainsets metadata: {self.path}")
            self.info = self._metadata()
        except BaseException:
            self.file.close()
            raise

    def _metadata(self):
        if "provenance" in self.file and "recording_json" in self.file["provenance"].attrs:
            info = json.loads(self.file["provenance"].attrs["recording_json"])
        else:
            animal = text(self.file["subject"].attrs["id"])
            session = text(self.file["session"].attrs["id"])
            bounds = self._interval("domain")
            info = dict(id=session, animal=animal, session=session, recording="passive",
                        session_offset_s=0.0, clock=dict(origin_s=float(bounds[0, 0]),
                        duration_s=float(bounds[-1, 1] - bounds[0, 0]), clock_method="declared_domain"))
        info["origin_s"] = info.get("clock", {}).get("origin_s")
        info["duration_s"] = info.get("clock", {}).get("duration_s")
        return info

    @property
    def metadata(self):
        return self.info

    @cached_property
    def sidecars(self):
        if "provenance" not in self.file:
            return {}
        return json.loads(self.file["provenance"].attrs.get("sidecars_json", "{}"))

    @cached_property
    def availability(self):
        values = (json.loads(self.file["provenance"].attrs.get("availability_json", "{}"))
                  if "provenance" in self.file else {})
        values.update(units=len(self.units) > 0, trials="trials/start" in self.file and "trials/end" in self.file,
                      cue="trials/goCue_start_time" in self.file)
        return values

    @cached_property
    def identity(self):
        """Portable dataset identity; active exports supply their checked source digest."""
        if "source_identity" in self.file.attrs:
            return ":".join(text(self.file.attrs.get(key, "")) for key in
                            ("active_schema_version", "source_identity", "implementation_sha256"))
        # Passive Brainsets files without a content identity get a streaming digest.
        # This runs only when a dataset is opened for an actual compute workflow.
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(block)
        return digest.hexdigest()

    @cached_property
    def units(self):
        frame = table(self.file["units"])
        has_labels = "predicted_cell_type" in frame
        count = len(frame)
        original = frame["source_id"] if "source_id" in frame else frame["id"]
        frame["id"] = original
        frame["unit_id"] = original
        frame["source_id"] = original
        frame["source_unit_row"] = np.arange(count, dtype=np.int64)
        for name in ("animal", "session", "recording"):
            frame[name] = self.info[name]
        if "global_unit_id" not in frame:
            frame["global_unit_id"] = [f"{self.info['animal']}/{self.info['session']}/{self.info['recording']}/{value}"
                                       for value in original]
        if frame.global_unit_id.duplicated().any() or original.duplicated().any():
            raise ValueError("Duplicate unit identities")
        if "quality" in self.file:
            for name, value in self.file["quality"].items():
                if isinstance(value, h5py.Dataset) and value.shape == (count,):
                    frame[name] = array(value)
        for name in ("default_qc", "hist_qc_failed"):
            if name in frame:
                values = frame[name].astype("string").str.strip().str.lower()
                frame[name] = values.map({"true": True, "1": True, "1.0": True,
                                          "false": False, "0": False, "0.0": False}).astype("boolean")
        if "ccf_acronym" not in frame and "brain_region" in frame:
            frame["ccf_acronym"] = frame.brain_region
        if "brain_region" not in frame and "ccf_acronym" in frame:
            frame["brain_region"] = frame.ccf_acronym
        for name in ("ccf_acronym", "brain_region", "decoder_label", "predicted_cell_type", "device_name"):
            if name not in frame:
                frame[name] = pd.Series(pd.NA, index=frame.index, dtype="string")
        if "depth" not in frame:
            frame["depth"] = np.nan
        labels = frame.get("predicted_cell_type", pd.Series(pd.NA, index=frame.index, dtype="string")).astype("string").str.strip()
        frame["original_label"] = labels
        frame["label"] = labels.mask(labels.isna() | labels.str.lower().isin(
            ["", "nan", "none", "unknown", "untagged", "unlabeled", "unlabelled"]), "Unlabeled")
        frame["is_labeled"] = frame.label.ne("Unlabeled")
        frame["label_source"] = "recorded label" if has_labels else "missing label column"
        if "resolved_peak_channel" in frame:
            frame["peak_channel"] = frame.get("peak_channel", pd.Series(np.nan, index=frame.index)).fillna(frame.resolved_peak_channel)
        return waveform_columns(frame)

    @cached_property
    def trials(self):
        if not self.availability["trials"]:
            raise ValueError(f"Trial labels are unavailable: {self.path.name}")
        group = self.file["trials"]
        count = len(group["start"])
        frame = table(group, count)
        frame["start_time"] = frame["start"]
        frame["stop_time"] = frame["end"]
        if "source_trial_row" not in frame:
            frame["source_trial_row"] = np.arange(count)
        return frame

    def _interval(self, path):
        if path not in self.file or not {"start", "end"}.issubset(self.file[path]):
            raise ValueError(f"Interval {path!r} is unavailable in {self.path.name}")
        group = self.file[path]
        return merge(np.column_stack([group["start"][:], group["end"][:]]))

    def intervals(self, domain="full", buffer_s=0):
        if not np.isfinite(buffer_s) or buffer_s < 0:
            raise ValueError("buffer_s must be finite and nonnegative")
        clock = self.info.get("clock", {})
        if clock.get("origin_s") is not None and clock.get("duration_s") is not None:
            bounds = merge([[clock["origin_s"], clock["origin_s"] + clock["duration_s"]]])
        else:
            bounds = self._interval("domain")
        if domain == "full":
            if buffer_s:
                raise ValueError("A full-recording selection cannot have an optotagging buffer")
            return bounds
        if domain == "pre_stimulus":
            result = intersect(self._interval("intervals/blocks/pre_stimulus"), bounds)
            result = intersect(result, self._interval("spikes/domain"))
            excluded = self._interval("intervals/blocks/stimulation")
            excluded = excluded + np.array([-buffer_s, buffer_s])
            return subtract(result, excluded)
        if domain == "pre_opto":
            if "intervals/pre_opto" in self.file:
                result = intersect(self._interval("intervals/pre_opto"), bounds)
            else:
                onset = self.opto_start
                if onset is None:
                    raise ValueError("A pre-opto selection requires an established optotagging onset")
                result = intersect(bounds, [[bounds[0, 0], onset]]) if onset > bounds[0, 0] else np.empty((0, 2))
        elif domain in {"baseline", "behavior", "optotagging"}:
            result = intersect(self._interval(f"intervals/{domain}"), bounds)
        else:
            raise ValueError(f"Unknown recording domain: {domain}")
        if buffer_s:
            if domain == "optotagging":
                raise ValueError("An optotagging selection cannot have a pre-opto buffer")
            onset = self.opto_start
            if onset is None:
                raise ValueError("An optotagging buffer requires an established onset")
            cutoff = onset - buffer_s
            return intersect(result, [[bounds[0, 0], cutoff]]) if cutoff > bounds[0, 0] else np.empty((0, 2))
        return result

    @cached_property
    def opto_start(self):
        if "intervals" in self.file:
            attrs = self.file["intervals"].attrs
            if attrs.get("opto_onset_available", False):
                return float(attrs["opto_onset_s"])
        candidates = []
        if self.has_events("opto_start"):
            values = self.events("opto_start")
            if len(values):
                candidates.append(float(np.min(values)))
        if "intervals/optotagging" in self.file:
            values = self._interval("intervals/optotagging")
            if len(values):
                candidates.append(float(values[0, 0]))
        timing, offset, clock = self.info.get("session_timing"), self.info.get("session_offset_s"), self.info.get("clock", {})
        if not candidates and timing and offset is not None and clock.get("origin_s") is not None:
            candidates.append(clock["origin_s"] + timing["first_opto_elapsed_s"] - offset)
        return min(candidates) if candidates else None

    def _event_path(self, name):
        return f"behavior/{name}" if name in {"running_onsets", "whisking_onsets"} and f"events/{name}" not in self.file else f"events/{name}"

    def has_events(self, name):
        path = self._event_path(name)
        return path in self.file and "timestamps" in self.file[path] and self._available(self.file[path])

    @staticmethod
    def _available(group):
        return bool(group.attrs.get("available", text(group.attrs.get("status", "present")) in {"present", "empty"}))

    def events(self, name):
        if not self.has_events(name):
            raise ValueError(f"Event signal {name!r} is unavailable in {self.path.name}")
        values = np.asarray(self.file[self._event_path(name) + "/timestamps"][:], dtype=float)
        if not np.isfinite(values).all() or np.any(np.diff(values) < 0):
            raise ValueError(f"Invalid {name} timestamps")
        return values

    def _signal_path(self, name, domain="full"):
        if domain not in {"full", "baseline", "behavior", "optotagging", "pre_opto"}:
            raise ValueError(f"Unknown signal domain: {domain}")
        prefix = "signals" if domain == "full" else f"signal_domains/{domain}/signals"
        path = f"{prefix}/{name}"
        if domain == "full" and path not in self.file and name in {"running", "whisking"}:
            path = f"behavior/{name}_speed"
        return path

    def has_signal(self, name, domain="full"):
        path = self._signal_path(name, domain)
        return (path in self.file and "timestamps" in self.file[path]
                and bool({"data", "speed"}.intersection(self.file[path]))
                and self._available(self.file[path]))

    def signal_names(self, domain="full"):
        """Names of readable signal leaves, including multi-series containers."""
        names = []
        def visit(group, prefix):
            for name, child in group.items():
                if not isinstance(child, h5py.Group):
                    continue
                key = prefix + name
                if self.has_signal(key, domain):
                    names.append(key)
                else:
                    visit(child, key + "/")
        root = self._signal_path("", domain).rstrip("/")
        if root in self.file:
            visit(self.file[root], "")
        for name in ("running", "whisking"):
            if self.has_signal(name, domain):
                names.append(name)
        return sorted(set(names))

    def signal_quality(self, name, domain="full"):
        """Inspect defects even when a signal/domain is unavailable."""
        path = self._signal_path(name, domain)
        if path not in self.file:
            return dict(status="missing", available=False, domain=domain)
        group = self.file[path]
        quality = json.loads(group.attrs.get("quality_json", "{}"))
        quality.update(status=text(group.attrs.get("status", "present")),
                       available=self._available(group), domain=domain)
        return quality

    def signal_info(self, name, domain="full"):
        if not self.has_signal(name, domain):
            quality = self.signal_quality(name, domain)
            raise ValueError(f"Continuous signal {name!r} is unavailable in {domain}: {quality['status']}")
        group = self.file[self._signal_path(name, domain)]
        key = "data" if "data" in group else "speed"
        info = {**dict(group.attrs), **dict(group[key].attrs)}
        info["unit"] = text(info.get("unit", info.get("units", "")))
        info["source_unit"] = info["unit"]
        if name == "speed":
            unit = info["unit"].lower().replace(" ", "")
            if unit not in {"m/s", "meters/second", "metres/second", "cm/s", "cm/sec", "centimeters/second", "centimetres/second"}:
                raise ValueError(f"Speed unit is not declared in cm/s or m/s: {info['unit']!r}")
            info["unit"] = "cm/s"
        info["units"] = info["unit"]
        return info

    def signal(self, name, domain="full"):
        """Read one validated domain on the unchanged source clock."""
        info = self.signal_info(name, domain)
        group = self.file[self._signal_path(name, domain)]
        key = "data" if "data" in group else "speed"
        times = np.asarray(group["timestamps"][:], dtype=float)
        values = np.asarray(group[key][:], dtype=float)
        if len(values) != len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError(f"Invalid time axis for {name}")
        values = values * float(info.get("conversion", 1.0)) + float(info.get("offset", 0.0))
        if name == "speed":
            unit = info["source_unit"].lower().replace(" ", "")
            if unit in {"m/s", "meters/second", "metres/second"}:
                values *= 100
        return times, values

    def spike_window(self, start, end):
        if "spikes" not in self.file:
            return np.empty(0), np.empty(0, dtype=np.int64)
        times = self.file["spikes/timestamps"]
        low, high = lower_bound(times, start), lower_bound(times, end)
        return np.asarray(times[low:high], dtype=float), np.asarray(self.file["spikes/unit_index"][low:high], dtype=np.int64)

    def _unit_row(self, source_unit_row):
        row = int(source_unit_row)
        if row != source_unit_row or not 0 <= row < len(self.units):
            raise IndexError("source_unit_row is outside the unit table")
        return row

    def spikes(self, source_unit_row, start=None, end=None):
        row = self._unit_row(source_unit_row)
        if "spike_index" in self.file:
            offsets = self.file["spike_index/offsets"]
            values = np.asarray(self.file["spike_index/timestamps"][int(offsets[row]):int(offsets[row + 1])], dtype=float)
        else:
            pieces = []
            if "spikes" not in self.file:
                return np.empty(0)
            times, indices = self.file["spikes/timestamps"], self.file["spikes/unit_index"]
            low = 0 if start is None else lower_bound(times, start)
            high = len(times) if end is None else lower_bound(times, end)
            for a in range(low, high, 1_048_576):
                b = min(a + 1_048_576, high)
                pieces.append(np.asarray(times[a:b])[indices[a:b] == row])
            values = np.concatenate(pieces) if pieces else np.empty(0)
        low = 0 if start is None else np.searchsorted(values, start, side="left")
        high = len(values) if end is None else np.searchsorted(values, end, side="left")
        return values[low:high]

    def spike_counts(self, intervals):
        result = np.zeros(len(self.units), dtype=np.int64)
        if "spikes" not in self.file:
            return result
        times, indices = self.file["spikes/timestamps"], self.file["spikes/unit_index"]
        for start, end in merge(intervals):
            low, high = lower_bound(times, start), lower_bound(times, end)
            for a in range(low, high, 1_048_576):
                rows = np.asarray(indices[a:min(a + 1_048_576, high)], dtype=np.int64)
                result += np.bincount(rows, minlength=len(result))
        return result

    def waveform(self, source_unit_row):
        row = self._unit_row(source_unit_row)
        if "units/waveform_mean" not in self.file:
            raise ValueError("Mean waveforms are unavailable")
        return np.asarray(self.file["units/waveform_mean"][row])

    def waveform_channels(self, source_unit_row):
        row = self._unit_row(source_unit_row)
        if ("electrodes" not in self.file or "unit_electrodes/mapping_status" not in self.file
                or "units/waveform_mean" not in self.file):
            raise ValueError("Waveform electrode mapping is unavailable")
        status = text(self.file["unit_electrodes/mapping_status"][row])
        if status not in {"source_references", "device_ap_order"}:
            raise ValueError(f"Waveform electrode mapping is unavailable for unit {row}: {status}")
        offsets = self.file["unit_electrodes/offsets"]
        rows = np.asarray(self.file["unit_electrodes/electrode_row"][int(offsets[row]):int(offsets[row + 1])], dtype=int)
        group = self.file["electrodes"]
        n = len(group["source_id"]) if "source_id" in group else len(group["id"])
        waveform = self.file["units/waveform_mean"]
        if (waveform.ndim != 3 or len(rows) != waveform.shape[-1]
                or np.any((rows < 0) | (rows >= n)) or len(np.unique(rows)) != len(rows)):
            raise ValueError("Waveform channel axis does not match its electrode references")
        frame = table(group, n).iloc[rows].copy()
        if "channel" not in frame and "channel_name" in frame:
            frame["channel"] = pd.to_numeric(frame.channel_name.astype(str).str.extract(r"^(?:AP|CH)(\d+)$", expand=False), errors="coerce")
        return frame

    @property
    def waveform_sampling_rate_hz(self):
        for attrs in (self.file["units"].attrs, self.file.attrs):
            for name in ("waveform_sampling_rate_hz", "sampling_rate_hz", "sampling_rate"):
                if name in attrs:
                    value = float(attrs[name])
                    if np.isfinite(value) and value > 0:
                        return value
        raise ValueError("Waveform sampling frequency is unavailable")

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
