"""Cohort and selection views over immutable Brainsets HDF5 recordings."""
from functools import cached_property
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.cohort import normalized_cohort, select_manifest
from src.data.io import resolve
from src.data.recording import Recording, discover_recordings


def selection_options(config):
    options = dict(domain=str(config.get("domain", "full")),
                   buffer_s=float(config.get("buffer_s", 0)), qc=str(config.get("qc", "none")),
                   exclude_noise=bool(config.get("exclude_noise", False)),
                   exclude_histology_failed=bool(config.get("exclude_histology_failed", False)),
                   tip_sites=config.get("tip_sites"),
                   include_unknown_regions=bool(config.get("include_unknown_regions", False)))
    for name in ("regions", "sorting", "labels"):
        values = config.get(name)
        if isinstance(values, str):
            raise ValueError(f"data.{name} must be a list or null")
        options[name] = None if values is None else sorted(set(map(str, values)))
    if options["qc"] not in {"none", "default", "pre_opto"}:
        raise ValueError("data.qc must be none, default, or pre_opto")
    if not np.isfinite(options["buffer_s"]) or options["buffer_s"] < 0:
        raise ValueError("data.buffer_s must be finite and nonnegative")
    tip = options["tip_sites"]
    if tip is not None and (int(tip) != tip or tip < 1):
        raise ValueError("data.tip_sites must be a positive integer or null")
    return options


def select_units(units, options):
    keep = pd.Series(True, index=units.index)
    for option, field in (("sorting", "decoder_label"), ("labels", "predicted_cell_type")):
        if options.get(option) is not None:
            if field not in units:
                raise ValueError(f"Requested {option} labels are unavailable")
            keep &= units[field].isin(options[option])
    if options.get("regions") is not None:
        region = units.get("ccf_acronym", pd.Series(pd.NA, index=units.index, dtype="string")).astype("string")
        unknown = region.isna() | region.str.lower().isin(["", "unknown", "unassigned"])
        keep &= region.isin(options["regions"]) | (options.get("include_unknown_regions", False) & unknown)
    if options.get("exclude_noise"):
        if "decoder_label" not in units:
            raise ValueError("Sorting labels are unavailable for noise exclusion")
        label = units.decoder_label.astype("string").str.strip().str.lower()
        keep &= label.notna() & ~label.isin(["noise", "", "nan", "none", "unknown"])
    qc = options.get("qc", "none")
    if qc != "none":
        field = "default_qc" if qc == "default" else "qc_pre_opto"
        if field not in units:
            raise ValueError(f"Requested QC field {field} is unavailable")
        keep &= units[field].eq(True if qc == "default" else 1).fillna(False)
    if options.get("exclude_histology_failed"):
        if "hist_qc_failed" not in units:
            raise ValueError("Histology QC is unavailable")
        keep &= units.hist_qc_failed.eq(False).fillna(False)
    if options.get("tip_sites") is not None:
        if "tip_site_rank" not in units:
            raise ValueError("Probe site mapping is unavailable")
        keep &= units.tip_site_rank.le(options["tip_sites"]).fillna(False)
    return units.loc[keep.fillna(False)].copy().reset_index(drop=True)


class RecordingView:
    """A unit/time selection for sampling; no filtered data are written to disk."""
    def __init__(self, path, options):
        self.recording = Recording(path)
        try:
            self.units = select_units(self.recording.units, options)
            source_intervals = self.recording.intervals(options["domain"], options["buffer_s"])
            if not len(source_intervals):
                raise ValueError("No eligible recording intervals")
            self.source_intervals = source_intervals
            self.origin_s = float(source_intervals[0, 0])
            self.eligible_intervals = source_intervals - self.origin_s
            self.t_start, self.t_end = 0.0, float(self.eligible_intervals[-1, 1])
            self.animal, self.session = self.recording.info["animal"], self.recording.info["session"]
            info = self.recording.info
            clock_origin = info.get("clock", {}).get("origin_s")
            offset = info.get("session_offset_s")
            session_offset = (float(offset) + self.origin_s - float(clock_origin)
                              if offset is not None and clock_origin is not None else None)
            self.metadata = dict(info, source=str(self.recording.path), origin_s=self.origin_s,
                                 session_offset_s=session_offset, retained_duration_s=self.t_end,
                                 eligible_duration_s=float(np.diff(source_intervals, axis=1).sum()),
                                 n_units=len(self.units), total_units=len(self.recording.units))
            self._row_map = np.full(len(self.recording.units), -1, dtype=np.int64)
            self._row_map[self.units.source_unit_row.to_numpy(int)] = np.arange(len(self.units))
        except BaseException:
            self.recording.close()
            raise

    @property
    def n_units(self):
        return len(self.units)

    @cached_property
    def firing_rates_hz(self):
        counts = self.recording.spike_counts(self.source_intervals)
        return counts[self.units.source_unit_row.to_numpy(int)] / np.diff(self.source_intervals, axis=1).sum()

    def make_view(self, start, duration, bin_size):
        if duration <= 0 or bin_size <= 0:
            raise ValueError("duration and bin_size must be positive")
        n_bins = round(duration / bin_size)
        if abs(n_bins * bin_size - duration) > 1e-9:
            raise ValueError("duration must be a multiple of bin_size")
        if not np.any((start >= self.eligible_intervals[:, 0]) &
                      (start + duration <= self.eligible_intervals[:, 1])):
            raise ValueError("Window crosses a recording boundary, gap, or excluded interval")
        times, source_rows = self.recording.spike_window(start + self.origin_s, start + duration + self.origin_s)
        rows = self._row_map[source_rows]
        keep = rows >= 0
        rows, times = rows[keep], times[keep]
        active, inverse = np.unique(rows, return_inverse=True)
        if not len(active):
            return active, np.empty((0, n_bins), dtype=np.float32)
        bins = np.floor((times - self.origin_s - start) / bin_size).astype(np.int64)
        bins = np.clip(bins, 0, n_bins - 1)
        counts = np.bincount(inverse * n_bins + bins, minlength=len(active) * n_bins)
        return active, counts.reshape(len(active), n_bins).astype(np.float32)

    def close(self):
        self.recording.close()


class BrainsetsDataset:
    def __init__(self, config, cohort=None):
        if config.get("format", "brainsets") != "brainsets":
            raise ValueError("Only direct Brainsets HDF5 recording inputs are supported")
        self.root = resolve(config.root)
        self.options = selection_options(config)
        records = discover_recordings(self.root)
        if not records:
            raise ValueError(f"No Brainsets HDF5 recordings at {self.root}; run preprocessing first")
        selected = select_manifest(pd.DataFrame(records), cohort)
        rows, self.rows = [], {}
        for item in selected.to_dict(orient="records"):
            row = {k: v for k, v in item.items() if k != "path"}
            try:
                view = RecordingView(item["path"], self.options)
                try:
                    if view.n_units < 2:
                        raise ValueError("Fewer than two eligible units")
                    row.update(status="ready", reason="", **{k: view.metadata[k] for k in
                               ("origin_s", "session_offset_s", "retained_duration_s", "eligible_duration_s", "n_units", "total_units")})
                finally:
                    view.close()
            except ValueError as error:
                row.update(status="excluded", reason=str(error))
            rows.append(row)
            if row["status"] == "ready":
                key = (str(row["animal"]), f"{row['session']}/{row['recording']}")
                if key in self.rows:
                    raise ValueError(f"Duplicate recording identity: {key}")
                self.rows[key] = row
        self.manifest = pd.DataFrame(rows)
        portable = [{k: v for k, v in row.items() if k != "source"} for row in rows]
        identity = dict(options=self.options, cohort=normalized_cohort(cohort), recordings=portable)
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        if not self.rows:
            reasons = self.manifest[["id", "reason"]].to_dict(orient="records")
            raise ValueError(f"No eligible Brainsets recordings: {reasons}")

    def keys(self):
        return list(self.rows)

    def load_session(self, animal, recording):
        return RecordingView(self.rows[(animal, recording)]["source"], self.options)
