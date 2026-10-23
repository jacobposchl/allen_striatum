"""Electrophysiology summaries with explicit measurement intervals."""
import numpy as np


def spike_features(spikes, intervals):
    """Rate and ISI features on eligible time only; never span interval gaps.

    prop_long_isi is the fraction of eligible time occupied by ISIs >100 ms.
    isi_violations_ratio uses a 1.5 ms refractory threshold and no censored bin:
    violations / (2 * threshold * spike_count * firing_rate).
    """
    intervals = np.asarray(intervals, dtype=float)
    duration = float(np.diff(intervals, axis=1).sum())
    if duration <= 0:
        raise ValueError("Eligible duration must be positive")
    count, pieces = 0, []
    for start, end in intervals:
        low, high = np.searchsorted(spikes, [start, end], side="left")
        count += high - low
        if high - low >= 2:
            pieces.append(np.diff(spikes[low:high]))
    isi = np.concatenate(pieces) if pieces else np.empty(0)
    rate = count / duration
    result = dict(firing_rate_hz=rate, eligible_spike_count=int(count),
                  isi_cv=np.nan, isi_mean_s=np.nan, isi_violations_ratio=np.nan,
                  frac_isi_lt_2ms=np.nan, frac_isi_lt_10ms=np.nan,
                  frac_isi_gt_500ms=np.nan, prop_long_isi=np.nan)
    if len(isi):
        mean = float(isi.mean())
        result.update(isi_cv=float(isi.std() / mean) if mean > 0 else np.nan,
                      isi_mean_s=mean,
                      isi_violations_ratio=float(np.sum(isi < .0015) / (.003 * count * rate)),
                      frac_isi_lt_2ms=float(np.mean(isi < .002)),
                      frac_isi_lt_10ms=float(np.mean(isi < .010)),
                      frac_isi_gt_500ms=float(np.mean(isi > .500)),
                      prop_long_isi=float(isi[isi > .100].sum() / duration))
    return result


def streamed_spike_features(recording, source_unit_rows, intervals, block_size=1_048_576):
    """Summarize a time-ordered spike table once, without a per-unit index.

    Memory is bounded by one spike chunk and per-unit accumulators. Stable
    grouping retains each unit's timestamp order; a previous spike is carried
    across chunks but reset between eligible intervals.
    """
    from src.data.recording import lower_bound

    source_unit_rows = np.asarray(source_unit_rows, dtype=np.int64)
    n_units = len(source_unit_rows)
    if not n_units:
        return []
    intervals = np.asarray(intervals, dtype=float)
    duration = float(np.diff(intervals, axis=1).sum())
    if duration <= 0:
        raise ValueError("Eligible duration must be positive")
    if block_size < 1 or int(block_size) != block_size:
        raise ValueError("Spike chunk size must be a positive integer")
    row_map = np.full(len(recording.units), -1, dtype=np.int64)
    if (np.any(source_unit_rows < 0) or np.any(source_unit_rows >= len(row_map))
            or len(np.unique(source_unit_rows)) != n_units):
        raise ValueError("Feature unit rows must be unique original unit-table rows")
    row_map[source_unit_rows] = np.arange(n_units)
    spike_count = np.zeros(n_units, dtype=np.int64)
    isi_count = np.zeros(n_units, dtype=np.int64)
    isi_mean = np.zeros(n_units)
    isi_m2 = np.zeros(n_units)
    violations = np.zeros(n_units, dtype=np.int64)
    below_2ms = np.zeros(n_units, dtype=np.int64)
    below_10ms = np.zeros(n_units, dtype=np.int64)
    above_500ms = np.zeros(n_units, dtype=np.int64)
    long_isi_time = np.zeros(n_units)
    if "spikes" in recording.file:
        timestamps = recording.file["spikes/timestamps"]
        unit_index = recording.file["spikes/unit_index"]
        for start, end in intervals:
            low, high = lower_bound(timestamps, start), lower_bound(timestamps, end)
            previous = np.full(n_units, np.nan)
            for first in range(low, high, int(block_size)):
                last = min(first + int(block_size), high)
                times = np.asarray(timestamps[first:last], dtype=float)
                source_rows = np.asarray(unit_index[first:last], dtype=np.int64)
                if np.any(source_rows < 0) or np.any(source_rows >= len(row_map)):
                    raise ValueError("A spike references an invalid unit row")
                rows = row_map[source_rows]
                keep = rows >= 0
                rows, times = rows[keep], times[keep]
                if not len(rows):
                    continue
                if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                    raise ValueError("Canonical spike timestamps must be finite and ordered")
                order = np.argsort(rows, kind="stable")
                rows, times = rows[order], times[order]
                boundaries = np.r_[0, np.flatnonzero(np.diff(rows)) + 1, len(rows)]
                for a, b in zip(boundaries[:-1], boundaries[1:]):
                    row = rows[a]
                    values = times[a:b]
                    spike_count[row] += b - a
                    isi = np.diff(np.r_[previous[row], values]) if np.isfinite(previous[row]) else np.diff(values)
                    previous[row] = values[-1]
                    if not len(isi):
                        continue
                    if np.any(isi < 0):
                        raise ValueError("Canonical unit spikes are not ordered across chunks")
                    # Combine population central moments without retaining ISIs.
                    count = len(isi)
                    mean = float(isi.mean())
                    m2 = float(np.square(isi - mean).sum())
                    old_count = int(isi_count[row])
                    total = old_count + count
                    delta = mean - isi_mean[row]
                    isi_m2[row] += m2 + delta * delta * old_count * count / total
                    isi_mean[row] += delta * count / total
                    isi_count[row] = total
                    violations[row] += np.count_nonzero(isi < .0015)
                    below_2ms[row] += np.count_nonzero(isi < .002)
                    below_10ms[row] += np.count_nonzero(isi < .010)
                    above_500ms[row] += np.count_nonzero(isi > .500)
                    long_isi_time[row] += isi[isi > .100].sum()
    summaries = []
    for row in range(n_units):
        rate = spike_count[row] / duration
        result = dict(firing_rate_hz=rate, eligible_spike_count=int(spike_count[row]),
                      isi_cv=np.nan, isi_mean_s=np.nan, isi_violations_ratio=np.nan,
                      frac_isi_lt_2ms=np.nan, frac_isi_lt_10ms=np.nan,
                      frac_isi_gt_500ms=np.nan, prop_long_isi=np.nan)
        count = int(isi_count[row])
        if count:
            mean = float(isi_mean[row])
            std = np.sqrt(max(0.0, isi_m2[row] / count))
            result.update(isi_cv=float(std / mean) if mean > 0 else np.nan,
                          isi_mean_s=mean,
                          isi_violations_ratio=float(violations[row] / (.003 * spike_count[row] * rate)),
                          frac_isi_lt_2ms=float(below_2ms[row] / count),
                          frac_isi_lt_10ms=float(below_10ms[row] / count),
                          frac_isi_gt_500ms=float(above_500ms[row] / count),
                          prop_long_isi=float(long_isi_time[row] / duration))
        summaries.append(result)
    return summaries


def recording_features(recording, units, intervals):
    """Calculate selected-time summaries directly from HDF5 spike timestamps."""
    result = units.copy()
    if "spike_index" in recording.file:
        rows = [spike_features(recording.spikes(row), intervals) for row in units.source_unit_row]
    else:
        rows = streamed_spike_features(recording, units.source_unit_row, intervals)
    for name in rows[0] if rows else []:
        if name in result:
            result[f"source_{name}"] = result[name]
        result[name] = [row[name] for row in rows]
    result["feature_interval_policy"] = "eligible_intervals_only"
    return result


def waveform_columns(frame):
    """Normalize available source scalars; missing waveform metrics stay missing."""
    import pandas as pd

    for target, sources, scale in (
        ("waveform_duration_ms", ("peak_to_valley",), 1000.0),
        ("waveform_spread_um", ("new_spread_interpolated", "spread_interpolated"), 1.0),
    ):
        if target not in frame:
            frame[target] = np.nan
        for source in sources:
            if source in frame:
                frame[target] = frame[target].fillna(pd.to_numeric(frame[source], errors="coerce") * scale)
    return frame
