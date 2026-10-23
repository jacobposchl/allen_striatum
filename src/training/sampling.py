from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.data.base import Session


@dataclass(frozen=True)
class PairIndex:
    key: tuple[str, str]
    start_a: float
    start_b: float
    seed: int


@dataclass(frozen=True)
class Sample:
    window_a: np.ndarray
    window_b: np.ndarray
    units_a: np.ndarray
    units_b: np.ndarray
    anchor_units: np.ndarray
    anchor_rows_a: np.ndarray
    anchor_rows_b: np.ndarray
    starts: np.ndarray

    @property
    def n_anchors(self) -> int:
        return len(self.anchor_units)


@dataclass(frozen=True)
class SampleBatch:
    window_a: np.ndarray
    window_b: np.ndarray
    unit_mask_a: np.ndarray
    unit_mask_b: np.ndarray
    anchor_rows_a: np.ndarray
    anchor_rows_b: np.ndarray
    anchor_mask: np.ndarray
    starts: np.ndarray

    @property
    def size(self) -> int:
        return len(self.window_a)


def sequential_starts(low: float, high: float, duration: float) -> np.ndarray:
    if duration <= 0:
        raise ValueError("duration must be positive")
    n_windows = max(0, int(np.floor((high - low) / duration + 1e-9)))
    starts = low + duration * np.arange(n_windows, dtype=np.float64)
    return starts[starts + duration <= high]


def eligible_intervals(session):
    """Use only retained recording intervals; never bridge stimulation or gaps."""
    intervals = np.asarray(getattr(session, "eligible_intervals", [[session.t_start, session.t_end]]),
                           dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all():
        raise ValueError("eligible_intervals must be finite [start, end] pairs")
    if np.any(intervals[:, 1] <= intervals[:, 0]) or np.any(intervals[1:, 0] < intervals[:-1, 1]):
        raise ValueError("eligible_intervals must be ordered, non-overlapping and positive")
    return intervals


def session_starts(session, duration):
    intervals = eligible_intervals(session)
    return np.concatenate([sequential_starts(low, high, duration) for low, high in intervals]) \
        if len(intervals) else np.empty(0, dtype=float)


def random_starts(low: float, high: float, duration: float, rng: np.random.Generator) -> np.ndarray:
    if duration <= 0:
        raise ValueError("duration must be positive")
    n_windows = max(0, int(np.floor((high - low) / duration + 1e-9)))
    if n_windows < 1 or high - low < duration:
        return np.empty(0, dtype=float)
    offset = rng.uniform(0.0, duration)
    starts = low + offset + duration * np.arange(n_windows, dtype=float)
    starts = starts[starts + duration <= high]
    if len(starts) < n_windows:
        left = starts[0] - low if len(starts) else offset
        right = high - starts[-1] - duration if len(starts) else high - low - offset
        starts = np.append(starts, high - duration if right > left else low)
    return starts[(starts >= low) & (starts + duration <= high)]


def sample_second_start(
    first: float,
    low: float,
    high: float,
    duration: float,
    max_distance: float,
    rng: np.random.Generator,
    min_distance: float = 0.0,
) -> float:
    if duration <= 0 or not 0 <= min_distance <= max_distance:
        raise ValueError("Invalid window duration or pairing distances")
    last_start = high - duration
    intervals = (
        (max(low, first - max_distance), min(last_start, first - min_distance)),
        (max(low, first + min_distance), min(last_start, first + max_distance)),
    )
    intervals = [(left, right) for left, right in intervals if right >= left]
    lengths = np.asarray([right - left for left, right in intervals])
    if not len(intervals):
        raise ValueError("no valid second window")
    if lengths.sum() == 0:
        return float(intervals[int(rng.integers(len(intervals)))][0])
    index = int(rng.choice(len(intervals), p=lengths / lengths.sum()))
    left, right = intervals[index]
    return float(rng.uniform(left, right))


def sample_population(n_units: int, min_fraction: float, rng: np.random.Generator) -> np.ndarray:
    if not 0 < min_fraction <= 1:
        raise ValueError("min_unit_fraction must be in (0, 1]")
    minimum = min(n_units, max(2, int(np.floor(min_fraction * n_units))))
    n_selected = int(rng.integers(minimum, n_units + 1))
    return rng.permutation(n_units)[:n_selected].astype(np.int64, copy=False)


def dense_window(active: np.ndarray, active_window: np.ndarray, units: np.ndarray, n_bins: int) -> np.ndarray:
    window = np.zeros((len(units), n_bins), dtype=np.float32)
    common, unit_rows, active_rows = np.intersect1d(units, active, assume_unique=True, return_indices=True)
    if len(common):
        window[unit_rows] = active_window[active_rows]
    return window


def pair_indices(
    sessions: Mapping[tuple[str, str], Session],
    keys: Sequence[tuple[str, str]],
    rng: np.random.Generator,
    sampling,
    random_offset: bool,
    min_distance: float = 0.0,
) -> list[PairIndex]:
    pairs = []
    for key in keys:
        session = sessions[key]
        for low, high in eligible_intervals(session):
            starts = (
                random_starts(low, high, sampling.duration, rng)
                if random_offset
                else sequential_starts(low, high, sampling.duration)
            )
            for start in starts:
                if max(float(start) - low, high - sampling.duration - float(start)) < min_distance:
                    continue
                second = sample_second_start(float(start), low, high, sampling.duration,
                                             sampling.max_distance, rng, min_distance)
                pairs.append(PairIndex(key, float(start), second, int(rng.integers(0, 2**32))))
    if random_offset and pairs:
        order = rng.permutation(len(pairs))
        pairs = [pairs[index] for index in order]
    return pairs


def sample_pair(session: Session, index: PairIndex, sampling, unit_dropout: bool) -> Sample:
    rng = np.random.default_rng(index.seed)
    starts = np.asarray([index.start_a, index.start_b], dtype=float)
    active_a, active_window_a = session.make_view(index.start_a, sampling.duration, sampling.bin_size)
    active_b, active_window_b = session.make_view(index.start_b, sampling.duration, sampling.bin_size)
    if unit_dropout:
        units_a = sample_population(session.n_units, sampling.min_unit_fraction, rng)
        units_b = sample_population(session.n_units, sampling.min_unit_fraction, rng)
        if not np.intersect1d(units_a, units_b).size:
            units_b[0] = units_a[0]
    else:
        units_a = np.arange(session.n_units, dtype=np.int64)
        units_b = units_a
    anchors, rows_a, rows_b = np.intersect1d(units_a, units_b, assume_unique=True, return_indices=True)
    n_bins = int(round(sampling.duration / sampling.bin_size))
    return Sample(
        window_a=dense_window(active_a, active_window_a, units_a, n_bins),
        window_b=dense_window(active_b, active_window_b, units_b, n_bins),
        units_a=units_a,
        units_b=units_b,
        anchor_units=anchors,
        anchor_rows_a=rows_a,
        anchor_rows_b=rows_b,
        starts=starts,
    )


def sample_batch(
    sessions: Mapping[tuple[str, str], Session],
    indices: Sequence[PairIndex],
    sampling,
    unit_dropout: bool,
) -> SampleBatch:
    samples = [sample_pair(sessions[index.key], index, sampling, unit_dropout) for index in indices]
    return collate_samples(samples)


def collate_samples(samples: Sequence[Sample]) -> SampleBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    n_bins = samples[0].window_a.shape[1]
    max_a = max(len(sample.units_a) for sample in samples)
    max_b = max(len(sample.units_b) for sample in samples)
    max_anchors = max(sample.n_anchors for sample in samples)
    window_a = np.zeros((batch_size, max_a, n_bins), dtype=np.float32)
    window_b = np.zeros((batch_size, max_b, n_bins), dtype=np.float32)
    unit_mask_a = np.zeros((batch_size, max_a), dtype=bool)
    unit_mask_b = np.zeros((batch_size, max_b), dtype=bool)
    anchor_rows_a = np.zeros((batch_size, max_anchors), dtype=np.int64)
    anchor_rows_b = np.zeros((batch_size, max_anchors), dtype=np.int64)
    anchor_mask = np.zeros((batch_size, max_anchors), dtype=bool)
    for row, sample in enumerate(samples):
        n_a, n_b, n_anchors = len(sample.units_a), len(sample.units_b), sample.n_anchors
        window_a[row, :n_a] = sample.window_a
        window_b[row, :n_b] = sample.window_b
        unit_mask_a[row, :n_a] = True
        unit_mask_b[row, :n_b] = True
        anchor_rows_a[row, :n_anchors] = sample.anchor_rows_a
        anchor_rows_b[row, :n_anchors] = sample.anchor_rows_b
        anchor_mask[row, :n_anchors] = True
    return SampleBatch(
        window_a=window_a,
        window_b=window_b,
        unit_mask_a=unit_mask_a,
        unit_mask_b=unit_mask_b,
        anchor_rows_a=anchor_rows_a,
        anchor_rows_b=anchor_rows_b,
        anchor_mask=anchor_mask,
        starts=np.stack([sample.starts for sample in samples]),
    )


def batches(indices: Sequence[PairIndex], batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]
