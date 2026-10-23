"""Temporal bases and event design matrices for response models."""
import numpy as np


def temporal_basis(lags, start, stop, count):
    centers = np.linspace(start, stop, count)
    phase = np.clip((np.asarray(lags)[..., None] - centers) / (centers[1] - centers[0]), -1, 1)
    values = (1 + np.cos(np.pi * phase)) / 2
    values[(np.asarray(lags) < start - 1e-10) | (np.asarray(lags) > stop + 1e-10)] = 0
    return values

def event_design(edges, times, start, stop, count):
    """Add exact-time kernels only to bins wholly inside their support."""
    design = np.zeros((len(edges) - 1, count))
    times = np.asarray(times, float)
    for event in times:
        if not np.isfinite(event):
            continue
        left, right = event + start, event + stop
        # A tiny tolerance handles exactly aligned floating-point clock values.
        first = max(0, int(np.searchsorted(edges[:-1], left - 1e-10)))
        last = min(len(design), int(np.searchsorted(edges[1:], right + 1e-10, side="right")))
        if first < last:
            lag = (edges[first:last] + edges[first + 1:last + 1]) / 2 - event
            design[first:last] += temporal_basis(lag, start, stop, count)
    return design
