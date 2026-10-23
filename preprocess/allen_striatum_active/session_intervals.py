"""Explicit interval availability; never invent missing behavioral labels."""
import numpy as np


def merge(intervals):
    result = []
    for low, high in sorted(intervals):
        if not np.isfinite([low, high]).all() or high <= low:
            continue
        if result and low <= result[-1][1]:
            result[-1][1] = max(high, result[-1][1])
        else:
            result.append([float(low), float(high)])
    return np.asarray(result, float).reshape(-1, 2)


def epochs(source):
    path = "intervals/epochs"
    if path not in source:
        return {}
    group = source[path]
    if not all(k in group for k in ("start_time", "stop_time", "tags")):
        return {}
    tags = group["tags"][:]
    ends = group["tags_index"][:] if "tags_index" in group else np.arange(1, len(tags)+1)
    begin = 0
    result = {}
    for low, high, end in zip(group["start_time"][:], group["stop_time"][:], ends):
        for tag in tags[begin:int(end)]:
            name = tag.decode() if isinstance(tag, bytes) else str(tag)
            result.setdefault(name.lower(), []).append([low, high])
        begin = int(end)
    return {k: merge(v) for k, v in result.items()}


def opto_start(source):
    candidates = []
    path = "stimulus/presentation/Opto Stim/start_time"
    if path in source:
        times = np.asarray(source[path][:], float)
        times = times[np.isfinite(times)]
        if len(times):
            candidates.append(float(times.min()))
    intervals = epochs(source).get("optotagging", [])
    if len(intervals):
        candidates.append(float(intervals[0][0]))
    return min(candidates) if candidates else None


def selection_opto_start(source, info):
    onset = opto_start(source)
    timing, offset = info.get("session_timing"), info.get("session_offset_s")
    origin = info.get("clock", {}).get("origin_s")
    if onset is None and timing and offset is not None and origin is not None:
        onset = origin + timing["first_opto_elapsed_s"] - offset
    return onset


def domain_intervals(source, info, domain, buffer_s=0):
    if not np.isfinite(buffer_s) or buffer_s < 0:
        raise ValueError("buffer_s must be finite and nonnegative")
    if buffer_s and domain in {"full", "optotagging"}:
        raise ValueError("Use an optotagging buffer with pre_opto, baseline, or behavior")
    clock = info.get("clock", {})
    origin, duration = clock.get("origin_s"), clock.get("duration_s")
    bounds = [[origin, origin+duration]] if origin is not None and duration is not None else []
    available = epochs(source)
    if domain == "full":
        intervals = bounds
    elif domain == "pre_opto":
        onset = selection_opto_start(source, info)
        if onset is None or not bounds:
            raise ValueError("Cannot establish pre-opto interval: onset or recording clock unavailable")
        intervals = [[origin, min(origin+duration, onset-buffer_s)]]
    elif domain in {"baseline", "behavior", "optotagging"}:
        intervals = available.get(domain, [])
        if buffer_s:
            onset = selection_opto_start(source, info)
            if onset is None:
                raise ValueError("Cannot apply an optotagging buffer without an onset")
            intervals = [[a, min(b, onset-buffer_s)] for a, b in intervals]
    else:
        raise ValueError(f"Unknown domain: {domain}")
    if not len(intervals):
        raise ValueError(f"Domain {domain!r} is unavailable; it is not an empty recording")
    if bounds:
        intervals = [[max(a, origin), min(b, origin+duration)] for a, b in intervals]
    return merge(intervals)


def qc_intervals(source, info=None):
    """Baseline + behavior, before the earliest optotagging epoch/pulse.

    Unlike a simple pre-opto spike selector, this metric excludes gaps between
    annotated epochs. Missing epochs/onset yield unavailable QC, not failure.
    """
    onset = opto_start(source)
    if onset is None:
        return np.empty((0, 2))
    blocks = epochs(source)
    values = [[a, min(b, onset)] for name in ("baseline", "behavior")
              for a, b in blocks.get(name, [])]
    clock = (info or {}).get("clock", {})
    if clock.get("origin_s") is not None and clock.get("duration_s") is not None:
        low, high = clock["origin_s"], clock["origin_s"] + clock["duration_s"]
        values = [[max(a, low), min(b, high)] for a, b in values]
    return merge(values)
