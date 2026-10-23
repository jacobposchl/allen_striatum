"""Selections on continuous source time; gaps never become adjacent samples."""
import numpy as np


def merge(values):
    values = np.asarray(values, dtype=float).reshape(-1, 2)
    if not np.isfinite(values).all() or np.any(values[:, 1] <= values[:, 0]):
        raise ValueError("Intervals must have finite, increasing boundaries")
    result = []
    for low, high in sorted(values.tolist()):
        if result and low <= result[-1][1]:
            result[-1][1] = max(result[-1][1], high)
        else:
            result.append([low, high])
    return np.asarray(result, dtype=float).reshape(-1, 2)


def intersect(left, right):
    return merge([[max(a, c), min(b, d)] for a, b in left for c, d in right
                  if min(b, d) > max(a, c)])


def subtract(values, excluded):
    result = merge(values)
    for a, b in merge(excluded):
        pieces = []
        for low, high in result:
            if b <= low or a >= high:
                pieces.append([low, high])
            else:
                if low < a:
                    pieces.append([low, min(a, high)])
                if b < high:
                    pieces.append([max(b, low), high])
        result = np.asarray(pieces, dtype=float).reshape(-1, 2)
    return result
