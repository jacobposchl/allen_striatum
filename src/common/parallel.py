"""CPU allocation and bounded, ordered process-pool submission."""

from collections import deque
from contextlib import contextmanager
import os


def available_cpus():
    """Respect process affinity and the current Slurm allocation when available."""
    counts = [len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)]
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name, "")
        if value.isdigit() and int(value) > 0:
            counts.append(int(value))
    return max(1, min(counts))


@contextmanager
def worker_thread_environment(threads):
    """Limit numerical libraries before spawned workers import NumPy/PyTorch."""
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def bounded_map(executor, function, items, max_pending):
    """Consume in input order, retaining at most max_pending submitted tasks."""
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    items, pending = iter(items), deque()
    try:
        for _ in range(max_pending):
            try:
                pending.append(executor.submit(function, next(items)))
            except StopIteration:
                break
        while pending:
            result = pending.popleft().result()
            yield result
            try:
                pending.append(executor.submit(function, next(items)))
            except StopIteration:
                pass
    finally:
        for future in pending:
            future.cancel()
