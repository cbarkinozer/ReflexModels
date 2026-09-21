"""Small dependency-free CPU timing primitive for baseline runners."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from collections.abc import Callable
from typing import Any


def measure_callable(fn: Callable[[], Any], *, warmup: int = 1, repetitions: int = 10) -> dict[str, float | int]:
    """Measure synchronous request latency and Python-managed peak allocation.

    Native framework memory (for example PyTorch tensor allocations) must be
    reported separately by model-specific runners.
    """
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions positive")
    for _ in range(warmup):
        fn()
    tracemalloc.start()
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        fn()
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(timings)
    p95_index = round((len(ordered) - 1) * 0.95)
    return {
        "requests": repetitions,
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_p50": statistics.median(timings),
        "latency_ms_p95": ordered[p95_index],
        "requests_per_second": repetitions / (sum(timings) / 1000),
        "python_peak_ram_bytes": peak_bytes,
    }
