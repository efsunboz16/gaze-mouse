"""Rolling rate and stage-latency measurements. Pure logic, no clocks of its own."""
from __future__ import annotations

from collections import deque

import numpy as np


class RateMeter:
    """Events per second from (timestamp, cumulative count) samples over a time window."""

    def __init__(self, window_s: float):
        self._window_ns = int(window_s * 1e9)
        self._samples: deque[tuple[int, int]] = deque()

    def update(self, timestamp_ns: int, count: int) -> None:
        self._samples.append((timestamp_ns, count))
        while timestamp_ns - self._samples[0][0] > self._window_ns:
            self._samples.popleft()

    def rate(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        (t0, c0), (t1, c1) = self._samples[0], self._samples[-1]
        return (c1 - c0) * 1e9 / (t1 - t0) if t1 > t0 else 0.0


class StageProfiler:
    """Keeps the last `window` durations per named stage."""

    def __init__(self, window: int):
        self._window = window
        self._stages: dict[str, deque[float]] = {}

    def add(self, stage: str, ms: float) -> None:
        self._stages.setdefault(stage, deque(maxlen=self._window)).append(ms)

    def percentile(self, stage: str, q: float) -> float:
        values = self._stages.get(stage)
        return float(np.percentile(values, q)) if values else float("nan")

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "mean": float(np.mean(v)),
                "n": len(v),
            }
            for name, v in self._stages.items()
        }
