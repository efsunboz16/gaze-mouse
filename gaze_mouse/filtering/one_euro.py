"""M7: 1-euro filter (Casiez et al., CHI 2012) for 2D screen coordinates."""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff_hz: np.ndarray | float, dt_s: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter:
    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float):
        self._min_cutoff = min_cutoff_hz
        self._beta = beta
        self._d_cutoff = d_cutoff_hz
        self.reset()

    def reset(self) -> None:
        self._t_ns: int | None = None
        self._x = np.zeros(2)
        self._dx = np.zeros(2)

    def update(self, timestamp_ns: int, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self._t_ns is None:
            self._t_ns, self._x = timestamp_ns, value
            return value.copy()
        dt_s = (timestamp_ns - self._t_ns) / 1e9
        if dt_s <= 0:
            return self._x.copy()
        self._dx += _alpha(self._d_cutoff, dt_s) * ((value - self._x) / dt_s - self._dx)
        cutoff = self._min_cutoff + self._beta * np.abs(self._dx)
        self._x = self._x + _alpha(cutoff, dt_s) * (value - self._x)
        self._t_ns = timestamp_ns
        return self._x.copy()
