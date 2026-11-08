"""M9: per-eye hysteresis and single-eye wink detection. Pure logic.

Phase 1 emits only short winks. Natural blinks close both eyes and are ignored;
long winks are reserved for dragging and scrolling in Phase 2.

The other eye narrows during a real wink, so a wink is told from a blink by how much
more open the other eye is at the deepest point of the closure, not by a fixed level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WinkEvent:
    side: str  # "left" or "right"
    onset_ns: int
    end_ns: int
    duration_ms: float


@dataclass
class _EyeState:
    closed: bool = False
    onset_ns: int = 0
    deepest_own: float = 0.0
    other_at_deepest: float = 0.0


class WinkClassifier:
    def __init__(self, cfg: dict[str, Any], baseline_left: float, baseline_right: float):
        self._close = cfg["close_ratio"]
        self._open = cfg["open_ratio"]
        self._min_asymmetry = cfg["wink_min_asymmetry"]
        self._min_ms = cfg["wink_min_ms"]
        self._long_ms = cfg["long_wink_ms"]
        self._baselines = (baseline_left, baseline_right)
        self.reset()

    def reset(self) -> None:
        self._eyes = (_EyeState(), _EyeState())

    def update(self, timestamp_ns: int, ear_left: float | None, ear_right: float | None) -> WinkEvent | None:
        """Feed one frame; a missing or invalid EAR cancels any closure in progress."""
        if ear_left is None or ear_right is None or math.isnan(ear_left) or math.isnan(ear_right):
            self.reset()
            return None
        ratios = (ear_left / self._baselines[0], ear_right / self._baselines[1])
        event = None
        for index, (side, eye) in enumerate(zip(("left", "right"), self._eyes)):
            own, other = ratios[index], ratios[1 - index]
            if not eye.closed:
                if own < self._close:
                    eye.closed, eye.onset_ns = True, timestamp_ns
                    eye.deepest_own, eye.other_at_deepest = own, other
                continue
            if own < eye.deepest_own:
                eye.deepest_own, eye.other_at_deepest = own, other
            if own > self._open:
                duration_ms = (timestamp_ns - eye.onset_ns) / 1e6
                asymmetric = eye.other_at_deepest - eye.deepest_own >= self._min_asymmetry
                if asymmetric and self._min_ms <= duration_ms < self._long_ms:
                    event = WinkEvent(side, eye.onset_ns, timestamp_ns, duration_ms)
                eye.closed = False
        return event
