"""M8 (Phase 1): turn per-frame gaze and eyelid data into cursor positions and clicks. Pure logic."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from gaze_mouse.eyelid.classifier import WinkClassifier
from gaze_mouse.filtering.one_euro import OneEuroFilter
from gaze_mouse.frame import FrameResult

_BUTTON_FOR_WINK = {"left": "left", "right": "right"}


@dataclass(frozen=True)
class Click:
    button: str
    x: int
    y: int


@dataclass(frozen=True)
class PointerOutput:
    status: str  # tracking | frozen | no_face
    cursor: tuple[int, int] | None = None  # None: leave the cursor where it is
    click: Click | None = None


class GazeHistory:
    """Recent gaze points, used to place a click from before the eye started closing."""

    def __init__(self, keep_ms: float):
        self._keep_ns = int(keep_ms * 1e6)
        self._points: deque[tuple[int, np.ndarray]] = deque()

    def clear(self) -> None:
        self._points.clear()

    def add(self, timestamp_ns: int, point: np.ndarray) -> None:
        self._points.append((timestamp_ns, point))
        while timestamp_ns - self._points[0][0] > self._keep_ns:
            self._points.popleft()

    def median_between(self, start_ns: int, end_ns: int) -> np.ndarray | None:
        points = [p for t, p in self._points if start_ns <= t <= end_ns]
        return np.median(points, axis=0) if points else None


class PointerController:
    def __init__(
        self,
        model: Any,
        cfg: dict[str, Any],
        baselines: tuple[float, float],
        screen: tuple[int, int],
    ):
        eyelid, one_euro, mask = cfg["eyelid"], cfg["filter"]["one_euro"], cfg["fixation"]["blink_mask"]
        self._model = model
        self._baselines = baselines
        self._wink = WinkClassifier(eyelid, *baselines)
        self._filter = OneEuroFilter(one_euro["min_cutoff_hz"], one_euro["beta"], one_euro["d_cutoff_hz"])
        self._lookback_ns = tuple(int(ms * 1e6) for ms in eyelid["click_lookback_ms"])
        self._history = GazeHistory(eyelid["long_wink_ms"] + eyelid["click_lookback_ms"][1])
        self._openness = mask["openness_ratio"]
        self._post_open_ns = int(mask["post_open_ms"] * 1e6)
        # Gaze never reaches the corner areas, so only the physical mouse can trigger the corner pause.
        inset = cfg["safety"]["corner_margin_px"] + 1
        self._low = np.array([inset, inset], dtype=np.float64)
        self._high = np.array([screen[0] - 1 - inset, screen[1] - 1 - inset], dtype=np.float64)
        self._valid_after_ns = 0

    def _eyes_open(self, result: FrameResult) -> bool:
        return (
            result.ear_left >= self._openness * self._baselines[0]
            and result.ear_right >= self._openness * self._baselines[1]
        )

    def _click_for(self, wink) -> Click | None:
        point = self._history.median_between(wink.onset_ns - self._lookback_ns[1], wink.onset_ns - self._lookback_ns[0])
        if point is None:
            return None
        return Click(_BUTTON_FOR_WINK[wink.side], int(round(point[0])), int(round(point[1])))

    def update(self, result: FrameResult) -> PointerOutput:
        if not result.face_found or result.gaze_features is None:
            self._wink.reset()
            self._filter.reset()
            self._history.clear()
            self._valid_after_ns = 0
            return PointerOutput("no_face")

        ts = result.timestamp_ns
        wink = self._wink.update(ts, result.ear_left, result.ear_right)
        click = self._click_for(wink) if wink is not None else None

        if not self._eyes_open(result):
            self._valid_after_ns = ts + self._post_open_ns
            return PointerOutput("frozen", click=click)
        if ts < self._valid_after_ns:
            return PointerOutput("frozen", click=click)

        raw = np.clip(self._model.predict(result.gaze_features[None])[0], self._low, self._high)
        self._history.add(ts, raw)
        x, y = self._filter.update(ts, raw)
        return PointerOutput("tracking", (int(round(x)), int(round(y))), click)
