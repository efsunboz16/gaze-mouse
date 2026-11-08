"""M1: background camera capture with a latest-frame ring buffer."""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2

from gaze_mouse.frame import CapturedFrame

_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}
_JOIN_TIMEOUT_S = 2.0


class LatestFrameBuffer:
    """Bounded frame store; consumers always get the newest frame and old ones fall off."""

    def __init__(self, capacity: int):
        self._frames: deque[CapturedFrame] = deque(maxlen=capacity)
        self._cond = threading.Condition()
        self._closed = False

    def put(self, frame: CapturedFrame) -> None:
        with self._cond:
            self._frames.append(frame)
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _newest_after(self, index: int) -> CapturedFrame | None:
        if self._frames and self._frames[-1].index > index:
            return self._frames[-1]
        return None

    def get_latest(self, after_index: int, timeout_s: float) -> CapturedFrame | None:
        """Return the newest frame with index > after_index, or None on timeout/close."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._closed or self._newest_after(after_index) is not None,
                timeout_s,
            )
            return self._newest_after(after_index)


@dataclass(frozen=True)
class CameraInfo:
    backend: str
    width: int
    height: int
    nominal_fps: float
    exposure_lock: str  # not_requested | unsupported | reverted | requested_unverified
    focus_lock: str  # not_requested | unsupported | requested_unverified
    brightness_before_lock: float
    brightness_after_lock: float


def open_capture(cfg: dict[str, Any]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg["index"], _BACKENDS[cfg["backend"]])
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {cfg['index']} with backend {cfg['backend']}")
    # FOURCC must be set before the size for the driver to negotiate it.
    if cfg["fourcc"]:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
    return cap


def _read_brightness(cap: Any, frames: int) -> tuple[float, Any]:
    """Read `frames` frames and return the last one's mean brightness and the frame."""
    image = None
    for _ in range(max(frames, 1)):
        ok, image = cap.read()
        if not ok or image is None:
            raise RuntimeError("Camera read failed during warm-up")
    return float(image[::16, ::16].mean()), image


class Camera:
    def __init__(
        self,
        cfg: dict[str, Any],
        capture_factory: Callable[[dict[str, Any]], Any] = open_capture,
    ):
        self._cfg = cfg
        self._factory = capture_factory
        self._buffer = LatestFrameBuffer(cfg["buffer_frames"])
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: Any = None
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> CameraInfo:
        self._cap = self._factory(self._cfg)
        before, image = _read_brightness(self._cap, self._cfg["warmup_frames"])
        exposure_lock, after = self._lock_exposure(before)
        focus_lock = self._lock_focus()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return CameraInfo(
            backend=self._cfg["backend"],
            width=image.shape[1],
            height=image.shape[0],
            nominal_fps=float(self._cap.get(cv2.CAP_PROP_FPS)),
            exposure_lock=exposure_lock,
            focus_lock=focus_lock,
            brightness_before_lock=before,
            brightness_after_lock=after,
        )

    def _lock_exposure(self, before: float) -> tuple[str, float]:
        cfg = self._cfg
        if not cfg["lock_exposure"]:
            return "not_requested", before
        if not self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, cfg["exposure_manual_value"]):
            return "unsupported", before
        after, _ = _read_brightness(self._cap, cfg["warmup_frames"])
        if abs(after - before) / max(before, 1.0) > cfg["exposure_lock_max_brightness_change"]:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, cfg["exposure_auto_value"])
            _read_brightness(self._cap, cfg["warmup_frames"])
            return "reverted", after
        return "requested_unverified", after

    def _lock_focus(self) -> str:
        if not self._cfg["lock_focus"]:
            return "not_requested"
        return "requested_unverified" if self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) else "unsupported"

    def _run(self) -> None:
        index = failures = 0
        while not self._stop.is_set():
            ok, image = self._cap.read()
            timestamp_ns = time.perf_counter_ns()
            if not ok or image is None:
                failures += 1
                if failures >= self._cfg["max_read_failures"]:
                    self._error = f"Camera stopped delivering frames after {failures} failed reads"
                    break
                continue
            failures = 0
            self._buffer.put(CapturedFrame(image, timestamp_ns, index))
            index += 1
        self._buffer.close()

    def get_latest(self, after_index: int, timeout_s: float) -> CapturedFrame | None:
        return self._buffer.get_latest(after_index, timeout_s)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(_JOIN_TIMEOUT_S)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
