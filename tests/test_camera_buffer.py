import threading
import time

import numpy as np

from gaze_mouse.capture.camera import Camera, LatestFrameBuffer
from gaze_mouse.frame import CapturedFrame


def frame(index: int) -> CapturedFrame:
    return CapturedFrame(np.zeros((2, 2, 3), np.uint8), index * 1000, index)


class FakeCapture:
    """Stands in for cv2.VideoCapture; fails after `frames` reads when set."""

    def __init__(self, brightness=(100,), set_result=True, frames=None):
        self._brightness = list(brightness)
        self._set_result = set_result
        self._remaining = frames
        self.sets = []
        self.released = False

    def read(self):
        time.sleep(0.001)
        if self._remaining is not None:
            if self._remaining <= 0:
                return False, None
            self._remaining -= 1
        level = self._brightness.pop(0) if len(self._brightness) > 1 else self._brightness[0]
        return True, np.full((720, 1280, 3), level, np.uint8)

    def set(self, prop, value):
        self.sets.append((prop, value))
        return self._set_result

    def get(self, prop):
        return 30.0

    def release(self):
        self.released = True


def cfg(**overrides):
    base = {
        "buffer_frames": 3, "warmup_frames": 1, "lock_exposure": True, "lock_focus": True,
        "exposure_manual_value": 0.25, "exposure_auto_value": 0.75,
        "exposure_lock_max_brightness_change": 0.25, "max_read_failures": 3, "backend": "dshow",
    }
    return base | overrides


def test_buffer_returns_newest_and_evicts_old():
    buffer = LatestFrameBuffer(capacity=2)
    for i in range(5):
        buffer.put(frame(i))
    assert buffer.get_latest(-1, 0).index == 4


def test_buffer_does_not_repeat_consumed_frame():
    buffer = LatestFrameBuffer(capacity=2)
    buffer.put(frame(7))
    assert buffer.get_latest(7, 0) is None


def test_buffer_wakes_waiting_consumer():
    buffer = LatestFrameBuffer(capacity=2)
    threading.Timer(0.02, buffer.put, args=(frame(1),)).start()
    assert buffer.get_latest(0, 2.0).index == 1


def test_buffer_close_releases_waiter():
    buffer = LatestFrameBuffer(capacity=2)
    threading.Timer(0.02, buffer.close).start()
    start = time.perf_counter()
    assert buffer.get_latest(0, 2.0) is None
    assert time.perf_counter() - start < 1.0


def test_camera_delivers_increasing_frames_and_releases():
    fake = FakeCapture()
    with Camera(cfg(), capture_factory=lambda _: fake) as camera:
        info = camera.start()
        first = camera.get_latest(-1, 2.0)
        second = camera.get_latest(first.index, 2.0)
    assert (info.width, info.height) == (1280, 720)
    assert second.index > first.index and second.timestamp_ns > first.timestamp_ns
    assert fake.released


def test_exposure_lock_reverted_when_brightness_jumps():
    fake = FakeCapture(brightness=(100, 20, 100))
    with Camera(cfg(lock_focus=False), capture_factory=lambda _: fake) as camera:
        info = camera.start()
    assert info.exposure_lock == "reverted"
    assert fake.sets[-1][1] == 0.75


def test_exposure_lock_kept_when_brightness_stable_is_not_claimed_verified():
    fake = FakeCapture(brightness=(100, 95))
    with Camera(cfg(), capture_factory=lambda _: fake) as camera:
        info = camera.start()
    assert info.exposure_lock == "requested_unverified"
    assert info.focus_lock == "requested_unverified"


def test_locks_report_unsupported():
    fake = FakeCapture(set_result=False)
    with Camera(cfg(), capture_factory=lambda _: fake) as camera:
        info = camera.start()
    assert (info.exposure_lock, info.focus_lock) == ("unsupported", "unsupported")


def test_read_failures_surface_as_error():
    fake = FakeCapture(frames=1)
    with Camera(cfg(lock_exposure=False, lock_focus=False), capture_factory=lambda _: fake) as camera:
        camera.start()
        assert camera.get_latest(-1, 2.0) is None
        assert "failed reads" in camera.error
