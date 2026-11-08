"""Background Qt thread: camera, landmarks and per-frame features."""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from gaze_mouse.capture.camera import Camera
from gaze_mouse.gaze.features import GazeFeatureExtractor
from gaze_mouse.perception.landmarks import FaceLandmarkTracker


class PipelineThread(QThread):
    camera_started = pyqtSignal(object)  # CameraInfo
    frame_ready = pyqtSignal(object)  # FrameResult
    failed = pyqtSignal(str)

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self._cfg = cfg

    def run(self) -> None:
        cfg = self._cfg
        extractor = GazeFeatureExtractor(cfg["landmarks"], cfg["landmarker"]["iris_diameter_mm"])
        timeout_s = cfg["preview"]["frame_timeout_s"]
        try:
            # MediaPipe stays on this thread for its whole lifetime.
            with FaceLandmarkTracker(cfg["landmarker"]) as tracker, Camera(cfg["camera"]) as camera:
                self.camera_started.emit(camera.start())
                last_index = -1
                while not self.isInterruptionRequested():
                    frame = camera.get_latest(last_index, timeout_s)
                    if frame is None:
                        raise RuntimeError(camera.error or f"No frame within {timeout_s} s")
                    last_index = frame.index
                    result = tracker.process(frame)
                    extractor.annotate(result)
                    self.frame_ready.emit(result)
        except Exception as exc:  # reported to the GUI thread, which owns shutdown
            self.failed.emit(f"{type(exc).__name__}: {exc}")
