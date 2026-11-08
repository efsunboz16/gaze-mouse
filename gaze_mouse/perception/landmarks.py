"""M2: MediaPipe Face Landmarker wrapper (Tasks API, VIDEO mode)."""
from __future__ import annotations

import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from gaze_mouse.config import resolve_path
from gaze_mouse.frame import CapturedFrame, FrameResult


class StrictMsClock:
    """Convert nanosecond timestamps to strictly increasing integer milliseconds."""

    def __init__(self) -> None:
        self._base_ns: int | None = None
        self._last_ms = 0

    def to_ms(self, timestamp_ns: int) -> int:
        if self._base_ns is None:
            self._base_ns = timestamp_ns
        ms = max((timestamp_ns - self._base_ns) // 1_000_000, self._last_ms + 1)
        self._last_ms = ms
        return ms


class FaceLandmarkTracker:
    def __init__(self, cfg: dict[str, Any]):
        if cfg["running_mode"] != "VIDEO":
            raise ValueError(f"Unsupported running_mode: {cfg['running_mode']}")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(resolve_path(cfg["model_path"])),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=cfg["num_faces"],
            min_face_detection_confidence=cfg["min_face_detection_confidence"],
            min_face_presence_confidence=cfg["min_face_presence_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._clock = StrictMsClock()

    def __enter__(self) -> "FaceLandmarkTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self._landmarker.close()

    def process(self, frame: CapturedFrame) -> FrameResult:
        height, width = frame.image.shape[:2]
        result = FrameResult(frame.timestamp_ns, frame.index, width, height)
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        start = time.perf_counter_ns()
        raw = self._landmarker.detect_for_video(image, self._clock.to_ms(frame.timestamp_ns))
        result.timings_ms["inference"] = (time.perf_counter_ns() - start) / 1e6

        if raw.face_landmarks:
            result.landmarks = np.array(
                [(p.x, p.y, p.z) for p in raw.face_landmarks[0]], dtype=np.float32
            )
            if raw.face_blendshapes:
                result.blendshapes = {c.category_name: c.score for c in raw.face_blendshapes[0]}
            if raw.facial_transformation_matrixes:
                result.face_transform = np.asarray(raw.facial_transformation_matrixes[0])
        return result
