"""Per-frame data contract shared between modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    image: np.ndarray  # BGR
    timestamp_ns: int  # time.perf_counter_ns() right after read() returned
    index: int  # counts every frame the camera delivered, including dropped ones


@dataclass(slots=True)
class FrameResult:
    timestamp_ns: int
    frame_index: int
    width: int
    height: int
    landmarks: np.ndarray | None = None  # Nx3 normalized MediaPipe coordinates
    points_px: np.ndarray | None = None  # Nx2 unmirrored image pixels
    blendshapes: dict[str, float] = field(default_factory=dict)
    face_transform: np.ndarray | None = None  # 4x4
    ear_left: float | None = None
    ear_right: float | None = None
    gaze_features: np.ndarray | None = None  # layout in gaze.features.FEATURE_SLICES
    distance_mm: float | None = None  # camera distance from iris size
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def face_found(self) -> bool:
        return self.landmarks is not None
