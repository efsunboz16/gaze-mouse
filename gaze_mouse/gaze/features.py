"""M4-A: geometric gaze features. Pure logic on arrays."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from gaze_mouse.eyelid.features import EyeLandmarkMap, compute_ears, to_pixels
from gaze_mouse.frame import FrameResult

FEATURE_SLICES = {
    "iris": slice(0, 4),  # left dx, dy, right dx, dy; relative to eye center, per eye width
    "pose": slice(4, 7),  # pitch, yaw, roll in radians
    "ear": slice(7, 8),  # mean of both eyes
    "distance": slice(8, 9),  # camera distance in mm
}
FEATURE_DIM = 9


def select_features(features: np.ndarray, groups: list[str]) -> np.ndarray:
    """Pick feature groups from Nx9 full feature vectors."""
    return np.hstack([features[:, FEATURE_SLICES[g]] for g in groups])


def iris_offset(points_px: np.ndarray, corners: tuple[int, int], iris: int) -> np.ndarray:
    """Iris center relative to the eye center, in eye widths."""
    a, b = points_px[corners[0]], points_px[corners[1]]
    width = np.linalg.norm(b - a)
    if width == 0:
        return np.full(2, np.nan, dtype=np.float32)
    return (points_px[iris] - (a + b) / 2) / width


def head_pose_angles(transform: np.ndarray) -> tuple[float, float, float]:
    """Pitch, yaw and roll in radians from a 4x4 face transform."""
    r = transform[:3, :3]
    return (
        math.atan2(r[2, 1], r[2, 2]),
        -math.asin(float(np.clip(r[2, 0], -1.0, 1.0))),
        math.atan2(r[1, 0], r[0, 0]),
    )


def distance_from_iris(iris_width_px: float, focal_px: float, iris_diameter_mm: float) -> float:
    return focal_px * iris_diameter_mm / iris_width_px if iris_width_px > 0 else float("nan")


class GazeFeatureExtractor:
    def __init__(self, landmarks_cfg: dict[str, Any], iris_diameter_mm: float):
        self._eye_map = EyeLandmarkMap.from_config(landmarks_cfg)
        self._width_pairs = (tuple(landmarks_cfg["iris_width_left"]), tuple(landmarks_cfg["iris_width_right"]))
        self._iris_mm = iris_diameter_mm

    def annotate(self, result: FrameResult) -> None:
        """Fill pixel points, EAR, distance and the gaze feature vector for a detected face."""
        if not result.face_found:
            return
        m = self._eye_map
        pts = to_pixels(result.landmarks, result.width, result.height)
        result.points_px = pts
        result.ear_left, result.ear_right = compute_ears(pts, m)

        widths = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in self._width_pairs]
        # Without camera intrinsics fx is approximated by the image width (report, M1).
        result.distance_mm = distance_from_iris(sum(widths) / 2, float(result.width), self._iris_mm)

        if result.face_transform is None:
            return
        vector = np.concatenate([
            iris_offset(pts, (m.ear_left[0], m.ear_left[3]), m.iris_left),
            iris_offset(pts, (m.ear_right[0], m.ear_right[3]), m.iris_right),
            head_pose_angles(result.face_transform),
            [(result.ear_left + result.ear_right) / 2, result.distance_mm],
        ]).astype(np.float32)
        if np.isfinite(vector).all():
            result.gaze_features = vector
