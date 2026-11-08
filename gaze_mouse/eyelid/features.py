"""M9: per-frame eyelid features. Pure logic on arrays."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EyeLandmarkMap:
    ear_left: tuple[int, ...]
    ear_right: tuple[int, ...]
    iris_left: int
    iris_right: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EyeLandmarkMap":
        return cls(tuple(cfg["ear_left"]), tuple(cfg["ear_right"]), cfg["iris_left"], cfg["iris_right"])


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalized landmarks to Nx2 pixels. EAR needs this: x and y scale differently."""
    return landmarks[:, :2] * np.array([width, height], dtype=np.float32)


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR from six ordered points p1..p6 (Soukupova & Cech). NaN for a degenerate eye."""
    p1, p2, p3, p4, p5, p6 = points
    width = np.linalg.norm(p1 - p4)
    if width == 0:
        return float("nan")
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width))


def compute_ears(points_px: np.ndarray, eye_map: EyeLandmarkMap) -> tuple[float, float]:
    """Return (left, right) EAR."""
    return (
        eye_aspect_ratio(points_px[list(eye_map.ear_left)]),
        eye_aspect_ratio(points_px[list(eye_map.ear_right)]),
    )


def eye_blink_scores(blendshapes: Mapping[str, float]) -> tuple[float | None, float | None]:
    """Return raw (eyeBlinkLeft, eyeBlinkRight). The side mapping is verified per user."""
    return blendshapes.get("eyeBlinkLeft"), blendshapes.get("eyeBlinkRight")
