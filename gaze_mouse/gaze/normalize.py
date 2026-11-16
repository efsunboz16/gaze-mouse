"""Face-image normalization for appearance-based gaze models."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    output_size: tuple[int, int] = (224, 224)
    target_eye_distance_px: float = 86.0
    target_eye_midpoint_px: tuple[float, float] = (112.0, 112.0)


@dataclass(frozen=True, slots=True)
class NormalizedFace:
    image: np.ndarray
    transform: np.ndarray  # 2x3 source-image to normalized-image transform
    inverse_transform: np.ndarray  # 2x3 normalized-image to source-image transform
    eye_centers_px: tuple[np.ndarray, np.ndarray]


def _eye_center(points_px: np.ndarray, indices: tuple[int, ...], name: str) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_px must have shape (N, 2), got {points.shape}")
    if not indices:
        raise ValueError(f"{name} indices must not be empty")
    if min(indices) < 0 or max(indices) >= len(points):
        raise ValueError(f"{name} contains an out-of-range landmark index")
    centers = points[list(indices)]
    if not np.isfinite(centers).all():
        raise ValueError(f"{name} points must be finite")
    return centers.mean(axis=0)


def normalize_face(
    image: np.ndarray,
    points_px: np.ndarray,
    left_eye_indices: tuple[int, ...],
    right_eye_indices: tuple[int, ...],
    spec: NormalizationSpec = NormalizationSpec(),
) -> NormalizedFace:
    """Rotate and scale a face so its eye geometry is stable for model input."""
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be grayscale or color, got {image.shape}")
    width, height = spec.output_size
    if width <= 0 or height <= 0 or spec.target_eye_distance_px <= 0:
        raise ValueError("Normalization output and target eye distance must be positive")
    left = _eye_center(points_px, left_eye_indices, "left_eye")
    right = _eye_center(points_px, right_eye_indices, "right_eye")
    delta = right - left
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        raise ValueError("Eye centers must not coincide")
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    scale = spec.target_eye_distance_px / distance
    midpoint = (left + right) / 2.0
    rotation = cv2.getRotationMatrix2D(tuple(midpoint), angle, scale)
    target = np.asarray(spec.target_eye_midpoint_px, dtype=np.float64)
    rotation[:, 2] += target - (rotation[:, :2] @ midpoint + rotation[:, 2])
    normalized = cv2.warpAffine(image, rotation, (width, height), flags=cv2.INTER_LINEAR)
    normalized_left = rotation[:, :2] @ left + rotation[:, 2]
    normalized_right = rotation[:, :2] @ right + rotation[:, 2]
    return NormalizedFace(
        normalized,
        rotation,
        cv2.invertAffineTransform(rotation),
        (normalized_left, normalized_right),
    )
