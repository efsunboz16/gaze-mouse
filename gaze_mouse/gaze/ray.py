"""3D gaze rays and ray-plane intersection helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain finite values")
    return vector


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    """Return a unit direction, rejecting zero and non-finite vectors."""
    vector = _vector3(direction, "direction")
    length = float(np.linalg.norm(vector))
    if length == 0:
        raise ValueError("direction must not be zero")
    return vector / length


@dataclass(frozen=True, slots=True)
class GazeRay:
    origin_mm: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        origin = _vector3(self.origin_mm, "origin_mm")
        direction = normalize_direction(self.direction)
        object.__setattr__(self, "origin_mm", origin)
        object.__setattr__(self, "direction", direction)

    def point_at(self, distance_mm: float) -> np.ndarray:
        if not np.isfinite(distance_mm):
            raise ValueError("distance_mm must be finite")
        return self.origin_mm + distance_mm * self.direction


@dataclass(frozen=True, slots=True)
class Plane:
    """A plane represented by a point and a unit normal in camera coordinates."""

    origin_mm: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_mm", _vector3(self.origin_mm, "origin_mm"))
        object.__setattr__(self, "normal", normalize_direction(self.normal))


def intersect_ray_plane(ray: GazeRay, plane: Plane) -> np.ndarray | None:
    """Return the forward ray intersection, or None for parallel/behind rays."""
    denominator = float(np.dot(ray.direction, plane.normal))
    if abs(denominator) < 1e-9:
        return None
    distance_mm = float(np.dot(plane.origin_mm - ray.origin_mm, plane.normal) / denominator)
    if distance_mm < 0:
        return None
    return ray.point_at(distance_mm)
