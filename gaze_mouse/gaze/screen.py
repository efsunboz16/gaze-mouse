"""M5 (partial): primary display size in physical pixels on Windows."""
from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

import numpy as np

from gaze_mouse.gaze.ray import GazeRay, Plane, intersect_ray_plane
from gaze_mouse.frame import FrameResult
_PROCESS_PER_MONITOR_DPI_AWARE = 2
_SM_CXSCREEN, _SM_CYSCREEN = 0, 1


def enable_dpi_awareness() -> None:
    """Call before creating any window so coordinates are physical pixels."""
    if sys.platform == "win32":
        ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE)


def primary_screen_size() -> tuple[int, int]:
    if sys.platform != "win32":
        raise NotImplementedError("Screen metrics are only implemented for Windows")
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(_SM_CXSCREEN), user32.GetSystemMetrics(_SM_CYSCREEN)


@dataclass(frozen=True, slots=True)
class ScreenPlane:
    """Physical screen rectangle expressed in camera coordinates."""

    origin_mm: np.ndarray
    right_mm: np.ndarray
    down_mm: np.ndarray
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_mm, dtype=np.float64)
        right = np.asarray(self.right_mm, dtype=np.float64)
        down = np.asarray(self.down_mm, dtype=np.float64)
        if origin.shape != (3,) or right.shape != (3,) or down.shape != (3,):
            raise ValueError("Screen plane vectors must have shape (3,)")
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("Screen dimensions must be positive")
        if abs(float(np.dot(right, down))) > 1e-6:
            raise ValueError("Screen basis vectors must be perpendicular")
        if not np.isclose(np.linalg.norm(right), 1.0):
            raise ValueError("right_mm must be a unit vector")
        if not np.isclose(np.linalg.norm(down), 1.0):
            raise ValueError("down_mm must be a unit vector")
        object.__setattr__(self, "origin_mm", origin)
        object.__setattr__(self, "right_mm", right)
        object.__setattr__(self, "down_mm", down)

    @property
    def plane(self) -> Plane:
        return Plane(self.origin_mm, np.cross(self.right_mm, self.down_mm))

    def intersect(self, ray: GazeRay) -> tuple[float, float] | None:
        point = intersect_ray_plane(ray, self.plane)
        if point is None:
            return None
        relative = point - self.origin_mm
        x_mm = float(np.dot(relative, self.right_mm))
        y_mm = float(np.dot(relative, self.down_mm))
        if not (0.0 <= x_mm <= self.width_mm and 0.0 <= y_mm <= self.height_mm):
            return None
        return x_mm, y_mm

    def to_pixels(self, point_mm: tuple[float, float], size_px: tuple[int, int]) -> np.ndarray:
        x_mm, y_mm = point_mm
        width_px, height_px = size_px
        return np.array(
            [x_mm / self.width_mm * width_px, y_mm / self.height_mm * height_px],
            dtype=np.float64,
        )

    def point_from_pixels(self, point_px: np.ndarray, size_px: tuple[int, int]) -> np.ndarray:
        """Convert physical screen pixels to a 3D point in camera coordinates."""
        pixels = np.asarray(point_px, dtype=np.float64)
        if pixels.shape != (2,) or not np.isfinite(pixels).all():
            raise ValueError("point_px must contain two finite values")
        width_px, height_px = size_px
        if width_px <= 0 or height_px <= 0:
            raise ValueError("Screen pixel dimensions must be positive")
        x_mm = pixels[0] / width_px * self.width_mm
        y_mm = pixels[1] / height_px * self.height_mm
        return self.origin_mm + x_mm * self.right_mm + y_mm * self.down_mm


def _fuse_points(
    points: list[np.ndarray], uncertainties: list[float | None]
) -> np.ndarray:
    weights = np.array([
        1.0 / max(float(sigma), 1e-6) ** 2 if sigma is not None and np.isfinite(sigma) else 1.0
        for sigma in uncertainties
    ])
    return np.average(np.stack(points), axis=0, weights=weights)


def project_frame_gaze(
    result: FrameResult, screen: ScreenPlane, screen_size_px: tuple[int, int]
) -> np.ndarray | None:
    """Project available eye gaze rays and fuse their screen intersections."""
    rays = []
    directions = (
        (result.left_eye_center_camera_mm, result.left_gaze_direction_camera),
        (result.right_eye_center_camera_mm, result.right_gaze_direction_camera),
    )
    for origin, direction in directions:
        if origin is None or direction is None:
            continue
        point_mm = screen.intersect(GazeRay(origin, direction))
        if point_mm is not None:
            rays.append(screen.to_pixels(point_mm, screen_size_px))
    if not rays:
        result.screen_gaze_px = None
        return None
    result.screen_gaze_px = _fuse_points(
        rays,
        [result.gaze_uncertainty] * len(rays),
    )
    return result.screen_gaze_px
