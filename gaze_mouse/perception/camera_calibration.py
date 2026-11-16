"""Camera calibration values used by the metric gaze geometry."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics and optional lens distortion coefficients."""

    focal_length_px: tuple[float, float]
    principal_point_px: tuple[float, float]
    distortion: np.ndarray | None = None

    def matrix(self) -> np.ndarray:
        fx, fy = self.focal_length_px
        cx, cy = self.principal_point_px
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, distortion: np.ndarray | None = None
    ) -> "CameraIntrinsics":
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (3, 3):
            raise ValueError(f"Camera matrix must have shape (3, 3), got {values.shape}")
        return cls(
            focal_length_px=(float(values[0, 0]), float(values[1, 1])),
            principal_point_px=(float(values[0, 2]), float(values[1, 2])),
            distortion=None if distortion is None else np.asarray(distortion, dtype=np.float64),
        )

    def validate(self) -> None:
        fx, fy = self.focal_length_px
        if fx <= 0 or fy <= 0:
            raise ValueError("Camera focal lengths must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        distortion = data.get("distortion")
        result = cls(
            tuple(data["focal_length_px"]),
            tuple(data["principal_point_px"]),
            None if distortion is None else np.asarray(distortion, dtype=np.float64),
        )
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraIntrinsics":
        with Path(path).open(encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


@dataclass(frozen=True, slots=True)
class CameraCalibrationResult:
    intrinsics: CameraIntrinsics
    reprojection_error_px: float
    views_used: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_length_px": list(self.intrinsics.focal_length_px),
            "principal_point_px": list(self.intrinsics.principal_point_px),
            "distortion": (
                None
                if self.intrinsics.distortion is None
                else self.intrinsics.distortion.tolist()
            ),
            "reprojection_error_px": self.reprojection_error_px,
            "views_used": self.views_used,
        }


def chessboard_object_points(
    board_size: tuple[int, int], square_size_mm: float
) -> np.ndarray:
    """Create planar chessboard corner coordinates in millimetres."""
    columns, rows = board_size
    if columns < 2 or rows < 2:
        raise ValueError("Chessboard dimensions must be at least 2 by 2")
    if square_size_mm <= 0:
        raise ValueError("Chessboard square size must be positive")
    grid = np.zeros((columns * rows, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_mm
    return grid


def detect_chessboard_corners(
    image: np.ndarray, board_size: tuple[int, int]
) -> np.ndarray | None:
    """Find and subpixel-refine inner chessboard corners in an image."""
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be grayscale or color, got {image.shape}")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = None
    found = False
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            board_size,
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    if not found or corners is None:
        return None
    if not hasattr(cv2, "findChessboardCornersSB"):
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)


def calibrate_camera(
    image_points: list[np.ndarray],
    image_size_px: tuple[int, int],
    board_size: tuple[int, int],
    square_size_mm: float,
) -> CameraCalibrationResult:
    """Estimate camera intrinsics from detected chessboard corners."""
    if not image_points:
        raise ValueError("At least one chessboard view is required")
    width, height = image_size_px
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    object_view = chessboard_object_points(board_size, square_size_mm)
    object_points = [object_view.copy() for _ in image_points]
    points = [np.asarray(view, dtype=np.float32) for view in image_points]
    expected_shape = (len(object_view), 1, 2)
    if any(view.shape != expected_shape for view in points):
        raise ValueError(f"Each image-point view must have shape {expected_shape}")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, points, (width, height), None, None
    )
    intrinsics = CameraIntrinsics.from_matrix(matrix, distortion.reshape(-1))
    intrinsics.validate()
    return CameraCalibrationResult(intrinsics, float(rms), len(points))

