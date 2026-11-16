"""Metric head-pose estimation from facial 2D/3D correspondences."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gaze_mouse.perception.camera_calibration import CameraIntrinsics


@dataclass(frozen=True, slots=True)
class HeadPose:
    rotation_vector: np.ndarray
    translation_vector_mm: np.ndarray
    transform_camera_from_face: np.ndarray


def _points(value: np.ndarray, dimensions: int, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != dimensions:
        raise ValueError(f"{name} must have shape (N, {dimensions}), got {points.shape}")
    if len(points) < 4:
        raise ValueError("At least four point correspondences are required")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite values")
    return points


def estimate_head_pose(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> HeadPose:
    """Estimate the face-to-camera transform with OpenCV solvePnP."""
    object_points = _points(object_points_mm, 3, "object_points_mm")
    image_points = _points(image_points_px, 2, "image_points_px")
    if len(object_points) != len(image_points):
        raise ValueError("Object and image point counts must match")
    intrinsics.validate()
    ok, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.matrix(),
        intrinsics.distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("OpenCV could not estimate the head pose")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_vector[:, 0]
    return HeadPose(
        rotation_vector[:, 0].copy(),
        translation_vector[:, 0].copy(),
        transform,
    )


def transform_points(transform_camera_from_face: np.ndarray, points_face_mm: np.ndarray) -> np.ndarray:
    """Transform Nx3 face-coordinate points into camera coordinates."""
    transform = np.asarray(transform_camera_from_face, dtype=np.float64)
    points = _points(points_face_mm, 3, "points_face_mm")
    if transform.shape != (4, 4):
        raise ValueError(f"transform_camera_from_face must have shape (4, 4), got {transform.shape}")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (homogeneous @ transform.T)[:, :3]


def pixel_to_camera_point(
    pixel_xy: np.ndarray, depth_mm: float, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Back-project one image pixel at a known camera-space depth."""
    pixel = np.asarray(pixel_xy, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_xy must contain two finite values")
    if not np.isfinite(depth_mm) or depth_mm <= 0:
        raise ValueError("depth_mm must be positive and finite")
    intrinsics.validate()
    fx, fy = intrinsics.focal_length_px
    cx, cy = intrinsics.principal_point_px
    return np.array(
        [(pixel[0] - cx) * depth_mm / fx, (pixel[1] - cy) * depth_mm / fy, depth_mm],
        dtype=np.float64,
    )
