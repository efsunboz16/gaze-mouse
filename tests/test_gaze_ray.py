import numpy as np
import pytest

from gaze_mouse.gaze.ray import GazeRay, Plane, intersect_ray_plane, normalize_direction
from gaze_mouse.gaze.screen import ScreenPlane
from gaze_mouse.perception.camera_calibration import CameraIntrinsics


def test_normalize_direction_returns_unit_vector():
    assert np.allclose(normalize_direction(np.array([3.0, 0.0, 4.0])), [0.6, 0.0, 0.8])


def test_normalize_direction_rejects_zero():
    with pytest.raises(ValueError, match="must not be zero"):
        normalize_direction(np.zeros(3))


def test_ray_plane_intersection_returns_forward_point():
    ray = GazeRay(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    plane = Plane(np.array([0.0, 0.0, 600.0]), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(intersect_ray_plane(ray, plane), [0.0, 0.0, 600.0])


def test_ray_plane_intersection_rejects_parallel_and_behind():
    plane = Plane(np.array([0.0, 0.0, 600.0]), np.array([0.0, 0.0, 1.0]))
    assert intersect_ray_plane(
        GazeRay(np.zeros(3), np.array([1.0, 0.0, 0.0])), plane
    ) is None
    assert intersect_ray_plane(
        GazeRay(np.zeros(3), np.array([0.0, 0.0, -1.0])), plane
    ) is None


def test_screen_plane_maps_ray_to_physical_and_pixel_coordinates():
    screen = ScreenPlane(
        origin_mm=np.array([-170.0, -105.0, 600.0]),
        right_mm=np.array([1.0, 0.0, 0.0]),
        down_mm=np.array([0.0, 1.0, 0.0]),
        width_mm=340.0,
        height_mm=210.0,
    )
    point = screen.intersect(GazeRay(np.zeros(3), np.array([0.0, 0.0, 1.0])))
    assert point == pytest.approx((170.0, 105.0))
    assert np.allclose(screen.to_pixels(point, (1920, 1080)), [960.0, 540.0])


def test_camera_intrinsics_round_trip():
    intrinsics = CameraIntrinsics((800.0, 810.0), (640.0, 360.0))
    restored = CameraIntrinsics.from_matrix(intrinsics.matrix())
    assert restored == intrinsics

