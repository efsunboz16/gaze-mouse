import cv2
import numpy as np
import pytest

from gaze_mouse.perception.camera_calibration import (
    CameraIntrinsics,
    calibrate_camera,
    chessboard_object_points,
    detect_chessboard_corners,
)
from gaze_mouse.perception.head_pose import (
    estimate_head_pose,
    pixel_to_camera_point,
    transform_points,
)


def test_chessboard_object_points_have_expected_metric_layout():
    points = chessboard_object_points((3, 2), 10.0)
    assert points.tolist() == [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 0.0],
        [20.0, 10.0, 0.0],
    ]


def test_detect_chessboard_corners_returns_inner_corners():
    squares = np.zeros((280, 400), dtype=np.uint8)
    for row in range(7):
        for column in range(10):
            if (row + column) % 2:
                squares[row * 40:(row + 1) * 40, column * 40:(column + 1) * 40] = 255
    corners = detect_chessboard_corners(squares, (9, 6))
    assert corners is not None
    assert corners.shape == (54, 1, 2)


def test_calibrate_camera_recovers_intrinsics_from_synthetic_views():
    intrinsics = CameraIntrinsics((800.0, 810.0), (320.0, 240.0))
    board = chessboard_object_points((6, 5), 20.0)
    image_points = []
    views = [
        ([-0.08, 0.05, 0.02], [0.0, 0.0, 700.0]),
        ([0.06, -0.07, -0.03], [30.0, -20.0, 760.0]),
        ([-0.05, -0.04, 0.06], [-25.0, 15.0, 820.0]),
        ([0.1, 0.02, -0.05], [15.0, 25.0, 740.0]),
        ([-0.07, 0.08, 0.04], [-35.0, -10.0, 880.0]),
    ]
    for rotation, translation in views:
        projected, _ = cv2.projectPoints(
            board, np.asarray(rotation, dtype=np.float64), np.asarray(translation, dtype=np.float64),
            intrinsics.matrix(), None,
        )
        image_points.append(projected.astype(np.float32))
    result = calibrate_camera(image_points, (640, 480), (6, 5), 20.0)
    assert result.views_used == 5
    assert result.reprojection_error_px < 1e-4
    assert result.intrinsics.focal_length_px == pytest.approx((800.0, 810.0), rel=1e-3)


def test_estimate_head_pose_recovers_translation_for_synthetic_face():
    intrinsics = CameraIntrinsics((800.0, 800.0), (320.0, 240.0))
    face = np.array([
        [-60.0, -40.0, 0.0], [60.0, -40.0, 0.0],
        [60.0, 40.0, 0.0], [-60.0, 40.0, 0.0],
        [0.0, 0.0, 40.0], [0.0, 30.0, 20.0],
    ])
    projected, _ = cv2.projectPoints(
        face, np.zeros(3), np.array([0.0, 0.0, 600.0]),
        intrinsics.matrix(), None,
    )
    pose = estimate_head_pose(face, projected[:, 0, :], intrinsics)
    assert pose.translation_vector_mm == pytest.approx([0.0, 0.0, 600.0], abs=1e-3)
    assert np.allclose(transform_points(pose.transform_camera_from_face, face), 
                       np.array([0.0, 0.0, 600.0]) + face, atol=1e-3)


def test_head_pose_rejects_mismatched_point_counts():
    intrinsics = CameraIntrinsics((800.0, 800.0), (320.0, 240.0))
    with pytest.raises(ValueError, match="counts must match"):
        estimate_head_pose(np.zeros((4, 3)), np.zeros((5, 2)), intrinsics)


def test_pixel_to_camera_point_back_projects_at_known_depth():
    intrinsics = CameraIntrinsics((800.0, 800.0), (320.0, 240.0))
    assert np.allclose(pixel_to_camera_point([400.0, 280.0], 600.0, intrinsics), [60.0, 30.0, 600.0])


def test_camera_intrinsics_load_from_calibration_json(tmp_path):
    path = tmp_path / "intrinsics.json"
    path.write_text(
        '{"focal_length_px": [800, 810], "principal_point_px": [320, 240], '
        '"distortion": [0, 0, 0, 0, 0]}',
        encoding="utf-8",
    )
    loaded = CameraIntrinsics.from_json(path)
    assert loaded.focal_length_px == (800.0, 810.0)
    assert loaded.distortion is not None
