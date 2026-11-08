import math

import numpy as np

from gaze_mouse.frame import FrameResult
from gaze_mouse.gaze.features import (
    FEATURE_DIM,
    GazeFeatureExtractor,
    distance_from_iris,
    head_pose_angles,
    iris_offset,
    select_features,
)

CFG = {
    "ear_left": [0, 1, 2, 3, 4, 5], "ear_right": [6, 7, 8, 9, 10, 11],
    "iris_left": 12, "iris_right": 13,
    "iris_width_left": [14, 15], "iris_width_right": [16, 17],
}


def face_result(transform=np.eye(4)) -> FrameResult:
    """Two 40 px wide eyes at 100 px-per-unit scale; iris 10 px right of each eye center."""
    w, h = 1000, 1000
    eye = lambda x0: [(x0, 100), (x0 + 13, 90), (x0 + 27, 90), (x0 + 40, 100), (x0 + 27, 110), (x0 + 13, 110)]
    px = eye(100) + eye(300) + [(130, 100), (330, 100), (125, 100), (135, 100), (325, 100), (335, 100)]
    landmarks = np.array([(x / w, y / h, 0.0) for x, y in px], dtype=np.float32)
    return FrameResult(0, 0, w, h, landmarks=landmarks, face_transform=transform)


def extractor() -> GazeFeatureExtractor:
    return GazeFeatureExtractor(CFG, 11.7)


def test_iris_offset_is_relative_to_eye_width():
    pts = np.array([[0, 0], [40, 0], [30, 4]], dtype=np.float32)
    assert np.allclose(iris_offset(pts, (0, 1), 2), [0.25, 0.1])


def test_iris_offset_degenerate_eye_is_nan():
    pts = np.zeros((3, 2), dtype=np.float32)
    assert np.isnan(iris_offset(pts, (0, 1), 2)).all()


def test_identity_transform_has_zero_angles():
    assert head_pose_angles(np.eye(4)) == (0.0, 0.0, 0.0)


def test_yaw_angle_sign_and_size():
    a = math.radians(20)
    transform = np.eye(4)
    transform[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    assert math.isclose(head_pose_angles(transform)[1], a, abs_tol=1e-6)


def test_distance_from_iris():
    assert math.isclose(distance_from_iris(20.0, 1280.0, 11.7), 748.8)
    assert math.isnan(distance_from_iris(0.0, 1280.0, 11.7))


def test_annotate_fills_feature_vector():
    result = face_result()
    extractor().annotate(result)
    assert result.gaze_features.shape == (FEATURE_DIM,)
    assert np.allclose(result.gaze_features[:4], [0.25, 0.0, 0.25, 0.0], atol=1e-4)
    assert np.allclose(result.gaze_features[4:7], 0.0)
    assert math.isclose(result.distance_mm, 1000 * 11.7 / 10, rel_tol=1e-3)


def test_annotate_ignores_missing_face():
    result = FrameResult(0, 0, 1000, 1000)
    extractor().annotate(result)
    assert result.gaze_features is None and result.ear_left is None


def test_annotate_without_transform_has_no_features_but_keeps_ear():
    result = face_result(transform=None)
    extractor().annotate(result)
    assert result.gaze_features is None and result.ear_left is not None


def test_select_features_groups():
    x = np.arange(9.0)[None, :]
    assert select_features(x, ["iris"]).tolist() == [[0, 1, 2, 3]]
    assert select_features(x, ["iris", "distance"]).tolist() == [[0, 1, 2, 3, 8]]
    assert select_features(x, ["pose", "ear"]).tolist() == [[4, 5, 6, 7]]
