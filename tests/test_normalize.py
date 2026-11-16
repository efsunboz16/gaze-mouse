import numpy as np
import pytest

from gaze_mouse.gaze.normalize import NormalizationSpec, normalize_face


def test_normalize_face_places_eye_geometry_at_target():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    points = np.array([[80.0, 100.0], [120.0, 100.0]])
    result = normalize_face(
        image, points, (0,), (1,),
        NormalizationSpec(target_eye_distance_px=80.0, target_eye_midpoint_px=(112.0, 112.0)),
    )
    left, right = result.eye_centers_px
    assert np.allclose((left + right) / 2.0, [112.0, 112.0])
    assert np.isclose(np.linalg.norm(right - left), 80.0)
    assert result.transform.shape == (2, 3)
    assert result.inverse_transform.shape == (2, 3)


def test_normalize_face_rotates_diagonal_eyes_to_horizontal():
    image = np.zeros((200, 300), dtype=np.uint8)
    points = np.array([[80.0, 80.0], [120.0, 120.0]])
    result = normalize_face(image, points, (0,), (1,))
    left, right = result.eye_centers_px
    assert abs(float(right[1] - left[1])) < 1e-6


def test_normalize_face_rejects_coincident_eyes():
    with pytest.raises(ValueError, match="must not coincide"):
        normalize_face(np.zeros((10, 10), dtype=np.uint8), np.zeros((2, 2)), (0,), (1,))
