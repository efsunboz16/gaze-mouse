import math

import numpy as np

from gaze_mouse.eyelid.features import (
    EyeLandmarkMap,
    compute_ears,
    eye_aspect_ratio,
    eye_blink_scores,
    to_pixels,
)


def eye(width: float, lid: float, x0: float = 0.0) -> np.ndarray:
    """Six points p1..p6 of a synthetic eye with half-opening `lid`."""
    return np.array(
        [[x0, 0], [x0 + width / 3, -lid], [x0 + 2 * width / 3, -lid],
         [x0 + width, 0], [x0 + 2 * width / 3, lid], [x0 + width / 3, lid]],
        dtype=np.float32,
    )


def test_open_eye():
    assert math.isclose(eye_aspect_ratio(eye(30, 6)), 0.4, abs_tol=1e-6)


def test_closed_eye_is_zero():
    assert eye_aspect_ratio(eye(30, 0)) == 0.0


def test_scale_invariant():
    assert math.isclose(eye_aspect_ratio(eye(30, 6)), eye_aspect_ratio(eye(90, 18)), abs_tol=1e-6)


def test_degenerate_eye_is_nan():
    assert math.isnan(eye_aspect_ratio(np.zeros((6, 2), dtype=np.float32)))


def test_compute_ears_uses_each_eye_indices():
    pts = np.vstack([eye(30, 6), eye(30, 3, x0=100)])
    eye_map = EyeLandmarkMap(tuple(range(6, 12)), tuple(range(6)), 0, 0)
    left, right = compute_ears(pts, eye_map)
    assert math.isclose(left, 0.2, abs_tol=1e-6)
    assert math.isclose(right, 0.4, abs_tol=1e-6)


def test_to_pixels_scales_axes_independently():
    px = to_pixels(np.array([[0.5, 0.5, 0.1]], dtype=np.float32), 1280, 720)
    assert px.tolist() == [[640.0, 360.0]]


def test_blink_scores_missing_are_none():
    assert eye_blink_scores({}) == (None, None)
    assert eye_blink_scores({"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}) == (0.9, 0.1)
