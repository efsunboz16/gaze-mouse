import numpy as np
import pytest

from gaze_mouse.gaze.model import (
    angles_to_direction,
    direction_to_angles,
    fit_kappa_from_screen_samples,
)
from gaze_mouse.gaze.screen import ScreenPlane


def test_fit_kappa_from_screen_targets():
    screen = ScreenPlane(
        np.array([-170.0, -105.0, 600.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        340.0,
        210.0,
    )
    centers = np.array([
        [[-30.0, 0.0, 500.0], [30.0, 0.0, 500.0]],
        [[-30.0, 0.0, 500.0], [30.0, 0.0, 500.0]],
    ])
    target_pixels = np.array([[960.0, 540.0], [1200.0, 540.0]])
    predicted = np.empty((2, 2, 3))
    for index in range(2):
        target = screen.point_from_pixels(target_pixels[index], (1920, 1080))
        for eye in range(2):
            predicted[index, eye] = angles_to_direction(
                *direction_to_angles(target - centers[index, eye])
            )
    calibration = fit_kappa_from_screen_samples(
        predicted, centers, target_pixels, screen, (1920, 1080)
    )
    assert np.allclose(calibration.offsets_rad, 0.0, atol=1e-8)


def test_fit_kappa_from_screen_samples_rejects_bad_shapes():
    screen = ScreenPlane(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 10, 10)
    with pytest.raises(ValueError, match="shape"):
        fit_kappa_from_screen_samples(
            np.zeros((2, 3)), np.zeros((2, 2, 3)), np.zeros((2, 2)), screen, (100, 100)
        )
