import numpy as np
import pytest

from gaze_mouse.gaze.model import (
    GazePrediction,
    KappaCalibration,
    angles_to_direction,
    direction_to_angles,
    fit_kappa_calibration,
)


def test_direction_angle_round_trip():
    angles = (0.12, -0.2)
    assert direction_to_angles(angles_to_direction(*angles)) == pytest.approx(angles)


def test_fit_kappa_calibration_and_apply():
    predicted = np.zeros((3, 2, 2))
    target = np.array([
        [[0.10, -0.05], [0.02, 0.04]],
        [[0.10, -0.05], [0.02, 0.04]],
        [[0.10, -0.05], [0.02, 0.04]],
    ])
    calibration = fit_kappa_calibration(predicted, target)
    prediction = GazePrediction(
        angles_to_direction(0.0, 0.0),
        angles_to_direction(0.0, 0.0),
        0.1,
    )
    corrected = calibration.apply(prediction)
    assert direction_to_angles(corrected.left_direction_camera) == pytest.approx((0.1, -0.05))
    assert direction_to_angles(corrected.right_direction_camera) == pytest.approx((0.02, 0.04))


def test_fit_kappa_calibration_wraps_angle_residuals():
    predicted = np.array([[[0.0, np.pi - 0.05], [0.0, np.pi - 0.05]]])
    target = np.array([[[0.0, -np.pi + 0.05], [0.0, -np.pi + 0.05]]])

    calibration = fit_kappa_calibration(predicted, target)

    assert calibration.offsets_rad[:, 1] == pytest.approx([0.1, 0.1])


def test_kappa_apply_preserves_negative_z_direction():
    prediction = GazePrediction(
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, -1.0]),
        0.1,
    )
    corrected = KappaCalibration(np.full((2, 2), 0.1)).apply(prediction)

    assert corrected.left_direction_camera[2] < 0
    assert corrected.right_direction_camera[2] < 0


def test_kappa_calibration_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        KappaCalibration(np.zeros((2, 3)))


def test_kappa_calibration_loads_from_json(tmp_path):
    path = tmp_path / "kappa.json"
    path.write_text('{"offsets_rad": [[0.1, 0.2], [0.3, 0.4]]}', encoding="utf-8")
    calibration = KappaCalibration.from_json(path)
    assert calibration.offsets_rad.tolist() == [[0.1, 0.2], [0.3, 0.4]]
