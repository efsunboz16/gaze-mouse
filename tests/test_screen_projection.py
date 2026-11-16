import numpy as np

from gaze_mouse.frame import FrameResult
from gaze_mouse.gaze.screen import ScreenPlane, project_frame_gaze


def screen() -> ScreenPlane:
    return ScreenPlane(
        np.array([-170.0, -105.0, 600.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        340.0,
        210.0,
    )


def test_project_frame_gaze_fuses_two_eye_rays():
    result = FrameResult(
        0, 0, 1920, 1080,
        left_eye_center_camera_mm=np.array([-30.0, 0.0, 500.0]),
        right_eye_center_camera_mm=np.array([30.0, 0.0, 500.0]),
        left_gaze_direction_camera=np.array([30.0, 0.0, 100.0]),
        right_gaze_direction_camera=np.array([-30.0, 0.0, 100.0]),
    )
    projected = project_frame_gaze(result, screen(), (1920, 1080))
    assert np.allclose(projected, [960.0, 540.0])
    assert np.allclose(result.screen_gaze_px, projected)


def test_project_frame_gaze_returns_none_without_valid_screen_intersection():
    result = FrameResult(
        0, 0, 1920, 1080,
        left_eye_center_camera_mm=np.zeros(3),
        left_gaze_direction_camera=np.array([1.0, 0.0, 0.0]),
    )
    assert project_frame_gaze(result, screen(), (1920, 1080)) is None
    assert result.screen_gaze_px is None


def test_project_frame_gaze_uses_one_eye_when_other_is_missing():
    result = FrameResult(
        0, 0, 1920, 1080,
        left_eye_center_camera_mm=np.array([0.0, 0.0, 500.0]),
        left_gaze_direction_camera=np.array([0.0, 0.0, 1.0]),
    )
    assert np.allclose(project_frame_gaze(result, screen(), (1920, 1080)), [960.0, 540.0])
