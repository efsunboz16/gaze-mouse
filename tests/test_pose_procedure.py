import numpy as np

from gaze_mouse.calibration.pose_procedure import PoseCalibrationProcedure, PosePhase, PoseVariant
from gaze_mouse.frame import FrameResult
from gaze_mouse.recording import Target


def test_pose_procedure_progresses_through_all_variants():
    procedure = PoseCalibrationProcedure(
        [Target(0, 100, 100)], (PoseVariant("a", "A"), PoseVariant("b", "B")), 1.0, 0.0
    )
    procedure.start(0)
    procedure.tick(0)
    assert procedure.phase == PosePhase.POINTS
    procedure.mark_shown(0)
    procedure.tick(1_000_000_000)
    assert procedure.phase == PosePhase.COUNTDOWN
    assert procedure.pose_variant.name == "b"
    procedure.start(1_000_000_000)
    procedure.tick(1_000_000_000)
    procedure.mark_shown(1_000_000_000)
    procedure.tick(2_000_000_000)
    assert procedure.phase == PosePhase.DONE


def test_pose_procedure_records_target_and_model_fields():
    procedure = PoseCalibrationProcedure(
        [Target(0, 100, 100)], (PoseVariant("a", "A"),), 1.0, 0.0
    )
    procedure.start(0)
    procedure.tick(0)
    procedure.mark_shown(0)
    result = FrameResult(
        500_000_000, 0, 200, 200,
        left_eye_center_camera_mm=np.ones(3),
        right_eye_center_camera_mm=np.ones(3),
        left_gaze_direction_camera=np.array([0.0, 0.0, 1.0]),
        right_gaze_direction_camera=np.array([0.0, 0.0, 1.0]),
    )
    procedure.add_frame(result)
    arrays = procedure.to_arrays()
    assert arrays["pose_index"].tolist() == [0]
    assert arrays["target_xy_px"].tolist() == [[100, 100]]
    assert arrays["left_gaze_direction_camera"].shape == (1, 3)


def test_pose_transition_starts_next_countdown_without_second_start_call():
    procedure = PoseCalibrationProcedure(
        [Target(0, 100, 100)], (PoseVariant("a", "A"), PoseVariant("b", "B")), 1.0, 3.0
    )
    procedure.start(0)
    procedure.tick(3_000_000_000)
    procedure.mark_shown(3_000_000_000)
    procedure.tick(4_000_000_000)

    assert procedure.phase == PosePhase.COUNTDOWN
    assert procedure.stimulus(4_000_000_000).countdown == 3

    procedure.tick(7_000_000_000)
    assert procedure.phase == PosePhase.POINTS
