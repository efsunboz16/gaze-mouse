import json

import numpy as np
import pytest

from gaze_mouse.frame import FrameResult
from gaze_mouse.recording import (
    SessionRecorder,
    Target,
    TargetSequence,
    build_targets,
    grid_positions,
    save_session,
)

SECOND = 1_000_000_000


def face_result(index: int, ear: float = 0.3) -> FrameResult:
    return FrameResult(
        timestamp_ns=index * SECOND, frame_index=index, width=1280, height=720,
        landmarks=np.full((478, 3), 0.5, np.float32),
        blendshapes={"eyeBlinkRight": 0.2, "eyeBlinkLeft": 0.1},
        face_transform=np.eye(4), ear_left=ear, ear_right=ear,
    )


def empty_result(index: int) -> FrameResult:
    return FrameResult(timestamp_ns=index * SECOND, frame_index=index, width=1280, height=720)


def test_grid_positions_cover_margins_and_center():
    points = grid_positions(3, 3, 1000, 500, 0.1)
    assert len(points) == 9
    assert points[0] == (100, 50) and points[4] == (500, 250) and points[8] == (900, 450)


def test_single_column_is_centered():
    assert grid_positions(1, 1, 1000, 500, 0.1) == [(500, 250)]


def test_each_repeat_visits_every_position_once():
    targets = build_targets(grid_positions(3, 3, 1000, 500, 0.1), repeats=2, seed=7)
    assert len(targets) == 18
    assert sorted(t.position_id for t in targets[:9]) == list(range(9))
    assert sorted(t.position_id for t in targets[9:]) == list(range(9))


def test_same_seed_gives_same_order():
    positions = grid_positions(3, 3, 1000, 500, 0.1)
    assert build_targets(positions, 2, 42) == build_targets(positions, 2, 42)


def test_sequence_timer_starts_only_when_shown():
    sequence = TargetSequence([Target(0, 1, 1), Target(1, 2, 2)], seconds_per_point=1.0)
    assert not sequence.advance_if_due(100 * SECOND)
    sequence.mark_shown(10 * SECOND)
    assert not sequence.advance_if_due(10 * SECOND + SECOND // 2)
    assert sequence.advance_if_due(11 * SECOND)
    assert sequence.current.position_id == 1 and sequence.onset_ns is None


def test_mark_shown_keeps_first_onset():
    sequence = TargetSequence([Target(0, 1, 1)], seconds_per_point=1.0)
    sequence.mark_shown(5)
    sequence.mark_shown(99)
    assert sequence.onset_ns == 5


def test_sequence_finishes():
    sequence = TargetSequence([Target(0, 1, 1)], seconds_per_point=1.0)
    sequence.mark_shown(0)
    assert sequence.advance_if_due(SECOND)
    assert sequence.finished and sequence.current is None
    sequence.mark_shown(5)
    assert sequence.onset_ns is None


def test_recorder_fills_missing_faces_with_nan():
    recorder = SessionRecorder()
    target = Target(3, 100, 200)
    recorder.add(empty_result(0), 0, target, 0)
    recorder.add(face_result(1), 0, target, 0)
    arrays, names = recorder.to_arrays()

    assert names == ["eyeBlinkLeft", "eyeBlinkRight"]
    assert arrays["face_found"].tolist() == [False, True]
    assert arrays["landmarks"].shape == (2, 478, 3)
    assert np.isnan(arrays["landmarks"][0]).all() and not np.isnan(arrays["landmarks"][1]).any()
    assert arrays["blendshapes"][1].tolist() == pytest.approx([0.1, 0.2])
    assert np.isnan(arrays["ear_left"][0])
    assert arrays["face_transform"].shape == (2, 4, 4)
    assert arrays["target_xy_px"].tolist() == [[100, 200], [100, 200]]
    assert arrays["ms_since_target_onset"].tolist() == pytest.approx([0.0, 1000.0])


def test_recorder_negative_offset_for_frame_before_onset():
    recorder = SessionRecorder()
    recorder.add(face_result(1), 0, Target(0, 1, 1), 2 * SECOND)
    arrays, _ = recorder.to_arrays()
    assert arrays["ms_since_target_onset"][0] == pytest.approx(-1000.0)


def test_recorder_all_frames_without_face():
    recorder = SessionRecorder()
    recorder.add(empty_result(0), 0, Target(0, 1, 1), 0)
    arrays, names = recorder.to_arrays()
    assert names == []
    assert arrays["blendshapes"].shape == (1, 0)
    assert not arrays["face_found"].any()


def test_save_session_roundtrip_and_refuses_overwrite(tmp_path):
    recorder = SessionRecorder()
    recorder.add(face_result(1), 0, Target(0, 1, 1), 0)
    arrays, _ = recorder.to_arrays()
    directory = tmp_path / "session"
    save_session(directory, arrays, {"note": "bakış"})

    with np.load(directory / "frames.npz") as loaded:
        assert loaded["landmarks"].shape == (1, 478, 3)
    assert json.loads((directory / "session.json").read_text(encoding="utf-8")) == {"note": "bakış"}
    with pytest.raises(FileExistsError):
        save_session(directory, arrays, {})
