import math

import numpy as np
import pytest

from gaze_mouse.cursor.controller import GazeHistory, PointerController
from gaze_mouse.eyelid.classifier import WinkClassifier
from gaze_mouse.filtering.one_euro import OneEuroFilter
from gaze_mouse.frame import FrameResult
from gaze_mouse.gaze.features import FEATURE_DIM

MS = 1_000_000
FRAME_MS = 33
EYELID = {
    "close_ratio": 0.6, "open_ratio": 0.8, "wink_min_ms": 200, "long_wink_ms": 1500, "wink_min_asymmetry": 0.15,
    "click_lookback_ms": [150, 300],
}
BASE = 0.3
OPEN, CLOSED = 0.3, 0.1


def run(pairs) -> list:
    """Feed (left, right) EAR pairs at 30 fps and collect events."""
    classifier = WinkClassifier(EYELID, BASE, BASE)
    events = [classifier.update(i * FRAME_MS * MS, left, right) for i, (left, right) in enumerate(pairs)]
    return [e for e in events if e is not None]


def closure(frames, left=CLOSED, right=OPEN):
    return [(OPEN, OPEN)] * 3 + [(left, right)] * frames + [(OPEN, OPEN)] * 3


def test_left_wink_detected_with_onset_and_duration():
    events = run(closure(9))
    assert len(events) == 1
    assert events[0].side == "left"
    assert events[0].onset_ns == 3 * FRAME_MS * MS
    assert events[0].duration_ms == pytest.approx(9 * FRAME_MS)


def test_right_wink_detected():
    assert [e.side for e in run(closure(9, left=OPEN, right=CLOSED))] == ["right"]


def test_natural_blink_with_both_eyes_is_ignored():
    assert run(closure(6, left=CLOSED, right=CLOSED)) == []


def test_short_closure_is_ignored():
    assert run(closure(3)) == []


def test_long_wink_is_reserved_for_phase_2():
    assert run(closure(46)) == []


def test_slightly_uneven_eyes_narrowing_together_is_ignored():
    pairs = closure(9, 0.45 * BASE, 0.59 * BASE)
    assert run(pairs) == []


def test_just_enough_asymmetry_is_a_wink():
    assert [e.side for e in run(closure(9, 0.45 * BASE, 0.61 * BASE))] == ["left"]


def test_wink_longer_than_a_second_is_still_a_short_wink_in_phase_1():
    assert [e.side for e in run(closure(31))] == ["left"]


@pytest.mark.parametrize("own, other", [(0.51, 0.73), (0.43, 0.70), (0.43, 0.65), (0.58, 0.94)])
def test_measured_left_winks_with_the_other_eye_narrowing(own, other):
    assert [e.side for e in run(closure(9, own * BASE, other * BASE))] == ["left"]


@pytest.mark.parametrize("own, other", [(0.46, 0.84), (0.42, 0.69), (0.26, 0.51), (0.47, 0.91), (0.41, 0.80)])
def test_measured_right_winks_fire_only_for_the_winking_eye(own, other):
    assert [e.side for e in run(closure(9, other * BASE, own * BASE))] == ["right"]


@pytest.mark.parametrize("left, right", [(0.28, 0.24), (0.35, 0.28), (0.29, 0.22), (0.24, 0.29), (0.28, 0.35)])
def test_measured_blinks_are_ignored(left, right):
    assert run(closure(9, left * BASE, right * BASE)) == []


def test_looking_down_narrows_both_eyes_without_a_click():
    assert run(closure(15, 0.55 * BASE, 0.55 * BASE)) == []


def test_hysteresis_band_does_not_reopen_the_eye():
    pairs = [(OPEN, OPEN)] * 2 + [(CLOSED, OPEN)] * 4 + [(0.7 * BASE, OPEN)] * 4 + [(CLOSED, OPEN)] * 2 + [(OPEN, OPEN)]
    events = run(pairs)
    assert len(events) == 1 and events[0].duration_ms == pytest.approx(10 * FRAME_MS)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_lost_tracking_cancels_a_closure(missing):
    pairs = [(OPEN, OPEN)] * 2 + [(CLOSED, OPEN)] * 6 + [(missing, missing)] + [(OPEN, OPEN)] * 3
    assert run(pairs) == []


def test_baseline_is_per_eye():
    classifier = WinkClassifier(EYELID, 0.2, 0.4)
    classifier.update(0, 0.2, 0.4)
    classifier.update(33 * MS, 0.1, 0.4)
    assert classifier.update(366 * MS, 0.2, 0.4).side == "left"


def test_one_euro_first_value_passes_through():
    f = OneEuroFilter(1.0, 0.0, 1.0)
    assert f.update(0, np.array([10.0, 20.0])).tolist() == [10.0, 20.0]


def test_one_euro_reduces_jitter_on_a_still_target():
    rng = np.random.default_rng(0)
    noisy = np.array([100.0, 100.0]) + rng.normal(0, 30, (300, 2))
    f = OneEuroFilter(1.0, 0.0, 1.0)
    out = np.array([f.update(i * FRAME_MS * MS, p) for i, p in enumerate(noisy)])
    assert out[30:].std(axis=0).max() < noisy[30:].std(axis=0).min() / 2


def test_one_euro_beta_tracks_a_jump_faster():
    def after_jump(beta):
        f = OneEuroFilter(1.0, beta, 1.0)
        f.update(0, np.zeros(2))
        return f.update(33 * MS, np.array([1000.0, 0.0]))[0]

    assert after_jump(1.0) > after_jump(0.0) * 2


def test_one_euro_ignores_repeated_timestamp():
    f = OneEuroFilter(1.0, 0.0, 1.0)
    f.update(5, np.array([1.0, 1.0]))
    assert f.update(5, np.array([500.0, 500.0])).tolist() == [1.0, 1.0]


def test_history_median_within_window_and_expiry():
    history = GazeHistory(keep_ms=100)
    for i in range(10):
        history.add(i * 50 * MS, np.array([float(i), 0.0]))
    assert history.median_between(300 * MS, 450 * MS).tolist() == [8.0, 0.0]
    assert history.median_between(0, 200 * MS) is None


class IdentityModel:
    """Reads the screen point straight from the first two feature columns."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, :2].astype(np.float64)


CFG = {
    "eyelid": EYELID,
    "filter": {"one_euro": {"min_cutoff_hz": 1000.0, "beta": 0.0, "d_cutoff_hz": 1.0}},  # effectively no smoothing
    "fixation": {"blink_mask": {"openness_ratio": 0.7, "post_open_ms": 100}},
    "safety": {"corner_margin_px": 5},
}


def controller() -> PointerController:
    return PointerController(IdentityModel(), CFG, (BASE, BASE), (1920, 1200))


def frame(i: int, xy=(500.0, 400.0), left=OPEN, right=OPEN) -> FrameResult:
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    features[:2] = xy
    return FrameResult(i * FRAME_MS * MS, i, 1280, 720, landmarks=np.zeros((478, 3)),
                       ear_left=left, ear_right=right, gaze_features=features)


def test_tracking_moves_cursor():
    pointer = controller()
    output = pointer.update(frame(0))
    assert output.status == "tracking" and output.cursor == (500, 400) and output.click is None


def test_gaze_is_clamped_outside_the_corner_area():
    assert controller().update(frame(0, xy=(-50.0, 5000.0))).cursor == (6, 1193)


def test_cursor_freezes_while_an_eye_closes_and_just_after_reopening():
    pointer = controller()
    pointer.update(frame(0))
    closing = pointer.update(frame(1, left=CLOSED))
    assert closing.status == "frozen" and closing.cursor is None
    assert pointer.update(frame(2)).status == "frozen"  # within post_open_ms
    assert pointer.update(frame(6)).status == "tracking"


def test_missing_face_produces_no_events():
    pointer = controller()
    pointer.update(frame(0))
    lost = FrameResult(33 * MS, 1, 1280, 720)
    assert pointer.update(lost).status == "no_face"
    assert pointer.update(lost).click is None


def test_wink_clicks_where_the_user_looked_before_closing():
    pointer = controller()
    for i in range(12):
        pointer.update(frame(i, xy=(800.0, 600.0)))
    # The iris drifts as the eye closes; the click must ignore these frames.
    outputs = [pointer.update(frame(i, xy=(100.0, 1100.0), left=CLOSED)) for i in range(12, 21)]
    assert all(o.click is None for o in outputs)
    click = pointer.update(frame(21, xy=(100.0, 1100.0))).click
    assert click is not None and (click.button, click.x, click.y) == ("left", 800, 600)


def test_right_wink_is_a_right_click():
    pointer = controller()
    for i in range(12):
        pointer.update(frame(i))
    for i in range(12, 21):
        pointer.update(frame(i, right=CLOSED))
    assert pointer.update(frame(21)).click.button == "right"


def test_wink_without_recent_gaze_does_not_click():
    pointer = controller()
    pointer.update(frame(0))
    for i in range(1, 10):
        pointer.update(frame(i, left=CLOSED))
    assert pointer.update(frame(10)).click is None


def test_natural_blink_does_not_click():
    pointer = controller()
    for i in range(12):
        pointer.update(frame(i))
    outputs = [pointer.update(frame(i, left=CLOSED, right=CLOSED)) for i in range(12, 18)]
    outputs += [pointer.update(frame(i)) for i in range(18, 22)]
    assert all(o.click is None for o in outputs)


def test_filter_constants_are_used():
    cfg = {**CFG, "filter": {"one_euro": {"min_cutoff_hz": 1.0, "beta": 0.0, "d_cutoff_hz": 1.0}}}
    pointer = PointerController(IdentityModel(), cfg, (BASE, BASE), (1920, 1200))
    pointer.update(frame(0, xy=(100.0, 100.0)))
    x = pointer.update(frame(1, xy=(1000.0, 100.0))).cursor[0]
    assert 100 < x < 1000 and not math.isclose(x, 1000)
