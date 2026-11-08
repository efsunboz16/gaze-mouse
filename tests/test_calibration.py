import math

import numpy as np
import pytest

from gaze_mouse.calibration.analysis import evaluate, fit_variant, training_set, usable_mask
from gaze_mouse.calibration.model import GazeModel, fit_gaze_model
from gaze_mouse.calibration.procedure import CalibrationProcedure, Phase, PursuitPath
from gaze_mouse.calibration.profile import load_profile, save_profile
from gaze_mouse.frame import FrameResult
from gaze_mouse.gaze.features import FEATURE_DIM
from gaze_mouse.recording import Target

SECOND = 1_000_000_000
CAL = {
    "discard_first_ms": 500, "pursuit_discard_first_ms": 500, "pursuit_weight": 1.0,
    "ridge_alpha": 1e-6, "validation_hit_box_px": 150,
}


def features_for(xy: np.ndarray) -> np.ndarray:
    """Full feature vectors whose iris columns encode the screen position linearly."""
    f = np.zeros((len(xy), FEATURE_DIM), dtype=np.float64)
    f[:, 0] = xy[:, 0] / 1000
    f[:, 1] = xy[:, 1] / 1000
    f[:, 2] = xy[:, 0] / 2000
    f[:, 3] = xy[:, 1] / 2000
    return f


GRID = np.array([(x, y) for y in (100, 500, 900) for x in (100, 500, 900)], dtype=np.float64)


def test_fit_recovers_linear_mapping():
    model = fit_gaze_model(features_for(GRID), GRID, ["iris"], 1, 1e-6)
    query = np.array([[300.0, 700.0]])
    assert model.predict(features_for(query)) == pytest.approx(query, abs=0.5)


def test_poly2_fits_quadratic_mapping_better_than_linear():
    x = np.linspace(-1, 1, 7)
    xy = np.array([(a, b) for a in x for b in x])
    f = np.zeros((len(xy), FEATURE_DIM))
    f[:, :2] = xy
    f[:, 2:4] = xy[:, ::-1] * 0.5 + 0.1 * np.sin(xy * 7)
    target = np.column_stack([xy[:, 0] ** 2 * 500, xy[:, 1] * 300])
    linear = np.abs(fit_gaze_model(f, target, ["iris"], 1, 1e-3).predict(f) - target).mean()
    quadratic = np.abs(fit_gaze_model(f, target, ["iris"], 2, 1e-3).predict(f) - target).mean()
    assert quadratic < linear / 5


def test_weights_pull_fit_toward_heavy_samples():
    f = np.zeros((2, FEATURE_DIM))
    target = np.array([[0.0, 0.0], [100.0, 100.0]])
    model = fit_gaze_model(f, target, ["iris"], 1, 1.0, np.array([9.0, 1.0]))
    assert model.predict(f[:1]) == pytest.approx(np.array([[10.0, 10.0]]))


def test_invalid_degree_rejected():
    with pytest.raises(ValueError):
        fit_gaze_model(features_for(GRID), GRID, ["iris"], 3, 1.0)


def test_model_roundtrip_through_profile(tmp_path):
    original = fit_gaze_model(features_for(GRID), GRID, ["iris"], 2, 1.0)
    path = tmp_path / "nested" / "user.yaml"
    save_profile(path, {"gaze_model": original.to_dict()})
    restored = GazeModel.from_dict(load_profile(path)["gaze_model"])
    assert restored.predict(features_for(GRID)) == pytest.approx(original.predict(features_for(GRID)))


def test_pursuit_starts_at_center_and_stays_inside_margin():
    path = PursuitPath(1000, 500, 0.1, 8.0, (1, 2))
    assert path.at(0.0) == (500, 250)
    points = np.array([path.at(t) for t in np.linspace(0, 8, 400)])
    assert points[:, 0].min() >= 100 and points[:, 0].max() <= 900
    assert points[:, 1].min() >= 50 and points[:, 1].max() <= 450


def make_procedure() -> CalibrationProcedure:
    return CalibrationProcedure(
        [Target(0, 10, 20), Target(1, 30, 40)], [Target(0, 50, 60)],
        seconds_per_point=1.0, pursuit=PursuitPath(1000, 500, 0.1, 2.0, (1, 2)), countdown_s=1.0,
    )


def frame(t_s: float, features: bool = True) -> FrameResult:
    r = FrameResult(int(t_s * SECOND), 0, 1280, 720, ear_left=0.3, ear_right=0.3, distance_mm=600.0)
    if features:
        r.gaze_features = np.ones(FEATURE_DIM, dtype=np.float32)
    return r


def step(p: CalibrationProcedure, t_s: float) -> None:
    now = int(t_s * SECOND)
    p.tick(now)
    p.mark_shown(now)


def test_procedure_walks_through_every_phase():
    p = make_procedure()
    assert p.phase == Phase.COUNTDOWN and p.stimulus(0).countdown == 1
    p.tick(100 * SECOND)
    assert p.phase == Phase.COUNTDOWN  # countdown has not been started

    p.start(0)
    step(p, 1.0)
    assert p.phase == Phase.POINTS and p.stimulus(SECOND).position == (10, 20)
    step(p, 2.0)
    assert p.stimulus(2 * SECOND).position == (30, 40)
    step(p, 3.0)
    assert p.phase == Phase.PURSUIT
    step(p, 5.0)
    assert p.phase == Phase.VALIDATION and p.stimulus(5 * SECOND).position == (50, 60)
    step(p, 6.0)
    assert p.phase == Phase.DONE and p.stimulus(6 * SECOND).position is None


def test_frames_are_tagged_with_phase_target_and_onset():
    p = make_procedure()
    p.start(0)
    p.add_frame(frame(0.5))  # countdown frames are not stored
    step(p, 1.0)
    p.add_frame(frame(1.75))
    step(p, 2.0)
    p.add_frame(frame(2.25))
    arrays = p.to_arrays()
    assert arrays["phase"].tolist() == [int(Phase.POINTS)] * 2
    assert arrays["seq"].tolist() == [0, 1]
    assert arrays["target_xy_px"].tolist() == [[10, 20], [30, 40]]
    assert arrays["ms_since_onset"].tolist() == pytest.approx([750.0, 250.0])


def test_frame_without_features_is_stored_as_nan():
    p = make_procedure()
    p.start(0)
    step(p, 1.0)
    p.add_frame(frame(1.75, features=False))
    assert np.isnan(p.to_arrays()["features"]).all()


def samples_from(rows) -> dict:
    """rows: (phase, seq, xy, ms, features, ear)."""
    return {
        "phase": np.array([int(r[0]) for r in rows]),
        "seq": np.array([r[1] for r in rows]),
        "target_xy_px": np.array([r[2] for r in rows]),
        "ms_since_onset": np.array([r[3] for r in rows], dtype=np.float32),
        "features": np.array([r[4] for r in rows], dtype=np.float32),
        "ear_left": np.array([r[5] for r in rows], dtype=np.float32),
        "ear_right": np.array([r[5] for r in rows], dtype=np.float32),
        "distance_mm": np.full(len(rows), 600.0, dtype=np.float32),
    }


def calibration_samples(validation_xy, offset=(0.0, 0.0)) -> dict:
    rows = []
    for seq, xy in enumerate(GRID):
        rows.append((Phase.POINTS, seq, xy, 800.0, features_for(xy[None])[0], 0.3))
    for seq, xy in enumerate(validation_xy):
        shifted = np.asarray(xy, float) - offset  # model will miss the target by `offset`
        rows.append((Phase.VALIDATION, seq, xy, 800.0, features_for(shifted[None])[0], 0.3))
    return samples_from(rows)


def test_usable_mask_drops_settling_blinks_and_missing_faces():
    good = np.ones(FEATURE_DIM)
    nan = np.full(FEATURE_DIM, np.nan)
    samples = samples_from([
        (Phase.POINTS, 0, (0, 0), 800.0, good, 0.3),
        (Phase.POINTS, 0, (0, 0), 100.0, good, 0.3),  # still settling
        (Phase.POINTS, 0, (0, 0), 800.0, good, 0.1),  # eyes closing
        (Phase.POINTS, 0, (0, 0), 800.0, nan, 0.3),  # no face features
        (Phase.VALIDATION, 0, (0, 0), 800.0, good, 0.3),  # other phase
    ])
    assert usable_mask(samples, Phase.POINTS, 500, 0.7).tolist() == [True, False, False, False, False]


def test_training_set_uses_point_medians_and_pursuit_weight():
    good = np.ones(FEATURE_DIM)
    rows = [(Phase.POINTS, 0, (10, 10), 800.0 + i, good * v, 0.3) for i, v in enumerate((1.0, 2.0, 30.0))]
    rows += [(Phase.POINTS, 1, (20, 20), 800.0, good, 0.3)]
    rows += [(Phase.PURSUIT, 0, (i, i), 800.0, good, 0.3) for i in range(4)]
    features, targets, weights = training_set(samples_from(rows), CAL, 0.7, use_pursuit=True)
    assert features.shape == (6, FEATURE_DIM)
    assert features[0, 0] == 2.0  # median rejects the outlier
    assert targets[:2].tolist() == [[10, 10], [20, 20]]
    assert weights.tolist() == [1.0, 1.0, 0.5, 0.5, 0.5, 0.5]


def test_training_set_without_pursuit():
    good = np.ones(FEATURE_DIM)
    rows = [(Phase.POINTS, 0, (10, 10), 800.0, good, 0.3), (Phase.PURSUIT, 0, (1, 1), 800.0, good, 0.3)]
    assert len(training_set(samples_from(rows), CAL, 0.7, use_pursuit=False)[0]) == 1


def test_training_set_without_usable_points_raises():
    with pytest.raises(ValueError):
        training_set(samples_from([(Phase.POINTS, 0, (1, 1), 100.0, np.ones(FEATURE_DIM), 0.3)]), CAL, 0.7, False)


def test_evaluate_perfect_model_hits_everything():
    samples = calibration_samples([(300, 300), (700, 700)])
    model = fit_variant(samples, CAL, 0.7, ["iris"], "ridge", False)
    metrics = evaluate(model, samples, CAL, 0.7, mm_per_px=0.177)
    assert metrics.frames == 2 and metrics.hit_rate == 1.0
    assert metrics.mean_px == pytest.approx(0.0, abs=0.5)
    assert metrics.passes(0.8)


def test_evaluate_reports_axes_degrees_and_box_misses():
    samples = calibration_samples([(300, 300), (700, 700)], offset=(0.0, 200.0))
    model = fit_variant(samples, CAL, 0.7, ["iris"], "ridge", False)
    metrics = evaluate(model, samples, CAL, 0.7, mm_per_px=0.2)
    assert metrics.mean_abs_dy_px == pytest.approx(200.0, abs=0.5)
    assert metrics.mean_abs_dx_px == pytest.approx(0.0, abs=0.5)
    assert metrics.mean_deg == pytest.approx(math.degrees(math.atan2(40.0, 600.0)), abs=0.05)
    assert metrics.hit_rate == 0.0 and not metrics.passes(0.8)


def test_evaluate_without_validation_frames_fails_gate():
    samples = calibration_samples([])
    model = fit_variant(samples, CAL, 0.7, ["iris"], "ridge", False)
    metrics = evaluate(model, samples, CAL, 0.7, mm_per_px=0.2)
    assert metrics.frames == 0 and not metrics.passes(0.0)
