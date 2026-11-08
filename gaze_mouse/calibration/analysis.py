"""M6: turn calibration samples into a fitted model and validation metrics. Pure logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gaze_mouse.calibration.model import GazeModel, fit_gaze_model
from gaze_mouse.calibration.procedure import Phase

_DEGREES = {"ridge": 1, "poly2": 2}


def ear_baselines(samples: dict[str, np.ndarray]) -> tuple[float, float]:
    """Median open-eye EAR per eye over the whole procedure."""
    return float(np.nanmedian(samples["ear_left"])), float(np.nanmedian(samples["ear_right"]))


def usable_mask(samples: dict[str, np.ndarray], phase: Phase, discard_ms: float, openness_ratio: float) -> np.ndarray:
    """Frames of `phase` past the settling time, with a face and open eyes."""
    left, right = ear_baselines(samples)
    return (
        (samples["phase"] == int(phase))
        & (samples["ms_since_onset"] > discard_ms)
        & np.isfinite(samples["features"]).all(axis=1)
        & (samples["ear_left"] >= openness_ratio * left)
        & (samples["ear_right"] >= openness_ratio * right)
    )


def training_set(
    samples: dict[str, np.ndarray], cal: dict[str, Any], openness_ratio: float, use_pursuit: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-point medians (weight 1) plus optional pursuit frames sharing a total weight."""
    mask = usable_mask(samples, Phase.POINTS, cal["discard_first_ms"], openness_ratio)
    seqs = np.unique(samples["seq"][mask])
    if len(seqs) == 0:
        raise ValueError("No usable calibration points: check that the face is visible")
    features = np.array([np.median(samples["features"][mask & (samples["seq"] == s)], axis=0) for s in seqs])
    targets = np.array([samples["target_xy_px"][mask & (samples["seq"] == s)][0] for s in seqs], dtype=np.float64)
    weights = np.ones(len(seqs))
    if use_pursuit:
        pursuit = usable_mask(samples, Phase.PURSUIT, cal["pursuit_discard_first_ms"], openness_ratio)
        count = int(pursuit.sum())
        if count:
            features = np.vstack([features, samples["features"][pursuit]])
            targets = np.vstack([targets, samples["target_xy_px"][pursuit].astype(np.float64)])
            weights = np.concatenate([weights, np.full(count, cal["pursuit_weight"] * len(seqs) / count)])
    return features, targets, weights


def fit_variant(
    samples: dict[str, np.ndarray],
    cal: dict[str, Any],
    openness_ratio: float,
    groups: list[str],
    regressor: str,
    use_pursuit: bool,
) -> GazeModel:
    features, targets, weights = training_set(samples, cal, openness_ratio, use_pursuit)
    return fit_gaze_model(features, targets, groups, _DEGREES[regressor], cal["ridge_alpha"], weights)


@dataclass(frozen=True)
class ValidationMetrics:
    frames: int
    mean_px: float
    p95_px: float
    mean_abs_dx_px: float
    mean_abs_dy_px: float
    mean_deg: float
    hit_rate: float  # raw frames inside the box
    click_hit_rate: float  # click-position estimates inside the box; this drives the gate

    def passes(self, min_hit_rate: float) -> bool:
        return self.frames > 0 and self.click_hit_rate >= min_hit_rate

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def click_point_estimates(
    predictions: np.ndarray, timestamps_ns: np.ndarray, seqs: np.ndarray, window_ms: float
) -> np.ndarray:
    """Causal median of the predictions in the window ending at each frame, within one target."""
    window_ns = int(window_ms * 1e6)
    estimates = np.empty_like(predictions)
    for i in range(len(predictions)):
        recent = (
            (seqs == seqs[i])
            & (timestamps_ns <= timestamps_ns[i])
            & (timestamps_ns > timestamps_ns[i] - window_ns)
        )
        estimates[i] = np.median(predictions[recent], axis=0)
    return estimates


def evaluate(
    model: GazeModel,
    samples: dict[str, np.ndarray],
    cal: dict[str, Any],
    openness_ratio: float,
    mm_per_px: float,
) -> ValidationMetrics:
    """Per-frame error on the validation targets, which the model never saw."""
    mask = usable_mask(samples, Phase.VALIDATION, cal["discard_first_ms"], openness_ratio)
    if not mask.any():
        nan = float("nan")
        return ValidationMetrics(0, nan, nan, nan, nan, nan, 0.0, 0.0)
    # The click window may reach back into frames that are excluded from the error metrics.
    stream = (samples["phase"] == int(Phase.VALIDATION)) & np.isfinite(samples["features"]).all(axis=1)
    predictions = model.predict(samples["features"][stream])
    clicks = click_point_estimates(
        predictions, samples["timestamp_ns"][stream], samples["seq"][stream], cal["validation_click_window_ms"]
    )
    in_stream = mask[stream]
    targets = samples["target_xy_px"][mask]
    delta = predictions[in_stream] - targets
    dist_px = np.linalg.norm(delta, axis=1)
    distance_mm = samples["distance_mm"][mask]
    half_box = cal["validation_hit_box_px"] / 2
    return ValidationMetrics(
        frames=int(mask.sum()),
        mean_px=float(dist_px.mean()),
        p95_px=float(np.percentile(dist_px, 95)),
        mean_abs_dx_px=float(np.abs(delta[:, 0]).mean()),
        mean_abs_dy_px=float(np.abs(delta[:, 1]).mean()),
        mean_deg=float(np.degrees(np.arctan2(dist_px * mm_per_px, distance_mm)).mean()),
        hit_rate=float((np.abs(delta) <= half_box).all(axis=1).mean()),
        click_hit_rate=float((np.abs(clicks[in_stream] - targets) <= half_box).all(axis=1).mean()),
    )
