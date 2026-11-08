"""M6: ridge regression from gaze features to screen pixels, serializable to YAML."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gaze_mouse.gaze.features import select_features


def _scale_params(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = z.std(axis=0)
    std[std == 0] = 1.0
    return z.mean(axis=0), std


def _quadratic_expand(z: np.ndarray) -> np.ndarray:
    n = z.shape[1]
    cross = [z[:, i] * z[:, j] for i in range(n) for j in range(i, n)]
    return np.column_stack([z, *cross])


@dataclass
class GazeModel:
    feature_groups: list[str]
    degree: int
    alpha: float
    mean: np.ndarray
    std: np.ndarray
    poly_mean: np.ndarray | None
    poly_std: np.ndarray | None
    coef: np.ndarray  # Dx2
    intercept: np.ndarray  # 2

    def _design(self, features: np.ndarray) -> np.ndarray:
        z = (select_features(features, self.feature_groups) - self.mean) / self.std
        if self.degree == 2:
            z = (_quadratic_expand(z) - self.poly_mean) / self.poly_std
        return z

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Map Nx9 full feature vectors to Nx2 screen pixels."""
        return self._design(features) @ self.coef + self.intercept

    def to_dict(self) -> dict[str, Any]:
        def lst(a: np.ndarray | None):
            return None if a is None else a.tolist()

        return {
            "feature_groups": self.feature_groups, "degree": self.degree, "alpha": self.alpha,
            "mean": lst(self.mean), "std": lst(self.std),
            "poly_mean": lst(self.poly_mean), "poly_std": lst(self.poly_std),
            "coef": lst(self.coef), "intercept": lst(self.intercept),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GazeModel":
        def arr(a):
            return None if a is None else np.array(a, dtype=np.float64)

        return cls(
            data["feature_groups"], data["degree"], data["alpha"],
            arr(data["mean"]), arr(data["std"]), arr(data["poly_mean"]), arr(data["poly_std"]),
            arr(data["coef"]), arr(data["intercept"]),
        )


def fit_gaze_model(
    features: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    degree: int,
    alpha: float,
    weights: np.ndarray | None = None,
) -> GazeModel:
    """Weighted ridge with an unpenalized intercept; degree is 1 (ridge) or 2 (poly2)."""
    if degree not in (1, 2):
        raise ValueError(f"Unsupported degree: {degree}")
    selected = select_features(features, groups).astype(np.float64)
    mean, std = _scale_params(selected)
    z = (selected - mean) / std
    poly_mean = poly_std = None
    if degree == 2:
        z = _quadratic_expand(z)
        poly_mean, poly_std = _scale_params(z)
        z = (z - poly_mean) / poly_std

    y = targets.astype(np.float64)
    w = np.ones(len(z)) if weights is None else np.asarray(weights, dtype=np.float64)
    z_center = (w[:, None] * z).sum(axis=0) / w.sum()
    y_center = (w[:, None] * y).sum(axis=0) / w.sum()
    zc, yc = z - z_center, y - y_center
    gram = (zc.T * w) @ zc + alpha * np.eye(z.shape[1])
    coef = np.linalg.solve(gram, (zc.T * w) @ yc)
    return GazeModel(groups, degree, alpha, mean, std, poly_mean, poly_std, coef, y_center - z_center @ coef)
