"""Adapters for appearance-based 3D gaze models."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from gaze_mouse.frame import FrameResult
from gaze_mouse.gaze.screen import ScreenPlane


@dataclass(frozen=True, slots=True)
class GazePrediction:
    left_direction_camera: np.ndarray
    right_direction_camera: np.ndarray
    uncertainty_rad: float


@dataclass(frozen=True, slots=True)
class KappaCalibration:
    """Per-eye pitch/yaw correction in the model's normalized head frame."""

    offsets_rad: np.ndarray  # shape (2, 2): left/right, pitch/yaw

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets_rad, dtype=np.float64)
        if offsets.shape != (2, 2) or not np.isfinite(offsets).all():
            raise ValueError("offsets_rad must have finite shape (2, 2)")
        object.__setattr__(self, "offsets_rad", offsets)

    def apply(self, prediction: GazePrediction) -> GazePrediction:
        directions = []
        for direction, offset in zip(
            (prediction.left_direction_camera, prediction.right_direction_camera),
            self.offsets_rad,
        ):
            pitch, yaw = direction_to_angles(direction)
            corrected = angles_to_direction(pitch + offset[0], yaw + offset[1])
            if direction[2] < 0:
                corrected = -corrected
            directions.append(corrected)
        return GazePrediction(directions[0], directions[1], prediction.uncertainty_rad)

    def to_dict(self) -> dict[str, Any]:
        return {"offsets_rad": self.offsets_rad.tolist()}

    @classmethod
    def from_json(cls, path: str | Path) -> "KappaCalibration":
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)
        return cls(np.asarray(data["offsets_rad"], dtype=np.float64))


def angles_to_direction(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Convert camera-relative pitch/yaw to a unit direction (+Z is forward)."""
    if not np.isfinite([pitch_rad, yaw_rad]).all():
        raise ValueError("Gaze angles must be finite")
    direction = np.array(
        [np.tan(yaw_rad), np.tan(pitch_rad), 1.0], dtype=np.float64
    )
    return direction / np.linalg.norm(direction)


def l2cs_angles_to_direction(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Convert L2CS angles to a camera ray pointing from the eye to the screen."""
    if not np.isfinite([pitch_rad, yaw_rad]).all():
        raise ValueError("Gaze angles must be finite")
    direction = np.array(
        [
            -np.cos(pitch_rad) * np.sin(yaw_rad),
            -np.sin(pitch_rad),
            -np.cos(pitch_rad) * np.cos(yaw_rad),
        ],
        dtype=np.float64,
    )
    return direction / np.linalg.norm(direction)


def direction_to_angles(direction: np.ndarray) -> tuple[float, float]:
    """Convert a unit camera direction to pitch/yaw using the +Z forward axis."""
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("direction must contain three finite values")
    if np.linalg.norm(vector) <= 1e-9:
        raise ValueError("direction must not be zero")
    vector = vector / np.linalg.norm(vector)
    return float(np.arctan2(vector[1], vector[2])), float(np.arctan2(vector[0], vector[2]))


def fit_kappa_calibration(
    predicted_angles_rad: np.ndarray,
    target_angles_rad: np.ndarray,
    weights: np.ndarray | None = None,
) -> KappaCalibration:
    """Fit per-eye angular offsets from pose-diverse calibration samples."""
    predicted = np.asarray(predicted_angles_rad, dtype=np.float64)
    target = np.asarray(target_angles_rad, dtype=np.float64)
    if predicted.ndim != 3 or predicted.shape[1:] != (2, 2) or target.shape != predicted.shape:
        raise ValueError("angle arrays must both have shape (N, 2, 2)")
    if len(predicted) == 0 or not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("angle arrays must be non-empty and finite")
    sample_weights = np.ones(len(predicted)) if weights is None else np.asarray(weights, dtype=np.float64)
    if sample_weights.shape != (len(predicted),) or not np.isfinite(sample_weights).all():
        raise ValueError("weights must have shape (N,) and be finite")
    if np.any(sample_weights < 0) or not np.any(sample_weights > 0):
        raise ValueError("weights must be non-negative with at least one positive value")
    residual = (target - predicted + np.pi) % (2.0 * np.pi) - np.pi
    offsets = np.average(residual, axis=0, weights=sample_weights)
    return KappaCalibration(offsets)


def fit_kappa_from_screen_samples(
    predicted_directions: np.ndarray,
    eye_centers_camera_mm: np.ndarray,
    target_pixels: np.ndarray,
    screen: ScreenPlane,
    screen_size_px: tuple[int, int],
    weights: np.ndarray | None = None,
) -> KappaCalibration:
    """Fit kappa offsets from predicted rays and known screen targets."""
    predicted = np.asarray(predicted_directions, dtype=np.float64)
    centers = np.asarray(eye_centers_camera_mm, dtype=np.float64)
    targets = np.asarray(target_pixels, dtype=np.float64)
    if predicted.ndim != 3 or predicted.shape[1:] != (2, 3):
        raise ValueError("predicted_directions must have shape (N, 2, 3)")
    if centers.shape != predicted.shape or targets.shape != (len(predicted), 2):
        raise ValueError("eye centers must be (N, 2, 3) and targets must be (N, 2)")
    target_angles = np.empty((len(predicted), 2, 2), dtype=np.float64)
    predicted_angles = np.empty_like(target_angles)
    for index in range(len(predicted)):
        target_point = screen.point_from_pixels(targets[index], screen_size_px)
        for eye in range(2):
            target_direction = target_point - centers[index, eye]
            target_angles[index, eye] = direction_to_angles(target_direction)
            predicted_angles[index, eye] = direction_to_angles(predicted[index, eye])
    return fit_kappa_calibration(predicted_angles, target_angles, weights)


class GazeBackend(Protocol):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return [[pitch, yaw, uncertainty], [pitch, yaw, uncertainty]]."""


def _validate_backend_output(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output, dtype=np.float64)
    if values.shape != (2, 3):
        raise ValueError(f"Gaze backend output must have shape (2, 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Gaze backend output must contain finite values")
    if np.any(values[:, 2] < 0):
        raise ValueError("Gaze uncertainty must not be negative")
    return values


class GazeModelAdapter:
    """Convert a model's per-eye angle output into camera-space directions."""

    def __init__(
        self,
        backend: GazeBackend,
        direction_converter: Callable[[float, float], np.ndarray] = angles_to_direction,
    ):
        self._backend = backend
        self._direction_converter = direction_converter

    def predict(self, normalized_face: np.ndarray) -> GazePrediction:
        image = np.asarray(normalized_face)
        if image.ndim not in (2, 3):
            raise ValueError(f"normalized_face must be an image, got {image.shape}")
        values = _validate_backend_output(self._backend(image))
        directions = [
            self._direction_converter(row[0], row[1]) for row in values
        ]
        uncertainty = float(np.mean(values[:, 2]))
        return GazePrediction(directions[0], directions[1], uncertainty)


def apply_prediction(result: FrameResult, prediction: GazePrediction) -> None:
    """Write a model prediction into the shared per-frame data contract."""
    result.left_gaze_direction_camera = prediction.left_direction_camera.copy()
    result.right_gaze_direction_camera = prediction.right_direction_camera.copy()
    result.gaze_uncertainty = prediction.uncertainty_rad


class OnnxGazeBackend:
    """Lazy ONNX Runtime backend for a model with one image input."""

    def __init__(
        self,
        model_path: str | Path,
        input_name: str | None = None,
        color_order: str = "RGB",
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        if color_order not in ("RGB", "BGR"):
            raise ValueError("color_order must be RGB or BGR")
        if len(mean) != 3 or len(std) != 3 or any(value == 0 for value in std):
            raise ValueError("mean and std must contain three non-zero channel values")
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required for ONNX gaze inference; install onnxruntime"
            ) from exc
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError("ONNX gaze model has no inputs")
        self._input_name = input_name or inputs[0].name
        self._color_order = color_order
        self._mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        tensor = np.asarray(image)
        if tensor.ndim != 3 or tensor.shape[2] != 3:
            raise ValueError(f"ONNX gaze input must have shape (H, W, 3), got {tensor.shape}")
        tensor = tensor.astype(np.float32) / 255.0 if tensor.dtype != np.float32 else tensor
        if self._color_order == "RGB":
            tensor = tensor[:, :, ::-1]
        tensor = (tensor - self._mean) / self._std
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.asarray(self._session.run(None, {self._input_name: tensor})[0])


class L2CSOnnxGazeBackend:
    """Run an exported L2CS-Net model and return the project gaze contract."""

    def __init__(self, model_path: str | Path, input_name: str | None = None):
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is required for L2CS gaze inference; install onnxruntime"
            ) from exc
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError("L2CS ONNX model has no inputs")
        self._input_name = input_name or inputs[0].name
        self._indices = np.arange(90, dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        tensor = np.asarray(image)
        if tensor.ndim != 3 or tensor.shape[2] != 3:
            raise ValueError(f"L2CS input must have shape (H, W, 3), got {tensor.shape}")
        tensor = tensor.astype(np.uint8, copy=False)
        tensor = cv2.resize(tensor, (448, 448), interpolation=cv2.INTER_LINEAR)
        tensor = tensor[:, :, ::-1].astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        outputs = self._session.run(None, {self._input_name: tensor})
        if len(outputs) < 2:
            raise ValueError("L2CS ONNX model must return yaw and pitch logits")
        yaw = self._angle_from_logits(outputs[0])
        pitch = self._angle_from_logits(outputs[1])
        row = np.array([pitch, yaw, 0.0], dtype=np.float64)
        return np.stack((row, row))

    def _angle_from_logits(self, logits: np.ndarray) -> float:
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (1, 90):
            raise ValueError(f"L2CS logits must have shape (1, 90), got {values.shape}")
        shifted = values[0] - np.max(values[0])
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        return float(np.sum(probabilities * self._indices) * 4.0 - 180.0) * np.pi / 180.0
