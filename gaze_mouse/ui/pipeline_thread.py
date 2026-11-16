"""Background Qt thread: camera, landmarks and per-frame features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from gaze_mouse.capture.camera import Camera
from gaze_mouse.config import resolve_path
from gaze_mouse.gaze.features import GazeFeatureExtractor
from gaze_mouse.gaze.model import (
    GazeModelAdapter,
    KappaCalibration,
    L2CSOnnxGazeBackend,
    OnnxGazeBackend,
    apply_prediction,
    angles_to_direction,
    l2cs_angles_to_direction,
)
from gaze_mouse.gaze.normalize import normalize_face
from gaze_mouse.gaze.screen import ScreenPlane, project_frame_gaze
from gaze_mouse.perception.camera_calibration import CameraIntrinsics
from gaze_mouse.perception.landmarks import FaceLandmarkTracker


class PipelineThread(QThread):
    camera_started = pyqtSignal(object)  # CameraInfo
    frame_ready = pyqtSignal(object)  # FrameResult
    failed = pyqtSignal(str)

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self._cfg = cfg

    def run(self) -> None:
        cfg = self._cfg
        intrinsics_cfg = cfg["camera"].get("intrinsics")
        intrinsics = None
        intrinsics_path = cfg["camera"].get("intrinsics_path")
        if intrinsics_path is not None:
            intrinsics = CameraIntrinsics.from_json(resolve_path(intrinsics_path))
        elif intrinsics_cfg is not None:
            intrinsics = CameraIntrinsics(
                tuple(intrinsics_cfg["focal_length_px"]),
                tuple(intrinsics_cfg["principal_point_px"]),
                intrinsics_cfg.get("distortion"),
            )
        model_cfg = cfg.get("gaze_model", {})
        model = None
        personal_calibration = None
        calibration_path = model_cfg.get("personal_calibration_path")
        if calibration_path is not None:
            personal_calibration = KappaCalibration.from_json(resolve_path(calibration_path))
        try:
            if model_cfg.get("enabled", False):
                model_path = model_cfg.get("path")
                if not model_path:
                    raise ValueError("gaze_model.path is required when gaze_model.enabled is true")
                resolved_model_path = resolve_path(model_path)
                if not Path(resolved_model_path).is_file():
                    raise FileNotFoundError(
                        f"gaze model file not found: {resolved_model_path}"
                    )
                backend_name = model_cfg.get("backend", "onnx")
                backend = (
                    L2CSOnnxGazeBackend(
                        resolved_model_path, input_name=model_cfg.get("input_name")
                    )
                    if backend_name == "l2cs"
                    else OnnxGazeBackend(
                        resolved_model_path,
                        input_name=model_cfg.get("input_name"),
                        color_order=model_cfg.get("color_order", "RGB"),
                        mean=tuple(model_cfg.get("mean", [0.5, 0.5, 0.5])),
                        std=tuple(model_cfg.get("std", [0.5, 0.5, 0.5])),
                    )
                )
                converter = (
                    l2cs_angles_to_direction
                    if backend_name == "l2cs"
                    else None
                )
                model = GazeModelAdapter(backend, converter or angles_to_direction)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        extractor = GazeFeatureExtractor(
            cfg["landmarks"], cfg["landmarker"]["iris_diameter_mm"], intrinsics
        )
        screen_cfg = cfg["screen"]
        camera_offset = np.asarray(screen_cfg.get("camera_offset_mm", [0.0, 0.0]), dtype=float)
        screen_plane = ScreenPlane(
            np.array([-screen_cfg["width_mm"] / 2 - camera_offset[0], -camera_offset[1], 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            screen_cfg["width_mm"],
            screen_cfg["height_mm"],
        )
        timeout_s = cfg["preview"]["frame_timeout_s"]
        try:
            # MediaPipe stays on this thread for its whole lifetime.
            with FaceLandmarkTracker(cfg["landmarker"]) as tracker, Camera(cfg["camera"]) as camera:
                self.camera_started.emit(camera.start())
                last_index = -1
                while not self.isInterruptionRequested():
                    frame = camera.get_latest(last_index, timeout_s)
                    if frame is None:
                        raise RuntimeError(camera.error or f"No frame within {timeout_s} s")
                    last_index = frame.index
                    result = tracker.process(frame)
                    extractor.annotate(result)
                    if model is not None and result.face_found and result.points_px is not None:
                        if backend_name == "l2cs":
                            points = result.points_px
                            x_min, y_min = np.min(points, axis=0)
                            x_max, y_max = np.max(points, axis=0)
                            margin = 0.2 * max(x_max - x_min, y_max - y_min)
                            left = max(0, int(np.floor(x_min - margin)))
                            top = max(0, int(np.floor(y_min - margin)))
                            right = min(result.width, int(np.ceil(x_max + margin)))
                            bottom = min(result.height, int(np.ceil(y_max + margin)))
                            model_image = frame.image[top:bottom, left:right]
                        else:
                            normalized = normalize_face(
                                frame.image,
                                result.points_px,
                                tuple(cfg["landmarks"]["ear_left"]),
                                tuple(cfg["landmarks"]["ear_right"]),
                            )
                            model_image = normalized.image
                        prediction = model.predict(model_image)
                        if personal_calibration is not None:
                            prediction = personal_calibration.apply(prediction)
                        apply_prediction(result, prediction)
                    if result.left_gaze_direction_camera is not None:
                        project_frame_gaze(result, screen_plane, (result.width, result.height))
                    self.frame_ready.emit(result)
        except Exception as exc:  # reported to the GUI thread, which owns shutdown
            self.failed.emit(f"{type(exc).__name__}: {exc}")
