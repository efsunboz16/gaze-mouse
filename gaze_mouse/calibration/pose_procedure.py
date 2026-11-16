"""Pose-diverse calibration procedure for the V2 gaze model."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gaze_mouse.calibration.procedure import Phase, Stimulus
from gaze_mouse.frame import FrameResult
from gaze_mouse.recording import Target, TargetSequence


PosePhase = Phase


@dataclass(frozen=True)
class PoseVariant:
    name: str
    instruction: str


class PoseCalibrationProcedure:
    """Collect the same target grid across explicitly instructed user poses."""

    def __init__(
        self,
        points: list[Target],
        variants: tuple[PoseVariant, ...],
        seconds_per_point: float,
        countdown_s: float,
    ):
        if not variants:
            raise ValueError("At least one pose variant is required")
        self._points = points
        self._variants = variants
        self._seconds_per_point = seconds_per_point
        self._sequence = TargetSequence(points, seconds_per_point)
        self._countdown_ns = int(countdown_s * 1e9)
        self._variant_index = 0
        self._phase = PosePhase.COUNTDOWN
        self._countdown_end_ns: int | None = None
        self._rows: list[tuple[int, int, tuple[int, int], float, int, FrameResult]] = []

    @property
    def phase(self) -> PosePhase:
        return self._phase

    @property
    def pose_variant(self) -> PoseVariant:
        return self._variants[self._variant_index]

    def start(self, now_ns: int) -> None:
        if self._countdown_end_ns is None:
            self._countdown_end_ns = now_ns + self._countdown_ns

    def tick(self, now_ns: int) -> None:
        if self._phase == PosePhase.COUNTDOWN:
            if self._countdown_end_ns is not None and now_ns >= self._countdown_end_ns:
                self._phase = PosePhase.POINTS
        elif self._phase == PosePhase.POINTS:
            if self._sequence.advance_if_due(now_ns) and self._sequence.finished:
                if self._variant_index + 1 == len(self._variants):
                    self._phase = PosePhase.DONE
                else:
                    self._variant_index += 1
                    self._sequence = TargetSequence(self._points, self._sequence_duration_s)
                    self._countdown_end_ns = now_ns + self._countdown_ns
                    self._phase = PosePhase.COUNTDOWN

    @property
    def _sequence_duration_s(self) -> float:
        return self._seconds_per_point

    def stimulus(self, now_ns: int) -> Stimulus:
        if self._phase == PosePhase.COUNTDOWN:
            end = self._countdown_end_ns if self._countdown_end_ns is not None else now_ns
            return Stimulus(
                self._phase,
                countdown=max(math.ceil((end - now_ns) / 1e9), 1),
            )
        if self._phase == PosePhase.POINTS and self._sequence.current is not None:
            target = self._sequence.current
            return Stimulus(self._phase, (target.x, target.y))
        return Stimulus(self._phase)

    def mark_shown(self, now_ns: int) -> None:
        if self._phase == PosePhase.POINTS:
            self._sequence.mark_shown(now_ns)

    def add_frame(self, result: FrameResult) -> None:
        if self._phase != PosePhase.POINTS or self._sequence.onset_ns is None:
            return
        target = self._sequence.current
        if target is None:
            return
        self._rows.append((
            self._variant_index,
            self._sequence.index,
            (target.x, target.y),
            (result.timestamp_ns - self._sequence.onset_ns) / 1e6,
            result.timestamp_ns,
            result,
        ))

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "pose_index": np.array([row[0] for row in self._rows], dtype=np.int32),
            "seq": np.array([row[1] for row in self._rows], dtype=np.int32),
            "target_xy_px": np.array([row[2] for row in self._rows], dtype=np.int32).reshape(-1, 2),
            "ms_since_onset": np.array([row[3] for row in self._rows], dtype=np.float32),
            "timestamp_ns": np.array([row[4] for row in self._rows], dtype=np.int64),
            "left_eye_center_camera_mm": np.array([
                np.full(3, np.nan) if row[5].left_eye_center_camera_mm is None
                else row[5].left_eye_center_camera_mm for row in self._rows
            ], dtype=np.float32).reshape(-1, 3),
            "right_eye_center_camera_mm": np.array([
                np.full(3, np.nan) if row[5].right_eye_center_camera_mm is None
                else row[5].right_eye_center_camera_mm for row in self._rows
            ], dtype=np.float32).reshape(-1, 3),
            "left_gaze_direction_camera": np.array([
                np.full(3, np.nan) if row[5].left_gaze_direction_camera is None
                else row[5].left_gaze_direction_camera for row in self._rows
            ], dtype=np.float32).reshape(-1, 3),
            "right_gaze_direction_camera": np.array([
                np.full(3, np.nan) if row[5].right_gaze_direction_camera is None
                else row[5].right_gaze_direction_camera for row in self._rows
            ], dtype=np.float32).reshape(-1, 3),
            "gaze_uncertainty": np.array([
                np.nan if row[5].gaze_uncertainty is None else row[5].gaze_uncertainty
                for row in self._rows
            ], dtype=np.float32),
        }
