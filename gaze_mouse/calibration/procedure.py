"""M6: calibration procedure as a time-driven state machine. No camera or UI here."""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from gaze_mouse.frame import FrameResult
from gaze_mouse.gaze.features import FEATURE_DIM
from gaze_mouse.recording import Target, TargetSequence


class Phase(enum.IntEnum):
    COUNTDOWN = 0
    POINTS = 1
    PURSUIT = 2
    VALIDATION = 3
    DONE = 4


@dataclass(frozen=True)
class PursuitPath:
    width: int
    height: int
    margin: float
    duration_s: float
    cycles: tuple[float, float]

    def at(self, t_s: float) -> tuple[int, int]:
        """Lissajous position; starts at the screen center."""
        ax, ay = self.width * (0.5 - self.margin), self.height * (0.5 - self.margin)
        x = self.width / 2 + ax * math.sin(2 * math.pi * self.cycles[0] * t_s / self.duration_s)
        y = self.height / 2 + ay * math.sin(2 * math.pi * self.cycles[1] * t_s / self.duration_s)
        return int(round(x)), int(round(y))


@dataclass(frozen=True)
class Stimulus:
    phase: Phase
    position: tuple[int, int] | None = None
    countdown: int = 0


class CalibrationProcedure:
    def __init__(
        self,
        points: list[Target],
        validation: list[Target],
        seconds_per_point: float,
        pursuit: PursuitPath,
        countdown_s: float,
    ):
        self._sequences = {
            Phase.POINTS: TargetSequence(points, seconds_per_point),
            Phase.VALIDATION: TargetSequence(validation, seconds_per_point),
        }
        self._pursuit = pursuit
        self._countdown_ns = int(countdown_s * 1e9)
        self._phase = Phase.COUNTDOWN
        self._countdown_end_ns: int | None = None
        self._pursuit_onset_ns: int | None = None
        self._rows: list[tuple] = []

    @property
    def phase(self) -> Phase:
        return self._phase

    def start(self, now_ns: int) -> None:
        if self._countdown_end_ns is None:
            self._countdown_end_ns = now_ns + self._countdown_ns

    def tick(self, now_ns: int) -> None:
        if self._phase == Phase.COUNTDOWN:
            if self._countdown_end_ns is not None and now_ns >= self._countdown_end_ns:
                self._phase = Phase.POINTS
        elif self._phase == Phase.POINTS:
            sequence = self._sequences[Phase.POINTS]
            if sequence.advance_if_due(now_ns) and sequence.finished:
                self._phase = Phase.PURSUIT
        elif self._phase == Phase.PURSUIT:
            onset = self._pursuit_onset_ns
            if onset is not None and (now_ns - onset) / 1e9 >= self._pursuit.duration_s:
                self._phase = Phase.VALIDATION
        elif self._phase == Phase.VALIDATION:
            sequence = self._sequences[Phase.VALIDATION]
            if sequence.advance_if_due(now_ns) and sequence.finished:
                self._phase = Phase.DONE

    def stimulus(self, now_ns: int) -> Stimulus:
        if self._phase == Phase.COUNTDOWN:
            end = self._countdown_end_ns if self._countdown_end_ns is not None else now_ns + self._countdown_ns
            return Stimulus(self._phase, countdown=max(math.ceil((end - now_ns) / 1e9), 1))
        if self._phase in self._sequences:
            target = self._sequences[self._phase].current
            return Stimulus(self._phase, (target.x, target.y) if target else None)
        if self._phase == Phase.PURSUIT:
            onset = self._pursuit_onset_ns
            return Stimulus(self._phase, self._pursuit.at(0.0 if onset is None else (now_ns - onset) / 1e9))
        return Stimulus(self._phase)

    def mark_shown(self, now_ns: int) -> None:
        if self._phase in self._sequences:
            self._sequences[self._phase].mark_shown(now_ns)
        elif self._phase == Phase.PURSUIT and self._pursuit_onset_ns is None:
            self._pursuit_onset_ns = now_ns

    def add_frame(self, result: FrameResult) -> None:
        ts = result.timestamp_ns
        if self._phase in self._sequences:
            sequence = self._sequences[self._phase]
            if sequence.onset_ns is None:
                return
            xy = (sequence.current.x, sequence.current.y)
            seq, onset = sequence.index, sequence.onset_ns
        elif self._phase == Phase.PURSUIT and self._pursuit_onset_ns is not None:
            onset = self._pursuit_onset_ns
            xy, seq = self._pursuit.at((ts - onset) / 1e9), 0
        else:
            return
        self._rows.append((int(self._phase), seq, xy, (ts - onset) / 1e6, ts, result))

    def to_arrays(self) -> dict[str, np.ndarray]:
        rows = self._rows
        blank = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
        return {
            "phase": np.array([r[0] for r in rows], dtype=np.int32),
            "seq": np.array([r[1] for r in rows], dtype=np.int32),
            "target_xy_px": np.array([r[2] for r in rows], dtype=np.int32).reshape(-1, 2),
            "ms_since_onset": np.array([r[3] for r in rows], dtype=np.float32),
            "timestamp_ns": np.array([r[4] for r in rows], dtype=np.int64),
            "features": np.array(
                [blank if r[5].gaze_features is None else r[5].gaze_features for r in rows], dtype=np.float32
            ).reshape(-1, FEATURE_DIM),
            "ear_left": np.array([np.nan if r[5].ear_left is None else r[5].ear_left for r in rows], np.float32),
            "ear_right": np.array([np.nan if r[5].ear_right is None else r[5].ear_right for r in rows], np.float32),
            "distance_mm": np.array(
                [np.nan if r[5].distance_mm is None else r[5].distance_mm for r in rows], np.float32
            ),
        }
