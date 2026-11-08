"""Session recording: target schedule and per-frame feature storage. No camera or UI here."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gaze_mouse.frame import FrameResult


@dataclass(frozen=True)
class Target:
    position_id: int  # grid position; shared by repeats of the same point
    x: int  # physical pixels, primary display, origin top-left
    y: int


def grid_positions(cols: int, rows: int, width: int, height: int, margin: float) -> list[tuple[int, int]]:
    def axis(count: int, size: int) -> list[int]:
        if count == 1:
            return [size // 2]
        return [int(round(v)) for v in np.linspace(margin * size, (1 - margin) * size, count)]

    return [(x, y) for y in axis(rows, height) for x in axis(cols, width)]


def build_targets(positions: list[tuple[int, int]], repeats: int, seed: int) -> list[Target]:
    rng = random.Random(seed)
    targets: list[Target] = []
    for _ in range(repeats):
        order = list(range(len(positions)))
        rng.shuffle(order)
        targets += [Target(i, *positions[i]) for i in order]
    return targets


class TargetSequence:
    """Advances through targets; each one's timer starts when it is actually on screen."""

    def __init__(self, targets: list[Target], seconds_per_point: float):
        self._targets = targets
        self._duration_ns = int(seconds_per_point * 1e9)
        self._index = 0
        self._onset_ns: int | None = None

    @property
    def index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        return self._index >= len(self._targets)

    @property
    def current(self) -> Target | None:
        return None if self.finished else self._targets[self._index]

    @property
    def onset_ns(self) -> int | None:
        return self._onset_ns

    def mark_shown(self, now_ns: int) -> None:
        if self._onset_ns is None and not self.finished:
            self._onset_ns = now_ns

    def advance_if_due(self, now_ns: int) -> bool:
        if self._onset_ns is None or now_ns - self._onset_ns < self._duration_ns:
            return False
        self._index += 1
        self._onset_ns = None
        return True


def _stack(rows: list[np.ndarray | None]) -> np.ndarray:
    """Stack optional arrays as float32, filling missing rows with NaN."""
    shape = next((r.shape for r in rows if r is not None), (0,))
    blank = np.full(shape, np.nan, dtype=np.float32)
    return np.stack([blank if r is None else r for r in rows]).astype(np.float32)


class SessionRecorder:
    def __init__(self) -> None:
        self._results: list[FrameResult] = []
        self._targets: list[tuple[int, Target, int]] = []

    def __len__(self) -> int:
        return len(self._results)

    def add(self, result: FrameResult, target_seq: int, target: Target, onset_ns: int) -> None:
        self._results.append(result)
        self._targets.append((target_seq, target, onset_ns))

    def to_arrays(self) -> tuple[dict[str, np.ndarray], list[str]]:
        results, targets = self._results, self._targets
        names = sorted(next((r.blendshapes for r in results if r.blendshapes), {}))
        blendshapes: list[np.ndarray | None] = [
            np.array([r.blendshapes.get(n, np.nan) for n in names], dtype=np.float32)
            if r.blendshapes else None
            for r in results
        ]
        arrays = {
            "timestamp_ns": np.array([r.timestamp_ns for r in results], dtype=np.int64),
            "frame_index": np.array([r.frame_index for r in results], dtype=np.int64),
            "face_found": np.array([r.face_found for r in results], dtype=bool),
            "landmarks": _stack([r.landmarks for r in results]),
            "blendshapes": _stack(blendshapes) if names else np.empty((len(results), 0), np.float32),
            "face_transform": _stack([r.face_transform for r in results]),
            "ear_left": np.array([np.nan if r.ear_left is None else r.ear_left for r in results], np.float32),
            "ear_right": np.array([np.nan if r.ear_right is None else r.ear_right for r in results], np.float32),
            "target_seq": np.array([t[0] for t in targets], dtype=np.int32),
            "target_position_id": np.array([t[1].position_id for t in targets], dtype=np.int32),
            "target_xy_px": np.array([(t[1].x, t[1].y) for t in targets], dtype=np.int32).reshape(-1, 2),
            # Negative when the frame was captured just before the target appeared.
            "ms_since_target_onset": np.array(
                [(r.timestamp_ns - t[2]) / 1e6 for r, t in zip(results, targets)], dtype=np.float32
            ),
        }
        return arrays, names


def save_session(directory: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Write a new session directory; refuses to overwrite an existing one."""
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "frames.npz", **arrays)
    (directory / "session.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
