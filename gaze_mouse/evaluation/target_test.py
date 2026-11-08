"""Phase 1 exit test: select large buttons with gaze and winks. Pure logic."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from gaze_mouse.recording import Target

HIT, WRONG_BUTTON, MISS, TIMEOUT = "hit", "wrong_button", "miss", "timeout"


@dataclass(frozen=True)
class Trial:
    target: Target
    button: str


@dataclass(frozen=True)
class TrialResult:
    trial: Trial
    outcome: str
    duration_ms: float


def build_trials(targets: list[Target], count: int) -> list[Trial]:
    """Alternate left and right buttons so both winks are exercised."""
    return [Trial(target, "left" if i % 2 == 0 else "right") for i, target in enumerate(targets[:count])]


class TargetTest:
    def __init__(self, trials: list[Trial], button_px: int, timeout_s: float):
        self._trials = trials
        self._half = button_px / 2
        self._timeout_ns = int(timeout_s * 1e9)
        self._index = 0
        self._trial_start_ns: int | None = None
        self.results: list[TrialResult] = []

    @property
    def current(self) -> Trial | None:
        return self._trials[self._index] if self._index < len(self._trials) else None

    @property
    def position(self) -> tuple[int, int]:
        """Current trial number and total."""
        return self._index + 1, len(self._trials)

    @property
    def started(self) -> bool:
        return self._trial_start_ns is not None

    @property
    def finished(self) -> bool:
        return self.current is None

    def start(self, now_ns: int) -> None:
        if self._trial_start_ns is None:
            self._trial_start_ns = now_ns

    def _finish_trial(self, outcome: str, now_ns: int) -> None:
        self.results.append(TrialResult(self.current, outcome, (now_ns - self._trial_start_ns) / 1e6))
        self._index += 1
        self._trial_start_ns = now_ns

    def on_click(self, button: str, x: int, y: int, now_ns: int) -> str | None:
        """Score a click against the current button; a click before the start is ignored."""
        trial = self.current
        if trial is None or self._trial_start_ns is None:
            return None
        inside = abs(x - trial.target.x) <= self._half and abs(y - trial.target.y) <= self._half
        outcome = MISS if not inside else HIT if button == trial.button else WRONG_BUTTON
        self._finish_trial(outcome, now_ns)
        return outcome

    def tick(self, now_ns: int) -> None:
        if self.current is not None and self._trial_start_ns is not None:
            if now_ns - self._trial_start_ns >= self._timeout_ns:
                self._finish_trial(TIMEOUT, now_ns)

    def summary(self) -> dict[str, float]:
        counts = Counter(r.outcome for r in self.results)
        hits = [r.duration_ms for r in self.results if r.outcome == HIT]
        total = len(self.results)
        return {
            "trials": total,
            HIT: counts[HIT],
            WRONG_BUTTON: counts[WRONG_BUTTON],
            MISS: counts[MISS],
            TIMEOUT: counts[TIMEOUT],
            "hit_rate": counts[HIT] / total if total else 0.0,
            "mean_hit_ms": sum(hits) / len(hits) if hits else float("nan"),
        }
