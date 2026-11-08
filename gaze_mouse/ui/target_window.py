"""Full-screen display for the large-button exit test."""
from __future__ import annotations

import math
import time
from collections.abc import Callable

from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QWidget

from gaze_mouse.cursor.controller import Click
from gaze_mouse.evaluation.target_test import TargetTest

_BACKGROUND, _WHITE, _LEFT, _RIGHT, _GRAY = (
    QColor(40, 40, 40), QColor(255, 255, 255), QColor(0, 120, 215), QColor(200, 50, 150), QColor(150, 150, 150),
)
_TICK_MS = 33
_BUTTON_NAMES = {Qt.MouseButton.LeftButton: "left", Qt.MouseButton.RightButton: "right"}
_LABELS = {"left": "SOL GÖZ", "right": "SAĞ GÖZ"}


class TargetWindow(QWidget):
    def __init__(
        self,
        test: TargetTest,
        button_px: int,
        countdown_s: float,
        on_done: Callable[[], None],
        on_cancel: Callable[[], None],
    ):
        super().__init__()
        self._test = test
        self._button_px = button_px
        self._on_done = on_done
        self._on_cancel = on_cancel
        self._countdown_end_ns = time.perf_counter_ns() + int(countdown_s * 1e9)
        self._result_lines: list[str] | None = None
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(_TICK_MS)

    def show_results(self, lines: list[str]) -> None:
        self._result_lines = lines
        self.update()

    def on_dry_click(self, click: Click) -> None:
        self._test.on_click(click.button, click.x, click.y, time.perf_counter_ns())

    def _on_tick(self) -> None:
        now = time.perf_counter_ns()
        if now >= self._countdown_end_ns:
            self._test.start(now)
        self._test.tick(now)
        self.update()
        if self._test.finished:
            self._timer.stop()
            self._on_done()

    def mousePressEvent(self, event) -> None:
        button = _BUTTON_NAMES.get(event.button())
        if button is not None and self._result_lines is None:
            point = event.globalPosition().toPoint()
            self._test.on_click(button, point.x(), point.y(), time.perf_counter_ns())

    def keyPressEvent(self, event) -> None:
        if self._result_lines is not None:
            self.close()
        elif event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self._timer.stop()
            self._on_cancel()

    def _centered(self, painter: QPainter, text: str, y: int, size: int, color: QColor) -> None:
        painter.setFont(QFont("Segoe UI", size))
        painter.setPen(color)
        painter.drawText(QRect(0, y, self.width(), 2 * size + 20), Qt.AlignmentFlag.AlignHCenter, text)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        if self._result_lines is not None:
            y = self.height() // 3
            for line in self._result_lines:
                self._centered(painter, line, y, 18, _WHITE)
                y += 50
        elif not self._test.started:
            remaining = max(math.ceil((self._countdown_end_ns - time.perf_counter_ns()) / 1e9), 1)
            middle = self.height() // 2
            self._centered(painter, "Düğmeye bak, üzerinde yazan gözle kısa bir wink yap.", middle - 90, 20, _WHITE)
            self._centered(painter, str(remaining), middle - 20, 40, _WHITE)
            self._centered(painter, "Esc: iptal", middle + 80, 14, _GRAY)
        elif self._test.current is not None:
            trial = self._test.current
            half = self._button_px // 2
            rect = QRect(trial.target.x - half, trial.target.y - half, self._button_px, self._button_px)
            painter.fillRect(rect, _LEFT if trial.button == "left" else _RIGHT)
            painter.setFont(QFont("Segoe UI", 26))
            painter.setPen(_WHITE)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, _LABELS[trial.button])
            number, total = self._test.position
            self._centered(painter, f"{number} / {total}", self.height() - 50, 14, _GRAY)
        painter.end()
