"""Full-screen calibration and validation display."""
from __future__ import annotations

import time
from collections.abc import Callable

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QWidget

from gaze_mouse.calibration.procedure import CalibrationProcedure, Phase, Stimulus
from gaze_mouse.frame import FrameResult

_BACKGROUND, _RED, _WHITE, _AMBER = QColor(40, 40, 40), QColor(255, 60, 60), QColor(255, 255, 255), QColor(255, 200, 0)
_TARGET_OUTER_PX, _TARGET_INNER_PX = 18, 4
_TICK_MS = 16
_PHASE_LABEL = {
    Phase.POINTS: "Kalibrasyon",
    Phase.PURSUIT: "Noktayı gözünle takip et",
    Phase.VALIDATION: "Doğrulama",
}


class CalibrationWindow(QWidget):
    def __init__(self, procedure: CalibrationProcedure, on_done: Callable[[], None], on_cancel: Callable[[], None]):
        super().__init__()
        self._procedure = procedure
        self._on_done = on_done
        self._on_cancel = on_cancel
        self._face_found = True
        self._started = False
        self._result_lines: list[str] | None = None
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(_TICK_MS)

    def on_frame(self, result: FrameResult) -> None:
        if not self._started:
            self._procedure.start(time.perf_counter_ns())
            self._started = True
        self._face_found = result.face_found
        self._procedure.add_frame(result)

    def show_results(self, lines: list[str]) -> None:
        self._result_lines = lines
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _on_tick(self) -> None:
        self._procedure.tick(time.perf_counter_ns())
        self.update()
        if self._procedure.phase == Phase.DONE:
            self._timer.stop()
            self._on_done()

    def keyPressEvent(self, event) -> None:
        if self._result_lines is not None:
            self.close()
        elif event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self._timer.stop()
            self._on_cancel()

    def _draw_text(self, painter: QPainter, text: str, y: int, size: int, color: QColor) -> None:
        painter.setFont(QFont("Segoe UI", size))
        painter.setPen(color)
        painter.drawText(QRect(0, y, self.width(), 2 * size + 20), Qt.AlignmentFlag.AlignHCenter, text)

    def _draw_stimulus(self, painter: QPainter, stimulus: Stimulus) -> None:
        center = self.height() // 2
        if stimulus.phase == Phase.COUNTDOWN:
            self._draw_text(painter, "Noktalara bak, nokta yer değiştirene kadar gözünü ayırma. Başını sabit tut.",
                            center - 90, 20, _WHITE)
            self._draw_text(painter, f"{stimulus.countdown}", center - 20, 40, _WHITE)
            self._draw_text(painter, "Esc: iptal", center + 80, 14, _AMBER)
        elif stimulus.position is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            point = QPoint(*stimulus.position)
            painter.setBrush(_RED)
            painter.drawEllipse(point, _TARGET_OUTER_PX, _TARGET_OUTER_PX)
            painter.setBrush(_WHITE)
            painter.drawEllipse(point, _TARGET_INNER_PX, _TARGET_INNER_PX)
            self._draw_text(painter, _PHASE_LABEL[stimulus.phase], 12, 12, QColor(130, 130, 130))
        if not self._face_found:
            self._draw_text(painter, "YÜZ GÖRÜNMÜYOR", self.height() - 60, 16, _AMBER)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BACKGROUND)
        if self._result_lines is not None:
            y = self.height() // 3
            for line in self._result_lines:
                self._draw_text(painter, line, y, 18, _WHITE)
                y += 50
        else:
            self._draw_stimulus(painter, self._procedure.stimulus(time.perf_counter_ns()))
        painter.end()
        self._procedure.mark_shown(time.perf_counter_ns())
