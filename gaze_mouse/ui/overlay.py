"""M11: transparent, click-through top window showing gaze and status."""
from __future__ import annotations

import time

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from gaze_mouse.cursor.controller import Click

_GAZE, _FROZEN, _LEFT, _RIGHT, _TEXT = (
    QColor(0, 220, 120), QColor(255, 200, 0), QColor(0, 160, 255), QColor(255, 80, 200), QColor(255, 255, 255),
)
_RADIUS_PX = 16
_CLICK_MS = 400
_STATUS_LABEL = {"tracking": "izleniyor", "frozen": "göz kapanıyor", "no_face": "YÜZ YOK"}


class Overlay(QWidget):
    def __init__(self, width: int, height: int, mode_label: str):
        super().__init__()
        self._mode = mode_label
        self._cursor: tuple[int, int] | None = None
        self._status = "no_face"
        self._paused = False
        self._click: Click | None = None
        self._click_ns = 0
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(0, 0, width, height)
        self._animation = QTimer(self)
        self._animation.timeout.connect(self.update)
        self._animation.start(33)

    def set_state(self, cursor: tuple[int, int] | None, status: str, paused: bool, click: Click | None) -> None:
        if cursor is not None:
            self._cursor = cursor
        self._status, self._paused = status, paused
        if click is not None and not paused:
            self._click, self._click_ns = click, time.perf_counter_ns()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._cursor is not None:
            painter.setPen(QPen(_FROZEN if self._status != "tracking" else _GAZE, 3))
            painter.drawEllipse(QPoint(*self._cursor), _RADIUS_PX, _RADIUS_PX)
        if self._click is not None:
            age_ms = (time.perf_counter_ns() - self._click_ns) / 1e6
            if age_ms < _CLICK_MS:
                painter.setPen(QPen(_LEFT if self._click.button == "left" else _RIGHT, 4))
                radius = int(_RADIUS_PX + age_ms / 8)
                painter.drawEllipse(QPoint(self._click.x, self._click.y), radius, radius)
        label = f"{self._mode}  |  {'DURAKLATILDI' if self._paused else _STATUS_LABEL[self._status]}  |  Ctrl+Alt+Q: çıkış"
        painter.setFont(QFont("Segoe UI", 11))
        painter.setPen(_TEXT)
        painter.fillRect(8, 8, 8 * len(label) + 20, 30, QColor(0, 0, 0, 170))
        painter.drawText(18, 29, label)
        painter.end()
