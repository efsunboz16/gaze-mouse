"""M11: mouse output. Real events come only from PynputMouse, which only --live selects."""
from __future__ import annotations

from typing import Protocol

from gaze_mouse.cursor.controller import PointerOutput


class MouseBackend(Protocol):
    def position(self) -> tuple[int, int] | None: ...
    def move(self, x: int, y: int) -> None: ...
    def click(self, button: str, x: int, y: int) -> None: ...


class DryRunMouse:
    """Sends nothing to the operating system."""

    def position(self) -> tuple[int, int] | None:
        return None

    def move(self, x: int, y: int) -> None:
        pass

    def click(self, button: str, x: int, y: int) -> None:
        pass


class PynputMouse:
    def __init__(self) -> None:
        from pynput.mouse import Button, Controller

        self._mouse = Controller()
        self._buttons = {"left": Button.left, "right": Button.right}

    def position(self) -> tuple[int, int] | None:
        x, y = self._mouse.position
        return int(x), int(y)

    def move(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def click(self, button: str, x: int, y: int) -> None:
        self._mouse.position = (x, y)
        self._mouse.click(self._buttons[button])


def in_corner(position: tuple[int, int] | None, screen: tuple[int, int], margin: int) -> bool:
    if position is None:
        return False
    x, y = position
    near_x = x <= margin or x >= screen[0] - 1 - margin
    near_y = y <= margin or y >= screen[1] - 1 - margin
    return near_x and near_y


class MouseDriver:
    """Applies pointer output unless the physical mouse is parked in a screen corner."""

    def __init__(self, backend: MouseBackend, screen: tuple[int, int], margin: int, corner_pause: bool):
        self._backend = backend
        self._screen = screen
        self._margin = margin
        self._corner_pause = corner_pause

    def apply(self, output: PointerOutput) -> bool:
        """Return True when control is paused and nothing was sent."""
        if self._corner_pause and in_corner(self._backend.position(), self._screen, self._margin):
            return True
        if output.cursor is not None:
            self._backend.move(*output.cursor)
        if output.click is not None:
            click = output.click
            self._backend.click(click.button, click.x, click.y)
        return False
