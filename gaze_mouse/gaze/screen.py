"""M5 (partial): primary display size in physical pixels on Windows."""
from __future__ import annotations

import ctypes
import sys

_PROCESS_PER_MONITOR_DPI_AWARE = 2
_SM_CXSCREEN, _SM_CYSCREEN = 0, 1


def enable_dpi_awareness() -> None:
    """Call before creating any window so coordinates are physical pixels."""
    if sys.platform == "win32":
        ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE)


def primary_screen_size() -> tuple[int, int]:
    if sys.platform != "win32":
        raise NotImplementedError("Screen metrics are only implemented for Windows")
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(_SM_CXSCREEN), user32.GetSystemMetrics(_SM_CYSCREEN)
