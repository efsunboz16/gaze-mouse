"""Main application: gaze cursor and wink clicks.

Run from the repository root: python -m gaze_mouse.app --user efsun
Dry-run by default: nothing is sent to the operating system. --live enables real mouse events.
Ctrl+Alt+Q quits at any time; parking the physical mouse in a screen corner pauses control.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication
from pynput import keyboard

from gaze_mouse.calibration.model import GazeModel
from gaze_mouse.calibration.profile import load_profile
from gaze_mouse.config import load_config, resolve_path
from gaze_mouse.cursor.controller import Click, PointerController
from gaze_mouse.frame import FrameResult
from gaze_mouse.os_io.mouse import DryRunMouse, MouseDriver, PynputMouse
from gaze_mouse.ui.overlay import Overlay
from gaze_mouse.ui.pipeline_thread import PipelineThread


def load_user_profile(cfg: dict[str, Any], user: str, screen: tuple[int, int]) -> dict[str, Any]:
    path = resolve_path(cfg["logging"]["profile_dir"]) / f"{user}.yaml"
    if not path.exists():
        raise SystemExit(f"No profile at {path}. Run: python -m tools.calibrate --user {user}")
    profile = load_profile(path)
    if tuple(profile["screen_px"]) != screen:
        raise SystemExit(f"Profile was calibrated for {profile['screen_px']}, screen is {list(screen)}. Recalibrate.")
    print(f"Profile {path.name}, calibrated {profile['created_at']}; "
          f"recalibrate after you change how you sit (a calibration does not carry over).")
    if not profile["validation"]["passed"]:
        print("WARNING: this calibration did not pass validation.")
    return profile


def primary_screen_size() -> tuple[int, int]:
    geometry = QGuiApplication.primaryScreen().geometry()
    return geometry.width(), geometry.height()


class Runtime(QObject):
    quit_requested = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        cfg: dict[str, Any],
        profile: dict[str, Any],
        screen: tuple[int, int],
        live: bool,
        on_dry_click: Callable[[Click], None] | None = None,
    ):
        super().__init__()
        self._live = live
        self._on_dry_click = on_dry_click
        baseline = profile["ear_baseline"]
        model = GazeModel.from_dict(profile["gaze_model"])
        self._controller = PointerController(model, cfg, (baseline["left"], baseline["right"]), screen)
        safety = cfg["safety"]
        backend = PynputMouse() if live else DryRunMouse()
        self._driver = MouseDriver(backend, screen, safety["corner_margin_px"], safety["corner_pause"])
        self._overlay = Overlay(*screen, "CANLI" if live else "DRY-RUN")
        self._pipeline = PipelineThread(cfg)
        self._pipeline.frame_ready.connect(self._on_frame)
        self._pipeline.failed.connect(self.failed)
        self._hotkey = keyboard.GlobalHotKeys({safety["kill_hotkey"]: self.quit_requested.emit})

    def start(self) -> None:
        self._hotkey.start()
        self._overlay.show()
        self._pipeline.start()

    def stop(self) -> None:
        self._hotkey.stop()
        self._pipeline.requestInterruption()
        self._pipeline.wait()
        self._overlay.close()

    def _on_frame(self, result: FrameResult) -> None:
        output = self._controller.update(result)
        paused = self._driver.apply(output)
        self._overlay.set_state(output.cursor, output.status, paused, output.click)
        if output.click is None or paused:
            return
        click = output.click
        print(f"[{'live' if self._live else 'dry-run'}] {click.button} click at ({click.x}, {click.y})")
        if not self._live and self._on_dry_click is not None:
            self._on_dry_click(click)


def prepare_application(argv: list[str]) -> QApplication:
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"  # work in physical pixels
    app = QApplication(argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    keep_python_responsive = QTimer(app)
    keep_python_responsive.timeout.connect(lambda: None)  # lets Python see Ctrl+C
    keep_python_responsive.start(200)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default="efsun", help="profile name")
    parser.add_argument("--live", action="store_true", help="send real mouse events")
    parser.add_argument("--config", help="YAML config path (default: config/default.yaml)")
    parser.add_argument("--camera", type=int, help="override camera index")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera

    app = prepare_application(sys.argv)
    screen = primary_screen_size()
    profile = load_user_profile(cfg, args.user, screen)
    print("LIVE: real mouse events are enabled." if args.live else "Dry-run: no mouse events are sent.")
    print(f"Quit with {cfg['safety']['kill_hotkey']} or Ctrl+C in this terminal.")

    app.setQuitOnLastWindowClosed(False)
    exit_code = {"value": 0}

    def fail(message: str) -> None:
        print(f"Pipeline error: {message}")
        exit_code["value"] = 1
        app.quit()

    runtime = Runtime(cfg, profile, screen, args.live)
    runtime.quit_requested.connect(app.quit)
    runtime.failed.connect(fail)
    runtime.start()
    app.exec()
    runtime.stop()
    return exit_code["value"]


if __name__ == "__main__":
    raise SystemExit(main())
