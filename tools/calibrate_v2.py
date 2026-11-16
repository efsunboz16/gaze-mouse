"""Collect pose-diverse V2 gaze calibration samples.

Requires gaze_model.enabled=true and a compatible ONNX model in the config.
The camera is opened and the resulting model/geometry samples are saved as NPZ.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

from gaze_mouse.calibration.pose_procedure import PoseCalibrationProcedure, PoseVariant
from gaze_mouse.config import load_config, resolve_path
from gaze_mouse.recording import build_targets, grid_positions
from gaze_mouse.ui.calibration_window import CalibrationWindow
from gaze_mouse.ui.pipeline_thread import PipelineThread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--camera", type=int, help="override camera index")
    parser.add_argument("--user", default="efsun")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.get("gaze_model", {}).get("enabled", False):
        parser.error("Enable gaze_model.enabled and provide an ONNX model in the config first")
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    app = QApplication(sys.argv)
    screen = QGuiApplication.primaryScreen().geometry()
    width, height = screen.width(), screen.height()
    cal = cfg["calibration"]
    positions = grid_positions(cal["grid_cols"], cal["grid_rows"], width, height, cal["margin_fraction"])
    points = build_targets(positions, 1, int(np.random.default_rng().integers(2**32)))
    variants = (
        PoseVariant("dik", "Laptopu dik tut. Başını doğal ve ortada tut."),
        PoseVariant("normal", "Laptopu normal kullanım açısına getir."),
        PoseVariant("geriye_yatik", "Laptop ekranını geriye yatır ve bu pozu koru."),
    )
    procedure = PoseCalibrationProcedure(points, variants, cal["seconds_per_point"], cal["countdown_s"])
    pipeline = PipelineThread(cfg)
    started_at = datetime.now()
    exit_code = {"value": 1}

    def finish() -> None:
        pipeline.requestInterruption()
        arrays = procedure.to_arrays()
        directory = resolve_path(cfg["logging"]["profile_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{args.user}_v2_calibration_{started_at:%Y%m%d_%H%M%S}.npz"
        np.savez_compressed(path, **arrays)
        print(f"Saved {path} ({len(arrays['timestamp_ns'])} frames)")
        exit_code["value"] = 0
        window.show_results(["V2 kalibrasyon kaydedildi", str(path), "Çıkmak için bir tuşa bas"])

    def cancel() -> None:
        pipeline.requestInterruption()
        window.close()
        exit_code["value"] = 0

    def fail(message: str) -> None:
        print(f"Pipeline error: {message}")
        window.close()

    window = CalibrationWindow(procedure, finish, cancel)
    pipeline.frame_ready.connect(window.on_frame)
    pipeline.failed.connect(fail)
    pipeline.start()
    window.showFullScreen()
    app.exec()
    pipeline.requestInterruption()
    pipeline.wait()
    return exit_code["value"]


if __name__ == "__main__":
    raise SystemExit(main())
