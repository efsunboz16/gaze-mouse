"""Calibrate the geometric gaze model and validate it on unseen targets.

Run from the repository root: python -m tools.calibrate --user efsun
Writes profiles/<user>.yaml and a features-only sample file; no images are stored.
Moves no cursor and sends no clicks.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

from gaze_mouse.calibration.analysis import ear_baselines, evaluate, fit_variant
from gaze_mouse.calibration.procedure import CalibrationProcedure, PursuitPath
from gaze_mouse.calibration.profile import save_profile
from gaze_mouse.config import load_config, resolve_path
from gaze_mouse.recording import Target, build_targets, grid_positions
from gaze_mouse.ui.calibration_window import CalibrationWindow
from gaze_mouse.ui.pipeline_thread import PipelineThread


def build_procedure(cfg: dict[str, Any], width: int, height: int, seed: int) -> CalibrationProcedure:
    cal = cfg["calibration"]
    positions = grid_positions(cal["grid_cols"], cal["grid_rows"], width, height, cal["margin_fraction"])
    validation = [
        Target(i, int(round(fx * width)), int(round(fy * height)))
        for i, (fx, fy) in enumerate(cal["validation_positions"])
    ]
    pursuit = PursuitPath(width, height, cal["margin_fraction"], cal["smooth_pursuit_seconds"],
                          tuple(cal["smooth_pursuit_cycles"]))
    return CalibrationProcedure(build_targets(positions, 1, seed), validation, cal["seconds_per_point"],
                                pursuit, cal["countdown_s"])


def summarize(cfg: dict[str, Any], samples: dict[str, np.ndarray], mm_per_px: float):
    """Fit the configured model, print all variants, return (model, metrics, result lines)."""
    cal = cfg["calibration"]
    ratio = cfg["fixation"]["blink_mask"]["openness_ratio"]
    min_rate = cal["validation_min_hit_rate"]

    print(f"\n{'features':<24}{'regressor':>10}{'pursuit':>9}{'px':>8}{'|dx|':>8}{'|dy|':>8}{'deg':>7}{'hit%':>8}")
    for groups, regressor, pursuit in itertools.product(cal["compare_features"], ("ridge", "poly2"), (False, True)):
        m = evaluate(fit_variant(samples, cal, ratio, groups, regressor, pursuit), samples, cal, ratio, mm_per_px)
        print(f"{'+'.join(groups):<24}{regressor:>10}{str(pursuit):>9}{m.mean_px:>8.1f}"
              f"{m.mean_abs_dx_px:>8.1f}{m.mean_abs_dy_px:>8.1f}{m.mean_deg:>7.2f}{100 * m.hit_rate:>8.1f}")

    model = fit_variant(samples, cal, ratio, cal["features"], cal["regressor"], cal["use_pursuit"])
    metrics = evaluate(model, samples, cal, ratio, mm_per_px)
    verdict = "GEÇTİ" if metrics.passes(min_rate) else "KALDI"
    lines = [
        f"Ortalama hata: {metrics.mean_px:.0f} px ({metrics.mean_deg:.2f}°)  |  p95: {metrics.p95_px:.0f} px",
        f"Yatay: {metrics.mean_abs_dx_px:.0f} px  |  Dikey: {metrics.mean_abs_dy_px:.0f} px",
        f"{cal['validation_hit_box_px']} px kutu isabeti: %{100 * metrics.hit_rate:.1f}  "
        f"(kapı: %{100 * min_rate:.0f})  ->  {verdict}",
        "Çıkmak için bir tuşa bas",
    ]
    return model, metrics, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default="efsun", help="profile name")
    parser.add_argument("--config", help="YAML config path (default: config/default.yaml)")
    parser.add_argument("--camera", type=int, help="override camera index")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"  # work in physical pixels
    app = QApplication(sys.argv)
    screen = QGuiApplication.primaryScreen().geometry()
    width, height = screen.width(), screen.height()
    print(f"Screen {width}x{height} physical px")
    mm_per_px = cfg["screen"]["width_mm"] / width

    started_at = datetime.now()
    seed = int(np.random.default_rng().integers(2**32))
    procedure = build_procedure(cfg, width, height, seed)
    pipeline = PipelineThread(cfg)
    exit_code = {"value": 1}
    camera_info: dict[str, Any] = {}

    def finish() -> None:
        pipeline.requestInterruption()
        samples = procedure.to_arrays()
        try:
            model, metrics, lines = summarize(cfg, samples, mm_per_px)
        except ValueError as exc:
            print(f"Calibration failed: {exc}")
            window.show_results([f"Kalibrasyon başarısız: {exc}", "Çıkmak için bir tuşa bas"])
            return
        left, right = ear_baselines(samples)
        directory = resolve_path(cfg["logging"]["profile_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        stamp = started_at.strftime("%Y%m%d_%H%M%S")
        np.savez_compressed(directory / f"{args.user}_calibration_{stamp}.npz", **samples)
        save_profile(directory / f"{args.user}.yaml", {
            "schema_version": 1,
            "created_at": started_at.isoformat(timespec="seconds"),
            "screen_px": [width, height],
            "ear_baseline": {"left": left, "right": right},
            "gaze_model": model.to_dict(),
            "validation": {**metrics.to_dict(), "passed": metrics.passes(cfg["calibration"]["validation_min_hit_rate"])},
            "camera": camera_info,
        })
        print("\n".join(lines[:3]))
        print(f"Profile: {directory / (args.user + '.yaml')}")
        exit_code["value"] = 0
        window.show_results(lines)

    def cancel() -> None:
        print("Cancelled by user; nothing saved.")
        exit_code["value"] = 0
        window.close()

    def fail(message: str) -> None:
        print(f"Pipeline error: {message}")
        window.close()

    def camera_started(info) -> None:
        print(info)
        camera_info.update(exposure_lock=info.exposure_lock, focus_lock=info.focus_lock,
                           brightness_before_lock=info.brightness_before_lock,
                           brightness_after_lock=info.brightness_after_lock)

    window = CalibrationWindow(procedure, finish, cancel)
    pipeline.camera_started.connect(camera_started)
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
