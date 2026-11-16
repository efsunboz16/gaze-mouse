"""Collect chessboard views and calibrate the camera intrinsics.

Run from the repository root:
    python -m tools.calibrate_camera --board-cols 9 --board-rows 6 --square-mm 25
Press q or Esc to cancel. Move the board between accepted views.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from gaze_mouse.capture.camera import Camera
from gaze_mouse.config import load_config, resolve_path
from gaze_mouse.perception.camera_calibration import (
    calibrate_camera,
    detect_chessboard_corners,
)

WINDOW = "gaze-mouse camera calibration"


def _draw_status(image, message: str, views: int, required: int) -> None:
    cv2.putText(image, f"{message}  views: {views}/{required}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Move board between captures; q/Esc cancels", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def collect_calibration(
    cfg: dict,
    board_size: tuple[int, int],
    required_views: int,
    interval_s: float,
) -> tuple[list, tuple[int, int]]:
    views = []
    last_index = -1
    last_capture = 0.0
    image_size = None
    with Camera(cfg["camera"]) as camera:
        info = camera.start()
        image_size = (info.width, info.height)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:
            while len(views) < required_views:
                frame = camera.get_latest(last_index, cfg["preview"]["frame_timeout_s"])
                if frame is None:
                    raise RuntimeError(camera.error or "Camera did not provide a frame")
                last_index = frame.index
                image = frame.image.copy()
                corners = detect_chessboard_corners(image, board_size)
                now = time.monotonic()
                if corners is not None:
                    cv2.drawChessboardCorners(image, board_size, corners, True)
                    if now - last_capture >= interval_s:
                        views.append(corners)
                        last_capture = now
                        status = "Accepted"
                    else:
                        status = "Board detected"
                else:
                    status = "Show full board"
                _draw_status(image, status, len(views), required_views)
                cv2.imshow(WINDOW, image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    raise KeyboardInterrupt
        finally:
            cv2.destroyWindow(WINDOW)
    if image_size is None:
        raise RuntimeError("No camera image was received")
    return views, image_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--camera", type=int, help="override camera index")
    parser.add_argument("--board-cols", type=int, default=9, help="inner corners horizontally")
    parser.add_argument("--board-rows", type=int, default=6, help="inner corners vertically")
    parser.add_argument("--square-mm", type=float, required=True, help="one square side in millimetres")
    parser.add_argument("--samples", type=int, default=15, help="number of distinct views")
    parser.add_argument("--interval", type=float, default=1.0, help="minimum seconds between captures")
    parser.add_argument("--output", default="config/camera_intrinsics.json", help="output JSON path")
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    try:
        image_points, image_size = collect_calibration(
            cfg, (args.board_cols, args.board_rows), args.samples, args.interval
        )
    except KeyboardInterrupt:
        print("Calibration cancelled.")
        return 1
    result = calibrate_camera(
        image_points, image_size, (args.board_cols, args.board_rows), args.square_mm
    )
    output = Path(args.output)
    output = output if output.is_absolute() else resolve_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved {output}")
    print(f"Views: {result.views_used}; reprojection error: {result.reprojection_error_px:.3f} px")
    print(f"focal length px: {result.intrinsics.focal_length_px}")
    print(f"principal point px: {result.intrinsics.principal_point_px}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
