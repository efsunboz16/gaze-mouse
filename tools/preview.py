"""Live preview of camera, landmarks, eyelid signals, fps and latency.

Run from the repository root: python -m tools.preview
Frames are never written to disk and no mouse events are sent.

Latency numbers start when cap.read() returns, so sensor and driver delay are not included.
  inference          detect_for_video only
  capture->result    read() return until landmarks and EAR are ready
  capture->display   read() return until imshow + waitKey returned (shown one frame late)
"""
from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np

from gaze_mouse.capture.camera import Camera
from gaze_mouse.config import load_config
from gaze_mouse.eyelid.features import EyeLandmarkMap, compute_ears, eye_blink_scores, to_pixels
from gaze_mouse.frame import FrameResult
from gaze_mouse.perception.landmarks import FaceLandmarkTracker
from gaze_mouse.profiling import RateMeter, StageProfiler

WINDOW = "gaze-mouse preview"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, CYAN, YELLOW, RED, GRAY = (
    (255, 255, 255), (80, 220, 80), (255, 220, 0), (0, 220, 255), (60, 60, 255), (120, 120, 120),
)
PANEL_W, PANEL_H = 500, 190
BAR_W, BAR_H = 200, 14


def _text(canvas: np.ndarray, text: str, x: int, y: int, color=WHITE, scale: float = 0.5) -> None:
    cv2.putText(canvas, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _bar(canvas: np.ndarray, x: int, y: int, value: float | None, color) -> None:
    cv2.rectangle(canvas, (x, y), (x + BAR_W, y + BAR_H), GRAY, 1)
    if value is not None:
        filled = int(np.clip(value, 0.0, 1.0) * BAR_W)
        cv2.rectangle(canvas, (x, y), (x + filled, y + BAR_H), color, -1)


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "--" if value is None or math.isnan(value) else format(value, spec)


def draw_face(canvas: np.ndarray, pts: np.ndarray, eye_map: EyeLandmarkMap, show_all: bool) -> None:
    """Draw on a canvas whose orientation already matches `pts`."""
    if show_all:
        for x, y in pts.astype(int):
            cv2.circle(canvas, (int(x), int(y)), 1, GREEN, -1, cv2.LINE_AA)
    for indices, color in ((eye_map.ear_left, CYAN), (eye_map.ear_right, YELLOW)):
        eye = pts[list(indices)].astype(np.int32)
        cv2.polylines(canvas, [eye], True, color, 1, cv2.LINE_AA)
        for x, y in eye:
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    for iris, label, color in ((eye_map.iris_left, "L", CYAN), (eye_map.iris_right, "R", YELLOW)):
        x, y = pts[iris].astype(int)
        cv2.drawMarker(canvas, (int(x), int(y)), RED, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        _text(canvas, label, int(x) - 6, int(y) - 26, color, 0.7)


def draw_panel(canvas: np.ndarray, result: FrameResult, stats: dict[str, float], dropped: int) -> None:
    canvas[:PANEL_H, :PANEL_W] = (canvas[:PANEL_H, :PANEL_W] * 0.3).astype(np.uint8)
    _text(canvas, f"camera {stats['camera_fps']:5.1f} fps   processing {stats['proc_fps']:5.1f} fps   "
                  f"dropped {dropped}", 10, 22)
    _text(canvas, f"inference {_fmt(stats['inference'], '.1f')} ms   "
                  f"capture->result {_fmt(stats['capture_to_result'], '.1f')} ms", 10, 46)
    _text(canvas, f"capture->display {_fmt(stats['capture_to_display'], '.1f')} ms   "
                  f"(medians, excl. sensor delay)", 10, 70)
    _text(canvas, f"EAR  L {_fmt(result.ear_left)}", 10, 98, CYAN)
    _text(canvas, f"R {_fmt(result.ear_right)}", 130, 98, YELLOW)
    left, right = eye_blink_scores(result.blendshapes)
    _text(canvas, f"eyeBlinkLeft  {_fmt(left, '.2f')}", 10, 124, CYAN)
    _bar(canvas, 190, 113, left, CYAN)
    _text(canvas, f"eyeBlinkRight {_fmt(right, '.2f')}", 10, 148, YELLOW)
    _bar(canvas, 190, 137, right, YELLOW)
    _text(canvas, "q/Esc quit   m mirror   l landmarks", 10, 176, GRAY)
    if not result.face_found:
        _text(canvas, "NO FACE", 400, 98, RED, 0.6)


def render(frame_image: np.ndarray, result: FrameResult, eye_map: EyeLandmarkMap,
           stats: dict[str, float], dropped: int, mirror: bool, show_all: bool) -> np.ndarray:
    canvas = cv2.flip(frame_image, 1) if mirror else frame_image.copy()
    if result.points_px is not None:
        pts = result.points_px.copy()
        if mirror:
            pts[:, 0] = result.width - 1 - pts[:, 0]
        draw_face(canvas, pts, eye_map, show_all)
    draw_panel(canvas, result, stats, dropped)
    return canvas


def print_summary(info, session: StageProfiler, processed: int, faces: int, captured: int,
                  dropped: int, elapsed_s: float) -> None:
    print(f"\nCamera: {info.width}x{info.height} backend={info.backend} nominal_fps={info.nominal_fps:.1f}")
    print(f"Exposure lock: {info.exposure_lock} (brightness {info.brightness_before_lock:.1f} -> "
          f"{info.brightness_after_lock:.1f}); focus lock: {info.focus_lock}")
    if elapsed_s > 0:
        print(f"Duration {elapsed_s:.1f} s | camera {captured / elapsed_s:.1f} fps | "
              f"processing {processed / elapsed_s:.1f} fps | dropped {dropped}")
    if processed:
        print(f"Face found in {faces}/{processed} processed frames ({100 * faces / processed:.1f}%)")
    print(f"{'stage':<20}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}{'n':>8}")
    for name, s in session.summary().items():
        print(f"{name:<20}{s['p50']:>10.1f}{s['p95']:>10.1f}{s['mean']:>10.1f}{s['n']:>8}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config path (default: config/default.yaml)")
    parser.add_argument("--camera", type=int, help="override camera index")
    parser.add_argument("--duration", type=float, help="exit after N seconds")
    parser.add_argument("--no-window", action="store_true", help="measure without a preview window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    preview = cfg["preview"]
    eye_map = EyeLandmarkMap.from_config(cfg["landmarks"])
    live = StageProfiler(preview["live_stats_window"])
    session = StageProfiler(preview["session_stats_window"])
    camera_rate = RateMeter(preview["fps_window_s"])
    proc_rate = RateMeter(preview["fps_window_s"])
    mirror, show_all = preview["mirror"], True
    show = not args.no_window

    processed = faces = dropped = 0
    last_index = -1
    first_ns = last_ns = None
    first_camera_index = 0
    info = None

    def record(stage: str, ms: float) -> None:
        live.add(stage, ms)
        session.add(stage, ms)

    with FaceLandmarkTracker(cfg["landmarker"]) as tracker, Camera(cfg["camera"]) as camera:
        info = camera.start()
        print(info)
        print("L/R labels follow MediaPipe's constants: wink each eye to confirm them.")
        if show:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:
            while True:
                frame = camera.get_latest(last_index, preview["frame_timeout_s"])
                if frame is None:
                    if camera.error:
                        raise RuntimeError(camera.error)
                    raise RuntimeError(f"No frame within {preview['frame_timeout_s']} s")
                if last_index >= 0:
                    dropped += frame.index - last_index - 1
                last_index = frame.index
                if first_ns is None:
                    first_ns, first_camera_index = frame.timestamp_ns, frame.index

                result = tracker.process(frame)
                if result.face_found:
                    faces += 1
                    result.points_px = to_pixels(result.landmarks, result.width, result.height)
                    result.ear_left, result.ear_right = compute_ears(result.points_px, eye_map)

                ready_ns = time.perf_counter_ns()
                processed += 1
                last_ns = frame.timestamp_ns
                record("inference", result.timings_ms["inference"])
                record("capture_to_result", (ready_ns - frame.timestamp_ns) / 1e6)
                camera_rate.update(frame.timestamp_ns, frame.index)
                proc_rate.update(ready_ns, processed)

                if show:
                    stats = {
                        "camera_fps": camera_rate.rate(),
                        "proc_fps": proc_rate.rate(),
                        "inference": live.percentile("inference", 50),
                        "capture_to_result": live.percentile("capture_to_result", 50),
                        "capture_to_display": live.percentile("capture_to_display", 50),
                    }
                    cv2.imshow(WINDOW, render(frame.image, result, eye_map, stats, dropped, mirror, show_all))
                    key = cv2.waitKey(1) & 0xFF
                    record("capture_to_display", (time.perf_counter_ns() - frame.timestamp_ns) / 1e6)
                    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord("m"):
                        mirror = not mirror
                    elif key == ord("l"):
                        show_all = not show_all

                if args.duration and (frame.timestamp_ns - first_ns) / 1e9 >= args.duration:
                    break
        finally:
            if show:
                cv2.destroyAllWindows()

    elapsed_s = (last_ns - first_ns) / 1e9 if first_ns is not None else 0.0
    print_summary(info, session, processed, faces, last_index - first_camera_index + 1, dropped, elapsed_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
