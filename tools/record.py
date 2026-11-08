"""Record a gaze session: full-screen targets plus per-frame face features.

Run from the repository root: python -m tools.record --save
Without --save nothing is written. Raw video is never written; only landmarks,
blendshapes, EAR, pose, target position and timestamps are stored under data/.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from gaze_mouse.capture.camera import Camera
from gaze_mouse.config import load_config, resolve_path
from gaze_mouse.eyelid.features import EyeLandmarkMap, compute_ears, to_pixels
from gaze_mouse.gaze.screen import enable_dpi_awareness, primary_screen_size
from gaze_mouse.perception.landmarks import FaceLandmarkTracker
from gaze_mouse.recording import (
    SessionRecorder,
    Target,
    TargetSequence,
    build_targets,
    grid_positions,
    save_session,
)

WINDOW = "gaze-mouse record"
FONT = cv2.FONT_HERSHEY_SIMPLEX
BACKGROUND, WHITE, RED, AMBER = (40, 40, 40), (255, 255, 255), (60, 60, 255), (0, 200, 255)
TARGET_OUTER_PX, TARGET_INNER_PX = 18, 4


def _centered_text(canvas: np.ndarray, text: str, y: int, color=WHITE, scale: float = 1.0) -> None:
    (w, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.putText(canvas, text, ((canvas.shape[1] - w) // 2, y), FONT, scale, color, 2, cv2.LINE_AA)


def draw_scene(size: tuple[int, int], target: Target | None, countdown: int, face_found: bool) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), BACKGROUND, np.uint8)
    if target is None:
        _centered_text(canvas, "Look at each dot until it moves. Keep your head still.", height // 2 - 60)
        _centered_text(canvas, f"Starting in {countdown}", height // 2, WHITE, 1.4)
        _centered_text(canvas, "q / Esc cancels", height // 2 + 60, AMBER, 0.7)
    else:
        cv2.circle(canvas, (target.x, target.y), TARGET_OUTER_PX, RED, -1, cv2.LINE_AA)
        cv2.circle(canvas, (target.x, target.y), TARGET_INNER_PX, WHITE, -1, cv2.LINE_AA)
    if not face_found:
        _centered_text(canvas, "NO FACE", height - 40, AMBER, 0.8)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config path (default: config/default.yaml)")
    parser.add_argument("--camera", type=int, help="override camera index")
    parser.add_argument("--save", action="store_true", help="write the session under the data directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["logging"]["record_raw_video"]:
        parser.error("logging.record_raw_video is not implemented; features are recorded instead")
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    rec = cfg["recording"]

    enable_dpi_awareness()
    screen = primary_screen_size()
    seed = rec["shuffle_seed"] if rec["shuffle_seed"] is not None else random.randrange(2**32)
    positions = grid_positions(rec["grid_cols"], rec["grid_rows"], *screen, rec["margin_fraction"])
    targets = build_targets(positions, rec["repeats"], seed)
    sequence = TargetSequence(targets, rec["seconds_per_point"])
    recorder = SessionRecorder()
    eye_map = EyeLandmarkMap.from_config(cfg["landmarks"])

    started_at = datetime.now()
    last_index, dropped, processed = -1, 0, 0
    window_rect = None
    info = None
    complete = False

    try:
        with FaceLandmarkTracker(cfg["landmarker"]) as tracker, Camera(cfg["camera"]) as camera:
            info = camera.start()
            print(info)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            countdown_end_ns = time.perf_counter_ns() + int(rec["countdown_s"] * 1e9)
            drawn_scene = None
            timeout_s = cfg["preview"]["frame_timeout_s"]

            while True:
                frame = camera.get_latest(last_index, timeout_s)
                if frame is None:
                    raise RuntimeError(camera.error or f"No frame within {timeout_s} s")
                if last_index >= 0:
                    dropped += frame.index - last_index - 1
                last_index = frame.index

                result = tracker.process(frame)
                processed += 1
                if result.face_found:
                    result.points_px = to_pixels(result.landmarks, result.width, result.height)
                    result.ear_left, result.ear_right = compute_ears(result.points_px, eye_map)
                    result.points_px = None  # derivable from landmarks; not worth storing

                now_ns = time.perf_counter_ns()
                if sequence.onset_ns is not None:
                    recorder.add(result, sequence.index, sequence.current, sequence.onset_ns)
                if sequence.advance_if_due(now_ns) and sequence.finished:
                    complete = True
                    break

                counting = now_ns < countdown_end_ns
                target = None if counting else sequence.current
                countdown = math.ceil((countdown_end_ns - now_ns) / 1e9) if counting else 0
                scene = (sequence.index, target is None, countdown, result.face_found)
                if scene != drawn_scene:
                    cv2.imshow(WINDOW, draw_scene(screen, target, countdown, result.face_found))
                    drawn_scene = scene
                key = cv2.waitKey(1) & 0xFF
                if window_rect is None:
                    window_rect = list(cv2.getWindowImageRect(WINDOW))
                if target is not None:
                    sequence.mark_shown(time.perf_counter_ns())
                if key in (ord("q"), 27):
                    print("Cancelled by user.")
                    break
    finally:
        cv2.destroyAllWindows()
        arrays, names = recorder.to_arrays()
        frames = len(recorder)
        faces = int(arrays["face_found"].sum())
        if frames:
            span_s = (arrays["timestamp_ns"][-1] - arrays["timestamp_ns"][0]) / 1e9
            print(f"Recorded {frames} frames over {span_s:.1f} s ({frames / max(span_s, 1e-9):.1f} fps), "
                  f"face in {faces} ({100 * faces / frames:.1f}%), dropped {dropped}, complete={complete}")
            print(f"Frames per target position: {np.bincount(arrays['target_position_id']).tolist()}")
            print(f"Screen {screen}, fullscreen window rect {window_rect}")
        if not args.save:
            print("Dry run: nothing written. Pass --save to store the session.")
        elif frames:
            directory = resolve_path(cfg["logging"]["data_dir"]) / started_at.strftime("session_%Y%m%d_%H%M%S")
            save_session(directory, arrays, {
                "schema_version": 1,
                "started_at": started_at.isoformat(timespec="seconds"),
                "complete": complete,
                "frame_count": frames,
                "face_found_count": faces,
                "dropped_frames": dropped,
                "processed_frames": processed,
                "screen_px": list(screen),
                "window_rect": window_rect,
                "coordinates": "target_xy_px: physical pixels, primary display, origin top-left; "
                               "landmarks: normalized, unmirrored camera image",
                "shuffle_seed": seed,
                "targets": [dataclasses.asdict(t) for t in targets],
                "blendshape_names": names,
                "camera": dataclasses.asdict(info) if info else None,
                "config": {k: cfg[k] for k in ("camera", "landmarker", "landmarks", "recording")},
                "versions": {"mediapipe": mp.__version__, "opencv": cv2.__version__},
            })
            print(f"Saved to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
