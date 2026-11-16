"""Fit per-eye kappa offsets from a saved V2 calibration NPZ session."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gaze_mouse.gaze.model import fit_kappa_from_screen_samples
from gaze_mouse.gaze.screen import ScreenPlane


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="calibration .npz file")
    parser.add_argument("--screen-size", nargs=2, type=int, required=True, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--screen-mm", nargs=2, type=float, required=True, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--origin-mm", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--output", type=Path, required=True, help="kappa JSON output path")
    args = parser.parse_args()
    with np.load(args.session) as data:
        required = (
            "left_eye_center_camera_mm", "right_eye_center_camera_mm",
            "left_gaze_direction_camera", "right_gaze_direction_camera",
            "target_xy_px",
        )
        missing = [name for name in required if name not in data]
        if missing:
            parser.error(f"session is missing V2 arrays: {', '.join(missing)}")
        centers = np.stack([data[required[0]], data[required[1]]], axis=1)
        directions = np.stack([data[required[2]], data[required[3]]], axis=1)
        targets = data["target_xy_px"]
    valid = np.isfinite(centers).all(axis=(1, 2)) & np.isfinite(directions).all(axis=(1, 2))
    if not valid.any():
        parser.error("session contains no complete V2 gaze samples")
    screen = ScreenPlane(
        np.asarray(args.origin_mm, dtype=np.float64),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        args.screen_mm[0],
        args.screen_mm[1],
    )
    calibration = fit_kappa_from_screen_samples(
        directions[valid], centers[valid], targets[valid], screen, tuple(args.screen_size)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved {args.output} from {int(valid.sum())} samples")
    print(calibration.offsets_rad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
