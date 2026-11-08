"""Phase 1 exit test: select 300 px buttons with gaze and winks.

Run from the repository root after calibrating: python -m tools.target_test --user efsun
Dry-run by default: clicks are scored inside this program and nothing reaches the operating
system. Add --live for real mouse events. Ctrl+Alt+Q quits at any time.
"""
from __future__ import annotations

import argparse
import random
import sys

from gaze_mouse.app import Runtime, load_user_profile, prepare_application, primary_screen_size
from gaze_mouse.config import load_config
from gaze_mouse.evaluation.target_test import TargetTest, build_trials
from gaze_mouse.recording import build_targets, grid_positions
from gaze_mouse.ui.target_window import TargetWindow


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
    settings = cfg["target_test"]

    app = prepare_application(sys.argv)
    screen = primary_screen_size()
    profile = load_user_profile(cfg, args.user, screen)
    print("LIVE: real mouse events are enabled." if args.live else "Dry-run: no mouse events are sent.")

    positions = grid_positions(settings["grid_cols"], settings["grid_rows"], *screen, settings["margin_fraction"])
    repeats = settings["trials"] // len(positions) + 1
    targets = build_targets(positions, repeats, random.randrange(2**32))
    test = TargetTest(build_trials(targets, settings["trials"]), settings["button_px"], settings["trial_timeout_s"])
    exit_code = {"value": 0}

    def done() -> None:
        s = test.summary()
        passed = s["hit_rate"] >= settings["pass_hit_rate"]
        for result in test.results:
            t = result.trial
            print(f"  ({t.target.x:>4}, {t.target.y:>4}) {t.button:<5} {result.outcome:<13} {result.duration_ms:7.0f} ms")
        lines = [
            f"İsabet: {s['hit']} / {s['trials']}  (%{100 * s['hit_rate']:.0f})  |  "
            f"kaçan: {s['miss']}  |  yanlış göz: {s['wrong_button']}  |  zaman aşımı: {s['timeout']}",
            f"İsabetlerde ortalama süre: {s['mean_hit_ms'] / 1000:.1f} sn" if s["hit"] else "İsabet yok",
            f"Çıkış ölçütü %{100 * settings['pass_hit_rate']:.0f}  ->  {'GEÇTİ' if passed else 'KALDI'}",
            "Çıkmak için bir tuşa bas",
        ]
        print("\n".join(lines[:3]))
        window.show_results(lines)

    def cancel() -> None:
        print("Cancelled by user.")
        window.close()

    window = TargetWindow(test, settings["button_px"], settings["countdown_s"], done, cancel)

    def fail(message: str) -> None:
        print(f"Pipeline error: {message}")
        exit_code["value"] = 1
        window.close()

    runtime = Runtime(cfg, profile, screen, args.live, window.on_dry_click)
    runtime.quit_requested.connect(app.quit)
    runtime.failed.connect(fail)
    runtime.start()
    window.showFullScreen()
    app.exec()
    runtime.stop()
    return exit_code["value"]


if __name__ == "__main__":
    raise SystemExit(main())
