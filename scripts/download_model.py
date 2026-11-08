"""Download the MediaPipe Face Landmarker model bundle into models/."""
from pathlib import Path
import urllib.request

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already present: {DEST}")
        return
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
