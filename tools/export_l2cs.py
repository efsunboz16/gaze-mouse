"""Export an L2CS-Net Gaze360 checkpoint to the ONNX format."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="L2CSNet_gaze360.pkl path")
    parser.add_argument("output", type=Path, help="destination .onnx path")
    args = parser.parse_args()

    try:
        import torch
        from l2cs.utils import getArch
    except ImportError as exc:
        parser.error(
            "Install the L2CS-Net package and PyTorch before exporting "
            "(pip install torch torchvision)"
        )
        raise AssertionError from exc

    model = getArch("ResNet50", 90)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 448, 448),
        args.output,
        input_names=["image"],
        output_names=["yaw_logits", "pitch_logits"],
        opset_version=18,
    )
    print(f"Exported {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
