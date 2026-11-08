"""Load project parameters from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
