"""Configuration loading. All paths are resolved relative to the repository root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = ROOT / value
    return cfg
