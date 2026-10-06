"""Shared utilities used by all pipeline scripts."""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_config(path=None):
    """Load config.yaml (single source of truth for classes, paths and thresholds)."""
    p = Path(path) if path else ROOT / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
