"""Shared utilities used by all pipeline scripts."""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_config(path=None):
    """Load config.yaml. Returns empty dict if file not found."""
    p = path or ROOT / "config.yaml"
    return yaml.safe_load(open(p, encoding="utf-8")) if p.exists() else {}
