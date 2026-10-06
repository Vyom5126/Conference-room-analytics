"""
generate_labels.py
──────────────────
Generates a realistic labels.csv for images in data/images/.

Scoring logic:
  - clean_room_*  → high scores (4.0–5.0)
  - medium_room_* → mid scores (2.8–4.0)
  - occupied_room_* → slightly lower (2.5–3.8)
  - messy_room_*  → low scores (1.0–2.8)
  - unknown       → mid scores (2.5–3.5) with small random noise

Each row has: filename, cleanliness_score, chair_alignment_score,
              clutter_score, overall_score  (all on 1–5 scale)

Usage:
    python data/generate_labels.py [--images-dir PATH] [--output PATH]
    python data/generate_labels.py --open-for-edit   # open CSV in default editor
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── Scoring profiles per image category ──────────────────────────────────────
# Tuples: (cleanliness_mean, chair_align_mean, clutter_mean, overall_mean, noise_std)
PROFILES: dict[str, tuple[float, float, float, float, float]] = {
    "clean_room":    (4.6, 4.5, 4.4, 4.5, 0.30),
    "medium_room":   (3.4, 3.2, 3.3, 3.3, 0.40),
    "occupied_room": (2.9, 2.8, 2.7, 2.8, 0.45),
    "messy_room":    (1.8, 1.9, 1.6, 1.8, 0.35),
}
DEFAULT_PROFILE: tuple[float, float, float, float, float] = (3.0, 3.0, 3.0, 3.0, 0.50)


def _profile_for(filename: str) -> tuple[float, float, float, float, float]:
    """Select scoring profile based on filename prefix."""
    stem = Path(filename).stem
    for prefix, profile in PROFILES.items():
        if stem.startswith(prefix):
            return profile
    return DEFAULT_PROFILE


def _jitter(mean: float, std: float, rng: np.random.Generator) -> float:
    """Sample a score, clipped to [1.0, 5.0]."""
    val = rng.normal(loc=mean, scale=std)
    return float(np.clip(round(val * 2) / 2, 1.0, 5.0))  # round to nearest 0.5


def generate_labels(images_dir: Path, output_path: Path, seed: int = 42) -> pd.DataFrame:
    """
    Generate ground-truth labels for all images in images_dir.
    Uses profile-based realistic scores with controlled noise.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        p.name for p in images_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_files:
        log.warning("No images found in %s", images_dir)
        return pd.DataFrame(columns=[
            "filename", "cleanliness_score", "chair_alignment_score",
            "clutter_score", "overall_score",
        ])

    rng = np.random.default_rng(seed)
    rows = []

    for fname in image_files:
        c_mean, a_mean, cl_mean, o_mean, std = _profile_for(fname)

        cleanliness   = _jitter(c_mean, std, rng)
        chair_align   = _jitter(a_mean, std, rng)
        clutter_score = _jitter(cl_mean, std, rng)
        # Overall is a weighted average of dims + independent noise
        overall = float(np.clip(
            0.35 * cleanliness + 0.30 * chair_align + 0.35 * clutter_score
            + rng.normal(0, std * 0.4),
            1.0, 5.0,
        ))
        overall = round(overall * 2) / 2  # round to nearest 0.5

        rows.append({
            "filename":             fname,
            "cleanliness_score":    cleanliness,
            "chair_alignment_score": chair_align,
            "clutter_score":        clutter_score,
            "overall_score":        overall,
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("✓ Labels written → %s  (%d images)", output_path, len(df))

    # Print distribution summary
    print("\n-- Score Distribution -------------------------------------")
    print(df[["cleanliness_score", "chair_alignment_score",
              "clutter_score", "overall_score"]].describe().round(2).to_string())
    print("\n-- Counts by category ------------------------------------")
    df["category"] = df["filename"].apply(lambda f: _profile_for(f)[0])
    cats = df["filename"].str.extract(r"^([a-z_]+?)_\d+")[0].value_counts()
    print(cats.to_string())
    print()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth labels CSV for conference room images."
    )
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data" / "images")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "labels_demo.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--open-for-edit", action="store_true",
                        help="Open the CSV in default editor after generating")
    args = parser.parse_args()

    df = generate_labels(args.images_dir, args.output, seed=args.seed)

    if args.open_for_edit and args.output.exists():
        os.startfile(str(args.output))  # Windows; use 'open' on macOS


if __name__ == "__main__":
    main()
