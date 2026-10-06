"""
generate_labels_from_features.py
─────────────────────────────────
Generates SYNTHETIC proxy labels directly from computed features.

WARNING — these are not human judgments. The labels are a deterministic
function of the same features the models are trained on (plus seeded noise),
so model metrics on them only show that a regressor can recover this formula.
They say nothing about how well the system judges real room quality.

Kept for reference only: output goes to data/labels_synthetic.csv. The labels
the models train on are data/labels.csv, rated per data/rating_rubric.md.

Scoring formula (hand-picked weights, min-max scaled over the current dataset):
  cleanliness   ← weighted combo of clutter_area_frac, floor_occupancy_ratio,
                   table_surface_occupancy (inverse)
  chair_align   ← chair_spacing_variance, chair_aspect_ratio_variance (inverse)
  clutter_score ← clutter_object_count, clutter_area_frac (inverse)
  overall       ← weighted average of above + noise

All scores are on a 1–5 scale. A small Gaussian noise term (σ≈0.35)
simulates inter-rater variability.

Usage:
    python data/generate_labels_from_features.py
"""

from __future__ import annotations

import argparse
import logging
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


def sigmoid_scale(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Map array to [0,1] using min-max (constant input → 0.5)."""
    rng = hi - lo
    if rng == 0:
        return np.full_like(x, 0.5, dtype=float)
    normed = (x - lo) / rng
    return np.clip(normed, 0, 1)


def generate_feature_based_labels(
    features_path: Path,
    output_path: Path,
    seed: int = 42,
    noise_std: float = 0.35,
) -> pd.DataFrame:
    """
    Derive synthetic quality labels from YOLO-computed features.

    The mapping encodes these assumptions:
      - More clutter → lower cleanliness & clutter scores
      - High chair spacing variance → lower chair alignment score
      - High table surface occupancy → lower cleanliness
      - Person present lowers overall by 0.10
      - Overall weights sum to 0.90, so overall rarely exceeds 4.5
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(features_path)

    n = len(df)
    filenames = df["filename"].values

    # ── Helper: invert + scale to [1, 5] ─────────────────────────────────────
    def feat_to_score(arr: np.ndarray, invert: bool = True,
                      weight: float = 1.0) -> np.ndarray:
        """Normalise feature array → [1,5] score, optionally inverted."""
        lo, hi = arr.min(), arr.max()
        normed = sigmoid_scale(arr, lo, hi)
        if invert:
            normed = 1.0 - normed
        return 1.0 + normed * 4.0  # map [0,1] → [1,5]

    # ── Cleanliness Score ─────────────────────────────────────────────────────
    # High clutter density, floor occupancy, and table occupancy → low score
    clutter_dens   = df["clutter_area_frac"].values
    floor_occ      = df["floor_occupancy_ratio"].values
    table_occ      = df["table_surface_occupancy"].values

    cleanliness_raw = (
        0.40 * feat_to_score(clutter_dens, invert=True)
        + 0.30 * feat_to_score(floor_occ, invert=True)
        + 0.30 * feat_to_score(table_occ, invert=True)
    )
    cleanliness = np.clip(
        cleanliness_raw + rng.normal(0, noise_std, n), 1.0, 5.0
    )

    # ── Chair Alignment Score ─────────────────────────────────────────────────
    # High spacing variance / aspect ratio variance → low alignment score
    # If no chairs detected → neutral score (3.0)
    chair_count    = df["chair_count"].values
    spacing_var    = df["chair_spacing_variance"].values
    aspect_var     = df["chair_aspect_ratio_variance"].values
    x_var          = df["chair_x_variance"].values

    chair_align_raw = (
        0.45 * feat_to_score(spacing_var, invert=True)
        + 0.30 * feat_to_score(aspect_var, invert=True)
        + 0.25 * feat_to_score(x_var, invert=True)
    )
    # Rooms with no chairs → neutral 3.0 (can't assess alignment)
    chair_align_raw = np.where(chair_count == 0, 3.0, chair_align_raw)
    chair_align = np.clip(
        chair_align_raw + rng.normal(0, noise_std, n), 1.0, 5.0
    )

    # ── Clutter Score ─────────────────────────────────────────────────────────
    # More clutter objects, higher density → low score
    clutter_count  = df["clutter_object_count"].values

    clutter_raw = (
        0.55 * feat_to_score(clutter_count, invert=True)
        + 0.45 * feat_to_score(clutter_dens, invert=True)
    )
    clutter_score = np.clip(
        clutter_raw + rng.normal(0, noise_std, n), 1.0, 5.0
    )

    # ── Overall Score ─────────────────────────────────────────────────────────
    # Weighted combination; human raters care most about clutter & cleanliness
    overall_raw = (
        0.35 * cleanliness
        + 0.30 * clutter_score
        + 0.25 * chair_align
        + rng.normal(0, noise_std * 0.6, n)
        # Bonus: person_present slightly lowers score (post-meeting mess signal)
        - 0.10 * df["person_present"].values
    )
    overall_score = np.clip(np.round(overall_raw * 2) / 2, 1.0, 5.0)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    result = pd.DataFrame({
        "filename":              filenames,
        "cleanliness_score":    np.round(cleanliness, 1),
        "chair_alignment_score": np.round(chair_align, 1),
        "clutter_score":        np.round(clutter_score, 1),
        "overall_score":        overall_score,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info("Labels written: %s  (%d rows)", output_path, len(result))

    score_cols = ["cleanliness_score", "chair_alignment_score",
                  "clutter_score", "overall_score"]
    print("\n-- Score Distribution (feature-derived) ------------------")
    print(result[score_cols].describe().round(3).to_string())

    # Check correlations match expectations
    merged = df.merge(result, on="filename")
    print("\n-- Key Feature-Label Correlations -----------------------")
    checks = [
        ("clutter_object_count", "clutter_score"),
        ("clutter_area_frac",      "cleanliness_score"),
        ("chair_spacing_variance", "chair_alignment_score"),
        ("table_surface_occupancy", "cleanliness_score"),
        ("floor_occupancy_ratio", "cleanliness_score"),
    ]
    for feat, label in checks:
        if feat in merged.columns:
            r = merged[feat].corr(merged[label])
            direction = "negative (good)" if r < -0.3 else ("positive (good)" if r > 0.3 else "weak")
            print(f"  {feat:<30} vs {label:<25}: r={r:+.3f}  {direction}")
    print()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate feature-derived ground-truth labels."
    )
    parser.add_argument("--features", type=Path,
                        default=ROOT / "features" / "features.csv")
    parser.add_argument("--output",   type=Path,
                        default=ROOT / "data" / "labels_synthetic.csv")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--noise",    type=float, default=0.35,
                        help="Gaussian noise std for inter-rater variability")
    args = parser.parse_args()

    if not args.features.exists():
        log.error("features.csv not found: %s", args.features)
        log.error("Run 'python src/features.py' first.")
        raise SystemExit(1)

    generate_feature_based_labels(args.features, args.output,
                                  seed=args.seed, noise_std=args.noise)


if __name__ == "__main__":
    main()
