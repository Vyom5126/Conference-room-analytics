"""Tests for score.load_data, evaluate helpers and human_check (no YOLO or CLIP needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import observations, stars  # noqa: E402
from human_check import agreement, make_sample, report  # noqa: E402
from score import load_data  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path):
    names = [f"img_{i}.jpg" for i in range(4)]
    pd.DataFrame({"filename": names, "overall_score": [1, 2, 3, 4], "valid_room": [1, 1, 1, 0]}) \
        .to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({"filename": names, "chair_count": [1.0, 2.0, 3.0, 4.0]}).to_csv(tmp_path / "features.csv", index=False)
    np.savez(tmp_path / "emb.npz", filenames=np.array(names[::-1]), embeddings=np.arange(8, dtype=np.float32).reshape(4, 2))
    return tmp_path


def test_load_data_drops_invalid_rooms_and_aligns_rows(tiny_dataset):
    labels, X = load_data(tiny_dataset / "features.csv", tiny_dataset / "emb.npz", tiny_dataset / "labels.csv")
    assert list(labels["filename"]) == ["img_0.jpg", "img_1.jpg", "img_2.jpg"]
    assert list(X["detection"]["chair_count"]) == [1.0, 2.0, 3.0]
    # embeddings were saved in reverse order; rows must still line up with labels
    assert X["clip"].iloc[0].tolist() == [6.0, 7.0]
    assert X["hybrid"].shape == (3, 3)


def test_stars_rounds_half_up():
    assert stars(4.5) == "★★★★★"
    assert stars(4.49) == "★★★★☆"


def test_observations_flag_low_dimensions_only():
    notes = observations({"clutter_score": 2.0, "cleanliness_score": 4.5, "chair_alignment_score": 4.0},
                         {"chair_count": 6, "person_present": 0, "clutter_object_count": 0})
    assert notes == ["⚠ Low table clear of clutter (2.0)"]


def test_agreement_perfect_and_offset():
    a = agreement([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert a["spearman"] == pytest.approx(1.0) and a["mae"] == 0 and a["within_0.5"] == 1.0
    b = agreement([1, 2, 3, 4, 5], [2, 3, 4, 5, 5])
    assert b["spearman"] > 0.9 and b["within_1"] == 1.0 and b["within_0.5"] == pytest.approx(0.2)


def test_make_sample_skips_invalid_and_covers_low_scores():
    labels = pd.DataFrame({"filename": [f"f{i}" for i in range(100)],
                           "overall_score": [2.0] * 5 + [4.5] * 95,
                           "valid_room": [1] * 99 + [0]})
    s = make_sample(labels, n=20)
    assert len(s) == 20 and s.is_unique and "f99" not in set(s)
    assert set(s) >= {f"f{i}" for i in range(5)}   # all 5 rare low scores are kept (random would pick ~1)


def test_report_compares_human_with_ai_and_model():
    labels = pd.DataFrame({"filename": ["a", "b", "c"], "overall_score": [2.0, 3.0, 5.0]})
    human  = pd.DataFrame({"filename": ["a", "b", "c"], "overall_score": [2.5, 3.0, 4.5]})
    oof    = pd.DataFrame({"filename": ["a", "b", "c"], "clip_ridge": [2.0, 3.5, 4.0]})
    r = report(labels, human, oof, "clip_ridge")
    assert len(r) == 2 and r["spearman"].tolist() == pytest.approx([1.0, 1.0])
