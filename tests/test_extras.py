"""Tests for score ranges, the out-of-domain check, heatmap maths and table crops (no model downloads)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from embed import table_box  # noqa: E402
from evaluate import resolve_models  # noqa: E402
from explain import patch_boxes, patch_contributions, view_geometry  # noqa: E402
from ood import fit_reference, knn_similarity  # noqa: E402
from intervals import apply_interval, fit_interval, heldout_coverage  # noqa: E402
from score import select_model  # noqa: E402


def test_interval_is_asymmetric_and_split_by_prediction():
    rng = np.random.default_rng(0)
    pred = np.r_[np.full(100, 3.0), np.full(100, 4.5)]
    y = pred + np.r_[rng.normal(-0.5, 0.6, 100), rng.normal(0, 0.2, 100)]   # low predictions: biased, wider
    iv = fit_interval(y, pred, level=0.8, split_at=4.0)
    lo_b, hi_b = iv["below"]; lo_a, hi_a = iv["above"]
    assert hi_b - lo_b > hi_a - lo_a                       # wider below the split
    assert lo_b < -abs(hi_b)                               # skewed downwards (model over-predicts there)
    assert apply_interval(4.9, iv)[1] == 5.0               # clipped to the scale
    cov, _ = heldout_coverage(y, pred, 0.8, 4.0, reps=50)
    assert cov == pytest.approx(0.8, abs=0.06)


def test_interval_falls_back_to_global_when_a_side_is_small():
    pred = np.r_[np.full(5, 3.0), np.full(100, 4.5)]
    iv = fit_interval(pred + 0.1, pred, split_at=4.0)
    assert iv["split_at"] is None and iv["below"] == iv["above"]


def test_select_model_prefers_simplest_within_tolerance():
    import pandas as pd
    r = pd.DataFrame({"target": ["t"] * 4, "model": ["dummy_mean", "embed_ridge", "table_ridge", "hybrid_ridge"],
                      "MAE_mean": [0.6, 0.324, 0.321, 0.320]})
    assert select_model(r, "t", 0.005) == "embed_ridge"
    assert select_model(r, "t", 0.001) == "table_ridge"
    assert select_model(r, "t", 0.0) == "hybrid_ridge"


def test_knn_similarity_and_reference_threshold():
    rng = np.random.default_rng(0)
    u = np.ones(8) / np.sqrt(8)                                              # training photos cluster around u
    ref = u + 0.3 * rng.normal(size=(50, 8)); ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert knn_similarity(ref[0], ref, k=1)[0] == pytest.approx(1.0)          # itself is the nearest
    assert knn_similarity(ref, ref, k=1, exclude_self=True).max() < 1.0      # leave-one-out drops it
    fit = fit_reference(ref, percentile=10)
    loo = knn_similarity(ref, ref, exclude_self=True)
    assert np.mean(loo < fit["threshold"]) == pytest.approx(0.1, abs=0.03)
    assert knn_similarity(-u, ref)[0] < fit["threshold"]                    # a photo far from the cluster is flagged


def test_view_geometry_pad_flip_and_tiles():
    geo = view_geometry(["pad", "flip", "tiles"], 400, 300)
    assert len(geo) == 6
    assert geo[0] == ((0, 0, 400, 300), (400, 400), (0, 50), False)
    assert geo[1][3] is True
    assert geo[2][0] == (0, 0, 200, 150) and geo[2][1] == (200, 200) and geo[2][2] == (0, 25)


def test_patch_boxes_cover_centre_crop_and_mirror():
    # 400x400 canvas (pad view of a 400x300 photo), resize shortest edge to 256, crop 224, patch 14 → 16x16 grid
    pad, flip = view_geometry(["pad", "flip"], 400, 300)
    boxes = patch_boxes(pad, 256, 224, 14)
    assert len(boxes) == 256
    l, t = boxes[0][:2]; r, b = boxes[-1][2:]
    margin = (256 - 224) / 2 * 400 / 256                                   # crop margin in canvas pixels
    assert l == pytest.approx(margin) and r == pytest.approx(400 - margin)
    assert t == pytest.approx(margin - 50) and b == pytest.approx(400 - margin - 50)
    fb = patch_boxes(flip, 256, 224, 14)
    assert fb[0][2] == pytest.approx(400 - boxes[0][0])                    # first flipped patch mirrors the first pad patch


def test_patch_contributions_sum_to_patch_term():
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(3, 1 + 9, 4))                                # 3 views, 9 patches, d = 4
    w = rng.normal(size=4)
    contrib, x = patch_contributions(hidden, w)
    assert np.linalg.norm(x) == pytest.approx(1.0, abs=1e-6)
    assert contrib.sum() == pytest.approx(w @ x[4:], rel=1e-5)            # exact decomposition of the patch half


def test_table_box_picks_largest_confident_table():
    dets = [{"class": "dining table", "confidence": 0.9, "x_center": 0.5, "y_center": 0.5, "width": 0.4, "height": 0.2},
            {"class": "dining table", "confidence": 0.2, "x_center": 0.5, "y_center": 0.5, "width": 0.9, "height": 0.9},
            {"class": "chair", "confidence": 0.9, "x_center": 0.1, "y_center": 0.1, "width": 0.1, "height": 0.1}]
    assert table_box(dets, 1000, 1000) == pytest.approx((280, 390, 720, 610), abs=1)   # 0.4 x 0.2 box enlarged by 10%
    assert table_box(dets[2:], 1000, 1000) is None


def test_resolve_models_auto_and_fixed(tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"overall_score": "embed_ridge", "clutter_score": "table_ridge"}))
    assert resolve_models(tmp_path, "auto", ["overall_score", "clutter_score"]) == \
        {"overall_score": "embed_ridge", "clutter_score": "table_ridge"}
    assert resolve_models(tmp_path, "xgboost", ["overall_score"]) == {"overall_score": "xgboost"}
