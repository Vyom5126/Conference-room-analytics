"""Unit tests for src/features.py using hand-built detections (no YOLO needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import compute_features, union_mask  # noqa: E402

CLUTTER   = {"handbag", "backpack", "bottle", "cup", "laptop", "cell phone", "remote", "book", "keyboard", "mouse"}
FURNITURE = {"chair", "dining table", "tv"}


def box(cls, xc, yc, w, h, conf=0.9):
    return {"class": cls, "x_center": xc, "y_center": yc, "width": w, "height": h, "confidence": conf}


def feats(dets):
    return compute_features(dets, CLUTTER, FURNITURE, floor_frac=0.45)


def test_empty_detections_give_all_zero_features():
    f = feats([])
    assert len(f) == 13
    assert all(v == 0.0 for v in f.values())


def test_person_over_table_is_not_table_clutter():
    table  = box("dining table", 0.5, 0.5, 0.4, 0.2)
    person = box("person",       0.5, 0.5, 0.2, 0.6)
    assert feats([table, person])["table_surface_occupancy"] == 0.0


def test_item_on_table_counts_toward_occupancy():
    table = box("dining table", 0.5, 0.5, 0.4, 0.2)   # area 0.08
    cup   = box("cup",          0.5, 0.5, 0.1, 0.1)   # area 0.01, fully on the table
    assert feats([table, cup])["table_surface_occupancy"] == pytest.approx(0.01 / 0.08, rel=0.05)


def test_overlapping_tables_do_not_double_count():
    t1  = box("dining table", 0.5, 0.5, 0.4, 0.2)
    t2  = box("dining table", 0.5, 0.5, 0.4, 0.2)     # duplicate detection of the same table
    cup = box("cup",          0.5, 0.5, 0.1, 0.1)
    one = feats([t1, cup])["table_surface_occupancy"]
    two = feats([t1, t2, cup])["table_surface_occupancy"]
    assert two == pytest.approx(one)


def test_tv_is_a_fixture_not_clutter_or_floor():
    tv = box("tv", 0.5, 0.9, 0.3, 0.2)
    f = feats([tv])
    assert f["clutter_object_count"] == 0.0
    assert f["floor_occupancy_ratio"] == 0.0


def test_clutter_area_frac_sums_box_areas():
    f = feats([box("bottle", 0.2, 0.2, 0.1, 0.2), box("laptop", 0.7, 0.3, 0.2, 0.1)])
    assert f["clutter_object_count"] == 2.0
    assert f["clutter_area_frac"] == pytest.approx(0.02 + 0.02)


def test_chair_features_need_two_chairs():
    one = feats([box("chair", 0.3, 0.7, 0.1, 0.2)])
    assert one["chair_count"] == 1.0
    assert one["chair_spacing_variance"] == 0.0
    evenly = feats([box("chair", x, 0.7, 0.1, 0.2) for x in (0.2, 0.4, 0.6, 0.8)])
    uneven = feats([box("chair", x, 0.7, 0.1, 0.2) for x in (0.1, 0.15, 0.6, 0.9)])
    assert evenly["chair_spacing_variance"] == pytest.approx(0.0, abs=1e-12)
    assert uneven["chair_spacing_variance"] > evenly["chair_spacing_variance"]


def test_union_mask_clips_boxes_to_image():
    m = union_mask([box("cup", 0.0, 0.0, 0.2, 0.2)], n=100)
    assert m.sum() == 10 * 10   # only the in-image quarter of the box
