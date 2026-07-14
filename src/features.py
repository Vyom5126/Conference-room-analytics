"""
features.py
Read detections.json and compute 13 features per image, saved to features.csv.

Features describe chair alignment, clutter level, and overall room scene.

Usage:
    python src/features.py
"""
import argparse, json, logging
from pathlib import Path
import numpy as np, pandas as pd
from utils import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_CLUTTER   = ["handbag", "backpack", "bottle", "cup", "laptop", "cell phone", "remote", "book"]
DEFAULT_FURNITURE = ["chair", "dining table"]


def nn_distances(centroids):
    """For each point, return its distance to the nearest other point."""
    if len(centroids) < 2:
        return []
    arr = np.array(centroids)
    return [float(np.min(np.linalg.norm(np.delete(arr, i, axis=0) - c, axis=1)))
            for i, c in enumerate(arr)]

def overlap_fraction(table, obj):
    """Fraction of obj's bounding box that overlaps with the table's bounding box."""
    # Convert (center, size) → (left, right, top, bottom)
    def corners(d):
        return d["x_center"]-d["width"]/2, d["x_center"]+d["width"]/2, \
               d["y_center"]-d["height"]/2, d["y_center"]+d["height"]/2
    tx1,tx2,ty1,ty2 = corners(table)
    ox1,ox2,oy1,oy2 = corners(obj)
    iw = max(0.0, min(tx2,ox2) - max(tx1,ox1))
    ih = max(0.0, min(ty2,oy2) - max(ty1,oy1))
    area = obj["width"] * obj["height"]
    return (iw * ih) / area if area > 0 else 0.0


def compute_features(dets, clutter_cls, furniture_cls, floor_frac=0.45):
    """
    Turn a list of object detections into 13 numerical features.
    Each detection is a dict with: class, x_center, y_center, width, height, confidence
    """
    # Sort detections into groups by class name
    by_cls  = {}
    for d in dets:
        by_cls.setdefault(d["class"], []).append(d)

    chairs  = by_cls.get("chair", [])
    tables  = by_cls.get("dining table", [])
    clutter = [d for d in dets if d["class"] in clutter_cls]
    people  = by_cls.get("person", [])
    feat    = {}

    # Chair features (how many, how aligned) 
    feat["chair_count"] = float(len(chairs))
    if len(chairs) >= 2:
        ratios = [c["width"]/c["height"] if c["height"] > 0 else 1.0 for c in chairs]
        dists  = nn_distances([(c["x_center"], c["y_center"]) for c in chairs])
        feat["chair_aspect_ratio_variance"] = float(np.var(ratios))           
        feat["chair_spacing_variance"]      = float(np.var(dists)) if dists else 0.0  
        feat["chair_x_variance"]            = float(np.var([c["x_center"] for c in chairs]))  
    else:
        feat["chair_aspect_ratio_variance"] = feat["chair_spacing_variance"] = feat["chair_x_variance"] = 0.0

    feat["chairs_per_table_ratio"] = float(len(chairs)) / max(len(tables), 1)

    # Clutter features (how messy) 
    feat["clutter_object_count"] = float(len(clutter))
    feat["clutter_density"]      = float(sum(d["width"]*d["height"] for d in clutter))  

    # Objects on the floor = non-furniture objects in the bottom portion of image
    floor_start = 1.0 - floor_frac
    floor_objs  = [d for d in dets if d["class"] not in furniture_cls|{"person"}
                   and (d["y_center"] + d["height"]/2) > floor_start]
    floor_covered = sum(d["width"] * min(d["height"], d["y_center"]+d["height"]/2 - floor_start)
                        for d in floor_objs)
    feat["floor_occupancy_ratio"] = float(min(floor_covered / floor_frac, 1.0))

    # Scene features (room-level context)
    feat["person_present"] = 1.0 if people else 0.0

    # How much of each table is covered by objects on top of it
    if tables:
        table_area = sum(t["width"]*t["height"] for t in tables)
        non_furn   = [d for d in dets if d["class"] not in furniture_cls]
        covered    = sum(overlap_fraction(t, o) * o["width"]*o["height"]
                        for t in tables for o in non_furn)
        feat["table_surface_occupancy"] = float(min(covered/table_area, 1.0)) if table_area > 0 else 0.0
    else:
        feat["table_surface_occupancy"] = 0.0

    feat["total_object_count"] = float(len(dets))
    feat["mean_confidence"]    = float(np.mean([d["confidence"] for d in dets])) if dets else 0.0
    feat["unique_class_count"] = float(len(by_cls))

    return feat


def build_features_csv(detections_path, output_path, config):
    """Load detections.json → compute features for each image → save features.csv."""
    if not detections_path.exists():
        raise FileNotFoundError(f"Run detect.py first. File not found: {detections_path}")

    all_dets  = json.load(open(detections_path, encoding="utf-8"))
    cfg       = config.get("features", {})
    clutter   = set(cfg.get("clutter_classes",   DEFAULT_CLUTTER))
    furniture = set(cfg.get("furniture_classes",  DEFAULT_FURNITURE))
    floor_frac = float(cfg.get("floor_region_fraction", 0.45))

    log.info("Computing features for %d images…", len(all_dets))

    rows = [{"filename": fname, **compute_features(dets, clutter, furniture, floor_frac)}
            for fname, dets in all_dets.items()]

    df = pd.DataFrame(rows)
    df = df[["filename"] + [c for c in df.columns if c != "filename"]].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("✓ Saved → %s  (%d images × %d features)", output_path, len(df), len(df.columns)-1)
    log.info("\n%s", df.select_dtypes("number").describe().round(3).to_string())
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path)
    p.add_argument("--output",     type=Path)
    p.add_argument("--config",     type=Path)
    args = p.parse_args()

    config = load_config(args.config)
    paths  = config.get("paths", {})
    build_features_csv(
        detections_path = args.detections or ROOT / paths.get("detections_dir", "outputs/detections") / "detections.json",
        output_path     = args.output     or ROOT / paths.get("features_csv",   "features/features.csv"),
        config          = config,
    )

if __name__ == "__main__":
    main()
