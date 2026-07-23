"""
evaluate.py
Score one conference room image end-to-end:
  image → detect objects → compute features → predict quality → print report

Also saves an annotated image and a score bar chart.

Usage:
    python src/evaluate.py --image path/to/room.jpg
    python src/evaluate.py --image path/to/room.jpg --model xgboost --no-save
"""
import argparse, json, logging, sys
from collections import Counter
from pathlib import Path
import cv2, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Bounding box colors per class (BGR for OpenCV)
BOX_COLORS = {
    "chair":(100,200,60), "dining table":(200,150,50), "person":(30,160,255),
    "bottle":(0,90,200),  "cup":(0,90,200),  "laptop":(180,40,200),
    "handbag":(30,30,220),"backpack":(30,30,220), "cell phone":(180,40,200),
    "remote":(0,140,140), "book":(140,80,0),
}

def stars(score):
    """Convert a score 1-5 into a star string like ★★★☆☆."""
    n = int(round(score))
    return "★"*n + "☆"*(5-n)


_yolo_cache: dict = {}   # loaded once per process

def detect(image_path, config):
    """Run YOLOv8 on a single image. Returns a list of detected objects."""
    from ultralytics import YOLO
    cfg  = config.get("detection", {})
    keep = set(cfg.get("relevant_classes", ["chair","dining table","handbag","backpack",
                                            "bottle","cup","laptop","person","cell phone","remote","book"]))
    model_name = cfg.get("model_name", "yolov8n.pt")
    if model_name not in _yolo_cache:
        _yolo_cache[model_name] = YOLO(model_name)
    model    = _yolo_cache[model_name]
    keep_ids = {cid for cid, name in model.names.items() if name in keep}
    res = model.predict(source=str(image_path),
                        conf=float(cfg.get("confidence_threshold", 0.30)),
                        iou=float(cfg.get("iou_threshold", 0.45)),
                        imgsz=int(cfg.get("image_size", 640)),
                        device=cfg.get("device",""), verbose=False)[0]
    h, w = res.orig_shape[:2]
    dets = []
    for box in res.boxes:
        cid = int(box.cls.item())
        if cid not in keep_ids: continue
        xc, yc, bw, bh = box.xywhn[0].tolist()
        dets.append({"class": model.names[cid],
                     "x_center":round(xc,5), "y_center":round(yc,5),
                     "width":round(bw,5),    "height":round(bh,5),
                     "confidence":round(float(box.conf.item()),4),
                     "img_width":w, "img_height":h})
    return dets


def get_features(dets, config):
    """Turn raw detections into 13 features using features.py."""
    sys.path.insert(0, str(ROOT/"src"))
    from features import compute_features
    cfg = config.get("features", {})
    return compute_features(
        dets,
        clutter_cls   = set(cfg.get("clutter_classes",   ["handbag","backpack","bottle","cup","laptop","cell phone","remote","book"])),
        furniture_cls = set(cfg.get("furniture_classes",  ["chair","dining table"])),
        floor_frac    = float(cfg.get("floor_region_fraction", 0.45)),
    )


_ml_cache: dict = {}     # loaded once per process

def predict(feat, models_dir, model_name="xgboost"):
    """Load a saved model (cached) and return (quality_score, list_of_warnings)."""
    path      = models_dir / f"{model_name.lower().replace(' ','_')}.joblib"
    cache_key = str(path)
    if cache_key not in _ml_cache:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}\nRun score.py first.")
        _ml_cache[cache_key] = joblib.load(path)
    payload = _ml_cache[cache_key]
    X = pd.DataFrame([{c: feat.get(c, 0.0) for c in payload["feature_cols"]}])
    score = float(np.clip(payload["model"].predict(X)[0], 1.0, 5.0))

    warns = []
    if feat.get("clutter_object_count", 0) >= 5: warns.append("⚠ High clutter (≥5 items)")
    if feat.get("chair_spacing_variance", 0) > 0.02: warns.append("⚠ Chairs unevenly spaced")
    if feat.get("person_present", 0):   warns.append("ℹ Room is occupied")
    if feat.get("chair_count", 0) == 0: warns.append("ℹ No chairs detected")
    return score, warns


def annotate(image_path, dets, score, feat, out_path):
    """Draw bounding boxes and a score banner on a copy of the image."""
    img = cv2.imread(str(image_path))
    if img is None: return
    h, w = img.shape[:2]
    thick = max(2, int(min(h,w)/300))
    fscale = max(0.5, min(h,w)/1200)

    for d in dets:
        col = BOX_COLORS.get(d["class"], (180,180,180))
        xc, yc = d["x_center"]*w, d["y_center"]*h
        bw, bh = d["width"]*w,   d["height"]*h
        x1,y1,x2,y2 = int(xc-bw/2),int(yc-bh/2),int(xc+bw/2),int(yc+bh/2)
        cv2.rectangle(img, (x1,y1),(x2,y2), col, thick)
        label = f"{d['class']} {d['confidence']:.0%}"
        (lw,lh),bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
        ly = max(y1-4, lh+bl+4)
        cv2.rectangle(img, (x1,ly-lh-bl),(x1+lw,ly+bl), col, -1)
        cv2.putText(img, label, (x1,ly-bl//2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (255,255,255), 1, cv2.LINE_AA)

    # Top banner with overall score
    bh2 = max(60, int(h*0.08))
    ov  = img.copy()
    cv2.rectangle(ov,(0,0),(w,bh2),(15,15,30),-1)
    cv2.addWeighted(ov,0.8,img,0.2,0,img)
    bf = max(0.6, bh2/80)
    cv2.putText(img, f"Quality: {score:.1f}/5.0  {stars(score)}", (12,int(bh2*0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, bf, (220,200,100), 2, cv2.LINE_AA)
    cv2.putText(img, f"Clutter: {int(feat.get('clutter_object_count',0))}  Chairs: {int(feat.get('chair_count',0))}",
                (12,int(bh2*0.92)), cv2.FONT_HERSHEY_SIMPLEX, bf*0.75, (160,200,255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    log.info("✓ Annotated image → %s", out_path)


def score_breakdown(feat):
    """Compute approximate sub-scores from features (for display only)."""
    return {
        "Cleanliness":     min(5.0,max(1.0,round(5.0 - feat.get("table_surface_occupancy",0)*3 - feat.get("floor_occupancy_ratio",0)*2, 1))),
        "Chair Alignment": min(5.0,max(1.0,round(5.0 - feat.get("chair_spacing_variance",0)*40 - feat.get("chair_aspect_ratio_variance",0)*20, 1))),
        "Clutter Level":   min(5.0,max(1.0,round(5.0 - feat.get("clutter_object_count",0)*0.6, 1))),
    }

def save_chart(overall, breakdown, path):
    """Save a horizontal bar chart showing each quality dimension."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(list(breakdown.keys()), list(breakdown.values()), edgecolor="none")
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Score")
    ax.set_title(f"Quality: {overall:.1f}/5.0  {stars(overall)}")
    ax.axvline(x=overall, color="gray", linestyle="--", lw=1, label=f"Overall {overall:.1f}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Score chart → %s", path)


def print_report(image_path, overall, breakdown, feat, dets, warns):
    """Print a formatted quality report to the terminal."""
    obj_str = ", ".join(f"{v}x {k}" for k,v in Counter(d["class"] for d in dets).most_common())
    W = 52; B = "="*W; T = "-"*W
    print(f"\n+{B}+")
    print(f"|  {'Conference Room Quality Report':^{W-2}}  |")
    print(f"+{B}+")
    print(f"|  Image: {image_path.name:<{W-9}}|")
    print(f"+{B}+")
    print(f"|  {'Overall Score:':16} {overall:.1f} / 5.0   {stars(overall):<{W-32}}|")
    print(f"+{B}+")
    for dim, val in breakdown.items():
        print(f"|  {dim+':':22} {val:.1f} / 5.0   {stars(val):<{W-36}}|")
    print(f"+{B}+")
    print(f"|  Objects: {obj_str[:W-12]:<{W-12}}|")
    if warns:
        print(f"+{T}+")
        for w in warns: print(f"|  {w[:W-2]:<{W-2}}  |")
    print(f"+{B}+\n")


def run(image_path, model_name, save_outputs, config):
    """Run the complete evaluation pipeline on one image."""
    paths      = config.get("paths", {})
    models_dir = ROOT / paths.get("models_dir",     "models")
    dets_dir   = ROOT / paths.get("detections_dir", "outputs/detections")

    log.info("Detecting objects…")
    dets    = detect(image_path, config)
    log.info("%d objects found", len(dets))
    feat    = get_features(dets, config)
    overall, warns = predict(feat, models_dir, model_name)
    bd      = score_breakdown(feat)
    print_report(image_path, overall, bd, feat, dets, warns)

    result = {"image": str(image_path), "model": model_name,
              "overall_score": overall, "breakdown": bd,
              "features": feat, "detections_count": len(dets), "warnings": warns}

    if save_outputs:
        stem = image_path.stem
        annotate(image_path, dets, overall, feat, dets_dir/f"{stem}_annotated.jpg")
        save_chart(overall, bd, ROOT/"outputs"/f"{stem}_score_chart.png")
        jp = ROOT/"outputs"/f"{stem}_result.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(jp,"w"), indent=2)
        log.info("✓ JSON → %s", jp)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",   type=Path, required=True)
    p.add_argument("--model",   type=str, default="xgboost",
                   choices=["linear_regression","ridge_regression","random_forest","xgboost"])
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--config",  type=Path)
    args = p.parse_args()
    if not args.image.exists():
        log.error("Image not found: %s", args.image); sys.exit(1)
    run(args.image, args.model, not args.no_save, load_config(args.config))

if __name__ == "__main__":
    main()
