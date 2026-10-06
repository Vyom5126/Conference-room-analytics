"""
evaluate.py
Score one conference room image end-to-end:
  image → CLIP embedding + YOLO detections → per-dimension quality models → report

Also saves an annotated image and a score bar chart.

Usage:
    python src/evaluate.py --image path/to/room.jpg
    python src/evaluate.py --image path/to/room.jpg --model ridge_regression --no-save
"""
import argparse, json, logging, sys
from collections import Counter
from pathlib import Path
import cv2, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import ROOT, load_config
from detect import detect_image
from embed import embed_images, embedding_cols
from features import compute_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Bounding box colors per class (BGR for OpenCV)
BOX_COLORS = {
    "chair":(100,200,60), "dining table":(200,150,50), "person":(30,160,255),
    "bottle":(0,90,200),  "cup":(0,90,200),  "laptop":(180,40,200),
    "handbag":(30,30,220),"backpack":(30,30,220), "cell phone":(180,40,200),
    "remote":(0,140,140), "book":(140,80,0), "keyboard":(180,40,200),
    "mouse":(180,40,200), "tv":(160,160,60),
}

# Display names for the per-dimension targets (5 = best for all of them)
DIMENSIONS = {
    "cleanliness_score":     "Cleanliness",
    "chair_alignment_score": "Chair Alignment",
    "clutter_score":         "Table Clear of Clutter",
}
MODEL_CHOICES = ["clip_ridge", "hybrid_ridge", "linear_regression", "ridge_regression", "random_forest", "xgboost"]


def stars(score):
    """Convert a score 1-5 into a star string like ★★★☆☆ (x.5 rounds up)."""
    n = int(score + 0.5)
    return "★"*n + "☆"*(5-n)


def get_features(dets, config):
    """Turn raw detections into the feature dict using features.py."""
    cfg = config["features"]
    return compute_features(
        dets,
        clutter_cls   = set(cfg["clutter_classes"]),
        furniture_cls = set(cfg["furniture_classes"]),
        floor_frac    = float(cfg.get("floor_region_fraction", 0.45)),
    )


_ml_cache: dict = {}     # loaded once per process

def load_model(models_dir, target, model_name):
    """Load a saved model payload (cached)."""
    path = models_dir / target / f"{model_name}.joblib"
    if str(path) not in _ml_cache:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}\nRun score.py first.")
        _ml_cache[str(path)] = joblib.load(path)
    return _ml_cache[str(path)]


def predict(inputs, models_dir, model_name, targets):
    """
    Predict every target with one model family.
    `inputs` maps column name → value (detection features and/or emb_### columns).
    Returns {target: score clipped to [1, 5]}.
    """
    scores = {}
    for target in targets:
        payload = load_model(models_dir, target, model_name)
        missing = [c for c in payload["feature_cols"] if c not in inputs]
        if missing:
            raise ValueError(f"Model {target}/{model_name} expects {len(missing)} inputs that are not "
                             f"produced (e.g. {missing[:3]}). Retrain with: python src/score.py")
        X = pd.DataFrame([[inputs[c] for c in payload["feature_cols"]]], columns=payload["feature_cols"])
        scores[target] = float(np.clip(payload["model"].predict(X)[0], 1.0, 5.0))
    return scores


def observations(scores, feat):
    """Human-readable notes from predicted sub-scores and detections."""
    notes = [f"⚠ Low {name.lower()} ({scores[t]:.1f})" for t, name in DIMENSIONS.items() if scores.get(t, 5) < 3.0]
    if feat.get("clutter_object_count", 0) >= 3:   # YOLO can't tell place settings from leftovers
        notes.append(f"ℹ {int(feat['clutter_object_count'])} small objects detected (cups, bottles, laptops…)")
    if feat.get("person_present", 0):   notes.append("ℹ Room is occupied")
    if feat.get("chair_count", 0) == 0: notes.append("ℹ No chairs detected")
    return notes


def annotate(img, dets, score, feat):
    """Return a copy of the BGR image with bounding boxes and a score banner drawn on it."""
    img = img.copy()
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
    cv2.putText(img, f"Quality: {score:.1f}/5.0", (12,int(bh2*0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, bf, (220,200,100), 2, cv2.LINE_AA)
    cv2.putText(img, f"Clutter items: {int(feat.get('clutter_object_count',0))}  Chairs: {int(feat.get('chair_count',0))}",
                (12,int(bh2*0.92)), cv2.FONT_HERSHEY_SIMPLEX, bf*0.75, (160,200,255), 1, cv2.LINE_AA)
    return img


def save_chart(overall, breakdown, path):
    """Save a horizontal bar chart showing each quality dimension."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(list(breakdown.keys()), list(breakdown.values()), edgecolor="none")
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Predicted score (5 = best)")
    ax.set_title(f"Quality: {overall:.1f}/5.0  {stars(overall)}")
    ax.axvline(x=overall, color="gray", linestyle="--", lw=1, label=f"Overall {overall:.1f}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Score chart → %s", path)


def print_report(name, overall, breakdown, dets, notes):
    """Print a formatted quality report to the terminal."""
    obj_str = ", ".join(f"{v}x {k}" for k,v in Counter(d["class"] for d in dets).most_common())
    W = 52; B = "="*W; T = "-"*W
    print(f"\n+{B}+")
    print(f"|  {'Conference Room Quality Report':^{W-2}}  |")
    print(f"+{B}+")
    print(f"|  Image: {name[:W-9]:<{W-9}}|")
    print(f"+{B}+")
    print(f"|  {'Overall Score:':24} {overall:.1f} / 5.0   {stars(overall):<{W-40}}|")
    print(f"+{B}+")
    for dim, val in breakdown.items():
        print(f"|  {dim+':':24} {val:.1f} / 5.0   {stars(val):<{W-40}}|")
    print(f"+{B}+")
    print(f"|  Objects: {obj_str[:W-12]:<{W-12}}|")
    if notes:
        print(f"+{T}+")
        for n in notes: print(f"|  {n[:W-2]:<{W-2}}  |")
    print(f"+{B}+\n")


def run(image, model_name, save_outputs, config, name=None, quiet=False):
    """
    Run the complete evaluation pipeline on one image.
    `image` is a file path or a BGR numpy array (pass `name` for arrays).
    Returns the result dict; result["annotated"] holds the annotated BGR image.
    """
    if isinstance(image, (str, Path)):
        image = Path(image)
        name  = name or image.name
        img   = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        img, name = image, name or "upload"

    paths       = config.get("paths", {})
    models_dir  = ROOT / paths.get("models_dir",     "models")
    dets_dir    = ROOT / paths.get("detections_dir", "outputs/detections")
    outputs_dir = ROOT / paths.get("outputs_dir",    "outputs")
    primary     = config["model"].get("target_column", "overall_score")
    targets     = [primary] + [t for t in DIMENSIONS if t != primary]

    log.info("Detecting objects…")
    dets = detect_image(img, config)
    feat = get_features(dets, config)
    inputs = dict(feat)
    if load_model(models_dir, primary, model_name)["feature_set"] in ("clip", "hybrid"):
        log.info("Embedding image…")
        emb = embed_images([img], config)[0]
        inputs.update(zip(embedding_cols(len(emb)), emb))

    scores    = predict(inputs, models_dir, model_name, targets)
    overall   = scores[primary]
    breakdown = {DIMENSIONS[t]: round(scores[t], 2) for t in DIMENSIONS}
    notes     = observations(scores, feat)
    if not quiet:
        print_report(name, overall, breakdown, dets, notes)

    result = {"image": name, "model": model_name,
              "overall_score": overall, "breakdown": breakdown,
              "features": feat, "detections_count": len(dets), "detections": dets, "warnings": notes}
    annotated = annotate(img, dets, overall, feat)

    if save_outputs:
        stem = Path(name).stem
        ann_path = dets_dir/f"{stem}_annotated.jpg"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ann_path), annotated)
        log.info("✓ Annotated image → %s", ann_path)
        save_chart(overall, breakdown, outputs_dir/f"{stem}_score_chart.png")
        jp = outputs_dir/f"{stem}_result.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w") as f:
            json.dump(result, f, indent=2)
        log.info("✓ JSON → %s", jp)
    return {**result, "annotated": annotated}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",   type=Path, required=True)
    p.add_argument("--model",   type=str, choices=MODEL_CHOICES)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--config",  type=Path)
    args = p.parse_args()
    if not args.image.exists():
        log.error("Image not found: %s", args.image); sys.exit(1)
    config = load_config(args.config)
    run(args.image, args.model or config["model"].get("default_model", "clip_ridge"), not args.no_save, config)

if __name__ == "__main__":
    main()
