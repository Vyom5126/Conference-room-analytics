"""
evaluate.py
Score one conference room image end-to-end:
  image → YOLO detections + image embeddings → per-dimension quality models → report

Each score comes with a range calibrated on held-out errors, a warning if the photo
looks unlike the training photos, and (optionally) a heatmap of the regions that
pull the score down.

Usage:
    python src/evaluate.py --image path/to/room.jpg
    python src/evaluate.py --image path/to/room.jpg --explain
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
from embed import embed_images, embed_tables, embedding_cols
from features import compute_features
from intervals import apply_interval
from ood import knn_similarity

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
MODEL_CHOICES = ["auto", "embed_ridge", "table_ridge", "hybrid_ridge",
                 "linear_regression", "ridge_regression", "random_forest", "xgboost"]
EMBED_SETS = ("embed", "embed_table", "hybrid")


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


def resolve_models(models_dir, model_name, targets):
    """{target: model key}. 'auto' uses the best model per target saved by score.py."""
    if model_name != "auto":
        return {t: model_name for t in targets}
    path = models_dir / "selection.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run score.py first.")
    with open(path, encoding="utf-8") as f:
        selection = json.load(f)
    return {t: selection[t] for t in targets}


def load_reference(models_dir):
    """Training embeddings + threshold for the out-of-domain check (cached); None if missing."""
    path = models_dir / "reference.npz"
    if "reference" not in _ml_cache:
        _ml_cache["reference"] = dict(np.load(path)) if path.exists() else None
    return _ml_cache["reference"]


def predict(inputs, models_dir, chosen):
    """
    Predict every target with its chosen model.
    `inputs` maps column name → value (detection features, emb_### and tab_### columns).
    Returns {target: (score, low, high)}, all clipped to [1, 5].
    """
    out = {}
    for target, key in chosen.items():
        payload = load_model(models_dir, target, key)
        missing = [c for c in payload["feature_cols"] if c not in inputs]
        if missing:
            raise ValueError(f"Model {target}/{key} expects {len(missing)} inputs that are not "
                             f"produced (e.g. {missing[:3]}). Retrain with: python src/score.py")
        X = pd.DataFrame([[inputs[c] for c in payload["feature_cols"]]], columns=payload["feature_cols"])
        s = float(np.clip(payload["model"].predict(X)[0], 1.0, 5.0))
        out[target] = (s, *apply_interval(s, payload["interval"])) if "interval" in payload else (s, s, s)
    return out


def observations(scores, feat, ood=None):
    """Human-readable notes from predicted sub-scores, detections and the out-of-domain check."""
    notes = []
    if ood and ood["flagged"]:
        notes.append("⚠ Photo unlike the training rooms: scores unreliable")
    notes += [f"⚠ Low {name.lower()} ({scores[t]:.1f})" for t, name in DIMENSIONS.items() if scores.get(t, 5) < 3.0]
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
    cv2.putText(img, f"Small items: {int(feat.get('clutter_object_count',0))}  Chairs: {int(feat.get('chair_count',0))}",
                (12,int(bh2*0.92)), cv2.FONT_HERSHEY_SIMPLEX, bf*0.75, (160,200,255), 1, cv2.LINE_AA)
    return img


def save_chart(overall, breakdown, ranges, path):
    """Save a horizontal bar chart of each dimension with its calibrated range."""
    names = list(breakdown)
    vals = [breakdown[n] for n in names]
    errs = np.array([[v - ranges[n][0] for n, v in zip(names, vals)], [ranges[n][1] - v for n, v in zip(names, vals)]])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(names, vals, xerr=errs, capsize=4, edgecolor="none")
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Predicted score (5 = best), bars show the ~80% range")
    ax.set_title(f"Quality: {overall:.1f}/5.0  {stars(overall)}")
    ax.axvline(x=overall, color="gray", linestyle="--", lw=1, label=f"Overall {overall:.1f}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Score chart → %s", path)


def print_report(name, scores, primary, dets, notes):
    """Print a formatted quality report to the terminal."""
    obj_str = ", ".join(f"{v}x {k}" for k,v in Counter(d["class"] for d in dets).most_common())
    W = 60; B = "="*W; T = "-"*W
    line = lambda label, t: f"|  {label+':':24} {scores[t][0]:.1f} / 5.0  ({scores[t][1]:.1f}–{scores[t][2]:.1f})  {stars(scores[t][0])}"
    print(f"\n+{B}+")
    print(f"|  {'Conference Room Quality Report':^{W-2}}  |")
    print(f"+{B}+")
    print(f"|  Image: {name[:W-9]:<{W-9}}|")
    print(f"+{B}+")
    print(f"{line('Overall Score', primary):<{W+1}}|")
    print(f"+{B}+")
    for t, label in DIMENSIONS.items():
        print(f"{line(label, t):<{W+1}}|")
    print(f"+{B}+")
    print(f"|  Objects: {obj_str[:W-12]:<{W-12}}|")
    if notes:
        print(f"+{T}+")
        for n in notes: print(f"|  {n[:W-2]:<{W-2}}  |")
    print(f"+{B}+\n")


def run(image, model_name, save_outputs, config, name=None, quiet=False, explain=False):
    """
    Run the complete evaluation pipeline on one image.
    `image` is a file path or a BGR numpy array (pass `name` for arrays).
    Returns the result dict; result["annotated"] holds the annotated BGR image and
    result["heatmap"] the explanation overlay (None unless explain=True and the model supports it).
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
    chosen      = resolve_models(models_dir, model_name, targets)
    payloads    = {t: load_model(models_dir, t, k) for t, k in chosen.items()}
    sets        = {p["feature_set"] for p in payloads.values()}

    log.info("Detecting objects…")
    dets = detect_image(img, config)
    feat = get_features(dets, config)
    inputs, x, heatmap = dict(feat), None, None

    if sets & set(EMBED_SETS):
        emb_cfg = next(p["embedding"] for p in payloads.values() if p["feature_set"] in EMBED_SETS)
        if explain:
            from explain import explain as explain_fn
            res = explain_fn(img, payloads[primary])
            if res is not None:
                heatmap, _, x = res
        if x is None:
            log.info("Embedding image…")
            x = embed_images([img], emb_cfg)[0]   # embed exactly as the models were trained
        inputs.update(zip(embedding_cols(len(x)), x))
        if "embed_table" in sets:
            tab, _ = embed_tables([img], [dets], emb_cfg)
            inputs.update(zip(embedding_cols(tab.shape[1], "tab"), tab[0]))

    scores = predict(inputs, models_dir, chosen)

    ood, ref = None, load_reference(models_dir)
    if x is not None and ref is not None:
        sim = float(knn_similarity(x, ref["embeddings"].astype(np.float32), int(ref["k"]))[0])
        ood = {"similarity": sim, "threshold": float(ref["threshold"]), "flagged": sim < float(ref["threshold"])}

    overall   = scores[primary][0]
    breakdown = {DIMENSIONS[t]: round(scores[t][0], 2) for t in DIMENSIONS}
    ranges    = {DIMENSIONS[t]: (round(scores[t][1], 2), round(scores[t][2], 2)) for t in DIMENSIONS}
    notes     = observations({t: v[0] for t, v in scores.items()}, feat, ood)
    if not quiet:
        print_report(name, scores, primary, dets, notes)

    result = {"image": name, "model": model_name, "models_used": chosen,
              "overall_score": overall, "overall_range": (round(scores[primary][1], 2), round(scores[primary][2], 2)),
              "breakdown": breakdown, "ranges": ranges, "ood": ood,
              "features": feat, "detections_count": len(dets), "detections": dets, "warnings": notes}
    annotated = annotate(img, dets, overall, feat)

    if save_outputs:
        stem = Path(name).stem
        ann_path = dets_dir/f"{stem}_annotated.jpg"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ann_path), annotated)
        log.info("✓ Annotated image → %s", ann_path)
        if heatmap is not None:
            cv2.imwrite(str(outputs_dir/f"{stem}_heatmap.jpg"), heatmap)
            log.info("✓ Heatmap → %s", outputs_dir/f"{stem}_heatmap.jpg")
        save_chart(overall, breakdown, ranges, outputs_dir/f"{stem}_score_chart.png")
        jp = outputs_dir/f"{stem}_result.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info("✓ JSON → %s", jp)
    return {**result, "annotated": annotated, "heatmap": heatmap}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",   type=Path, required=True)
    p.add_argument("--model",   type=str, choices=MODEL_CHOICES)
    p.add_argument("--explain", action="store_true", help="also save the experimental score heatmap")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--config",  type=Path)
    args = p.parse_args()
    if not args.image.exists():
        log.error("Image not found: %s", args.image); sys.exit(1)
    config = load_config(args.config)
    run(args.image, args.model or config["model"].get("default_model", "auto"), not args.no_save, config,
        explain=args.explain)

if __name__ == "__main__":
    main()
