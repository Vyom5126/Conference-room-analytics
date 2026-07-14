"""
clip_baseline.py
Score images using CLIP (zero-shot, no training needed) and compare
how well it ranks rooms against our trained XGBoost pipeline.

CLIP works by comparing each image to two text descriptions:
  - Positive: "a clean, organized conference room…"
  - Negative: "a messy, cluttered conference room…"
The score is how well the image matches the positive description.

Usage:
    python src/clip_baseline.py
    python src/clip_baseline.py --images-dir data/images --labels data/labels.csv
"""
import argparse, logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import ROOT, load_config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)




def load_clip(model_name):
    """Download and return the CLIP model and text/image processor."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        log.error("Run: pip install transformers"); sys.exit(1)
    log.info("Loading CLIP: %s", model_name)
    model = CLIPModel.from_pretrained(model_name); model.eval()
    return model, CLIPProcessor.from_pretrained(model_name)

def clip_score(img_path, model, processor, pos, neg):
    """
    Return a 0-1 score for how well the image matches the positive text description.
    Uses CLIP's softmax over [positive, negative] similarity logits.
    """
    import torch
    from PIL import Image
    inputs = processor(text=[pos, neg], images=Image.open(img_path).convert("RGB"),
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image  
    return float(logits.softmax(dim=1).squeeze()[0].item())

def score_all(images_dir, labels_df, model, processor, pos, neg):
    """Compute CLIP scores for every labelled image. Returns a DataFrame."""
    from tqdm import tqdm
    rows = []
    for fname in tqdm(labels_df["filename"].tolist(), desc="CLIP scoring"):
        img = images_dir / fname
        if not img.exists(): 
            hits = [p for p in images_dir.iterdir() if p.name.lower() == fname.lower()]
            if not hits: log.warning("Skipping (not found): %s", fname); continue
            img = hits[0]
        try:
            rows.append({"filename": fname, "clip_score_raw": clip_score(img, model, processor, pos, neg)})
        except Exception as e:
            log.warning("Failed %s: %s", fname, e)
    return pd.DataFrame(rows)


def compare(clip_df, labels_df, features_path, models_dir, config, outputs_dir):
    """Compare CLIP and pipeline ranking. Print table, save charts."""
    import joblib
    from scipy.stats import pearsonr, spearmanr

    target = config.get("model", {}).get("target_column", "overall_score")

    # Try to load pipeline (XGBoost) predictions for comparison
    pipe_scores = None
    if features_path.exists():
        feat  = pd.read_csv(features_path)
        mpath = models_dir / "xgboost.joblib"
        if mpath.exists():
            p = joblib.load(mpath)
            X = feat.merge(labels_df[["filename"]], on="filename")
            pipe_scores = pd.DataFrame({
                "filename":       X["filename"].values,
                "pipeline_score": np.clip(p["model"].predict(X[[c for c in p["feature_cols"] if c in X.columns]].fillna(0.0)), 1.0, 5.0),
            })

    # Rescale CLIP's raw 0-1 scores to 1-5 for a fair comparison
    merged = clip_df.merge(labels_df[["filename", target]], on="filename")
    raw    = merged["clip_score_raw"].values
    merged["clip_1_5"] = 1 + (raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 4
    if pipe_scores is not None:
        merged = merged.merge(pipe_scores, on="filename", how="left")

    y = merged[target].values
    cs, _ = spearmanr(y, merged["clip_1_5"].values)
    cp, _ = pearsonr(y,  merged["clip_1_5"].values)

    print("\n" + "-"*50)
    print(f"{'Method':<24}{'Spearman-rho':>12}{'Pearson r':>12}")
    print("-"*50)
    print(f"{'CLIP (zero-shot)':<24}{cs:>12.3f}{cp:>12.3f}")
    ps = pp = None
    if pipe_scores is not None and "pipeline_score" in merged.columns:
        v = merged.dropna(subset=["pipeline_score"])
        if len(v) >= 3:
            ps,_ = spearmanr(v[target].values, v["pipeline_score"].values)
            pp,_ = pearsonr(v[target].values,  v["pipeline_score"].values)
            print(f"{'Engineered Pipeline':<24}{ps:>12.3f}{pp:>12.3f}")
    print("-"*50 + "\n")

    _plot_scatter(merged, y, target, outputs_dir)
    if ps is not None:
        _plot_bars({"CLIP":(cs,cp),"Pipeline":(ps,pp)}, outputs_dir)


def _plot_scatter(df, y, target, out_dir):
    """Scatter + histogram comparison of CLIP scores vs ground truth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.scatter(y, df["clip_1_5"].values, alpha=0.6, s=40, edgecolors="none")
    mn, mx = y.min()-0.3, y.max()+0.3
    ax1.plot([mn,mx],[mn,mx],"--", color="gray", lw=1)
    ax1.set_xlabel(f"Ground Truth ({target})"); ax1.set_ylabel("CLIP Score (1-5)")
    ax1.set_title("CLIP vs Ground Truth"); ax1.grid(alpha=0.3)

    ax2.hist(df["clip_1_5"].values, bins=15, alpha=0.7, edgecolor="none", label="CLIP")
    ax2.hist(y, bins=15, alpha=0.5, edgecolor="none", label="Ground Truth")
    ax2.set_xlabel("Score (1-5)"); ax2.set_ylabel("Count")
    ax2.set_title("Score Distribution"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout(); out = out_dir/"clip_analysis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ CLIP scatter → %s", out)

def _plot_bars(corr, out_dir):
    """Bar chart of Spearman rho and Pearson r for CLIP vs. Pipeline."""
    methods = list(corr.keys())
    x, w = np.arange(len(methods)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x-w/2, [corr[m][0] for m in methods], w, label="Spearman rho", alpha=0.85, edgecolor="none")
    ax.bar(x+w/2, [corr[m][1] for m in methods], w, label="Pearson r",    alpha=0.85, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Correlation"); ax.set_ylim(0, 1.15)
    ax.set_title("CLIP vs Pipeline — Ranking Correlation")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); out = out_dir/"clip_vs_pipeline.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Comparison chart → %s", out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path); p.add_argument("--labels", type=Path)
    p.add_argument("--config",     type=Path)
    args = p.parse_args()

    config   = load_config(args.config)
    paths    = config.get("paths", {})
    clip_cfg = config.get("clip", {})

    images_dir    = args.images_dir or ROOT / paths.get("images_dir",   "data/images")
    labels_path   = args.labels     or ROOT / paths.get("labels_csv",   "data/labels.csv")
    features_path = ROOT / paths.get("features_csv", "features/features.csv")
    models_dir    = ROOT / paths.get("models_dir",   "models")
    outputs_dir   = ROOT / paths.get("outputs_dir",  "outputs")

    if not labels_path.exists():
        log.error("labels.csv not found: %s", labels_path); sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    log.info("Loaded %d labelled images", len(labels_df))

    model, proc = load_clip(clip_cfg.get("model_name", "openai/clip-vit-base-patch32"))
    clip_df = score_all(images_dir, labels_df, model, proc,
                        clip_cfg.get("positive_prompt","a clean, organized, professional conference room with neatly arranged chairs"),
                        clip_cfg.get("negative_prompt","a messy, cluttered, disorganized conference room with objects scattered"))

    if clip_df.empty:
        log.error("No CLIP scores — check image paths."); sys.exit(1)

    out = outputs_dir/"clip_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    clip_df.to_csv(out, index=False); log.info("✓ Scores → %s", out)
    compare(clip_df, labels_df, features_path, models_dir, config, outputs_dir)

if __name__ == "__main__":
    main()
