"""
clip_baseline.py
Score images using CLIP (zero-shot, no training needed) and compare
how well it ranks rooms against our trained XGBoost pipeline.

CLIP compares each image to an ensemble of positive ("clean, tidy room…")
and negative ("messy, cluttered room…") prompts. The score is the mean
positive logit minus the mean negative logit. Unlike a 2-way softmax this
does not saturate near 1.0, so rankings stay informative.

The trained models are compared using out-of-fold predictions written by
score.py (outputs/oof_predictions.csv), so no model is scored on images it
was trained on. Zero-shot CLIP uses no labels at all.

Usage:
    python src/clip_baseline.py
    python src/clip_baseline.py --images-dir data/images --labels data/labels.csv
"""
import argparse, logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from score import MODEL_INFO
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
    Return mean(positive-prompt logits) − mean(negative-prompt logits) for one image.
    Higher = looks more like the positive descriptions.
    """
    import torch
    from PIL import Image
    inputs = processor(text=list(pos) + list(neg), images=Image.open(img_path).convert("RGB"),
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.squeeze(0)
    return float(logits[:len(pos)].mean() - logits[len(pos):].mean())

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
            rows.append({"filename": fname, "clip_score": clip_score(img, model, processor, pos, neg)})
        except Exception as e:
            log.warning("Failed %s: %s", fname, e)
    return pd.DataFrame(rows)


def compare(clip_df, labels_df, oof_path, config, outputs_dir):
    """Compare CLIP and out-of-fold pipeline rankings. Print table, save charts."""
    from scipy.stats import pearsonr, spearmanr

    target = config.get("model", {}).get("target_column", "overall_score")
    merged = clip_df.merge(labels_df[["filename", target]], on="filename")
    model_cols = []
    if oof_path.exists():
        # Same images for every method: the OOF file already excludes flagged images
        oof = pd.read_csv(oof_path)
        model_cols = [c for c in oof.columns if c not in ("filename", target) and not c.startswith("dummy")]
        merged = merged.merge(oof[["filename"] + model_cols], on="filename")
    else:
        log.warning("No %s — run score.py first to compare against the pipeline.", oof_path)

    raw = merged["clip_score"].values
    # Linear rescale to 1-5 for plotting only (rank metrics are unaffected)
    merged["clip_1_5"] = 1 + (raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 4

    y = merged[target].values
    corr = {"CLIP (zero-shot)": (spearmanr(y, raw)[0], pearsonr(y, raw)[0])}
    for c in model_cols:
        corr[f"{MODEL_INFO[c][0] if c in MODEL_INFO else c} (OOF)"] = (spearmanr(y, merged[c])[0], pearsonr(y, merged[c])[0])
    log.info("Comparing on %d images", len(merged))

    print("\n" + "-"*58)
    print(f"{'Method':<32}{'Spearman-rho':>13}{'Pearson r':>13}")
    print("-"*58)
    for name, (sp, pe) in corr.items():
        print(f"{name:<32}{sp:>13.3f}{pe:>13.3f}")
    print("-"*58)
    print("Labels: AI-judge ratings (data/rating_rubric.md). Trained models use out-of-fold predictions.\n")

    _plot_scatter(merged, y, target, outputs_dir)
    _plot_bars(corr, outputs_dir)
    return corr


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
    """Bar chart of Spearman rho and Pearson r for CLIP vs. pipeline models."""
    methods = list(corr.keys())
    x, w = np.arange(len(methods)), 0.35
    fig, ax = plt.subplots(figsize=(max(7, 2*len(methods)), 4))
    ax.bar(x-w/2, [corr[m][0] for m in methods], w, label="Spearman rho", alpha=0.85, edgecolor="none")
    ax.bar(x+w/2, [corr[m][1] for m in methods], w, label="Pearson r",    alpha=0.85, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("Correlation"); ax.set_ylim(min(0, *[v for c in corr.values() for v in c]) - 0.05, 1.15)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("CLIP vs Pipeline (out-of-fold) — Ranking Correlation")
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
    outputs_dir   = ROOT / paths.get("outputs_dir",  "outputs")

    if not labels_path.exists():
        log.error("labels.csv not found: %s", labels_path); sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    log.info("Loaded %d labelled images", len(labels_df))

    model, proc = load_clip(clip_cfg.get("model_name", "openai/clip-vit-base-patch32"))
    clip_df = score_all(images_dir, labels_df, model, proc,
                        clip_cfg["positive_prompts"], clip_cfg["negative_prompts"])

    if clip_df.empty:
        log.error("No CLIP scores — check image paths."); sys.exit(1)

    out = outputs_dir/"clip_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    clip_df.to_csv(out, index=False); log.info("✓ Scores → %s", out)
    compare(clip_df, labels_df, outputs_dir/"oof_predictions.csv", config, outputs_dir)

if __name__ == "__main__":
    main()
