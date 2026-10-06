"""
human_check.py
Validate the AI-judge labels against a small set of human ratings.

  python src/human_check.py sample   # pick 50 images, stratified by AI overall score
  streamlit run rate_app.py          # rate them blind → data/labels_human.csv
  python src/human_check.py report   # agreement: human vs AI judge vs trained model

If the AI judge agrees with you about as well as a second human would
(Spearman ρ around 0.6–0.8 and most scores within ±0.5), the AI labels are a
reasonable training target. If not, rate more images yourself and retrain.
"""
import argparse, logging
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from utils import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SCORES = ["overall_score", "cleanliness_score", "chair_alignment_score", "clutter_score"]
SAMPLE_CSV = ROOT / "data" / "human_check_sample.csv"
HUMAN_CSV  = ROOT / "data" / "labels_human.csv"


def make_sample(labels, n=50, seed=0):
    """Pick n valid-room images, spread across the AI overall-score range (low scores are rare)."""
    valid = labels[labels.get("valid_room", 1) != 0]
    bins = pd.cut(valid["overall_score"], bins=[0, 2.75, 3.75, 4.25, 4.75, 5.0])
    per_bin = max(1, n // bins.nunique())
    picks = (valid.groupby(bins, observed=True, group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), per_bin), random_state=seed)))
    rest = valid.drop(picks.index)
    if len(picks) < n:
        picks = pd.concat([picks, rest.sample(n - len(picks), random_state=seed)])
    return picks["filename"].sample(frac=1, random_state=seed).reset_index(drop=True)


def agreement(a, b):
    """Agreement stats between two score vectors on a 1–5 scale."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return {"n": len(a), "spearman": spearmanr(a, b)[0], "mae": float(np.mean(np.abs(a - b))),
            "within_0.5": float(np.mean(np.abs(a - b) <= 0.5)), "within_1": float(np.mean(np.abs(a - b) <= 1.0))}


def report(labels, human, oof=None, model_col=None):
    """Return a DataFrame of agreement rows: human vs AI judge (per score) and human vs model."""
    m = human.merge(labels, on="filename", suffixes=("_human", "_ai"))
    rows = [{"comparison": f"human vs AI judge — {s}", **agreement(m[f"{s}_human"], m[f"{s}_ai"])}
            for s in SCORES if f"{s}_human" in m and m[f"{s}_human"].notna().all()]
    if oof is not None and model_col in oof:
        mo = human.merge(oof[["filename", model_col]], on="filename")
        rows.append({"comparison": f"human vs {model_col} (out-of-fold) — overall_score",
                     **agreement(mo["overall_score"], mo[model_col])})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["sample", "report"])
    p.add_argument("-n", type=int, default=50)
    args = p.parse_args()

    config = load_config()
    labels = pd.read_csv(ROOT / config["paths"]["labels_csv"])

    if args.command == "sample":
        make_sample(labels, args.n).to_frame().to_csv(SAMPLE_CSV, index=False)
        log.info("✓ %d images → %s. Next: streamlit run rate_app.py", args.n, SAMPLE_CSV)
        return

    if not HUMAN_CSV.exists():
        raise SystemExit(f"No human ratings yet: {HUMAN_CSV}. Run `streamlit run rate_app.py` first.")
    human = pd.read_csv(HUMAN_CSV)
    oof_path = ROOT / config["paths"].get("outputs_dir", "outputs") / "oof_predictions.csv"
    oof = pd.read_csv(oof_path) if oof_path.exists() else None
    r = report(labels, human, oof, config["model"].get("default_model", "clip_ridge"))
    print("\n" + r.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")


if __name__ == "__main__":
    main()
