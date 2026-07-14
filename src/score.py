"""
score.py
Train 4 regression models on the features, evaluate them, and save the best one.
Also saves feature importance and comparison charts to outputs/.

Usage:
    python src/score.py
    python src/score.py --target cleanliness_score
"""
import argparse, logging, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from utils import ROOT, load_config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)



def metrics(y_true, y_pred):
    """Compute MAE, RMSE, R², Pearson-r, Spearman-r."""
    return {"MAE":       round(mean_absolute_error(y_true, y_pred), 4),
            "RMSE":      round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "R2":        round(r2_score(y_true, y_pred), 4),
            "Pearson_r": round(pearsonr(y_true, y_pred)[0], 4),
            "Spearman_r":round(spearmanr(y_true, y_pred)[0], 4)}


def load_data(features_path, labels_path, target_col):
    """Merge features.csv and labels.csv. Return X (features), y (target), feature names."""
    feat  = pd.read_csv(features_path)
    label = pd.read_csv(labels_path)
    df    = feat.merge(label, on="filename", how="inner")
    if len(df) == 0:
        raise ValueError("No matching filenames between features.csv and labels.csv")
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found. Options: {list(label.columns)}")
    cols = [c for c in feat.columns if c != "filename" and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols].fillna(0.0), df[target_col].astype(float), cols


def make_models(config):
    """Return the 4 models we will train and compare."""
    xgb = config.get("model", {}).get("xgboost", {})
    rf  = config.get("model", {}).get("random_forest", {})
    return {
        "Linear Regression": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Ridge Regression":  Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=int(rf.get("n_estimators",200)),
                            max_depth=rf.get("max_depth",6), random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=int(xgb.get("n_estimators",200)),
                    max_depth=int(xgb.get("max_depth",4)),
                    learning_rate=float(xgb.get("learning_rate",0.08)),
                    subsample=float(xgb.get("subsample",0.8)),
                    colsample_bytree=float(xgb.get("colsample_bytree",0.8)),
                    min_child_weight=int(xgb.get("min_child_weight",2)),
                    random_state=42, verbosity=0),
    }


def plot_importance(model, feat_names, path, title):
    """Horizontal bar chart showing which features matter most."""
    if not hasattr(model, "feature_importances_"): return
    imp = model.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feat_names[i] for i in idx], imp[idx], color="steelblue", edgecolor="none")
    ax.set_xlabel("Importance"); ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Feature importance → %s", path)

def plot_scatter(y_true, preds, path):
    """Scatter plot of predicted vs actual scores for all models."""
    fig, axes = plt.subplots(1, len(preds), figsize=(5*len(preds), 4), squeeze=False)
    for ax, (name, yp) in zip(axes[0], preds.items()):
        ax.scatter(y_true, yp, alpha=0.6, s=40, edgecolors="none")
        mn, mx = min(y_true.min(), yp.min())-0.2, max(y_true.max(), yp.max())+0.2
        ax.plot([mn,mx],[mn,mx],"--", color="gray", lw=1)
        ax.set_title(f"{name}\nMAE={mean_absolute_error(y_true,yp):.3f}  R²={r2_score(y_true,yp):.3f}", fontsize=9)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)
    fig.suptitle("Predicted vs Actual", fontsize=12)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Scatter chart → %s", path)

def plot_comparison(results, path):
    """Bar chart comparing key metrics across all 4 models."""
    met   = ["MAE", "RMSE", "R2", "Pearson_r"]
    names = list(results.keys())
    x, w  = np.arange(len(met)), 0.8/len(names)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        vals   = [results[name][m] for m in met]
        offset = (i - len(names)/2 + 0.5) * w
        ax.bar(x+offset, vals, w*0.9, label=name, alpha=0.85, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(met)
    ax.set_title("Model Comparison — Test Set")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Comparison chart → %s", path)


def train_and_evaluate(features_path, labels_path, models_dir, outputs_dir, config):
    """Train every model, print results, save models and charts."""
    cfg    = config.get("model", {})
    target = cfg.get("target_column", "overall_score")
    X, y, feat_cols = load_data(features_path, labels_path, target)
    log.info("Target: %s | %d features | %d samples", target, len(feat_cols), len(X))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=float(cfg.get("test_size",0.20)), random_state=42)
    log.info("Train: %d  Test: %d", len(X_tr), len(X_te))

    models_dir.mkdir(parents=True, exist_ok=True)
    models, results, preds = make_models(config), {}, {}

    for name, est in models.items():
        log.info("Training %s…", name)
        est.fit(X_tr, y_tr)
        yp = np.clip(est.predict(X_te), 1.0, 5.0)
        cv_mae = -cross_val_score(est, X_tr, y_tr, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1).mean()
        results[name] = {**metrics(y_te.values, yp), "CV_MAE": round(cv_mae, 4)}
        preds[name]   = yp
        safe = name.lower().replace(" ", "_")
        joblib.dump({"model": est, "feature_cols": feat_cols, "target": target}, models_dir/f"{safe}.joblib")
        log.info("  ✓ Saved → %s", models_dir/f"{safe}.joblib")

    # Print summary table
    print("\n" + "-"*70)
    print(f"{'Model':<22}{'MAE':>6}{'RMSE':>7}{'R2':>7}{'Pearson-r':>10}{'CV-MAE':>8}")
    print("-"*70)
    for name, m in results.items():
        print(f"{name:<22}{m['MAE']:>6.3f}{m['RMSE']:>7.3f}{m['R2']:>7.3f}{m['Pearson_r']:>10.3f}{m['CV_MAE']:>8.3f}")
    print("-"*70)
    log.info("Best: %s (MAE=%.3f)", min(results, key=lambda n: results[n]["MAE"]),
             min(m["MAE"] for m in results.values()))

    # Save charts
    for name, est in models.items():
        raw = est.steps[-1][1] if hasattr(est, "steps") else est
        plot_importance(raw, feat_cols, outputs_dir/f"feature_importance_{name.lower().replace(' ','_')}.png", name)
    plot_scatter(y_te.values, preds, outputs_dir/"predicted_vs_actual.png")
    plot_comparison(results, outputs_dir/"model_comparison.png")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path); p.add_argument("--labels", type=Path)
    p.add_argument("--target",   type=str);  p.add_argument("--config", type=Path)
    args = p.parse_args()

    config = load_config(args.config)
    paths  = config.get("paths", {})
    if args.target:
        config.setdefault("model", {})["target_column"] = args.target
    train_and_evaluate(
        features_path = args.features or ROOT / paths.get("features_csv", "features/features.csv"),
        labels_path   = args.labels   or ROOT / paths.get("labels_csv",   "data/labels.csv"),
        models_dir    = ROOT / paths.get("models_dir",  "models"),
        outputs_dir   = ROOT / paths.get("outputs_dir", "outputs"),
        config        = config,
    )

if __name__ == "__main__":
    main()
