"""
score.py
For every target in config.model.targets, cross-validate a predict-the-mean
baseline and models built on three feature sets, using repeated K-fold:

  detection  13 YOLO features          → Linear, Ridge, Random Forest, XGBoost
  clip       CLIP image embeddings     → Ridge ("linear probe")
  hybrid     embeddings + YOLO features → Ridge

Each model is then refit on all labelled data and saved to
models/<target>/<model>.joblib. Writes outputs/cv_results.csv, out-of-fold
predictions for the primary target, and comparison charts.

Images rated valid_room = 0 in labels.csv (not conference rooms) are left out.

Usage:
    python src/score.py
"""
import argparse, logging, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from embed import load_embeddings
from utils import ROOT, load_config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# model key → (display name, feature set)
MODEL_INFO = {
    "dummy_mean":        ("Dummy (mean)",                "detection"),
    "linear_regression": ("Linear Regression (YOLO)",    "detection"),
    "ridge_regression":  ("Ridge (YOLO)",                "detection"),
    "random_forest":     ("Random Forest (YOLO)",        "detection"),
    "xgboost":           ("XGBoost (YOLO)",              "detection"),
    "clip_ridge":        ("Ridge (CLIP embeddings)",     "clip"),
    "hybrid_ridge":      ("Ridge (CLIP + YOLO)",         "hybrid"),
}


def metrics(y_true, y_pred):
    """Compute MAE, RMSE, R², Spearman-ρ, Kendall-τ (rank metrics are NaN for constant predictions)."""
    const = np.ptp(y_pred) == 0
    return {"MAE":        mean_absolute_error(y_true, y_pred),
            "RMSE":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2":         r2_score(y_true, y_pred),
            "Spearman_r": np.nan if const else spearmanr(y_true, y_pred)[0],
            "Kendall_t":  np.nan if const else kendalltau(y_true, y_pred)[0]}


def load_data(features_path, embeddings_path, labels_path):
    """
    Join labels with YOLO features and CLIP embeddings, dropping images rated valid_room = 0.
    Returns (labels DataFrame, {feature set: X DataFrame}).
    """
    label = pd.read_csv(labels_path)
    if "valid_room" in label.columns:
        n_bad = int((label["valid_room"] == 0).sum())
        label = label[label["valid_room"] != 0]
        log.info("Excluding %d images rated valid_room = 0 (not conference rooms)", n_bad)
    feat = pd.read_csv(features_path).set_index("filename")
    emb  = load_embeddings(embeddings_path)

    names = [f for f in label["filename"] if f in feat.index and f in emb.index]
    if not names:
        raise ValueError("No filenames shared by labels, features and embeddings")
    if len(names) < len(label):
        log.warning("%d labelled images have no features/embeddings — skipped", len(label) - len(names))
    label = label.set_index("filename").loc[names].reset_index()

    det = feat.loc[names].select_dtypes("number").fillna(0.0)
    clip = emb.loc[names]
    X = {"detection": det, "clip": clip, "hybrid": pd.concat([clip, det], axis=1)}
    return label, {k: v.reset_index(drop=True) for k, v in X.items()}


def make_models(config, seed):
    """Return {model key: unfitted estimator}."""
    cfg    = config.get("model", {})
    xgb    = cfg.get("xgboost", {})
    rf     = cfg.get("random_forest", {})
    alphas = [float(a) for a in cfg.get("ridge_alphas", [0.1, 1, 10, 100, 1000])]
    ridge  = lambda: Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=alphas))])
    return {
        "dummy_mean":        DummyRegressor(strategy="mean"),
        "linear_regression": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "ridge_regression":  ridge(),
        "random_forest": RandomForestRegressor(n_estimators=int(rf.get("n_estimators",200)),
                            max_depth=rf.get("max_depth",6), random_state=seed, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=int(xgb.get("n_estimators",200)),
                    max_depth=int(xgb.get("max_depth",4)),
                    learning_rate=float(xgb.get("learning_rate",0.08)),
                    subsample=float(xgb.get("subsample",0.8)),
                    colsample_bytree=float(xgb.get("colsample_bytree",0.8)),
                    min_child_weight=int(xgb.get("min_child_weight",2)),
                    random_state=seed, verbosity=0),
        "clip_ridge":   ridge(),
        "hybrid_ridge": ridge(),
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
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2*len(preds), 4), squeeze=False)
    for ax, (name, yp) in zip(axes[0], preds.items()):
        ax.scatter(y_true, yp, alpha=0.5, s=30, edgecolors="none")
        ax.plot([1, 5], [1, 5], "--", color="gray", lw=1)
        ax.set_xlim(0.8, 5.2); ax.set_ylim(0.8, 5.2)
        ax.set_title(f"{name}\nMAE={mean_absolute_error(y_true,yp):.3f}  ρ={spearmanr(y_true,yp)[0]:.3f}", fontsize=9)
        ax.set_xlabel("Rated"); ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)
    fig.suptitle("Predicted vs Rated (out-of-fold)", fontsize=12)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Scatter chart → %s", path)

def plot_comparison(results, target, path):
    """Bar chart of CV MAE and Spearman-ρ (mean ± std) for every model on one target."""
    r = results[results["target"] == target].set_index("model")
    keys = [k for k in MODEL_INFO if k in r.index]
    names = [MODEL_INFO[k][0] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, met in zip(axes, ["MAE", "Spearman_r"]):
        vals = np.nan_to_num(r.loc[keys, f"{met}_mean"].values)
        errs = np.nan_to_num(r.loc[keys, f"{met}_std"].values)
        colors = ["#9aa0a6" if k == "dummy_mean" else "#4c72b0" if MODEL_INFO[k][1] == "detection" else "#dd8452" for k in keys]
        ax.barh(names, vals, xerr=errs, color=colors, capsize=3, edgecolor="none")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
        ax.set_title(f"{met} — {target} (lower is better)" if met == "MAE" else f"{met} — {target} (higher is better)", fontsize=10)
    fig.suptitle("Repeated K-fold CV (mean ± std). Blue = YOLO features, orange = CLIP embeddings", fontsize=11)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info("✓ Comparison chart → %s", path)


def train_and_evaluate(features_path, embeddings_path, labels_path, models_dir, outputs_dir, config):
    """Cross-validate every model on every target, print results, refit on all data, save models and charts."""
    cfg      = config.get("model", {})
    primary  = cfg.get("target_column", "overall_score")
    targets  = cfg.get("targets", [primary])
    seed     = int(cfg.get("random_state", 42))
    folds    = int(cfg.get("cv_folds", 5))
    repeats  = int(cfg.get("cv_repeats", 5))
    labels, X = load_data(features_path, embeddings_path, labels_path)
    log.info("%d images | targets: %s | %d×%d-fold CV", len(labels), targets, repeats, folds)

    rkf, rows, oof = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed), [], {}
    for target in targets:
        y = labels[target].astype(float)
        (models_dir/target).mkdir(parents=True, exist_ok=True)
        for key, est in make_models(config, seed).items():
            Xs = X[MODEL_INFO[key][1]]
            fold_scores = []
            for tr, te in rkf.split(Xs):
                m = clone(est).fit(Xs.iloc[tr], y.iloc[tr])
                fold_scores.append(metrics(y.iloc[te].values, m.predict(Xs.iloc[te])))
            fs = pd.DataFrame(fold_scores)
            rows.append({"target": target, "model": key, "feature_set": MODEL_INFO[key][1],
                         **{f"{c}_mean": fs[c].mean() for c in fs.columns},
                         **{f"{c}_std":  fs[c].std()  for c in fs.columns}})
            if target == primary:   # one K-fold pass of out-of-fold predictions for plots and the CLIP comparison
                oof[key] = cross_val_predict(clone(est), Xs, y, cv=KFold(folds, shuffle=True, random_state=seed))

            if key == "dummy_mean":
                continue
            est.fit(Xs, y)   # final model is refit on all labelled data
            joblib.dump({"model": est, "feature_set": MODEL_INFO[key][1], "feature_cols": list(Xs.columns),
                         "target": target}, models_dir/target/f"{key}.joblib")
        log.info("✓ %s: models saved → %s", target, models_dir/target)

    results = pd.DataFrame(rows)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(outputs_dir/"cv_results.csv", index=False)

    # Print summary tables
    cols = ["MAE", "RMSE", "R2", "Spearman_r", "Kendall_t"]
    for target in targets:
        r = results[results["target"] == target]
        print(f"\n{target}\n" + "-"*104)
        print(f"{'Model':<29}" + "".join(f"{c:>15}" for c in cols))
        print("-"*104)
        for _, row in r.iterrows():
            print(f"{MODEL_INFO[row['model']][0]:<29}" + "".join(f"{row[c+'_mean']:>8.3f} ±{row[c+'_std']:>5.3f}" for c in cols))
        best = r[r["model"] != "dummy_mean"].sort_values("MAE_mean").iloc[0]
        dummy = r[r["model"] == "dummy_mean"].iloc[0]
        log.info("Best for %s: %s (CV MAE=%.3f vs dummy %.3f)", target, MODEL_INFO[best["model"]][0],
                 best["MAE_mean"], dummy["MAE_mean"])
    log.info("✓ CV results → %s", outputs_dir/"cv_results.csv")

    oof_df = pd.DataFrame({"filename": labels["filename"], primary: labels[primary].values, **oof})
    oof_df.to_csv(outputs_dir/"oof_predictions.csv", index=False)
    log.info("✓ Out-of-fold predictions → %s", outputs_dir/"oof_predictions.csv")

    # Charts for the primary target
    for key in ["random_forest", "xgboost"]:
        payload = joblib.load(models_dir/primary/f"{key}.joblib")
        plot_importance(payload["model"], payload["feature_cols"],
                        outputs_dir/f"feature_importance_{key}.png", f"{MODEL_INFO[key][0]} — {primary}")
    plot_scatter(labels[primary].values,
                 {MODEL_INFO[k][0]: oof[k] for k in ["ridge_regression", "xgboost", "clip_ridge", "hybrid_ridge"]},
                 outputs_dir/"predicted_vs_actual.png")
    plot_comparison(results, primary, outputs_dir/"model_comparison.png")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features",   type=Path); p.add_argument("--embeddings", type=Path)
    p.add_argument("--labels",     type=Path); p.add_argument("--config",     type=Path)
    args = p.parse_args()

    config = load_config(args.config)
    paths  = config.get("paths", {})
    train_and_evaluate(
        features_path   = args.features   or ROOT / paths.get("features_csv",   "features/features.csv"),
        embeddings_path = args.embeddings or ROOT / paths.get("embeddings_npz", "features/clip_embeddings.npz"),
        labels_path     = args.labels     or ROOT / paths.get("labels_csv",     "data/labels.csv"),
        models_dir      = ROOT / paths.get("models_dir",  "models"),
        outputs_dir     = ROOT / paths.get("outputs_dir", "outputs"),
        config          = config,
    )

if __name__ == "__main__":
    main()
