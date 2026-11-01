"""ML property prediction (Phase 5).

For each target property, trains RandomForestRegressor and XGBRegressor,
evaluates with 5-fold CV plus a held-out 20% test split. Splits are
source-stratified (synthetic vs non-synthetic) so synthetic rows never leak
between train/test in a misleading way, and metrics are reported on
non-synthetic rows only when there are >=10 of them.

Saves:
  - reports/metrics.json       (MAE, RMSE, R2 per target per model)
  - reports/metrics.md         (markdown table)
  - models/<target>.joblib     (best model per target)
  - reports/figures/<target>_pred_vs_actual.png

Run: python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from config import (  # noqa: E402
    INPUT_FEATURES,
    TARGET_PROPERTIES,
)

FEATURES_PATH = Path("data/processed/features.csv")
METRICS_JSON = Path("reports/metrics.json")
METRICS_MD = Path("reports/metrics.md")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2
NON_SYNTHETIC_MIN = 10

MODELS: dict[str, object] = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
}


def _mae_rmse_r2(y_true, y_pred) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _cv_scores(model, X, y, n_splits: int = N_SPLITS) -> dict[str, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    maes, rmses, r2s = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        pred = model_clone.predict(X_val)
        m = _mae_rmse_r2(y_val, pred)
        maes.append(m["MAE"]); rmses.append(m["RMSE"]); r2s.append(m["R2"])
    return {
        "CV_MAE": float(np.mean(maes)),
        "CV_RMSE": float(np.mean(rmses)),
        "CV_R2": float(np.mean(r2s)),
    }


def _try_autogluon(X_train, y_train, X_test, y_test) -> dict | None:
    """Optional AutoGluon baseline (wrapped in try/except)."""
    try:
        from autogluon.tabular import TabularPredictor  # noqa: E402
    except ImportError:
        return None
    train_df = X_train.copy()
    train_df["target"] = y_train.values
    try:
        predictor = TabularPredictor(label="target", problem_type="regression").fit(
            train_df, presets="medium_quality", verbosity=0
        )
        test_df = X_test.copy()
        pred = predictor.predict(test_df)
        return _mae_rmse_r2(y_test, pred)
    except Exception as exc:  # noqa: BLE001
        print(f"  AutoGluon baseline failed: {exc}")
        return None


def _try_h2o(X_train, y_train, X_test, y_test) -> dict | None:
    """Optional H2O baseline (wrapped in try/except)."""
    try:
        import h2o  # noqa: E402
        from h2o.automl import H2OAutoML  # noqa: E402
    except ImportError:
        return None
    try:
        h2o.init(max_mem_size="2G", nthreads=-1)
        train = X_train.copy()
        train["target"] = y_train.values
        test = X_test.copy()
        test["target"] = y_test.values
        train_h = h2o.H2OFrame(train)
        test_h = h2o.H2OFrame(test)
        aml = H2OAutoML(max_models=5, seed=RANDOM_STATE, verbosity="warning")
        aml.train(x=list(X_train.columns), y="target", training_frame=train_h)
        pred = aml.leader.predict(test_h).as_data_frame().values.flatten()
        h2o.shutdown(prompt=False)
        return _mae_rmse_r2(y_test, pred)
    except Exception as exc:  # noqa: BLE001
        print(f"  H2O baseline failed: {exc}")
        return None


def train_target(target: str, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train models for one target. Returns metrics dict."""
    print(f"\n=== Target: {target} ===")
    cols = feature_cols + [target]
    if "source" in df.columns:
        cols.append("source")
    data = df[cols].dropna()
    if len(data) < 20:
        print(f"  SKIPPED: only {len(data)} rows")
        return {}

    source_col = "source" if "source" in df.columns else None
    is_syn = data["source"].eq("synthetic") if source_col else pd.Series(False, index=data.index)

    # Source-stratified split: train/test each get a share of synthetic &
    # non-synthetic so neither side is purely one source.
    syn_idx = data.index[is_syn]
    non_syn_idx = data.index[~is_syn]

    train_parts, test_parts = [], []
    for idx in (syn_idx, non_syn_idx):
        if len(idx) == 0:
            continue
        tr, te = train_test_split(
            idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        train_parts.append(tr); test_parts.append(te)

    train_idx = np.concatenate(train_parts) if train_parts else np.array([])
    test_idx = np.concatenate(test_parts) if test_parts else np.array([])

    X_train = data.loc[train_idx, feature_cols]
    y_train = data.loc[train_idx, target]
    X_test = data.loc[test_idx, feature_cols]
    y_test = data.loc[test_idx, target]

    print(f"  train={len(X_train)} test={len(X_test)} "
          f"(non-synthetic test={int((~data.loc[test_idx, 'source'].eq('synthetic')).sum()) if source_col else 0})")

    results: dict[str, dict] = {}
    best_name = None
    best_rmse = np.inf
    best_model = None

    for name, model in MODELS.items():
        cv = _cv_scores(model, X_train, y_train)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        test_m = _mae_rmse_r2(y_test, pred)
        results[name] = {"cv": cv, "test": test_m}
        print(f"  {name}: CV_R2={cv['CV_R2']:.3f} Test_R2={test_m['R2']:.3f} "
              f"Test_RMSE={test_m['RMSE']:.3f}")
        if test_m["RMSE"] < best_rmse:
            best_rmse = test_m["RMSE"]
            best_name = name
            best_model = model

    # Non-synthetic-only test metrics (if enough rows)
    non_syn_test_mask = ~data.loc[test_idx, "source"].eq("synthetic") if source_col else pd.Series(False, index=y_test.index)
    non_syn_metrics = None
    if non_syn_test_mask.sum() >= NON_SYNTHETIC_MIN:
        y_ns = y_test[non_syn_test_mask]
        # Re-predict on non-synthetic subset using best model
        X_ns = X_test[non_syn_test_mask]
        pred_ns = best_model.predict(X_ns)
        non_syn_metrics = _mae_rmse_r2(y_ns, pred_ns)
        print(f"  non-synthetic test (n={len(y_ns)}): "
              f"R2={non_syn_metrics['R2']:.3f} RMSE={non_syn_metrics['RMSE']:.3f}")

    # Optional baselines
    ag = _try_autogluon(X_train, y_train, X_test, y_test)
    h2o_res = _try_h2o(X_train, y_train, X_test, y_test)
    if ag:
        print(f"  AutoGluon test: R2={ag['R2']:.3f} RMSE={ag['RMSE']:.3f}")
    if h2o_res:
        print(f"  H2O test: R2={h2o_res['R2']:.3f} RMSE={h2o_res['RMSE']:.3f}")

    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{target}.joblib"
    joblib.dump(best_model, model_path)
    print(f"  best model ({best_name}) -> {model_path}")

    # Plot predicted vs actual
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"{target}_pred_vs_actual.png"
    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, best_model.predict(X_test), alpha=0.6, edgecolor="k", linewidth=0.3)
    lo = min(y_test.min(), best_model.predict(X_test).min())
    hi = max(y_test.max(), best_model.predict(X_test).max())
    plt.plot([lo, hi], [lo, hi], "r--", label="ideal")
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{target} — {best_name} (R2={results[best_name]['test']['R2']:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"  plot -> {fig_path}")

    return {
        "target": target,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_non_synthetic_test": int(non_syn_test_mask.sum()) if source_col else 0,
        "best_model": best_name,
        "models": results,
        "non_synthetic_metrics": non_syn_metrics,
    }


def main() -> None:
    """CLI entry: python -m src.train"""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH} (run src.features first)"
        )
    df = pd.read_csv(FEATURES_PATH)

    # feature columns: everything except targets + source/reference/sample
    exclude = set(TARGET_PROPERTIES.keys()) | {"source", "reference", "sample"}
    feature_cols = [c for c in df.columns if c not in exclude]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    for target in TARGET_PROPERTIES:
        m = train_target(target, df, feature_cols)
        if m:
            all_metrics.append(m)

    # Write metrics.json
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics -> {METRICS_JSON}")

    # Write metrics.md
    lines = ["# ML Metrics", "", "| Target | Model | CV_R2 | Test_R2 | Test_RMSE | Test_MAE |", "|---|---|---|---|---|---|"]
    for m in all_metrics:
        for name, res in m["models"].items():
            lines.append(
                f"| {m['target']} | {name} | {res['cv']['CV_R2']:.3f} | "
                f"{res['test']['R2']:.3f} | {res['test']['RMSE']:.3f} | "
                f"{res['test']['MAE']:.3f} |"
            )
    METRICS_MD.write_text("\n".join(lines))
    print(f"Metrics table -> {METRICS_MD}")


if __name__ == "__main__":
    main()
