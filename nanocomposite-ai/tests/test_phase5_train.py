"""Phase 5 tests: ML training, metrics, model saving."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TARGET_PROPERTIES  # noqa: E402

FEATURES_PATH = Path("data/processed/features.csv")
METRICS_JSON = Path("reports/metrics.json")
METRICS_MD = Path("reports/metrics.md")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")


def test_features_csv_exists():
    assert FEATURES_PATH.exists(), "features.csv missing (run src.features)"
    df = pd.read_csv(FEATURES_PATH)
    assert len(df) == 418, f"expected 418 rows, got {len(df)}"
    assert "source" in df.columns


def test_models_saved():
    if not METRICS_JSON.exists():
        pytest.skip("metrics.json not built yet (run src.train)")
    metrics = json_load(METRICS_JSON)
    for target in TARGET_PROPERTIES:
        model_path = MODELS_DIR / f"{target}.joblib"
        assert model_path.exists(), f"missing model {model_path}"
        m = joblib.load(model_path)
        # model should predict
        assert hasattr(m, "predict")


def test_metrics_json_structure():
    if not METRICS_JSON.exists():
        pytest.skip("metrics.json not built yet")
    metrics = json_load(METRICS_JSON)
    assert isinstance(metrics, list) and len(metrics) > 0
    for m in metrics:
        assert "target" in m
        assert "best_model" in m
        assert "models" in m
        for name, res in m["models"].items():
            assert "cv" in res and "test" in res
            for k in ("CV_R2", "CV_RMSE", "CV_MAE"):
                assert k in res["cv"], f"missing {k} in cv for {m['target']}/{name}"
            for k in ("R2", "RMSE", "MAE"):
                assert k in res["test"], f"missing {k} in test for {m['target']}/{name}"


def test_metrics_md_exists():
    if not METRICS_MD.exists():
        pytest.skip("metrics.md not built yet")
    text = METRICS_MD.read_text()
    assert "| Target |" in text


def test_pred_vs_actual_figures_exist():
    if not METRICS_JSON.exists():
        pytest.skip("metrics not built yet")
    metrics = json_load(METRICS_JSON)
    for m in metrics:
        fig = FIGURES_DIR / f"{m['target']}_pred_vs_actual.png"
        assert fig.exists(), f"missing figure {fig}"


def test_model_predicts_sensibly():
    if not METRICS_JSON.exists():
        pytest.skip("metrics not built yet")
    metrics = json_load(METRICS_JSON)
    df = pd.read_csv(FEATURES_PATH)
    exclude = set(TARGET_PROPERTIES.keys()) | {"source", "reference", "sample"}
    feature_cols = [c for c in df.columns if c not in exclude]
    for m in metrics:
        model_path = MODELS_DIR / f"{m['target']}.joblib"
        model = joblib.load(model_path)
        X = df[feature_cols].fillna(0.0).head(5)
        pred = model.predict(X)
        assert len(pred) == 5
        assert all(np.isfinite(p) for p in pred)


def json_load(p):
    import json
    return json.loads(p.read_text())


def test_non_synthetic_metrics_when_available():
    if not METRICS_JSON.exists():
        pytest.skip("metrics not built yet")
    metrics = json_load(METRICS_JSON)
    # With only 18 seed rows, non-synthetic test n may be < 10, so metrics may
    # be None. Just check structure doesn't crash.
    for m in metrics:
        ns = m.get("non_synthetic_metrics")
        if ns is not None:
            assert "R2" in ns and "RMSE" in ns
