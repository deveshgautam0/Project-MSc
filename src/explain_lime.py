"""src/explain_lime.py — Gap 6: LIME explainability for ML predictions.

Provides local interpretable model-agnostic explanations for the trained
Random Forest / XGBoost models, complementing the global SHAP explanations in
src/explain.py.

Wrapped in try/except so the module imports without the `lime` package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Optional LIME — wrapped so the module imports without it.
try:
    from lime.lime_tabular import LimeTabularExplainer
    _HAS_LIME = True
except Exception:  # noqa: BLE001
    _HAS_LIME = False


class LIMEExplainer:
    """Local explanations for individual formulation predictions."""

    def __init__(self, model, X_train: np.ndarray,
                 feature_names: list[str],
                 class_names: list[str] | None = None,
                 mode: str = "regression"):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.explainer: Any = None

        if _HAS_LIME:
            self.explainer = LimeTabularExplainer(
                training_data=X_train,
                feature_names=feature_names,
                class_names=class_names,
                mode=mode,
                random_state=42,
            )

    def explain_instance(self, X_row: np.ndarray,
                         num_features: int = 10) -> dict | None:
        """Return a LIME explanation dict for a single row.

        Returns None if lime is not installed.
        """
        if not _HAS_LIME or self.explainer is None:
            return None

        exp = self.explainer.explain_instance(
            X_row, self.model.predict, num_features=num_features
        )
        return {
            "feature": [f for f, _ in exp.as_list()],
            "weight": [w for _, w in exp.as_list()],
            "intercept": exp.intercept,
            "prediction": exp.predicted_value,
        }

    def explain_dataframe(self, df: pd.DataFrame,
                          num_features: int = 10) -> list[dict | None]:
        """Explain every row of a DataFrame."""
        X = df[self.feature_names].values
        return [self.explain_instance(row, num_features) for row in X]


def make_dummy_lime_data(n: int = 50, n_features: int = 8) -> tuple[np.ndarray, list[str]]:
    """Create synthetic data for smoke-testing LIME."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, n_features))
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, feature_names