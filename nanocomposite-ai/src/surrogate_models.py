"""src/surrogate_models.py — Gap 4: Gaussian Process / Polynomial Regression
surrogates for nanoprocess simulation.

Provides cheap surrogate models that map processing parameters (extrusion
temperature, screw speed, sonication time, filler loading, etc.) to predicted
properties. Used for fast "what-if" simulation before committing to lab trials.

Backed by scikit-learn (GaussianProcessRegressor + PolynomialFeatures), so no
extra dependencies beyond the base requirements.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class NanoprocessSurrogate:
    """Surrogate model for nanoprocess parameter → property prediction.

    Supports two backends:
      - "gp": Gaussian Process Regression (uncertainty-aware, good for small data)
      - "poly": Polynomial Regression + Ridge (good for larger data, faster)
    """

    def __init__(self, backend: str = "gp", degree: int = 2, alpha: float = 1.0):
        self.backend = backend
        self.degree = degree
        self.alpha = alpha
        self.model: BaseEstimator | RegressorMixin | None = None
        self.scaler_x: Any = StandardScaler()
        self.scaler_y: Any = StandardScaler()
        self._feature_names: list[str] = []

    def _build_model(self):
        if self.backend == "gp":
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
                noise_level=0.1
            )
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
            )
        elif self.backend == "poly":
            return Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=self.alpha, random_state=42)),
                ]
            )
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Use 'gp' or 'poly'.")

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None) -> "NanoprocessSurrogate":
        self._feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self.model = self._build_model()
        self.model.fit(Xs, ys)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        Xs = self.scaler_x.transform(X)
        ys = self.model.predict(Xs)
        return self.scaler_y.inverse_transform(ys.reshape(-1, 1)).ravel()

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty (GP only; Poly returns zeros for std)."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        Xs = self.scaler_x.transform(X)
        if self.backend == "gp" and hasattr(self.model, "predict"):
            mean, std = self.model.predict(Xs, return_std=True)
            mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).ravel()
            std = std * self.scaler_y.scale_[0]
            return mean, std
        else:
            mean = self.predict(X)
            return mean, np.zeros_like(mean)

    def feature_importance(self) -> dict[str, float] | None:
        """Approximate feature importance (Poly backend only, via ridge coefs)."""
        if self.model is None or self.backend != "poly":
            return None
        ridge = self.model.named_steps["ridge"]
        coefs = np.abs(ridge.coef_)
        names = self.model.named_steps["poly"].get_feature_names_out(
            self._feature_names
        )
        return dict(zip(names, coefs))


def make_dummy_process_data(n: int = 100) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic nanoprocess data for smoke-testing."""
    rng = np.random.default_rng(42)
    X = np.column_stack([
        rng.uniform(160, 230, n),   # extrusion_temp_C
        rng.uniform(30, 150, n),    # screw_speed_rpm
        rng.uniform(0, 60, n),      # sonication_min
        rng.uniform(0, 40, n),      # filler_wt_pct
        rng.uniform(0, 5, n),       # nano_wt_pct
    ])
    # Synthetic property: tensile strength depends non-linearly on params
    y = (
        20.0
        + 0.05 * X[:, 0]
        + 0.1 * X[:, 1]
        + 0.2 * X[:, 2]
        + 0.5 * X[:, 3]
        + 0.3 * X[:, 4]
        - 0.001 * X[:, 0] * X[:, 3]
        + rng.normal(0, 2.0, n)
    )
    feature_names = [
        "extrusion_temp_C",
        "screw_speed_rpm",
        "sonication_min",
        "filler_wt_pct",
        "nano_wt_pct",
    ]
    return X, y, feature_names