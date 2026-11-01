"""Explainability (Phase 6).

SHAP summary + bar plots per target using TreeExplainer, saved to
reports/figures/<target>_shap_summary.png and _shap_bar.png.

Also provides explain_ontology(formulation) -> plain-English reasoning from
KG facts using the rule table in config.ONTOLOGY_RULES.

Run: python -m src.explain
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import (  # noqa: E402
    INPUT_FEATURES,
    ONTOLOGY_RULES,
    TARGET_PROPERTIES,
)

FEATURES_PATH = Path("data/processed/features.csv")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")

EXCLUDE = set(TARGET_PROPERTIES.keys()) | {"source", "reference", "sample"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE]


def _load_data_and_model(target: str) -> tuple[pd.DataFrame, object, list[str]]:
    df = pd.read_csv(FEATURES_PATH)
    model_path = MODELS_DIR / f"{target}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path} (run src.train first)")
    model = joblib.load(model_path)
    feature_cols = _feature_cols(df)
    X = df[feature_cols].fillna(0.0)
    return df, model, feature_cols


def explain_target(target: str) -> None:
    """SHAP summary + bar plots for one target."""
    df, model, feature_cols = _load_data_and_model(target)
    X = df[feature_cols].fillna(0.0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Summary plot (beeswarm)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X, feature_names=feature_cols, show=False, max_display=15
    )
    plt.title(f"{target} — SHAP summary")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{target}_shap_summary.png", dpi=120)
    plt.close()

    # Bar plot (mean |SHAP|)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X, feature_names=feature_cols, plot_type="bar",
        show=False, max_display=15,
    )
    plt.title(f"{target} — SHAP bar")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{target}_shap_bar.png", dpi=120)
    plt.close()

    print(f"SHAP plots saved for {target}")


def explain_ontology(formulation: dict) -> list[str]:
    """Return plain-English ontology reasoning for a formulation.

    Uses config.ONTOLOGY_RULES matched against formulation fields.
    """
    reasoning: list[str] = []
    for condition, explanation in ONTOLOGY_RULES:
        # Simple substring/keyword matching on condition text against formulation.
        matched = _condition_matches(condition, formulation)
        if matched:
            reasoning.append(f"[{condition}] {explanation}")
    if not reasoning:
        reasoning.append("No specific ontology rules triggered for this formulation.")
    return reasoning


def _condition_matches(condition: str, formulation: dict) -> bool:
    """Evaluate a simple condition string against a formulation dict."""
    cond = condition.strip()
    if " == " in cond:
        key, val = cond.split(" == ", 1)
        key = key.strip(); val = val.strip()
        return str(formulation.get(key, "")) == val
    if " != " in cond:
        key, val = cond.split(" != ", 1)
        key = key.strip(); val = val.strip()
        return str(formulation.get(key, "")) != val
    if " in " in cond:
        key, rest = cond.split(" in ", 1)
        key = key.strip()
        vals = [v.strip() for v in rest.strip("()").split(",")]
        return str(formulation.get(key, "")) in vals
    if " > " in cond:
        key, val = cond.split(" > ", 1)
        try:
            return float(formulation.get(key, 0)) > float(val)
        except (TypeError, ValueError):
            return False
    if " < " in cond:
        key, val = cond.split(" < ", 1)
        try:
            return float(formulation.get(key, 0)) < float(val)
        except (TypeError, ValueError):
            return False
    return False


def main() -> None:
    """CLI entry: python -m src.explain"""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH} (run src.features first)"
        )
    if not list(MODELS_DIR.glob("*.joblib")):
        raise FileNotFoundError(
            f"No models in {MODELS_DIR} (run src.train first)"
        )

    for target in TARGET_PROPERTIES:
        model_path = MODELS_DIR / f"{target}.joblib"
        if not model_path.exists():
            print(f"SKIP {target}: no model")
            continue
        explain_target(target)

    # Demo explain_ontology on a sample formulation
    demo = {
        "matrix_polymer": "PET_recycled",
        "agro_filler": "rice_husk",
        "nanoparticle": "SiO2",
        "surface_treatment": "silane",
        "nano_wt_pct": 2.0,
    }
    print("\n--- Ontology reasoning (demo) ---")
    for line in explain_ontology(demo):
        print("  -", line)


if __name__ == "__main__":
    main()
