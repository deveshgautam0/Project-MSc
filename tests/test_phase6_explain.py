"""Phase 6 tests: SHAP plots + explain_ontology."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import explain  # noqa: E402
from config import TARGET_PROPERTIES  # noqa: E402

FIGURES_DIR = Path("reports/figures")
MODELS_DIR = Path("models")


def test_models_exist():
    assert any(MODELS_DIR.glob("*.joblib")), "no models (run src.train)"


def test_shap_plots_exist():
    if not any(MODELS_DIR.glob("*.joblib")):
        pytest.skip("no models yet")
    for target in TARGET_PROPERTIES:
        summary = FIGURES_DIR / f"{target}_shap_summary.png"
        bar = FIGURES_DIR / f"{target}_shap_bar.png"
        assert summary.exists(), f"missing {summary}"
        assert bar.exists(), f"missing {bar}"


def test_explain_ontology_triggered():
    formulation = {
        "matrix_polymer": "PET_recycled",
        "agro_filler": "rice_husk",
        "nanoparticle": "SiO2",
        "surface_treatment": "silane",
        "nano_wt_pct": 2.0,
    }
    out = explain.explain_ontology(formulation)
    assert isinstance(out, list) and len(out) > 0
    joined = " ".join(out)
    assert "rice_husk" in joined
    assert "PET_recycled" in joined


def test_explain_ontology_empty():
    out = explain.explain_ontology({"matrix_polymer": "HDPE", "agro_filler": "none"})
    assert isinstance(out, list) and len(out) >= 1
