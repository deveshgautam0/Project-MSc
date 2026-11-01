"""Phase 7 tests: optimization returns non-empty Pareto front."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import optimize  # noqa: E402
from config import USE_CASES  # noqa: E402

PARETO_CSV = Path("reports/pareto.csv")
FIG_2D = Path("reports/figures/pareto_2d.html")
FIG_3D = Path("reports/figures/pareto_3d.html")


def test_models_exist_for_optimize():
    from pathlib import Path as P
    assert any(P("models").glob("*.joblib")), "no models (run src.train)"


def test_optimize_returns_nonempty_pareto():
    pareto = optimize.optimize(use_case="insulation_panel", n_trials=50)
    assert isinstance(pareto, pd.DataFrame)
    assert len(pareto) > 0, "Pareto front empty"
    for c in ["tensile_strength_MPa", "thermal_conductivity_W_mK",
              "gwp_kgCO2e_per_kg", "water_absorption_pct"]:
        assert c in pareto.columns


def test_pareto_respects_constraints():
    pareto = optimize.optimize(use_case="structural_decking", n_trials=50)
    if pareto.empty:
        pytest.skip("empty pareto, cannot check constraints")
    for _, row in pareto.iterrows():
        # structural_decking: tensile >= 25, water_absorption <= 5
        assert row["tensile_strength_MPa"] >= 25 - 1e-6, \
            f"tensile {row['tensile_strength_MPa']} < 25"
        assert row["water_absorption_pct"] <= 5 + 1e-6, \
            f"water {row['water_absorption_pct']} > 5"


def test_pareto_files_saved():
    if not PARETO_CSV.exists():
        pytest.skip("pareto not built yet")
    assert PARETO_CSV.exists()
    assert FIG_2D.exists(), "2D pareto figure missing"
    assert FIG_3D.exists(), "3D pareto figure missing"


def test_invalid_use_case_raises():
    with pytest.raises(ValueError):
        optimize.optimize(use_case="nonexistent_use_case", n_trials=10)
