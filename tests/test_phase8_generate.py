"""Phase 8 tests: constrained candidate generation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import generate  # noqa: E402
from config import USE_CASES  # noqa: E402


def test_pareto_exists():
    from pathlib import Path as P
    assert P("reports/pareto.csv").exists(), "run src.optimize first"


def test_generate_candidates_returns_ranked():
    df = generate.generate_candidates("insulation_panel", n=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "no candidates generated"
    assert "sustainability_score" in df.columns
    for c in ["tensile_strength_MPa", "thermal_conductivity_W_mK",
              "gwp_kgCO2e_per_kg", "water_absorption_pct"]:
        assert c in df.columns
    # ranked: first row has the max sustainability score
    assert df["sustainability_score"].iloc[0] == df["sustainability_score"].max()


def test_generate_meets_use_case_constraints():
    df = generate.generate_candidates("structural_decking", n=10)
    if df.empty:
        pytest.skip("no candidates for structural_decking")
    for _, row in df.iterrows():
        assert row["tensile_strength_MPa"] >= 25 - 1e-6
        assert row["water_absorption_pct"] <= 5 + 1e-6


def test_invalid_use_case_raises():
    with pytest.raises(ValueError):
        generate.generate_candidates("nonexistent_use_case", n=5)
