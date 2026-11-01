"""Phase 4 tests: SPARQL queries + features from KG."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import kg_query  # noqa: E402
from src import features  # noqa: E402

QUERY_NAMES = [
    "a_insulation_panel",
    "b_avg_tensile_by_filler",
    "c_recycled_matrix_high_filler",
    "d_top10_lowest_gwp",
]


def test_kg_graph_exists():
    assert Path("kg/graph.ttl").exists(), "kg/graph.ttl missing (run src.kg_build)"


@pytest.mark.parametrize("name", QUERY_NAMES)
def test_run_query_returns_rows(name):
    df = kg_query.run_query(name)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, f"query {name} returned no rows"


def test_run_query_invalid_name_raises():
    with pytest.raises(FileNotFoundError):
        kg_query.run_query("nonexistent_query")


def test_features_csv_exists():
    assert Path("data/processed/features.csv").exists(), "features.csv missing"


def test_features_built_from_kg():
    df = features.load_features_from_kg()
    assert len(df) > 0
    # All input features + targets present
    for c in [
        "matrix_polymer",
        "recycled_content_pct",
        "agro_filler",
        "filler_wt_pct",
        "nanoparticle",
        "tensile_strength_MPa",
        "gwp_kgCO2e_per_kg",
    ]:
        assert c in df.columns, f"missing column {c}"


def test_features_has_graph_derived_flags():
    df = pd.read_csv("data/processed/features.csv")
    assert "matrix_is_recycled" in df.columns
    assert "filler_is_lignocellulosic" in df.columns
    assert set(df["matrix_is_recycled"].unique()).issubset({0, 1})
    assert set(df["filler_is_lignocellulosic"].unique()).issubset({0, 1})


def test_features_one_hot_encoded():
    df = pd.read_csv("data/processed/features.csv")
    assert any(c.startswith("matrix_polymer_") for c in df.columns)
    assert any(c.startswith("agro_filler_") for c in df.columns)
    assert any(c.startswith("nanoparticle_") for c in df.columns)
    # Raw categoricals dropped after encoding
    assert "matrix_polymer" not in df.columns


def test_encoder_saved():
    assert Path("models/feature_encoder.joblib").exists()
    cols = __import__("joblib").load("models/feature_encoder.joblib")
    assert isinstance(cols, list) and len(cols) > 0
