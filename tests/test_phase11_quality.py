"""Phase 11: final quality / end-to-end checks.

Covers: config validator, data ranges, SHACL conformance, SPARQL returns rows,
models load and predict, optimizer returns non-empty Pareto front.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pyshacl import validate
from rdflib import Graph, Namespace, RDF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    Formulation,
    NUMERIC_FEATURES,
    TARGET_PROPERTIES,
    USE_CASES,
    validate_formulation,
)
from src import kg_query  # noqa: E402
from src import optimize  # noqa: E402

NC = Namespace("http://example.org/nanocomposite#")


# --- config validator ---
def test_config_validator_accepts_valid():
    f = Formulation(
        matrix_polymer="HDPE", recycled_content_pct=50.0, agro_filler="none",
        filler_wt_pct=0.0, filler_particle_size_um=0.0, nanoparticle="SiO2",
        nano_wt_pct=2.0, nano_size_nm=30.0, surface_treatment="silane",
        extrusion_temp_C=190.0, screw_speed_rpm=100.0, sonication_min=20.0,
        source="user", reference="q",
    )
    assert validate_formulation(f) == []


def test_config_validator_rejects_range_violation():
    f = Formulation(
        matrix_polymer="HDPE", recycled_content_pct=150.0, agro_filler="none",
        filler_wt_pct=0.0, filler_particle_size_um=0.0, nanoparticle="none",
        nano_wt_pct=0.0, nano_size_nm=0.0, surface_treatment="none",
        extrusion_temp_C=190.0, screw_speed_rpm=100.0, sonication_min=20.0,
        source="synthetic", reference="q",
    )
    assert validate_formulation(f)


# --- data ranges ---
def test_dataset_within_ranges():
    df = pd.read_csv("data/processed/dataset.csv")
    for name, (lo, hi, _unit) in NUMERIC_FEATURES.items():
        if name not in df.columns:
            continue
        col = df[name]
        # 0 is a legitimate N/A sentinel (no filler / no nanoparticle);
        # all non-zero values must lie within [lo, hi].
        nonzero = col[col != 0]
        if len(nonzero):
            assert nonzero.min() >= lo, f"{name} below {lo}"
            assert nonzero.max() <= hi, f"{name} above {hi}"


# --- SHACL conformance ---
def test_shacl_conformance():
    g = Graph(); g.parse("kg/graph.ttl", format="turtle")
    sg = Graph(); sg.parse("kg/shapes.ttl", format="turtle")
    conforms, _, _ = validate(g, shacl_graph=sg, abort_on_error=False)
    assert conforms, "KG not SHACL-conformant"


# --- SPARQL returns rows ---
def test_sparql_queries_return_rows():
    for name in [
        "a_insulation_panel", "b_avg_tensile_by_filler",
        "c_recycled_matrix_high_filler", "d_top10_lowest_gwp",
    ]:
        df = kg_query.run_query(name)
        assert len(df) > 0, f"{name} returned no rows"


# --- models load and predict ---
def test_models_load_and_predict():
    df = pd.read_csv("data/processed/features.csv")
    exclude = set(TARGET_PROPERTIES.keys()) | {"source", "reference", "sample"}
    fcols = [c for c in df.columns if c not in exclude]
    X = df[fcols].fillna(0.0).head(3)
    for target in TARGET_PROPERTIES:
        p = Path("models") / f"{target}.joblib"
        assert p.exists(), f"missing {p}"
        model = joblib.load(p)
        pred = model.predict(X)
        assert len(pred) == 3
        assert all(np.isfinite(pred))


# --- optimizer returns non-empty Pareto front ---
def test_optimizer_nonempty_pareto():
    pareto = optimize.optimize(use_case="insulation_panel", n_trials=40)
    assert isinstance(pareto, pd.DataFrame)
    assert len(pareto) > 0
    for c in ["tensile_strength_MPa", "thermal_conductivity_W_mK",
              "gwp_kgCO2e_per_kg", "water_absorption_pct"]:
        assert c in pareto.columns
