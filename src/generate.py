"""Constrained design generation (Phase 8).

generate_candidates(use_case, n=20):
  1. Reads the Pareto front from reports/pareto.csv (src.optimize).
  2. Adds Latin-hypercube random samples from config.py ranges.
  3. Predicts properties using saved surrogate models (src.train).
  4. FILTERS with ontology rules:
       - valid ranges (config validator)
       - no forbidden additives (config.FORBIDDEN_ADDITIVES)
       - SHACL-valid when temporarily added to the knowledge graph
       - predicted properties meet use-case thresholds (config.USE_CASES)
  5. Returns a ranked DataFrame with predicted properties + a sustainability
     score (composite: low GWP, high recyclability, high biodegradability).

Run: python -m src.generate [use_case] [n]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FORBIDDEN_ADDITIVES,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    TARGET_PROPERTIES,
    USE_CASES,
    Formulation,
    validate_formulation,
)
from src import optimize  # noqa: E402
from src import kg_build  # noqa: E402  (for graph access / SHACL)

PARETO_CSV = Path("reports/pareto.csv")
RANDOM_STATE = 42

# Sustainability score weights (configurable, sum not required)
SUST_WEIGHTS = {
    "gwp_kgCO2e_per_kg": -1.0,   # lower GWP = better
    "recyclability_score": 1.0,   # higher = better
    "biodegradability_score": 1.0,  # higher = better
}


def _lhs_samples(n: int, rng: np.random.Generator) -> list[dict]:
    """Latin-hypercube random samples across config ranges."""
    samples: list[dict] = []
    cat_opts = CATEGORICAL_FEATURES
    cont_names = list(NUMERIC_FEATURES.keys())

    def lhs_col(n_bins: int) -> np.ndarray:
        """One LHS column stratified into n bins (values in [0,1))."""
        cut = np.linspace(0, 1, n + 1)
        u = rng.random(n)
        a = cut[:n]
        b = cut[1:n + 1]
        strat = a + (b - a) * u
        return rng.permutation(strat)

    n_cat = len(cat_opts["matrix_polymer"])
    n_af = len(cat_opts["agro_filler"])
    n_np = len(cat_opts["nanoparticle"])
    n_st = len(cat_opts["surface_treatment"])

    # Categorical LHS columns (each in [0,1))
    mat_col = lhs_col(n_cat)
    af_col = lhs_col(n_af)
    np_col = lhs_col(n_np)
    st_col = lhs_col(n_st)

    # Continuous LHS columns
    cont_cols = {}
    for name in cont_names:
        cont_cols[name] = lhs_col(len(cont_names))

    for i in range(n):
        form: dict = {}
        form["matrix_polymer"] = cat_opts["matrix_polymer"][
            min(int(mat_col[i] * n_cat), n_cat - 1)
        ]
        form["agro_filler"] = cat_opts["agro_filler"][
            min(int(af_col[i] * n_af), n_af - 1)
        ]
        form["nanoparticle"] = cat_opts["nanoparticle"][
            min(int(np_col[i] * n_np), n_np - 1)
        ]
        form["surface_treatment"] = cat_opts["surface_treatment"][
            min(int(st_col[i] * n_st), n_st - 1)
        ]
        for name in cont_names:
            lo, hi, _ = NUMERIC_FEATURES[name]
            form[name] = float(lo + cont_cols[name][i] * (hi - lo))
        # enforce N/A zeros
        if form["agro_filler"] == "none":
            form["filler_particle_size_um"] = 0.0
        if form["nanoparticle"] == "none":
            form["nano_wt_pct"] = 0.0
            form["nano_size_nm"] = 0.0
        form["source"] = "synthetic"
        form["reference"] = "generated"
        samples.append(form)
    return samples


def _is_shacl_valid(form: dict, pred: dict, g) -> bool:
    """Temporarily add a sample (inputs + predicted targets) to the graph and run SHACL."""
    from rdflib import Graph, Literal, Namespace, RDF, XSD

    NC = Namespace("http://example.org/nanocomposite#")
    g_test = Graph()
    g_test += g  # copy

    sample = NC["candidate_temp"]
    g_test.add((sample, RDF.type, NC.Nanocomposite))
    g_test.add((sample, RDF.type, NC.Material))

    matrix_map = {
        "HDPE": "HDPE", "LDPE": "LDPE", "PP": "PP",
        "PET_recycled": "PET_recycled", "PLA": "PLA",
    }
    filler_map = {
        "none": None, "rice_husk": "rice_husk", "wheat_straw": "wheat_straw",
        "sugarcane_bagasse": "sugarcane_bagasse", "coconut_coir": "coconut_coir",
    }
    nano_map = {
        "none": None, "SiO2": "SiO2", "nanoclay": "nanoclay",
        "TiO2": "TiO2", "cellulose_nanocrystal": "cellulose_nanocrystal",
    }

    g_test.add((sample, NC.hasMatrix, NC[matrix_map[form["matrix_polymer"]]]))
    filler_name = filler_map[form["agro_filler"]]
    if filler_name:
        g_test.add((sample, NC.hasFiller, NC[filler_name]))
        g_test.add((sample, NC.weightPercent,
                    Literal(float(form["filler_wt_pct"]), datatype=XSD.decimal)))
        if form["filler_particle_size_um"] and form["filler_particle_size_um"] > 0:
            g_test.add((sample, NC.particleSize,
                        Literal(float(form["filler_particle_size_um"]),
                                datatype=XSD.decimal)))
    nano_name = nano_map[form["nanoparticle"]]
    if nano_name:
        g_test.add((sample, NC.hasNanoparticle, NC[nano_name]))

    from src.kg_build import PROPERTY_CLASS
    for prop, unit in TARGET_PROPERTIES.items():
        pn = NC[f"candidate_temp_{prop}"]
        pc = PROPERTY_CLASS[prop]
        g_test.add((pn, RDF.type, NC[pc]))
        g_test.add((pn, RDF.type, NC.Property))
        g_test.add((sample, NC.hasProperty, pn))
        g_test.add((pn, NC.hasValue, Literal(float(pred[prop]), datatype=XSD.decimal)))
        g_test.add((pn, NC.hasUnit, Literal(unit)))

    # validate
    from pyshacl import validate
    shapes_g = Graph()
    shapes_g.parse(kg_build.SHAPES_PATH, format="turtle")
    conforms, _, _ = validate(
        g_test, shacl_graph=shapes_g, inference="rdfs",
        abort_on_error=False, meta_shacl=False, advanced=False, js=False,
        serialize_report=False,
    )
    return conforms


def _sustainability_score(pred: dict) -> float:
    """Composite sustainability score (higher = better)."""
    gwp = pred.get("gwp_kgCO2e_per_kg", 0.0)
    rec = pred.get("recyclability_score", 0.0)
    bio = pred.get("biodegradability_score", 0.0)
    # Normalize GWP by a rough scale; clip
    gwp_term = max(0.0, 1.0 - gwp / 5.0)
    return float(SUST_WEIGHTS["gwp_kgCO2e_per_kg"] * gwp
                 + SUST_WEIGHTS["recyclability_score"] * rec
                 + SUST_WEIGHTS["biodegradability_score"] * bio)


def generate_candidates(use_case: str, n: int = 20) -> pd.DataFrame:
    """Generate and filter candidate formulations for a use case.

    Returns a ranked DataFrame (top n) with predicted properties and a
    sustainability_score column.
    """
    if use_case not in USE_CASES:
        raise ValueError(f"Unknown use_case: {use_case}")

    feature_cols = optimize._feature_cols()

    # --- 1. Pareto solutions ---
    pareto: pd.DataFrame
    if PARETO_CSV.exists():
        pareto = pd.read_csv(PARETO_CSV)
        pareto = pareto[pareto.columns[: len(INPUT_FEATURES)]].copy()
        pareto["source"] = "pareto"
    else:
        pareto = pd.DataFrame(columns=INPUT_FEATURES)

    # --- 2. Latin-hypercube random samples ---
    rng = np.random.default_rng(RANDOM_STATE)
    lhs = _lhs_samples(n * 3, rng)
    lhs_df = pd.DataFrame(lhs)
    for c in INPUT_FEATURES:
        if c not in lhs_df.columns:
            lhs_df[c] = 0.0

    pool = pd.concat([pareto, lhs_df], ignore_index=True)

    # --- 3. Predict properties + filter ---
    from src.optimize import _predict

    accepted: list[dict] = []
    # Load graph once for SHACL checks
    g = Graph_load()

    for _, row in pool.iterrows():
        form = {c: row[c] for c in INPUT_FEATURES if c in row}
        # ensure all input features present
        for c in INPUT_FEATURES:
            if c not in form:
                form[c] = 0.0

        # Range + categorical validation
        errs = validate_formulation(Formulation(**form, source="synthetic"))
        if errs:
            continue

        # Forbidden additives
        if form["nanoparticle"] in FORBIDDEN_ADDITIVES:
            continue

        # Predict
        try:
            pred = _predict(form, feature_cols)
        except Exception:  # noqa: BLE001
            continue

        # Use-case thresholds
        ok = True
        for prop, (op, value) in USE_CASES[use_case].items():
            v = pred.get(prop)
            if v is None:
                ok = False
                break
            if op == "<=" and not (v <= value):
                ok = False
                break
            if op == ">=" and not (v >= value):
                ok = False
                break
        if not ok:
            continue

        # SHACL validity when added to graph
        if not _is_shacl_valid(form, pred, g):
            continue

        candidate = {**form, **pred}
        candidate["sustainability_score"] = _sustainability_score(pred)
        accepted.append(candidate)

    df = pd.DataFrame(accepted)
    if df.empty:
        print("No candidates passed all filters.")
        return df

    # --- 5. Rank by sustainability score (desc), then tensile (desc) ---
    df = df.sort_values(
        ["sustainability_score", "tensile_strength_MPa"],
        ascending=[False, False],
    ).reset_index(drop=True)

    top = df.head(n).copy()
    print(f"Generated {len(top)} candidates for {use_case} (from {len(accepted)} accepted).")
    return top


# ---------------------------------------------------------------------------
# Helpers to (re)load the graph
# ---------------------------------------------------------------------------
def Graph_load():
    from rdflib import Graph
    g = Graph()
    g.parse("kg/graph.ttl", format="turtle")
    return g


def main() -> None:
    """CLI entry: python -m src.generate [use_case] [n]"""
    import sys

    use_case = sys.argv[1] if len(sys.argv) > 1 else "insulation_panel"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    df = generate_candidates(use_case, n=n)
    if not df.empty:
        print(top_columns(df))


def top_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["sustainability_score", "tensile_strength_MPa",
            "thermal_conductivity_W_mK", "gwp_kgCO2e_per_kg",
            "water_absorption_pct", "matrix_polymer", "agro_filler",
            "nanoparticle"]
    return df[[c for c in cols if c in df.columns]]


if __name__ == "__main__":
    main()
