"""Multi-objective optimization (Phase 7).

Uses the trained surrogate models (RandomForest/XGBoost saved by src.train)
as objective functions. Optuna with NSGAIISampler optimizes:

  maximize tensile_strength_MPa
  minimize thermal_conductivity_W_mK
  minimize gwp_kgCO2e_per_kg
  minimize water_absorption_pct

Search space from config.py ranges. Trials violating the chosen use-case
constraints (config.USE_CASES) are rejected.

Inputs via function args: matrix (optional fix), filler (optional fix),
use_case name, n_trials (default 300).

Output:
  - reports/pareto.csv  (Pareto front DataFrame with predicted properties)
  - reports/figures/pareto_2d.html  (plotly scatter matrix / 2D)
  - reports/figures/pareto_3d.html  (plotly 3D scatter)

Run: python -m src.optimize [use_case] [matrix] [filler] [n_trials]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    TARGET_PROPERTIES,
    USE_CASES,
)

MODELS_DIR = Path("models")
FEATURES_PATH = Path("data/processed/features.csv")
PARETO_CSV = Path("reports/pareto.csv")
FIG_2D = Path("reports/figures/pareto_2d.html")
FIG_3D = Path("reports/figures/pareto_3d.html")

RANDOM_STATE = 42
OBJECTIVES: list[tuple[str, str]] = [
    ("tensile_strength_MPa", "maximize"),
    ("thermal_conductivity_W_mK", "minimize"),
    ("gwp_kgCO2e_per_kg", "minimize"),
    ("water_absorption_pct", "minimize"),
]

# Models cached per target
_MODEL_CACHE: dict[str, object] = {}


def _load_model(target: str):
    if target not in _MODEL_CACHE:
        path = MODELS_DIR / f"{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} (run src.train first)")
        _MODEL_CACHE[target] = joblib.load(path)
    return _MODEL_CACHE[target]


def _feature_cols() -> list[str]:
    """Feature columns matching src.train (all non-target, non-meta columns)."""
    df = pd.read_csv(FEATURES_PATH)
    exclude = set(TARGET_PROPERTIES.keys()) | {"source", "reference", "sample"}
    return [c for c in df.columns if c not in exclude]


def _build_vector(
    trial,
    feature_cols: list[str],
    matrix_fix: str | None,
    filler_fix: str | None,
) -> dict:
    """Build a full formulation feature dict from an Optuna trial."""
    cat_opts = CATEGORICAL_FEATURES

    # matrix
    if matrix_fix:
        matrix = matrix_fix
    else:
        matrix = trial.suggest_categorical(
            "matrix_polymer", cat_opts["matrix_polymer"]
        )

    # agro filler
    if filler_fix:
        agro_filler = filler_fix
    else:
        agro_filler = trial.suggest_categorical(
            "agro_filler", cat_opts["agro_filler"]
        )

    # numeric inputs from config ranges
    def num(name: str) -> float:
        lo, hi, _ = NUMERIC_FEATURES[name]
        return trial.suggest_float(name, lo, hi)

    recycled_content_pct = num("recycled_content_pct")
    filler_wt_pct = num("filler_wt_pct")
    filler_particle_size_um = (
        0.0 if agro_filler == "none" else num("filler_particle_size_um")
    )
    nanoparticle = trial.suggest_categorical(
        "nanoparticle", cat_opts["nanoparticle"]
    )
    nano_wt_pct = 0.0 if nanoparticle == "none" else num("nano_wt_pct")
    nano_size_nm = 0.0 if nanoparticle == "none" else num("nano_size_nm")
    surface_treatment = trial.suggest_categorical(
        "surface_treatment", cat_opts["surface_treatment"]
    )
    extrusion_temp_C = num("extrusion_temp_C")
    screw_speed_rpm = num("screw_speed_rpm")
    sonication_min = num("sonication_min")

    return {
        "matrix_polymer": matrix,
        "recycled_content_pct": recycled_content_pct,
        "agro_filler": agro_filler,
        "filler_wt_pct": filler_wt_pct,
        "filler_particle_size_um": filler_particle_size_um,
        "nanoparticle": nanoparticle,
        "nano_wt_pct": nano_wt_pct,
        "nano_size_nm": nano_size_nm,
        "surface_treatment": surface_treatment,
        "extrusion_temp_C": extrusion_temp_C,
        "screw_speed_rpm": screw_speed_rpm,
        "sonication_min": sonication_min,
    }


def _vector_to_df_row(form: dict, feature_cols: list[str]) -> pd.DataFrame:
    """One-hot encode a formulation dict into a single-row DataFrame matching feature_cols."""
    row = {c: 0.0 for c in feature_cols}
    # numeric
    for c, v in form.items():
        if c in NUMERIC_FEATURES:
            row[c] = float(v)
    # categorical one-hot
    cat_cols = [c for c in CATEGORICAL_FEATURES]
    for c in cat_cols:
        val = form[c]
        col = f"{c}_{val}"
        if col in row:
            row[col] = 1.0
    # graph-derived flags
    row["matrix_is_recycled"] = 1.0 if form["matrix_polymer"] == "PET_recycled" else 0.0
    ligno = {"rice_husk", "wheat_straw", "sugarcane_bagasse", "coconut_coir"}
    row["filler_is_lignocellulosic"] = (
        1.0 if form["agro_filler"] in ligno else 0.0
    )
    return pd.DataFrame([row], columns=feature_cols)


def _predict(form: dict, feature_cols: list[str]) -> dict[str, float]:
    """Predict all targets for a formulation using saved surrogate models."""
    X = _vector_to_df_row(form, feature_cols)
    out: dict[str, float] = {}
    for target in TARGET_PROPERTIES:
        model = _load_model(target)
        out[target] = float(model.predict(X)[0])
    return out


def _constraints_ok(form: dict, pred: dict, use_case: str | None) -> bool:
    """Check use-case thresholds from config.USE_CASES."""
    if not use_case or use_case not in USE_CASES:
        return True
    for prop, (op, value) in USE_CASES[use_case].items():
        v = pred.get(prop)
        if v is None:
            return False
        if op == "<=" and not (v <= value):
            return False
        if op == ">=" and not (v >= value):
            return False
    return True


def optimize(
    use_case: str | None = None,
    matrix: str | None = None,
    filler: str | None = None,
    n_trials: int = 300,
) -> pd.DataFrame:
    """Run NSGA-II multi-objective optimization. Returns Pareto front DataFrame."""
    feature_cols = _feature_cols()

    # Validate fixes
    if matrix and matrix not in CATEGORICAL_FEATURES["matrix_polymer"]:
        raise ValueError(f"Invalid matrix: {matrix}")
    if filler and filler not in CATEGORICAL_FEATURES["agro_filler"]:
        raise ValueError(f"Invalid filler: {filler}")
    if use_case and use_case not in USE_CASES:
        raise ValueError(f"Unknown use_case: {use_case}")

    direction = [
        "maximize" if d == "maximize" else "minimize" for _, d in OBJECTIVES
    ]

    def objective(trial: optuna.Trial):
        form = _build_vector(trial, feature_cols, matrix, filler)
        pred = _predict(form, feature_cols)
        if not _constraints_ok(form, pred, use_case):
            # Return worst values so Optuna discards (constraint via penalty)
            return tuple(
                1e6 if d == "minimize" else -1e6 for _, d in OBJECTIVES
            )
        return tuple(pred[t] for t, _ in OBJECTIVES)

    study = optuna.create_study(
        directions=direction,
        sampler=optuna.samplers.NSGAIISampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Extract feasible trials (non-penalized)
    feasible: list[dict] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        vals = t.values
        if vals is None:
            continue
        # skip penalized
        if any(abs(v) > 1e5 for v in vals):
            continue
        form = _build_vector_from_params(t.params, feature_cols, matrix, filler)
        pred = dict(zip([t for t, _ in OBJECTIVES], vals))
        row = {**form, **pred}
        feasible.append(row)

    pareto = pd.DataFrame(feasible)
    if pareto.empty:
        print("WARNING: no feasible Pareto solutions found.")
        pareto = pd.DataFrame()

    # Rank: simple crowding-free sort; keep all non-dominated
    PARETO_CSV.parent.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(PARETO_CSV, index=False)
    print(f"Pareto front ({len(pareto)} solutions) -> {PARETO_CSV}")

    # Visualizations
    _plot_pareto(pareto)

    return pareto


def _build_vector_from_params(
    params: dict, feature_cols: list[str], matrix: str | None, filler: str | None
) -> dict:
    """Rebuild a formulation dict from trial params (mirrors _build_vector)."""
    nano = params["nanoparticle"]
    return {
        "matrix_polymer": matrix or params["matrix_polymer"],
        "recycled_content_pct": float(params["recycled_content_pct"]),
        "agro_filler": filler or params["agro_filler"],
        "filler_wt_pct": float(params["filler_wt_pct"]),
        "filler_particle_size_um": (
            0.0 if (filler or params["agro_filler"]) == "none"
            else float(params["filler_particle_size_um"])
        ),
        "nanoparticle": nano,
        "nano_wt_pct": 0.0 if nano == "none" else float(params["nano_wt_pct"]),
        "nano_size_nm": 0.0 if nano == "none" else float(params["nano_size_nm"]),
        "surface_treatment": params["surface_treatment"],
        "extrusion_temp_C": float(params["extrusion_temp_C"]),
        "screw_speed_rpm": float(params["screw_speed_rpm"]),
        "sonication_min": float(params["sonication_min"]),
    }


def _plot_pareto(pareto: pd.DataFrame) -> None:
    if pareto.empty:
        return
    FIG_2D.parent.mkdir(parents=True, exist_ok=True)

    # 2D: scatter matrix of the 4 objectives
    obj_cols = [t for t, _ in OBJECTIVES]
    fig = px.scatter_matrix(
        pareto,
        dimensions=obj_cols,
        color="gwp_kgCO2e_per_kg",
        title="Pareto Front — Objective Scatter Matrix",
    )
    fig.write_html(str(FIG_2D))
    print(f"2D Pareto figure -> {FIG_2D}")

    # 3D: tensile vs thermal vs gwp (color = water absorption)
    fig3 = go.Figure(
        data=go.Scatter3d(
            x=pareto["tensile_strength_MPa"],
            y=pareto["thermal_conductivity_W_mK"],
            z=pareto["gwp_kgCO2e_per_kg"],
            mode="markers",
            marker=dict(
                size=5,
                color=pareto["water_absorption_pct"],
                colorscale="Viridis",
                colorbar=dict(title="Water absorption %"),
            ),
            text=[
                f"tensile={t:.2f}<br>thermal={th:.3f}<br>gwp={g:.2f}<br>water={w:.2f}"
                for t, th, g, w in zip(
                    pareto["tensile_strength_MPa"],
                    pareto["thermal_conductivity_W_mK"],
                    pareto["gwp_kgCO2e_per_kg"],
                    pareto["water_absorption_pct"],
                )
            ],
        )
    )
    fig3.update_layout(
        title="Pareto Front — 3D (tensile ↑, thermal ↓, gwp ↓)",
        scene=dict(
            xaxis_title="Tensile Strength (MPa)",
            yaxis_title="Thermal Conductivity (W/mK)",
            zaxis_title="GWP (kgCO2e/kg)",
        ),
    )
    fig3.write_html(str(FIG_3D))
    print(f"3D Pareto figure -> {FIG_3D}")


def main() -> None:
    """CLI entry: python -m src.optimize [use_case] [matrix] [filler] [n_trials]"""
    use_case = sys.argv[1] if len(sys.argv) > 1 else None
    matrix = sys.argv[2] if len(sys.argv) > 2 else None
    filler = sys.argv[3] if len(sys.argv) > 3 else None
    n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    optimize(use_case=use_case, matrix=matrix, filler=filler, n_trials=n_trials)


if __name__ == "__main__":
    main()
