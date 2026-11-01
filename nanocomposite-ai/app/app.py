"""Streamlit app (Phase 9).

AI + Ontology-driven design of nanocomposites from recycled polymers and
agro-waste for construction.

Tabs:
  1. Predict        - formulation -> predicted properties + uncertainty + SHAP + KG reasoning
  2. Optimize       - Pareto multi-objective search -> plotly + table
  3. Design suggestions - top constrained candidates + CSV/JSON download
  4. Knowledge graph - pyvis graph + read-only SPARQL SELECT box
  5. About / Limitations

Run: streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root (where config.py lives) is importable when Streamlit
# runs this script (Streamlit adds the script's own dir, not the root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

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

FEATURES_PATH = Path("data/processed/features.csv")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
PARETO_CSV = Path("reports/pareto.csv")
KG_VIZ = Path("reports/figures/kg_subgraph.html")

st.set_page_config(page_title="Nanocomposite AI Design", layout="wide")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH)


@st.cache_resource
def load_model(target: str):
    import joblib

    path = MODELS_DIR / f"{target}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_graph():
    from rdflib import Graph

    g = Graph()
    g.parse("kg/graph.ttl", format="turtle")
    return g


@st.cache_data
def synthetic_fraction() -> float:
    df = load_features()
    if "source" not in df.columns or len(df) == 0:
        return 0.0
    return float((df["source"] == "synthetic").mean() * 100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def feature_cols() -> list[str]:
    df = load_features()
    exclude = set(TARGET_PROPERTIES.keys()) | {"source", "reference", "sample"}
    return [c for c in df.columns if c not in exclude]


def formulation_to_row(form: dict, fcols: list[str]) -> pd.DataFrame:
    row = {c: 0.0 for c in fcols}
    for c, v in form.items():
        if c in NUMERIC_FEATURES:
            row[c] = float(v)
    for c in CATEGORICAL_FEATURES:
        col = f"{c}_{form[c]}"
        if col in row:
            row[col] = 1.0
    row["matrix_is_recycled"] = 1.0 if form["matrix_polymer"] == "PET_recycled" else 0.0
    ligno = {"rice_husk", "wheat_straw", "sugarcane_bagasse", "coconut_coir"}
    row["filler_is_lignocellulosic"] = (
        1.0 if form["agro_filler"] in ligno else 0.0
    )
    return pd.DataFrame([row], columns=fcols)


def predict_properties(form: dict) -> dict[str, float]:
    fcols = feature_cols()
    X = formulation_to_row(form, fcols)
    out: dict[str, float] = {}
    for target in TARGET_PROPERTIES:
        model = load_model(target)
        if model is None:
            continue
        out[target] = float(model.predict(X)[0])
    return out


def prediction_uncertainty(form: dict) -> dict[str, float | None]:
    """RF tree-variance uncertainty (only if best model is a RandomForest)."""
    fcols = feature_cols()
    X = formulation_to_row(form, fcols)
    out: dict[str, float | None] = {}
    for target in TARGET_PROPERTIES:
        model = load_model(target)
        if model is None or not hasattr(model, "estimators_"):
            out[target] = None
            continue
        preds = np.array([est.predict(X)[0] for est in model.estimators_])
        out[target] = float(preds.std())
    return out


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Nanocomposite AI")
st.sidebar.caption("Ontology-driven design for construction materials")

frac = synthetic_fraction()
st.sidebar.markdown(
    f"**Training data:** {frac:.0f}% synthetic (remainder seed/literature examples)"
)

st.sidebar.header("Formulation inputs")
matrix = st.sidebar.selectbox("Matrix polymer", CATEGORICAL_FEATURES["matrix_polymer"])
recycled = st.sidebar.slider(
    "Recycled content %", *NUMERIC_FEATURES["recycled_content_pct"][:2], 50.0
)
agro_filler = st.sidebar.selectbox("Agro filler", CATEGORICAL_FEATURES["agro_filler"])
filler_wt = st.sidebar.slider(
    "Filler wt %", *NUMERIC_FEATURES["filler_wt_pct"][:2], 20.0
)
filler_size = st.sidebar.slider(
    "Filler particle size (um)", *NUMERIC_FEATURES["filler_particle_size_um"][:2], 150.0
)
nanoparticle = st.sidebar.selectbox("Nanoparticle", CATEGORICAL_FEATURES["nanoparticle"])
nano_wt = st.sidebar.slider(
    "Nano wt %", *NUMERIC_FEATURES["nano_wt_pct"][:2], 2.0
)
nano_size = st.sidebar.slider(
    "Nano size (nm)", *NUMERIC_FEATURES["nano_size_nm"][:2], 30.0
)
treatment = st.sidebar.selectbox(
    "Surface treatment", CATEGORICAL_FEATURES["surface_treatment"]
)
temp = st.sidebar.slider(
    "Extrusion temp (C)", *NUMERIC_FEATURES["extrusion_temp_C"][:2], 190.0
)
screw = st.sidebar.slider(
    "Screw speed (rpm)", *NUMERIC_FEATURES["screw_speed_rpm"][:2], 100.0
)
sonication = st.sidebar.slider(
    "Sonication (min)", *NUMERIC_FEATURES["sonication_min"][:2], 20.0
)

use_case = st.sidebar.selectbox("Construction use case", list(USE_CASES.keys()))

current_form = {
    "matrix_polymer": matrix,
    "recycled_content_pct": recycled,
    "agro_filler": agro_filler,
    "filler_wt_pct": filler_wt,
    "filler_particle_size_um": filler_size,
    "nanoparticle": nanoparticle,
    "nano_wt_pct": nano_wt,
    "nano_size_nm": nano_size,
    "surface_treatment": treatment,
    "extrusion_temp_C": temp,
    "screw_speed_rpm": screw,
    "sonication_min": sonication,
    "source": "user",
    "reference": "app_input",
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_optimize, tab_design, tab_kg, tab_feedback, tab_about = st.tabs(
    ["Predict", "Optimize", "Design suggestions", "Knowledge graph", "Submit data", "About / Limitations"]
)

# ===== Tab 1: Predict =====
with tab_predict:
    st.header("1. Predict properties")
    st.caption(
        "Enter a full formulation (sidebar). Predictions come from the trained "
        "surrogate models; uncertainty is RF tree-variance (when the best model "
        "is a RandomForest)."
    )
    errs = validate_formulation(Formulation(**current_form))
    if errs:
        st.error("Invalid formulation: " + "; ".join(errs))
    else:
        preds = predict_properties(current_form)
        unc = prediction_uncertainty(current_form)
        if preds:
            cols = st.columns(3)
            for i, (target, val) in enumerate(preds.items()):
                unit = TARGET_PROPERTIES[target]
                u = unc.get(target)
                u_txt = f" ± {u:.3f} {unit}" if u is not None else ""
                with cols[i % 3]:
                    st.metric(target, f"{val:.3f} {unit}{u_txt}")
        else:
            st.warning("No trained models found (run src.train).")

    st.subheader("SHAP explanation")
    shap_bar = FIGURES_DIR / f"{list(TARGET_PROPERTIES.keys())[0]}_shap_bar.png"
    if shap_bar.exists():
        st.image(str(shap_bar), caption="SHAP bar (tensile_strength_MPa example)")
    else:
        st.info("Run src.explain to generate SHAP plots.")

    st.subheader("Knowledge-graph reasoning")
    from src import explain  # noqa: E402

    reasoning = explain.explain_ontology(current_form)
    for line in reasoning:
        st.write("- " + line)

# ===== Tab 2: Optimize =====
with tab_optimize:
    st.header("2. Multi-objective optimization")
    st.caption(
        "Optuna NSGA-II: maximize tensile strength, minimize thermal conductivity, "
        "GWP, and water absorption. Trials violating the selected use-case "
        "constraints are rejected."
    )
    n_trials = st.slider("n_trials", 50, 1000, 200, step=50)
    opt_matrix = st.selectbox("Fix matrix (optional)", ["(none)"] + CATEGORICAL_FEATURES["matrix_polymer"])
    opt_filler = st.selectbox("Fix filler (optional)", ["(none)"] + CATEGORICAL_FEATURES["agro_filler"])
    if st.button("Run optimization"):
        from src import optimize  # noqa: E402

        m = None if opt_matrix == "(none)" else opt_matrix
        f = None if opt_filler == "(none)" else opt_filler
        with st.spinner("Optimizing..."):
            pareto = optimize.optimize(
                use_case=use_case, matrix=m, filler=f, n_trials=n_trials
            )
        if not pareto.empty:
            st.success(f"{len(pareto)} Pareto solutions found.")
            st.dataframe(pareto.head(30), use_container_width=True)
            fig2 = FIGURES_DIR / "pareto_2d.html"
            fig3 = FIGURES_DIR / "pareto_3d.html"
            if fig2.exists():
                st.components.v1.html(fig2.read_text(), height=500, scrolling=True)
            if fig3.exists():
                st.components.v1.html(fig3.read_text(), height=500, scrolling=True)
        else:
            st.warning("No feasible Pareto solutions.")

# ===== Tab 3: Design suggestions =====
with tab_design:
    st.header("3. Design suggestions")
    st.caption(
        "Top constrained candidates (Pareto + Latin-hypercube, filtered by "
        "ontology rules + SHACL + use-case thresholds)."
    )
    n_design = st.slider("Number of candidates", 5, 50, 20)
    if st.button("Generate candidates"):
        from src import generate  # noqa: E402

        with st.spinner("Generating & filtering candidates..."):
            cands = generate.generate_candidates(use_case, n=n_design)
        if not cands.empty:
            st.dataframe(cands, use_container_width=True)
            csv = cands.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV", csv, f"candidates_{use_case}.csv", "text/csv"
            )
            spec = cands.to_dict(orient="records")
            st.download_button(
                "Download JSON spec sheet",
                json.dumps(spec, indent=2).encode("utf-8"),
                f"spec_{use_case}.json",
                "application/json",
            )
        else:
            st.warning("No candidates passed all filters.")

# ===== Tab 4: Knowledge graph =====
with tab_kg:
    st.header("4. Knowledge graph")
    st.caption("Pyvis visualization of a 50-node subgraph + read-only SPARQL SELECT.")
    if KG_VIZ.exists():
        st.components.v1.html(KG_VIZ.read_text(), height=600, scrolling=True)
    else:
        st.info("Run src.kg_build to generate the KG visualization.")

    st.subheader("SPARQL query (SELECT only)")
    from src import kg_query  # noqa: E402

    query_names = [p.stem for p in Path("kg/queries").glob("*.rq")]
    qname = st.selectbox("Saved query", query_names)
    if st.button("Run saved query"):
        try:
            dfq = kg_query.run_query(qname)
            st.dataframe(dfq, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    custom_q = st.text_area("Custom SELECT query", height=120)
    if st.button("Run custom query"):
        if not custom_q.strip().lower().startswith("select"):
            st.error("Only SELECT queries are allowed (read-only).")
        else:
            from rdflib import Graph

            g = load_graph()
            try:
                rows = g.query(custom_q)
                data = []
                for row in rows:
                    data.append({str(v): row[v].toPython() if hasattr(row[v], "toPython") else str(row[v]) for v in rows.vars})
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

# ===== Tab 5: Submit data (feedback loop) =====
with tab_feedback:
    st.header("5. Submit data (feedback loop)")
    st.caption(
        "Add a measured sample. It is validated with the Formulation validator "
        "and appended to data/raw/user_data.csv with source='user'."
    )
    from src import feedback  # noqa: E402

    with st.form("submit_form"):
        fb_matrix = st.selectbox("Matrix polymer", CATEGORICAL_FEATURES["matrix_polymer"])
        fb_recycled = st.slider(
            "Recycled content %", *NUMERIC_FEATURES["recycled_content_pct"][:2], 50.0
        )
        fb_agro = st.selectbox("Agro filler", CATEGORICAL_FEATURES["agro_filler"])
        fb_filler_wt = st.slider(
            "Filler wt %", *NUMERIC_FEATURES["filler_wt_pct"][:2], 20.0
        )
        fb_filler_size = st.slider(
            "Filler particle size (um)", *NUMERIC_FEATURES["filler_particle_size_um"][:2], 150.0
        )
        fb_nano = st.selectbox("Nanoparticle", CATEGORICAL_FEATURES["nanoparticle"])
        fb_nano_wt = st.slider(
            "Nano wt %", *NUMERIC_FEATURES["nano_wt_pct"][:2], 2.0
        )
        fb_nano_size = st.slider(
            "Nano size (nm)", *NUMERIC_FEATURES["nano_size_nm"][:2], 30.0
        )
        fb_treatment = st.selectbox(
            "Surface treatment", CATEGORICAL_FEATURES["surface_treatment"]
        )
        fb_temp = st.slider(
            "Extrusion temp (C)", *NUMERIC_FEATURES["extrusion_temp_C"][:2], 190.0
        )
        fb_screw = st.slider(
            "Screw speed (rpm)", *NUMERIC_FEATURES["screw_speed_rpm"][:2], 100.0
        )
        fb_sonication = st.slider(
            "Sonication (min)", *NUMERIC_FEATURES["sonication_min"][:2], 20.0
        )
        # Targets
        st.subheader("Measured properties")
        fb_targets = {}
        for target, unit in TARGET_PROPERTIES.items():
            fb_targets[target] = st.number_input(f"{target} ({unit})", value=0.0)
        fb_reference = st.text_input("Reference (free text)", value="user submission")
        fb_submit = st.form_submit_button("Submit sample")

    if fb_submit:
        from src import feedback  # noqa: E402

        form = {
            "matrix_polymer": fb_matrix,
            "recycled_content_pct": fb_recycled,
            "agro_filler": fb_agro,
            "filler_wt_pct": fb_filler_wt,
            "filler_particle_size_um": fb_filler_size,
            "nanoparticle": fb_nano,
            "nano_wt_pct": fb_nano_wt,
            "nano_size_nm": fb_nano_size,
            "surface_treatment": fb_treatment,
            "extrusion_temp_C": fb_temp,
            "screw_speed_rpm": fb_screw,
            "sonication_min": fb_sonication,
            "source": "user",
            "reference": fb_reference,
        }
        form.update(fb_targets)
        ok, msg = feedback.submit_sample(form)
        if ok:
            st.success(msg)
        else:
            st.error("Validation failed: " + "; ".join(msg))

    st.subheader("Retrain pipeline")
    st.caption(
        "Back up old models to models/archive/<timestamp>/, then re-run "
        "clean -> kg_build -> features -> train with all data (incl. user samples)."
    )
    if st.button("Retrain"):
        from src import feedback  # noqa: E402

        with st.spinner("Retraining (this may take a while)..."):
            out = feedback.retrain()
        st.success("Retraining complete.")
        st.write("Archive:", out["archive"])
        st.write("User samples in data:", out["user_samples"])
        if out["before"] and out["after"]:
            st.subheader("Before / After metrics (test R2)")
            rows = []
            for b, a in zip(out["before"], out["after"]):
                rows.append(
                    {
                        "target": b["target"],
                        "before_R2": b["models"][b["best_model"]]["test"]["R2"],
                        "after_R2": a["models"][a["best_model"]]["test"]["R2"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("User-submitted samples")
    ud = feedback.list_user_samples()
    if len(ud):
        st.dataframe(ud, use_container_width=True)
        csv = ud.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download user data CSV", csv, "user_data.csv", "text/csv"
        )
    else:
        st.info("No user-submitted samples yet.")

# ===== Tab 6: About / Limitations =====
with tab_about:
    st.header("5. About / Limitations")
    st.write(
        """
        **Project:** AI + Ontology-driven design of nanocomposites from recycled
        polymers and agro-waste for construction.

        **Pipeline:** curated/synthetic data -> ontology Knowledge Graph (RDF) ->
        ML property prediction -> multi-objective optimization -> constrained
        design generation -> Streamlit app with feedback loop.

        **Limitations:**
        - Part of the training data is synthetic (physics-based surrogates, not
          measured literature values). A banner shows the synthetic fraction.
        - `data/raw/seed_data.csv` contains example rows marked
          `source="seed_example"` — replace with real literature data before any
          design decision.
        - Ontology external links (`EXTERNAL_ONTOLOGY_LINKS`) now point to real
          portals (EMMO, MatOnto, CHMO, ENVO, LCA). Term IRIs still use
          `example.org` placeholders — VERIFY against the real portals before
          publishing.
        - SHACL validation and SPARQL are read-only for the KG tab.
        """
    )
    st.write(f"Synthetic rows in training data: **{frac:.0f}%**")
