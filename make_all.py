"""make_all.py — run phases 2-8 end to end.

Usage: python make_all.py

Runs, in order:
  src.data_gen     (synthetic data, N=400)
  src.clean        (merge + validate -> data/processed/dataset.csv)
  src.kg_build     (ontology knowledge graph -> kg/graph.ttl, pyshacl validate)
  src.kg_query     (run the 4 saved SPARQL queries to prove KG is used)
  src.features     (features from KG -> data/processed/features.csv + encoder)
  src.train        (RF + XGB per target -> models/*.joblib + metrics + plots)
  src.explain      (SHAP plots per target + ontology reasoning demo)
  src.optimize     (NSGA-II Pareto -> reports/pareto.csv + figures)

Gap-phase modules (run manually, optional deps):
  src.data_ingestion   (Gap 1: live open-source ingestion)
  src.gnn_model        (Gap 3: GNN property predictor)
  src.surrogate_models (Gap 4: GP / Polynomial surrogates)
  src.generative_models (Gap 5: VAE/GAN formulation generator)
  src.explain_lime     (Gap 6: LIME explainability)

Prints a summary at the end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Phase 2: data_gen", "src.data_gen"),
    ("Phase 2: clean", "src.clean"),
    ("Phase 3: kg_build", "src.kg_build"),
    ("Phase 4: features", "src.features"),
    ("Phase 5: train", "src.train"),
    ("Phase 6: explain", "src.explain"),
    ("Phase 7: optimize", "src.optimize"),
]

QUERY_NAMES = [
    "a_insulation_panel",
    "b_avg_tensile_by_filler",
    "c_recycled_matrix_high_filler",
    "d_top10_lowest_gwp",
]


def run(module: str) -> None:
    print(f"\n{'='*60}\nRunning {module}\n{'='*60}")
    result = subprocess.run(
        [sys.executable, "-m", module], capture_output=True, text=True
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{module} FAILED (exit {result.returncode})")


def main() -> None:
    for label, module in STEPS:
        run(module)

    # Phase 4: run each saved SPARQL query to confirm KG returns rows
    print(f"\n{'='*60}\nRunning Phase 4 SPARQL queries\n{'='*60}")
    for name in QUERY_NAMES:
        print(f"\n--- {name} ---")
        result = subprocess.run(
            [sys.executable, "-m", "src.kg_query", name],
            capture_output=True, text=True,
        )
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            raise SystemExit(f"kg_query {name} FAILED")

    print(f"\n{'='*60}\nALL PHASES 2-8 COMPLETE\n{'='*60}")
    print("Next:")
    print("  - streamlit run app/app.py        (Phase 9 app)")
    print("  - pytest                          (Phase 11 quality)")
    print("  - python -m src.feedback retrain  (Phase 10 retrain)")


if __name__ == "__main__":
    main()
