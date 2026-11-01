# Nanocomposite-AI

**AI + Ontology-driven design of nanocomposites from recycled polymers and agro-waste for construction.**

Pipeline:

```
curated/synthetic data
    -> ontology Knowledge Graph (RDF + SHACL)
    -> ML property prediction (RF / XGBoost)
    -> multi-objective optimization (Optuna NSGA-II)
    -> constrained design generation
    -> Streamlit app with feedback loop
```

## What's included

| Component | Files | Purpose |
|---|---|---|
| Config & schema | `config.py` | All constants: features, targets, units, ranges, constraints, ontology rules, `Formulation` dataclass + validator |
| Data | `src/data_gen.py`, `src/clean.py`, `data/raw/seed_data.csv` | Physics-inspired synthetic generator (N=400), merge/validate, seed-example template |
| Ontology + KG | `kg/ontology.ttl`, `kg/shapes.ttl`, `src/kg_build.py` | Turtle ontology (`nc:` namespace), SHACL shapes, RDF graph build + pyshacl validation |
| SPARQL + features | `kg/queries/*.rq`, `src/kg_query.py`, `src/features.py` | 4 SPARQL queries, features built **from the KG**, one-hot encoding + graph-derived flags |
| ML models | `src/train.py` | RF + XGBoost per target, 5-fold CV + held-out test, metrics + pred-vs-actual plots |
| Explainability | `src/explain.py` | SHAP summary/bar per target, `explain_ontology()` plain-English KG reasoning |
| Optimization | `src/optimize.py` | Optuna NSGA-II multi-objective, use-case constraints, Pareto CSV + Plotly figures |
| Constrained generation | `src/generate.py` | Pareto + Latin-hypercube candidates, filtered by ontology rules + SHACL + thresholds, ranked |
| Streamlit app | `app/app.py` | 6 tabs: Predict, Optimize, Design suggestions, Knowledge graph, Submit data, About |
| Feedback loop | `src/feedback.py` | Submit measured samples (`source="user"`), retrain pipeline with versioned model archive |
| Optional "Gap" modules | `src/data_ingestion.py`, `src/gnn_model.py`, `src.surrogate_models.py`, `src.generative_models.py`, `src.explain_lime.py` | Live open-source ingestion, GNN predictor, GP/poly surrogates, VAE/GMM generator, LIME explanations |
| Tests | `tests/test_phase*.py` | 72 pytest tests covering every phase |
| Orchestration | `make_all.py` | Run phases 2-8 end-to-end with one command |

## Tech stack (fixed)

Python 3.11, pandas, numpy, scikit-learn, xgboost, shap, optuna, rdflib, pyshacl,
streamlit, plotly, matplotlib, joblib, pytest, pyyaml, networkx, pyvis, requests.

Optional (wrapped in `try/except`, project runs without them): autogluon, h2o,
torch + torch-geometric, lime.

## Quick start

```bash
# 1. Clone / open the project in VS Code
cd /path/to/Project-MSc

# 2. Create + activate the virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify core imports
python -c "import rdflib, xgboost, optuna, streamlit, shap, pyshacl; print('imports OK')"

# 5. Run the full pipeline (phases 2-8)
python make_all.py

# 6. Launch the Streamlit app
streamlit run app/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Run phases individually

```bash
python -m src.data_gen          # Phase 2: synthetic data
python -m src.clean             # Phase 2: merge + validate -> data/processed/dataset.csv
python -m src.kg_build          # Phase 3: build KG + SHACL validate
python -m src.kg_query a_insulation_panel   # Phase 4: SPARQL
python -m src.features          # Phase 4: features from KG
python -m src.train             # Phase 5: train models
python -m src.explain           # Phase 6: SHAP + KG reasoning
python -m src.optimize insulation_panel     # Phase 7: Pareto optimization
python -m src.generate insulation_panel 20  # Phase 8: constrained candidates
streamlit run app/app.py        # Phase 9: app
python -m src.feedback retrain  # Phase 10: retrain with user feedback
pytest                          # Phase 11: quality tests
```

## Project structure (at repo root)

```
Project-MSc/
  config.py              # ALL constants (features, targets, units, ranges, constraints)
  requirements.txt
  README.md
  make_all.py
  .gitignore
  data/raw/seed_data.csv          # 18 example rows (source="seed_example", replace with real literature)
  data/raw/user_data.csv          # user-submitted feedback (created on submit; not committed)
  data/processed/                 # dataset.csv, features.csv, data_report.md (generated)
  kg/ontology.ttl                 # Turtle ontology (nc: namespace)
  kg/shapes.ttl                   # SHACL shapes
  kg/graph.ttl                    # built RDF knowledge graph (generated)
  kg/queries/*.rq                 # 4 SPARQL queries
  src/*.py                        # pipeline modules
  app/app.py                      # Streamlit app
  models/*.joblib                 # best model per target (generated)
  models/archive/<ts>/            # versioned backups of old models
  reports/metrics.json            # per-target per-model metrics
  reports/metrics.md              # metrics table
  reports/data_report.md          # row counts per source, missing values, describe()
  reports/kg_stats.md             # graph statistics
  reports/figures/                # SHAP, pred-vs-actual, Pareto, KG viz (generated)
  tests/                          # pytest suite (one file per phase)
```

## Architecture (text diagram)

```
data/raw/seed_data.csv  +  src/data_gen.py  (synthetic, N=400)
        \
         --> src/clean.py  --> data/processed/dataset.csv (+ reports/data_report.md)
                                               |
                                               v
                                        src/kg_build.py
                                               |
                                               v
                                         kg/graph.ttl   (rdflib, pyshacl-validated)
                                         kg/shapes.ttl
                                               |
                                               v
                                        src/kg_query.py  (SPARQL)
                                               |
                                               v
                                        src/features.py  --> data/processed/features.csv
                                               |
                                               v
                                        src/train.py     --> models/*.joblib
                                                             reports/metrics.json/md
                                                             reports/figures/*_pred_vs_actual.png
                                               |
                                               v
                                        src/explain.py   --> reports/figures/*_shap_*.png
                                               |
                                               v
                                        src/optimize.py  --> reports/pareto.csv
                                                             reports/figures/pareto_2d/3d.html
                                               |
                                               v
                                        src/generate.py  --> reports/pareto.csv filtered + ranked
                                               |
                                               v
                                        app/app.py       (streamlit, 6 tabs)
                                               |
                                               v
                                        src/feedback.py  (submit + retrain + model archive)
```

## Replace seed data with real literature data

`data/raw/seed_data.csv` contains 18 example rows marked `source="seed_example"` — **not** confirmed literature values. Replace them before any design decision.

Export from PoLyInfo / NanoMine / PubChem and map columns to this schema:

| nanocomposite-ai column | Source equivalent | Notes |
|---|---|---|
| `matrix_polymer` | polymer / matrix | HDPE, LDPE, PP, PET_recycled, PLA |
| `recycled_content_pct` | recycled content % | 0-100 |
| `agro_filler` | filler type | rice_husk, wheat_straw, sugarcane_bagasse, coconut_coir, none |
| `filler_wt_pct` | filler loading wt% | 0-40 |
| `filler_particle_size_um` | filler size um | 20-500; 0 if no filler |
| `nanoparticle` | nanoparticle type | SiO2, nanoclay, TiO2, cellulose_nanocrystal, none |
| `nano_wt_pct` | nanoparticle loading wt% | 0-5; 0 if none |
| `nano_size_nm` | nanoparticle size nm | 10-100; 0 if none |
| `surface_treatment` | coupling agent | none, silane, NaOH, MAPE_compatibilizer |
| `extrusion_temp_C` | extrusion temp C | 160-230 |
| `screw_speed_rpm` | screw speed rpm | 30-150 |
| `sonication_min` | sonication min | 0-60 |
| `tensile_strength_MPa` ... | target properties | MPa, GPa, W/mK, %, kgCO2e/kg, 0-1 |
| `source` | provenance | literature / user / synthetic / seed_example |
| `reference` | citation | free text |

Then re-run:
```bash
python -m src.clean && python -m src.kg_build && python -m src.features && python -m src.train
```

## Limitations

- Part of the training data is synthetic (documented physics-based surrogates in `src/data_gen.py`). A banner in the Streamlit app shows the synthetic fraction.
- Seed rows are marked `source="seed_example"` — replace with real literature data before publication/design decisions.
- Ontology external links in `config.py` (`EXTERNAL_ONTOLOGY_LINKS`) are aligned to EMMO/ENVO/CHMO where possible; verify against live portals before publishing.
- All numeric values are stored as plain floats with units defined in `config.py` (never `"35 MPa"` strings).
- Every row has a `source` column: `literature`, `user`, `synthetic`, or `seed_example`. Counts per source are reported in `reports/data_report.md` and via the app banner.
- Fixed random seed (42) used throughout.
- Optional Gap modules (GNN, VAE, LIME, autogluon/h2o) require extra dependencies and are wrapped in `try/except`.

## Push to GitHub

From VS Code (or terminal), in this folder:

```bash
git add -A
git commit -m "Full nanocomposite-AI project: ontology KG, ML, optimization, Streamlit app, feedback loop"
git push origin main
```

GitHub will display `README.md` on the repository's main page automatically.
