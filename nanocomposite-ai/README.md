# nanocomposite-ai

AI + Ontology-driven design of nanocomposites from recycled polymers and agro-waste for construction.

Pipeline: curated/synthetic data -> ontology Knowledge Graph (RDF) -> ML property prediction
-> multi-objective optimization -> constrained design generation -> Streamlit app
with feedback loop.

## Tech stack (fixed)

Python 3.11, pandas, numpy, scikit-learn, xgboost, shap, optuna, rdflib, pyshacl,
streamlit, plotly, matplotlib, joblib, pytest, pyyaml, networkx, pyvis.
Optional: autogluon OR h2o (Phase 5 extra baseline, wrapped in try/except).

Companion ontology tool: Protege (https://protege.stanford.edu)

## Setup

```bash
# 1. Create and activate the virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify core imports (macOS needs libomp: brew install libomp)
python -c "import rdflib, xgboost, optuna, streamlit, shap, pyshacl; print('imports OK')"
```

## Run order (commands run from the project root)

Phase 0: setup                                          (this README + requirements + venv)
Phase 1: config & schema                                src/config.py
Phase 2: data                                           python -m src.data_gen && python -m src.clean
Phase 3: ontology + KG                                  python -m src.kg_build
Phase 4: SPARQL + features                              python -m src.kg_query <name> && python -m src.features
Phase 5: ML models                                      python -m src.train
Phase 6: explainability                                 python -m src.explain
Phase 7: optimization                                   python -m src.optimize [use_case] [n_trials]
Phase 8: constrained generation                         python -m src.generate [use_case] [n]
Phase 9: Streamlit app                                  streamlit run app/app.py
Phase 10: feedback loop                                 python -m src.feedback {submit|retrain}
Phase 11: quality                                       pytest

End-to-end (phases 2-8):
    python make_all.py

Individual SPARQL queries (kg/queries/):
    python -m src.kg_query a_insulation_panel
    python -m src.kg_query b_avg_tensile_by_filler
    python -m src.kg_query c_recycled_matrix_high_filler
    python -m src.kg_query d_top10_lowest_gwp

## Folder structure

nanocomposite-ai/
  config.py            # ALL constants: features, targets, units, ranges, namespaces,
                       # constraints (forbidden additives, use cases), ontology rules
  requirements.txt
  README.md
  data/raw/seed_data.csv                  # 18 example rows (source="seed_example")
  data/raw/user_data.csv                  # user-submitted feedback (created on submit)
  data/processed/dataset.csv              # merged + validated dataset
  data/processed/features.csv             # features built FROM THE KG via SPARQL
  data/processed/removed_rows.csv         # rows rejected by the validator (logged)
  kg/ontology.ttl                         # Turtle ontology (nc: namespace)
  kg/shapes.ttl                           # SHACL shapes (forbidden additives injected)
  kg/graph.ttl                            # built RDF knowledge graph
  kg/queries/*.rq                         # 4 SPARQL queries
  src/*.py                                # pipeline modules
  app/app.py                              # Streamlit app (5 tabs + Submit data)
  models/*.joblib                         # best model per target
  models/archive/<ts>/                    # versioned backups of old models
  reports/metrics.json                    # per-target per-model metrics
  reports/metrics.md                      # metrics table
  reports/data_report.md                  # row counts per source, missing values, describe()
  reports/kg_stats.md                     # graph statistics
  reports/figures/                        # SHAP, pred-vs-actual, Pareto, KG viz
  tests/                                  # pytest suite (one test file per phase)

## Architecture (text diagram)

data/raw/seed_data.csv  +  src/data_gen.py (synthetic, N=400)
        \
         --> src/clean.py --> data/processed/dataset.csv (+ reports/data_report.md)
                                               |
                                               v
                                        src/kg_build.py
                                               |
                                               v
                                         kg/graph.ttl  (rdflib)
                                         kg/shapes.ttl  (pyshacl validation)
                                               |
                                               v
                                        src/kg_query.py (SPARQL)
                                               |
                                               v
                                        src/features.py --> data/processed/features.csv
                                               |
                                               v
                                        src/train.py  --> models/*.joblib
                                                        reports/metrics.json/md
                                                        reports/figures/*_pred_vs_actual.png
                                               |
                                               v
                                        src/explain.py  --> reports/figures/*_shap_*.png
                                               |
                                               v
                                        src/optimize.py --> reports/pareto.csv
                                                            reports/figures/pareto_2d/3d.html
                                               |
                                               v
                                        src/generate.py --> reports/pareto.csv filtered
                                               |
                                               v
                                        app/app.py  (streamlit)
                                               |
                                               v
                                        src/feedback.py  (submit + retrain)

## Limitations

- `data/raw/seed_data.csv` currently contains example rows marked
  `source="seed_example"`, NOT confirmed literature values. Replace with real
  literature data before any publication or design decision.
- Part of the training data is synthetic (documented physics-based surrogates in
  src/data_gen.py). A banner in the Streamlit app shows the synthetic fraction.
- `EXTERNAL_ONTOLOGY_LINKS` in config.py now points to real ontology portals
  (EMMO, MatOnto, CHMO, ENVO, LCA). Term IRIs (e.g. emmo:#, envo:#) still use
  `example.org` placeholders — VERIFY and replace them against the real
  portals before publishing.
- All numeric values are stored as plain floats with units defined in config.py
  (never as "35 MPa" strings).
- Every row has a `source` column: "literature", "user", "synthetic", or
  "seed_example". Counts per source are reported in reports/data_report.md and
  via the Streamlit banner.
- Fixed random seed (42) used throughout.

## Replace seed data with real data (PoLyInfo / NanoMine export)

Export your literature data to CSV and map columns to the schema below. Set
`source="literature"` only after confirming provenance of each row; otherwise
use `source="seed_example"`.

Required columns (order does not matter):

| nanocomposite-ai column | PoLyInfo / NanoMine equivalent | notes |
|---|---|---|
| matrix_polymer | polymer / matrix | one of HDPE, LDPE, PP, PET_recycled, PLA |
| recycled_content_pct | recycled content % | 0-100 |
| agro_filler | filler type | rice_husk, wheat_straw, sugarcane_bagasse, coconut_coir, none |
| filler_wt_pct | filler loading wt% | 0-40 |
| filler_particle_size_um | filler size um | 20-500; 0 if no filler |
| nanoparticle | nanoparticle type | SiO2, nanoclay, TiO2, cellulose_nanocrystal, none |
| nano_wt_pct | nanoparticle loading wt% | 0-5; 0 if none |
| nano_size_nm | nanoparticle size nm | 10-100; 0 if none |
| surface_treatment | coupling agent | none, silane, NaOH, MAPE_compatibilizer |
| extrusion_temp_C | extrusion temp C | 160-230 |
| screw_speed_rpm | screw speed rpm | 30-150 |
| sonication_min | sonication min | 0-60 |
| tensile_strength_MPa | tensile strength MPa | target |
| youngs_modulus_GPa | Young's modulus GPa | target |
| thermal_conductivity_W_mK | thermal conductivity W/mK | target |
| water_absorption_pct | water absorption % | target |
| limiting_oxygen_index_pct | LOI % | target |
| density_g_cm3 | density g/cm3 | target |
| gwp_kgCO2e_per_kg | GWP kgCO2e/kg | target |
| recyclability_score | recyclability 0-1 | target |
| biodegradability_score | biodegradability 0-1 | target |
| source | provenance | literature / user / synthetic / seed_example |
| reference | citation | free text |

Then re-run: `python -m src.clean && python -m src.kg_build && python -m src.features && python -m src.train`.

## Data Acquisition from Open Sources

Per the thesis methodology (Section 3), the project is designed to pull training
data from public open sources. The source URLs are registered in
`config.py` under `DATA_ACQUISITION_SOURCES`:

| Category | Source | URL |
|----------|--------|-----|
| Polymer / Bioplastic data | PoLyInfo (NIMS) | https://polymer.nims.go.jp |
| Polymer / Bioplastic data | Polymer Genome | https://polymergenome.org |
| Agri-waste materials | PubChem | https://pubchem.ncbi.nlm.nih.gov |
| Agri-waste materials | ChEBI (EBI) | https://www.ebi.ac.uk/chebi/ |
| Nanocomposite properties | NanoMine | https://nanomine.org |
| Nanocomposite properties | Materials Project (Next-Gen) | https://next-gen.materialsproject.org |
| Environmental / LCA data | Ecoinvent | https://ecoinvent.org |
| Environmental / LCA data | openLCA Nexus | https://nexus.openlca.org |

### Feature groups extracted from these sources
- Composition (polymer matrix, fillers, agri-waste type)
- Nanoparticle characteristics (size, aspect ratio, functionalization)
- Processing conditions (extrusion, sonication, curing)
- Material properties (strength, thermal, water resistance)
- Sustainability metrics (LCA, recyclability, CO₂ eq.)

> **Note:** The current `data/raw/seed_data.csv` and `src/data_gen.py` use
> physics-inspired synthetic surrogates. Real data from the sources above must
> be exported to CSV, mapped to the schema in the "Replace seed data" section,
> and loaded with `source="literature"` before any publication or design decision.

## 📋 What's Still Missing to Fully Match the Thesis

If you want the project to be **exactly** per the thesis methodology, these are
the gaps that still need to be implemented:

1. **Live data ingestion pipelines** — Code to actually pull from PoLyInfo,
   Polymer Genome, PubChem, ChEBI, NanoMine, Materials Project, Ecoinvent, and
   OpenLCA Nexus (currently only the URLs are listed in `config.py` and README).

2. **Real ontology term IRIs** — Replace the `example.org` term IRIs with actual
   EMMO/ENVO/CHMO/MatOnto/LCA terms verified from those portals, and populate
   `kg/ontology.ttl` with real imports/alignments.

3. **GNN models** — Graph Neural Networks for molecular structures of polymers
   and nanoparticles (Thesis Section 5).

4. **Gaussian Process / Polynomial Regression surrogates** — For nanoprocess
   simulation (Thesis Section 6).

5. **VAE/GAN generative models** — For novel formulation generation (Thesis
   Section 7).

6. **LIME explainability** — Alongside SHAP (Thesis Section 8).

## Quality

Run the full pytest suite:

    pytest

One test file per phase lives in tests/. The suite checks config validation, data
ranges, SHACL conformance, SPARQL row returns, model load+predict, and a
non-empty Pareto front from the optimizer.
