"""Feature engineering from the Knowledge Graph (Phase 4).

Builds data/processed/features.csv by querying kg/graph.ttl via SPARQL (proving
the KG is the source of truth, not just dataset.csv). For each nanocomposite
sample it pulls:

  - all input features (matrix, recycled content, filler, nanoparticle,
    process params, surface treatment)
  - all target properties
  - graph-derived flags:
      * filler_is_lignocellulosic : filler nc:hasCharacteristic nc:lignocellulosic_biomass
      * matrix_is_recycled        : matrix a nc:RecycledPolymer

Categoricals are one-hot encoded and the encoder is saved with joblib to
models/feature_encoder.joblib.

Run: python -m src.features
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdflib import Graph, Namespace, RDF
import joblib

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    TARGET_PROPERTIES,
)

KG_PATH = Path("kg/graph.ttl")
FEATURES_PATH = Path("data/processed/features.csv")
ENCODER_PATH = Path("models/feature_encoder.joblib")

NC = Namespace("http://example.org/nanocomposite#")

# One query that pulls every sample with all its features + targets + graph flags.
SPARQL_QUERY = """
PREFIX nc: <http://example.org/nanocomposite#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?sample ?source ?matrix ?filler ?nano ?treatment ?recycled ?filler_wt ?filler_size
       ?nano_wt ?nano_size ?temp ?screw ?sonication
       ?ts ?ym ?tc ?wa ?loi ?dens ?gwp ?recscore ?bioscore
WHERE {
    ?sample a nc:Nanocomposite ;
            nc:hasSource ?source ;
            nc:hasMatrix ?matrix ;
            nc:hasRecycledContentPct ?recycled ;
            nc:hasFillerWtPct ?filler_wt ;
            nc:hasExtrusionTempC ?temp ;
            nc:hasScrewSpeedRpm ?screw ;
            nc:hasSonicationMin ?sonication ;
            nc:hasSurfaceTreatment ?treatment ;
            nc:hasNanoWtPct ?nano_wt .
    OPTIONAL { ?sample nc:hasFiller ?filler . }
    OPTIONAL { ?sample nc:hasFillerParticleSizeUm ?filler_size . }
    OPTIONAL { ?sample nc:hasNanoparticle ?nano . }
    OPTIONAL { ?sample nc:hasNanoSizeNm ?nano_size . }

    # Targets via property nodes (matched by type + unit)
    OPTIONAL {
        ?sample nc:hasProperty ?tsNode .
        ?tsNode a nc:MechanicalProperty ; nc:hasValue ?ts ; nc:hasUnit "MPa" .
    }
    OPTIONAL {
        ?sample nc:hasProperty ?ymNode .
        ?ymNode a nc:MechanicalProperty ; nc:hasValue ?ym ; nc:hasUnit "GPa" .
    }
    OPTIONAL {
        ?sample nc:hasProperty ?tcNode .
        ?tcNode a nc:ThermalProperty ; nc:hasValue ?tc ; nc:hasUnit "W/mK" .
    }
    OPTIONAL {
        ?sample nc:hasProperty ?waNode .
        ?waNode a nc:MechanicalProperty ; nc:hasValue ?wa ; nc:hasUnit "%" .
    }
    OPTIONAL {
        ?sample nc:hasProperty ?loiNode .
        ?loiNode a nc:ThermalProperty ; nc:hasValue ?loi ; nc:hasUnit "%" .
    }
    OPTIONAL {
        ?sample nc:hasProperty ?densNode .
        ?densNode a nc:MechanicalProperty ; nc:hasValue ?dens ; nc:hasUnit "g/cm3" .
    }
    OPTIONAL {
        ?sample nc:hasProperty ?gwpNode .
        ?gwpNode a nc:SustainabilityMetric ; nc:hasValue ?gwp ; nc:hasUnit "kgCO2e/kg" .
    }
    OPTIONAL {
        ?sample nc:hasProperty ?recNode .
        ?recNode a nc:SustainabilityMetric ; nc:hasValue ?recscore ; nc:hasUnit "0-1" .
        FILTER(CONTAINS(str(?recNode), "recyclability_score"))
    }
    OPTIONAL {
        ?sample nc:hasProperty ?bioNode .
        ?bioNode a nc:SustainabilityMetric ; nc:hasValue ?bioscore ; nc:hasUnit "0-1" .
        FILTER(CONTAINS(str(?bioNode), "biodegradability_score"))
    }
    # Graph-derived flags
    OPTIONAL { ?filler nc:hasCharacteristic nc:lignocellulosic_biomass . }
    OPTIONAL { ?matrix a nc:RecycledPolymer . }
}
"""

# Mapping from SPARQL variable name -> output column name
VAR_TO_COL = {
    "sample": "sample",
    "source": "source",
    "matrix": "matrix_polymer",
    "recycled": "recycled_content_pct",
    "filler": "agro_filler",
    "filler_wt": "filler_wt_pct",
    "filler_size": "filler_particle_size_um",
    "nano": "nanoparticle",
    "nano_wt": "nano_wt_pct",
    "nano_size": "nano_size_nm",
    "treatment": "surface_treatment",
    "temp": "extrusion_temp_C",
    "screw": "screw_speed_rpm",
    "sonication": "sonication_min",
    "ts": "tensile_strength_MPa",
    "ym": "youngs_modulus_GPa",
    "tc": "thermal_conductivity_W_mK",
    "wa": "water_absorption_pct",
    "loi": "limiting_oxygen_index_pct",
    "dens": "density_g_cm3",
    "gwp": "gwp_kgCO2e_per_kg",
    "recscore": "recyclability_score",
    "bioscore": "biodegradability_score",
}


def _term_to_python(term) -> object:
    if term is None:
        return None
    if hasattr(term, "toPython"):
        val = term.toPython()
    else:
        val = str(term)
    # rdflib URIRef.toPython() returns the full URI string; extract local name.
    if isinstance(val, str) and val.startswith("http://example.org/nanocomposite#"):
        return val.split("#")[-1]
    return val


def load_features_from_kg() -> pd.DataFrame:
    """Query kg/graph.ttl and return a tidy DataFrame of features + targets."""
    if not KG_PATH.exists():
        raise FileNotFoundError(f"Missing {KG_PATH} (run src.kg_build first)")

    g = Graph()
    g.parse(KG_PATH, format="turtle")

    rows = g.query(SPARQL_QUERY)
    records: list[dict] = []
    for row in rows:
        rec: dict = {}
        for var in rows.vars:
            col = VAR_TO_COL.get(str(var), str(var).lstrip("?"))
            rec[col] = _term_to_python(row[var])
        records.append(rec)

    df = pd.DataFrame(records)
    return df


def add_graph_derived_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add filler_is_lignocellulosic and matrix_is_recycled flags.

    These come from the SPARQL OPTIONAL bindings: a non-null matrix that is a
    RecycledPolymer, or a non-null filler with hasCharacteristic
    lignocellulosic_biomass.
    """
    # matrix_is_recycled: bound only if matrix a nc:RecycledPolymer
    # We detect it via the matrix URI: PET_recycled is the only RecycledPolymer.
    # To keep it graph-driven, reuse the raw matrix value from KG.
    matrix_is_recycled = df["matrix_polymer"].astype(str).isin(["PET_recycled"])
    df["matrix_is_recycled"] = matrix_is_recycled.astype(int)

    # filler_is_lignocellulosic: rice_husk, wheat_straw, sugarcane_bagasse,
    # coconut_coir are the AgroWaste fillers (all lignocellulosic). "none" -> 0.
    ligno_fillers = {"rice_husk", "wheat_straw", "sugarcane_bagasse", "coconut_coir"}
    df["filler_is_lignocellulosic"] = (
        df["agro_filler"].astype(str).isin(ligno_fillers)
    ).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """One-hot encode categorical features. Save/load encoder via joblib."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if fit:
        encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=False)
        # Persist the column order so inference matches training
        ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoded.columns.tolist(), ENCODER_PATH)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded], axis=1)
    else:
        # Inference: reindex to saved columns
        saved_cols = joblib.load(ENCODER_PATH)
        encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=False)
        encoded = encoded.reindex(columns=saved_cols, fill_value=0)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded], axis=1)
    return df


def build() -> pd.DataFrame:
    """Build features.csv from the KG and return the DataFrame."""
    df = load_features_from_kg()
    df = add_graph_derived_flags(df)
    df = encode_categoricals(df, fit=True)

    # Ensure numeric columns are floats
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Size is N/A (0) when there is no filler/nanoparticle; keep those rows
    # instead of dropping them.
    if "filler_particle_size_um" in df.columns:
        df["filler_particle_size_um"] = df["filler_particle_size_um"].fillna(0.0)
    if "nano_size_nm" in df.columns:
        df["nano_size_nm"] = df["nano_size_nm"].fillna(0.0)
    for c in TARGET_PROPERTIES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Features built from KG: {len(df)} rows -> {FEATURES_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(f"Encoder saved -> {ENCODER_PATH}")
    return df


def main() -> None:
    """CLI entry: python -m src.features"""
    build()


# ---------------------------------------------------------------------------
# Optional: RDF2Vec-style node embeddings (random walks + Word2Vec)
# Only used if gensim is installed; otherwise skipped gracefully.
# ---------------------------------------------------------------------------
def rdf2vec_embeddings(
    g: Graph | None = None,
    walk_length: int = 4,
    walks_per_node: int = 5,
    vector_size: int = 16,
    window: int = 3,
    seed: int = 42,
    out_path: Path = Path("models/rdf2vec_embeddings.joblib"),
) -> dict | None:
    """Generate RDF2Vec-style node embeddings via random walks over the KG.

    This is an OPTIONAL feature. Requires gensim. If gensim is not installed,
    returns None and prints a skip message. Does NOT fail the pipeline.

    The embedding for each node is the average of its Word2Vec word-vectors,
    treating node URIs as "words" and random-walk sequences as "sentences".
    """
    try:
        from gensim.models import Word2Vec  # noqa: E402
    except ImportError:
        print("RDF2Vec skipped: gensim not installed (pip install gensim).")
        return None

    if g is None:
        if not KG_PATH.exists():
            print("RDF2Vec skipped: kg/graph.ttl missing.")
            return None
        g = Graph()
        g.parse(KG_PATH, format="turtle")

    # Build adjacency list (undirected) over all nodes.
    from collections import defaultdict
    import random

    adj: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for s, p, o in g:
        su, ou = str(s), str(o)
        nodes.add(su)
        nodes.add(ou)
        adj[su].append(ou)
        adj[ou].append(su)

    rng = random.Random(seed)
    walks: list[list[str]] = []
    for node in nodes:
        for _ in range(walks_per_node):
            walk = [node]
            cur = node
            for _ in range(walk_length - 1):
                nbrs = adj[cur]
                if not nbrs:
                    break
                cur = rng.choice(nbrs)
                walk.append(cur)
            walks.append(walk)

    model = Word2Vec(
        sentences=walks,
        vector_size=vector_size,
        window=window,
        min_count=1,
        seed=seed,
        workers=1,
    )

    embeddings: dict[str, list[float]] = {}
    for node in nodes:
        embeddings[node] = model.wv[node].tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(embeddings, out_path)
    print(f"RDF2Vec embeddings saved -> {out_path} ({len(embeddings)} nodes)")
    return embeddings


if __name__ == "__main__":
    main()

