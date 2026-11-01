"""Knowledge graph builder (Phase 3).

Reads data/processed/dataset.csv and builds kg/graph.ttl with rdflib:

  - one nc:Nanocomposite individual per row (URI nc:sample_<id>)
  - matrix, filler, nanoparticle as shared individuals (reused from ontology.ttl)
  - each property measurement as a node with hasValue + hasUnit
  - link sample -> application via suitableFor using config USE_CASES thresholds
  - knowledge links (rice_husk -> lignocellulosic biomass -> insulation potential)
    come from ontology.ttl

Also writes kg/shapes.ttl (forbidden additives injected from config), validates
with pyshacl (fails loudly if non-conformant), exports graph stats to
reports/kg_stats.md, and renders a pyvis HTML visualization of a 50-node
subgraph.

Run: python -m src.kg_build
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD
from rdflib.namespace import OWL, SKOS

from config import (  # noqa: E402
    FORBIDDEN_ADDITIVES,
    NAMESPACE,
    TARGET_PROPERTIES,
    USE_CASES,
    Formulation,
    validate_formulation,
)

DATASET_PATH = Path("data/processed/dataset.csv")
ONTOLOGY_PATH = Path("kg/ontology.ttl")
SHAPES_PATH = Path("kg/shapes.ttl")
GRAPH_PATH = Path("kg/graph.ttl")
STATS_PATH = Path("reports/kg_stats.md")
VIZ_PATH = Path("reports/figures/kg_subgraph.html")

NC = Namespace(NAMESPACE)

PROPERTY_CLASS: dict[str, str] = {
    "tensile_strength_MPa": "MechanicalProperty",
    "youngs_modulus_GPa": "MechanicalProperty",
    "thermal_conductivity_W_mK": "ThermalProperty",
    "water_absorption_pct": "MechanicalProperty",
    "limiting_oxygen_index_pct": "ThermalProperty",
    "density_g_cm3": "MechanicalProperty",
    "gwp_kgCO2e_per_kg": "SustainabilityMetric",
    "recyclability_score": "SustainabilityMetric",
    "biodegradability_score": "SustainabilityMetric",
}

MATERIAL_MAP: dict[str, str] = {
    "HDPE": "HDPE",
    "LDPE": "LDPE",
    "PP": "PP",
    "PET_recycled": "PET_recycled",
    "PLA": "PLA",
}

FILLER_MAP: dict[str, str] = {
    "none": None,
    "rice_husk": "rice_husk",
    "wheat_straw": "wheat_straw",
    "sugarcane_bagasse": "sugarcane_bagasse",
    "coconut_coir": "coconut_coir",
}

NANO_MAP: dict[str, str] = {
    "none": None,
    "SiO2": "SiO2",
    "nanoclay": "nanoclay",
    "TiO2": "TiO2",
    "cellulose_nanocrystal": "cellulose_nanocrystal",
}

APP_URI: dict[str, str] = {
    "insulation_panel": "insulation_panel",
    "structural_decking": "structural_decking",
}


def suitable_applications(row: pd.Series) -> list[str]:
    """Return list of application names the sample satisfies per USE_CASES."""
    suitable: list[str] = []
    for app_name, thresholds in USE_CASES.items():
        ok = True
        for prop, (op, value) in thresholds.items():
            v = row[prop]
            if op == "<=" and not (v <= value):
                ok = False
                break
            if op == ">=" and not (v >= value):
                ok = False
                break
        if ok:
            suitable.append(app_name)
    return suitable


def _render_shapes() -> str:
    """Render shapes.ttl with forbidden additives injected from config."""
    forbidden_blocks: list[str] = []
    for name in FORBIDDEN_ADDITIVES:
        forbidden_blocks.append(
            f"""    sh:property [
        sh:path nc:hasNanoparticle ;
        sh:not [ sh:hasValue nc:{name} ] ;
        sh:message "Forbidden additive nc:{name} detected." ;
    ]"""
        )

    weight_percent_block = """    sh:property [
        sh:path nc:weightPercent ;
        sh:minInclusive 0.0 ;
        sh:maxInclusive 40.0 ;
        sh:message "filler_wt_pct must be between 0 and 40 wt%." ;
    ]"""

    nano_blocks = ""
    if forbidden_blocks:
        nano_blocks = " ;\n" + " ;\n".join(forbidden_blocks)

    return f"""@prefix nc: <http://example.org/nanocomposite#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

###############################################################################
# SHACL shapes for the Nanocomposite Knowledge Graph
# Forbidden additives injected from config.FORBIDDEN_ADDITIVES.
###############################################################################

nc:NanocompositeShape a sh:NodeShape ;
    sh:targetClass nc:Nanocomposite ;
    sh:comment "Every nanocomposite must have exactly one matrix and a valid filler weight percent (0-40 wt%)." ;

    sh:property [
        sh:path nc:hasMatrix ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:message "A nanocomposite must have exactly one hasMatrix link." ;
    ] ;
{weight_percent_block}{nano_blocks} .

nc:PropertyShape a sh:NodeShape ;
    sh:targetClass nc:Property ;
    sh:comment "Every property measurement must have a numeric value and a unit string." ;

    sh:property [
        sh:path nc:hasValue ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:message "A property must have exactly one hasValue." ;
    ] ;

    sh:property [
        sh:path nc:hasUnit ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:message "A property must have exactly one hasUnit." ;
    ] .
"""


def build() -> Graph:
    """Build the RDF graph and return it (also serialized to kg/graph.ttl)."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH} (run src.clean first)")

    df = pd.read_csv(DATASET_PATH)

    g = Graph()
    g.bind("nc", NC)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    g.bind("xsd", XSD)

    g.parse(ONTOLOGY_PATH, format="turtle")

    for _, row in df.iterrows():
        row_id = int(row.name)
        sample = NC[f"sample_{row_id}"]

        row_dict = {k: row[k] for k in Formulation.__dataclass_fields__ if k in row}
        errs = validate_formulation(Formulation(**row_dict))
        if errs:
            raise ValueError(f"Row {row_id} failed validation: {errs}")

        g.add((sample, RDF.type, NC.Nanocomposite))
        g.add((sample, RDF.type, NC.Material))

        matrix_name = MATERIAL_MAP[row["matrix_polymer"]]
        matrix_uri = NC[matrix_name]
        g.add((sample, NC.hasMatrix, matrix_uri))
        if row["matrix_polymer"] == "PET_recycled":
            g.add((matrix_uri, RDF.type, NC.RecycledPolymer))

        filler_name = FILLER_MAP[row["agro_filler"]]
        if filler_name:
            filler_uri = NC[filler_name]
            g.add((sample, NC.hasFiller, filler_uri))
            g.add((sample, NC.weightPercent, Literal(float(row["filler_wt_pct"]), datatype=XSD.decimal)))
            if row["filler_particle_size_um"] and row["filler_particle_size_um"] > 0:
                g.add((sample, NC.particleSize, Literal(float(row["filler_particle_size_um"]), datatype=XSD.decimal)))

        nano_name = NANO_MAP[row["nanoparticle"]]
        if nano_name:
            nano_uri = NC[nano_name]
            g.add((sample, NC.hasNanoparticle, nano_uri))

        g.add((sample, NC.processedBy, NC.extrusion_process))
        if row["sonication_min"] and row["sonication_min"] > 0:
            g.add((sample, NC.processedBy, NC.sonication_process))
        if row["surface_treatment"] and row["surface_treatment"] != "none":
            g.add((sample, NC.processedBy, NC.surface_treatment_process))

        # --- all input features as datatype properties on the sample ---
        g.add((sample, NC.hasRecycledContentPct, Literal(float(row["recycled_content_pct"]), datatype=XSD.decimal)))
        g.add((sample, NC.hasFillerWtPct, Literal(float(row["filler_wt_pct"]), datatype=XSD.decimal)))
        if row["filler_particle_size_um"] and row["filler_particle_size_um"] > 0:
            g.add((sample, NC.hasFillerParticleSizeUm, Literal(float(row["filler_particle_size_um"]), datatype=XSD.decimal)))
        g.add((sample, NC.hasNanoWtPct, Literal(float(row["nano_wt_pct"]), datatype=XSD.decimal)))
        if row["nano_size_nm"] and row["nano_size_nm"] > 0:
            g.add((sample, NC.hasNanoSizeNm, Literal(float(row["nano_size_nm"]), datatype=XSD.decimal)))
        g.add((sample, NC.hasExtrusionTempC, Literal(float(row["extrusion_temp_C"]), datatype=XSD.decimal)))
        g.add((sample, NC.hasScrewSpeedRpm, Literal(float(row["screw_speed_rpm"]), datatype=XSD.decimal)))
        g.add((sample, NC.hasSonicationMin, Literal(float(row["sonication_min"]), datatype=XSD.decimal)))
        g.add((sample, NC.hasSurfaceTreatment, Literal(str(row["surface_treatment"]), datatype=XSD.string)))
        g.add((sample, NC.hasSource, Literal(str(row["source"]), datatype=XSD.string)))

        for prop, unit in TARGET_PROPERTIES.items():
            prop_class = PROPERTY_CLASS[prop]
            prop_node = NC[f"sample_{row_id}_{prop}"]
            g.add((prop_node, RDF.type, NC[prop_class]))
            g.add((prop_node, RDF.type, NC.Property))
            g.add((sample, NC.hasProperty, prop_node))
            g.add((prop_node, NC.hasValue, Literal(float(row[prop]), datatype=XSD.decimal)))
            g.add((prop_node, NC.hasUnit, Literal(unit)))

        for app_name in suitable_applications(row):
            g.add((sample, NC.suitableFor, NC[APP_URI[app_name]]))

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=GRAPH_PATH, format="turtle")
    return g


def validate_with_shacl(g: Graph) -> tuple[bool, str]:
    """Validate graph against shapes.ttl using pyshacl. Returns (conforms, report)."""
    try:
        from pyshacl import validate  # noqa: E402
    except ImportError as exc:
        raise ImportError("pyshacl is required for SHACL validation") from exc

    shapes_g = Graph()
    shapes_g.parse(SHAPES_PATH, format="turtle")

    conforms, results_graph, results_text = validate(
        g,
        shacl_graph=shapes_g,
        inference="rdfs",
        abort_on_error=False,
        meta_shacl=False,
        advanced=False,
        js=False,
        serialize_report=False,
    )
    return conforms, results_text


def export_stats(g: Graph) -> None:
    """Export graph statistics to reports/kg_stats.md."""
    from collections import Counter

    class_counts: Counter = Counter()
    for s, p, o in g:
        if p == RDF.type:
            class_counts[str(o)] += 1

    lines: list[str] = []
    lines.append("# Knowledge Graph Statistics")
    lines.append("")
    lines.append(f"Total triples: {len(g)}")
    lines.append(f"Unique subjects: {len(set(g.subjects()))}")
    lines.append("")
    lines.append("## Class counts (rdf:type)")
    for cls, cnt in class_counts.most_common():
        name = cls.split("#")[-1] if "#" in cls else cls
        lines.append(f"- {name}: {cnt}")
    lines.append("")
    lines.append("## Namespaces")
    for prefix, ns in g.namespaces():
        lines.append(f"- {prefix}: {ns}")
    lines.append("")
    lines.append("## Forbidden additives")
    if FORBIDDEN_ADDITIVES:
        for a in FORBIDDEN_ADDITIVES:
            lines.append(f"- {a}")
    else:
        lines.append("- (none configured)")

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text("\n".join(lines))


def visualize_subgraph(g: Graph, max_nodes: int = 50) -> None:
    """Render a pyvis HTML visualization of a subgraph (first max_nodes subjects)."""
    try:
        from pyvis.network import Network  # noqa: E402
    except ImportError as exc:
        raise ImportError("pyvis is required for visualization") from exc

    net = Network(notebook=False, directed=True, cdn_resources="remote")
    net.toggle_physics(True)

    subjects = list(g.subjects())[:max_nodes]
    subject_set = set(str(s) for s in subjects)

    for s in subjects:
        label = str(s).split("#")[-1]
        net.add_node(str(s), label=label, title=str(s))

    for s, p, o in g:
        if str(s) in subject_set and str(o) in subject_set:
            net.add_edge(str(s), str(o), label=str(p).split("#")[-1])

    VIZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(VIZ_PATH))
    print(f"Pyvis subgraph saved -> {VIZ_PATH} ({len(subjects)} nodes)")


def main() -> None:
    """CLI entry: python -m src.kg_build"""
    g = build()
    print(f"Graph built: {len(g)} triples -> {GRAPH_PATH}")

    SHAPES_PATH.write_text(_render_shapes())
    print(f"SHACL shapes written -> {SHAPES_PATH}")

    conforms, report = validate_with_shacl(g)
    print("=== SHACL Conformance Report ===")
    print(report)
    print("=================================")
    if not conforms:
        raise RuntimeError("Knowledge graph does NOT conform to SHACL shapes")
    print("SHACL conformance: PASS")

    export_stats(g)
    print(f"Graph stats -> {STATS_PATH}")

    visualize_subgraph(g, max_nodes=50)


if __name__ == "__main__":
    main()
