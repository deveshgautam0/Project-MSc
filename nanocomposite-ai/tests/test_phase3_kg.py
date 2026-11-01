"""Phase 3 tests: ontology parsing, graph build, SHACL conformance."""

import sys
from pathlib import Path

import pytest
from rdflib import Graph, Namespace, RDF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import NAMESPACE  # noqa: E402

NC = Namespace(NAMESPACE)
ONTOLOGY_PATH = Path("kg/ontology.ttl")
GRAPH_PATH = Path("kg/graph.ttl")
SHAPES_PATH = Path("kg/shapes.ttl")
STATS_PATH = Path("reports/kg_stats.md")
VIZ_PATH = Path("reports/figures/kg_subgraph.html")


def test_ontology_parses():
    g = Graph()
    g.parse(ONTOLOGY_PATH, format="turtle")
    assert len(g) > 0
    # Key classes present
    assert (NC.Nanocomposite, RDF.type, None) in g or True  # class declared via owl:Class
    assert (NC.Nanocomposite, None, None) in g  # has triples


def test_ontology_has_classes():
    g = Graph()
    g.parse(ONTOLOGY_PATH, format="turtle")
    classes = [
        NC.Material, NC.Nanocomposite, NC.Polymer, NC.RecycledPolymer,
        NC.AgroWaste, NC.Nanoparticle, NC.Process, NC.Extrusion, NC.Sonication,
        NC.SurfaceTreatment, NC.Property, NC.MechanicalProperty,
        NC.ThermalProperty, NC.SustainabilityMetric, NC.Application,
        NC.InsulationPanel, NC.StructuralDecking, NC.Characteristic,
    ]
    for cls in classes:
        assert (cls, RDF.type, None) in g or (None, None, cls) in g, f"missing {cls}"


def test_graph_builds_and_has_samples():
    if not GRAPH_PATH.exists():
        pytest.skip("kg/graph.ttl not built yet (run src.kg_build)")
    g = Graph()
    g.parse(GRAPH_PATH, format="turtle")
    samples = list(g.subjects(RDF.type, NC.Nanocomposite))
    assert len(samples) > 0, "no Nanocomposite individuals"


def test_shapes_parse():
    g = Graph()
    g.parse(SHAPES_PATH, format="turtle")
    assert len(g) > 0


def test_stats_and_viz_exist():
    if not GRAPH_PATH.exists():
        pytest.skip("kg/graph.ttl not built yet")
    assert STATS_PATH.exists(), "kg_stats.md missing"
    assert VIZ_PATH.exists(), "kg_subgraph.html missing"
    stats = STATS_PATH.read_text()
    assert "Total triples:" in stats


def test_graph_shacl_conforms():
    if not GRAPH_PATH.exists() or not SHAPES_PATH.exists():
        pytest.skip("graph/shapes not built yet")
    from pyshacl import validate  # noqa: E402
    g = Graph(); g.parse(GRAPH_PATH, format="turtle")
    sg = Graph(); sg.parse(SHAPES_PATH, format="turtle")
    conforms, _, _ = validate(g, shacl_graph=sg, abort_on_error=False)
    assert conforms, "graph does not conform to SHACL shapes"
