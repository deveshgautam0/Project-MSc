"""SPARQL query runner (Phase 4).

Loads queries from kg/queries/<name>.rq and runs them against kg/graph.ttl,
returning a pandas DataFrame.

Usage:
    from src.kg_query import run_query
    df = run_query("a_insulation_panel")

CLI: python -m src.kg_query <name>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph

from config import NAMESPACE  # noqa: E402

QUERIES_DIR = Path("kg/queries")
GRAPH_PATH = Path("kg/graph.ttl")

NC = None  # not strictly needed; queries use full IRIs

# Load the graph once
_GRAPH: Graph | None = None


def _graph() -> Graph:
    global _GRAPH
    if _GRAPH is None:
        if not GRAPH_PATH.exists():
            raise FileNotFoundError(
                f"Missing {GRAPH_PATH} (run src.kg_build first)"
            )
        _GRAPH = Graph()
        _GRAPH.parse(GRAPH_PATH, format="turtle")
    return _GRAPH


def _query_path(name: str) -> Path:
    # Accept names with or without .rq extension
    if not name.endswith(".rq"):
        name = f"{name}.rq"
    path = QUERIES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Query not found: {path}")
    return path


def run_query(name: str) -> pd.DataFrame:
    """Run SPARQL query ``kg/queries/<name>.rq`` against kg/graph.ttl.

    Returns a pandas DataFrame. Raises FileNotFoundError if the query file or
    the graph is missing.
    """
    query_text = _query_path(name).read_text()
    g = _graph()
    rows = g.query(query_text)
    # rdflql query rows -> list of tuples; convert via pandas
    data: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for var in rows.vars:
            val = row[var]
            # unwrap rdflib term to python value
            record[str(var)] = val.toPython() if hasattr(val, "toPython") else str(val)
        data.append(record)
    return pd.DataFrame(data)


def main() -> None:
    """CLI entry: python -m src.kg_query <name>"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.kg_query <query_name>")
        print("Available queries:")
        for p in sorted(QUERIES_DIR.glob("*.rq")):
            print(f"  - {p.stem}")
        raise SystemExit(1)
    name = sys.argv[1]
    df = run_query(name)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
