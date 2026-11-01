"""src/data_ingestion.py -- Gap 1: Live ingestion pipelines for open sources."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import DATA_ACQUISITION_SOURCES, INPUT_FEATURES, TARGET_PROPERTIES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REQUEST_TIMEOUT = 20
USER_AGENT = "nanocomposite-ai/1.0"

TARGET_COLS = list(TARGET_PROPERTIES.keys())
ALL_COLS = INPUT_FEATURES + TARGET_COLS + ["source", "reference"]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_COLS)


def _safe_request(url, params=None, headers=None):
    try:
        h = {"User-Agent": USER_AGENT}
        if headers:
            h.update(headers)
        resp = requests.get(url, params=params, headers=h, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except Exception:
        return None


def fetch_polynims():
    df = _empty_df()
    url = DATA_ACQUISITION_SOURCES["Polymer/Bioplastic Data"]["PoLyInfo (NIMS)"]
    print(f"  [INFO] PoLyInfo: {url} -- manual export required.")
    return df


def fetch_polymergenome():
    df = _empty_df()
    api_key = os.getenv("POLYGENOME_API_KEY")
    if not api_key:
        print("  [WARN] Polymer Genome: POLYGENOME_API_KEY not set -- skipping.")
        return df
    url = "https://polymergenome.org/api/properties"
    resp = _safe_request(url, headers={"Authorization": f"Bearer {api_key}"})
    if resp is None:
        return df
    try:
        data = resp.json()
        rows = []
        for item in data:
            row = {c: item.get(c) for c in ALL_COLS}
            row["source"] = "literature"
            row["reference"] = "Polymer Genome API"
            rows.append(row)
        df = pd.DataFrame(rows, columns=ALL_COLS)
    except Exception:
        pass
    return df


def fetch_pubchem():
    df = _empty_df()
    cids = ["cellulose", "lignin", "hemicellulose", "rice husk silica",
            "wheat straw", "sugarcane bagasse", "coconut coir"]
    rows = []
    for cid in cids:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
        resp = _safe_request(url)
        if resp is None:
            continue
        try:
            props = resp.json()["PropertyTable"]["Properties"][0]
            row = {"matrix_polymer": "none",
                   "agro_filler": cid.replace(" ", "_"),
                   "nanoparticle": "none", "surface_treatment": "none",
                   "source": "literature",
                   "reference": f"PubChem CID {cid}"}
            for k in INPUT_FEATURES:
                if k not in row:
                    row[k] = 0.0
            for t in TARGET_COLS:
                row[t] = 0.0
            rows.append(row)
        except Exception:
            pass
        time.sleep(0.2)
    if rows:
        df = pd.DataFrame(rows, columns=ALL_COLS)
    return df