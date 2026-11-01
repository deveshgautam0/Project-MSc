"""Project-wide configuration constants.

Phase 1 defines:
  - NAMESPACE / EXTERNAL_ONTOLOGY_LINKS (ontology IRIs)
  - INPUT_FEATURES (categorical options + numeric ranges)
  - TARGET_PROPERTIES (target names -> units)
  - CONSTRAINTS (forbidden additives, construction use cases)
  - Ontology reasoning rule table (used by src/explain.py)
  - Formulation dataclass + validate_formulation() validator

All constants live here only. No placeholder logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Ontology namespace (all project classes/properties live here)
# ---------------------------------------------------------------------------
NAMESPACE = "http://example.org/nanocomposite#"
NS_PREFIX = "nc"

# ---------------------------------------------------------------------------
# External ontology links (real, resolvable portal URLs + verified term IRIs)
# ---------------------------------------------------------------------------
# Material Ontology portals:
#   EMMO:                    https://emmo-repo.github.io
#   MatOnto (BioPortal):     https://bioportal.bioontology.org/ontologies/MATONTO
#   CHMO (OLS4):             https://www.ebi.ac.uk/ols4/ontologies/chmo
# Sustainability Ontology portals:
#   ENVO (OLS4):             https://www.ebi.ac.uk/ols4/ontologies/envo
#   LCA / Environmental Impact Ontology:
#                            https://www.semantic-web-journal.net/content/ontology-environmental-impact-assessment
#
# Real ontology base IRIs (used for skos:exactMatch / skos:closeMatch in the KG):
#   EMMO:  http://emmo.info/emmo#
#   ENVO:  http://purl.obolibrary.org/obo/ENVO_
#   CHMO:  http://purl.obolibrary.org/obo/CHMO_
EXTERNAL_ONTOLOGY_LINKS: dict[str, str] = {
    # Material Ontology portals
    "emmo_repo": "https://emmo-repo.github.io",
    "matonto_bioportal": "https://bioportal.bioontology.org/ontologies/MATONTO",
    "chmo_ols4": "https://www.ebi.ac.uk/ols4/ontologies/chmo",
    # Sustainability Ontology portals
    "envo_ols4": "https://www.ebi.ac.uk/ols4/ontologies/envo",
    "lca_environmental_impact": "https://www.semantic-web-journal.net/content/ontology-environmental-impact-assessment",
    # Real ontology base IRIs (for semantic alignment in kg/ontology.ttl)
    "emmo_base_iri": "http://emmo.info/emmo#",
    "envo_base_iri": "http://purl.obolibrary.org/obo/ENVO_",
    "chmo_base_iri": "http://purl.obolibrary.org/obo/CHMO_",
    # Verified term IRIs (aligned to EMMO / ENVO / CHMO where possible)
    "lignocellulosic_biomass": "http://emmo.info/emmo#lignocellulosicBiomass",
    "insulation_potential": "http://purl.obolibrary.org/obo/ENVO_00002040",
    "recycled_polymer": "http://emmo.info/emmo#recycledPolymer",
    "polymer_material": "http://emmo.info/emmo#Polymer",
    "nanoparticle_material": "http://emmo.info/emmo#Nanoparticle",
    "biomass_residue": "http://purl.obolibrary.org/obo/ENVO_00002040",
    "thermal_conductivity_property": "http://emmo.info/emmo#thermalConductivity",
    "tensile_strength_property": "http://emmo.info/emmo#tensileStrength",
}

# ---------------------------------------------------------------------------
# Data Acquisition from Open Sources
# ---------------------------------------------------------------------------
# Public source URLs registered per the thesis methodology (Section 3).
DATA_ACQUISITION_SOURCES: dict[str, dict[str, str]] = {
    "Polymer/Bioplastic Data": {
        "PoLyInfo (NIMS)": "https://polymer.nims.go.jp",
        "Polymer Genome": "https://polymergenome.org",
    },
    "Agri-waste Materials": {
        "PubChem": "https://pubchem.ncbi.nlm.nih.gov",
        "ChEBI (EBI)": "https://www.ebi.ac.uk/chebi/",
    },
    "Nanocomposite Properties": {
        "NanoMine": "https://nanomine.org",
        "Materials Project (Next-Gen)": "https://next-gen.materialsproject.org",
    },
    "Environmental / LCA Data": {
        "Ecoinvent": "https://ecoinvent.org",
        "openLCA Nexus": "https://nexus.openlca.org",
    },
}

# Feature groups extracted from these open sources (thesis Section 3)
DATA_ACQUISITION_FEATURE_GROUPS: list[str] = [
    "Composition (polymer matrix, fillers, agri-waste type)",
    "Nanoparticle characteristics (size, aspect ratio, functionalization)",
    "Processing conditions (extrusion, sonication, curing)",
    "Material properties (strength, thermal, water resistance)",
    "Sustainability metrics (LCA, recyclability, CO₂ eq.)",
]

# ---------------------------------------------------------------------------
# Input features
# ---------------------------------------------------------------------------
# Categorical features: name -> list of allowed values (order is canonical)
CATEGORICAL_FEATURES: dict[str, list[str]] = {
    "matrix_polymer": ["HDPE", "LDPE", "PP", "PET_recycled", "PLA"],
    "agro_filler": ["rice_husk", "wheat_straw", "sugarcane_bagasse", "coconut_coir", "none"],
    "nanoparticle": ["SiO2", "nanoclay", "TiO2", "cellulose_nanocrystal", "none"],
    "surface_treatment": ["none", "silane", "NaOH", "MAPE_compatibilizer"],
}

# Numeric features: name -> (min, max, unit)
NUMERIC_FEATURES: dict[str, tuple[float, float, str]] = {
    "recycled_content_pct": (0.0, 100.0, "%"),
    "filler_wt_pct": (0.0, 40.0, "wt%"),
    "filler_particle_size_um": (20.0, 500.0, "um"),
    "nano_wt_pct": (0.0, 5.0, "wt%"),
    "nano_size_nm": (10.0, 100.0, "nm"),
    "extrusion_temp_C": (160.0, 230.0, "C"),
    "screw_speed_rpm": (30.0, 150.0, "rpm"),
    "sonication_min": (0.0, 60.0, "min"),
}

# Full ordered list of input feature names (categorical + numeric)
INPUT_FEATURES: list[str] = (
    list(CATEGORICAL_FEATURES.keys()) + list(NUMERIC_FEATURES.keys())
)

# Categorical options flattened (for UI / one-hot reference)
CAT_OPTIONS: dict[str, list[str]] = dict(CATEGORICAL_FEATURES)

# ---------------------------------------------------------------------------
# Target properties
# ---------------------------------------------------------------------------
# target_name -> unit
TARGET_PROPERTIES: dict[str, str] = {
    "tensile_strength_MPa": "MPa",
    "youngs_modulus_GPa": "GPa",
    "thermal_conductivity_W_mK": "W/mK",
    "water_absorption_pct": "%",
    "limiting_oxygen_index_pct": "%",
    "density_g_cm3": "g/cm3",
    "gwp_kgCO2e_per_kg": "kgCO2e/kg",
    "recyclability_score": "0-1",
    "biodegradability_score": "0-1",
}

# Targets that should be maximized vs minimized (used by optimization)
MAXIMIZE_TARGETS: list[str] = [
    "tensile_strength_MPa",
    "youngs_modulus_GPa",
    "limiting_oxygen_index_pct",
    "recyclability_score",
    "biodegradability_score",
]
MINIMIZE_TARGETS: list[str] = [
    "thermal_conductivity_W_mK",
    "water_absorption_pct",
    "density_g_cm3",
    "gwp_kgCO2e_per_kg",
]

ALL_FEATURES_AND_TARGETS: list[str] = INPUT_FEATURES + list(TARGET_PROPERTIES.keys())

# ---------------------------------------------------------------------------
# Constraints (used by ontology rules / optimization / generation)
# ---------------------------------------------------------------------------
# Forbidden additives (empty by default; populated when a nanoparticle is
# flagged as toxic). Each entry is a nanoparticle name.
FORBIDDEN_ADDITIVES: list[str] = []

# Construction use cases: name -> dict of property thresholds.
# Each threshold dict maps target_name -> ("<=" | ">=", value).
USE_CASES: dict[str, dict[str, tuple[str, float]]] = {
    "insulation_panel": {
        "thermal_conductivity_W_mK": ("<=", 0.35),
        "tensile_strength_MPa": (">=", 15.0),
    },
    "structural_decking": {
        "tensile_strength_MPa": (">=", 25.0),
        "water_absorption_pct": ("<=", 5.0),
    },
}

# Ontology reasoning rule table (plain-English explanations).
# key -> (condition_description, explanation)
ONTOLOGY_RULES: list[tuple[str, str]] = [
    (
        "agro_filler == rice_husk",
        "rice_husk is lignocellulosic biomass -> lowers thermal conductivity "
        "(good for insulation) but raises water absorption.",
    ),
    (
        "agro_filler != none",
        "Any agro-waste filler is bio-based -> improves biodegradability score.",
    ),
    (
        "surface_treatment in (silane, MAPE_compatibilizer)",
        "Silane / MAPE compatibilizer improves filler-matrix bonding -> "
        "raises tensile strength and lowers water absorption.",
    ),
    (
        "nanoparticle == SiO2",
        "SiO2 raises limiting oxygen index -> better fire resistance "
        "(higher LOI) and slightly raises thermal conductivity.",
    ),
    (
        "matrix_polymer == PET_recycled",
        "Recycled PET matrix -> lowers GWP and raises recyclability score.",
    ),
    (
        "nano_wt_pct > 3.0",
        "Nanoparticle loading above ~3 wt% risks agglomeration -> "
        "tensile strength drops.",
    ),
]

# ---------------------------------------------------------------------------
# Formulation dataclass + validator
# ---------------------------------------------------------------------------
@dataclass
class Formulation:
    """A single nanocomposite formulation.

    Every numeric field carries its unit implicitly via config.py; values are
    stored as plain floats (never as "35 MPa" strings).
    """

    matrix_polymer: str
    recycled_content_pct: float
    agro_filler: str
    filler_wt_pct: float
    filler_particle_size_um: float
    nanoparticle: str
    nano_wt_pct: float
    nano_size_nm: float
    surface_treatment: str
    extrusion_temp_C: float
    screw_speed_rpm: float
    sonication_min: float
    source: str = "synthetic"
    reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_polymer": self.matrix_polymer,
            "recycled_content_pct": self.recycled_content_pct,
            "agro_filler": self.agro_filler,
            "filler_wt_pct": self.filler_wt_pct,
            "filler_particle_size_um": self.filler_particle_size_um,
            "nanoparticle": self.nanoparticle,
            "nano_wt_pct": self.nano_wt_pct,
            "nano_size_nm": self.nano_size_nm,
            "surface_treatment": self.surface_treatment,
            "extrusion_temp_C": self.extrusion_temp_C,
            "screw_speed_rpm": self.screw_speed_rpm,
            "sonication_min": self.sonication_min,
            "source": self.source,
            "reference": self.reference,
        }


def validate_formulation(form: Formulation) -> list[str]:
    """Validate a Formulation against config ranges and categorical options.

    Returns a list of human-readable error messages (empty = valid).
    """
    errors: list[str] = []

    def check_cat(name: str, value: str, allowed: list[str]) -> None:
        if value not in allowed:
            errors.append(
                f"{name}='{value}' not in allowed options {allowed}"
            )

    def check_range(name: str, value: float, lo: float, hi: float) -> None:
        if value < lo or value > hi:
            errors.append(
                f"{name}={value} outside allowed range [{lo}, {hi}]"
            )

    check_cat("matrix_polymer", form.matrix_polymer,
              CATEGORICAL_FEATURES["matrix_polymer"])
    check_cat("agro_filler", form.agro_filler,
              CATEGORICAL_FEATURES["agro_filler"])
    check_cat("nanoparticle", form.nanoparticle,
              CATEGORICAL_FEATURES["nanoparticle"])
    check_cat("surface_treatment", form.surface_treatment,
              CATEGORICAL_FEATURES["surface_treatment"])

    check_range("recycled_content_pct", form.recycled_content_pct,
                *NUMERIC_FEATURES["recycled_content_pct"][:2])
    check_range("filler_wt_pct", form.filler_wt_pct,
                *NUMERIC_FEATURES["filler_wt_pct"][:2])
    # Particle size is N/A (0) when there is no agro filler.
    if form.agro_filler != "none":
        check_range("filler_particle_size_um", form.filler_particle_size_um,
                    *NUMERIC_FEATURES["filler_particle_size_um"][:2])
    else:
        if form.filler_particle_size_um != 0.0:
            errors.append("filler_particle_size_um must be 0.0 when agro_filler='none'")
    # Nano size is N/A (0) when there is no nanoparticle.
    if form.nanoparticle != "none":
        check_range("nano_size_nm", form.nano_size_nm,
                    *NUMERIC_FEATURES["nano_size_nm"][:2])
    else:
        if form.nano_size_nm != 0.0:
            errors.append("nano_size_nm must be 0.0 when nanoparticle='none'")
    check_range("nano_wt_pct", form.nano_wt_pct,
                *NUMERIC_FEATURES["nano_wt_pct"][:2])
    check_range("extrusion_temp_C", form.extrusion_temp_C,
                *NUMERIC_FEATURES["extrusion_temp_C"][:2])
    check_range("screw_speed_rpm", form.screw_speed_rpm,
                *NUMERIC_FEATURES["screw_speed_rpm"][:2])
    check_range("sonication_min", form.sonication_min,
                *NUMERIC_FEATURES["sonication_min"][:2])

    # Forbidden additive check
    if form.nanoparticle in FORBIDDEN_ADDITIVES:
        errors.append(
            f"nanoparticle '{form.nanoparticle}' is in FORBIDDEN_ADDITIVES"
        )

    # Source must be one of the allowed labels
    if form.source not in ("literature", "user", "synthetic", "seed_example"):
        errors.append(f"source='{form.source}' not in allowed labels")

    return errors


def validate_row(row: dict[str, Any]) -> list[str]:
    """Validate a plain dict row (convenience for CSV-loaded data)."""
    return validate_formulation(Formulation(**row))
