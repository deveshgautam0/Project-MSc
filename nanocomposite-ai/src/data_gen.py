"""Physics-inspired synthetic data generator (Phase 2).

Generates N=400 synthetic nanocomposite rows (source="synthetic") using simple,
documented physics-inspired models plus Gaussian noise (5% relative).

Every equation below is an approximate, illustrative surrogate --- NOT a
validated empirical model. Replace with real literature data before any design
decision.

All numeric outputs are stored as plain floats with units defined in config.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    TARGET_PROPERTIES,
    Formulation,
    validate_formulation,
)

RNG_SEED = 42
N_SYNTHETIC = 400
NOISE_RELATIVE = 0.05  # 5% relative Gaussian noise

# Base property values per matrix polymer (illustrative surrogates).
MATRIX_BASE: dict[str, dict[str, float]] = {
    "HDPE": {
        "tensile_strength_MPa": 25.0,
        "youngs_modulus_GPa": 1.0,
        "thermal_conductivity_W_mK": 0.42,
        "water_absorption_pct": 2.0,
        "limiting_oxygen_index_pct": 17.5,
        "density_g_cm3": 0.95,
        "gwp_kgCO2e_per_kg": 2.0,
        "recyclability_score": 0.80,
        "biodegradability_score": 0.05,
    },
    "LDPE": {
        "tensile_strength_MPa": 15.0,
        "youngs_modulus_GPa": 0.6,
        "thermal_conductivity_W_mK": 0.38,
        "water_absorption_pct": 1.8,
        "limiting_oxygen_index_pct": 17.4,
        "density_g_cm3": 0.92,
        "gwp_kgCO2e_per_kg": 3.2,
        "recyclability_score": 0.40,
        "biodegradability_score": 0.05,
    },
    "PP": {
        "tensile_strength_MPa": 35.0,
        "youngs_modulus_GPa": 1.4,
        "thermal_conductivity_W_mK": 0.35,
        "water_absorption_pct": 1.5,
        "limiting_oxygen_index_pct": 17.8,
        "density_g_cm3": 0.91,
        "gwp_kgCO2e_per_kg": 1.9,
        "recyclability_score": 0.85,
        "biodegradability_score": 0.05,
    },
    "PET_recycled": {
        "tensile_strength_MPa": 55.0,
        "youngs_modulus_GPa": 2.2,
        "thermal_conductivity_W_mK": 0.45,
        "water_absorption_pct": 1.2,
        "limiting_oxygen_index_pct": 18.5,
        "density_g_cm3": 1.35,
        "gwp_kgCO2e_per_kg": 1.5,
        "recyclability_score": 0.90,
        "biodegradability_score": 0.10,
    },
    "PLA": {
        "tensile_strength_MPa": 50.0,
        "youngs_modulus_GPa": 2.0,
        "thermal_conductivity_W_mK": 0.33,
        "water_absorption_pct": 3.0,
        "limiting_oxygen_index_pct": 18.8,
        "density_g_cm3": 1.25,
        "gwp_kgCO2e_per_kg": 1.2,
        "recyclability_score": 0.50,
        "biodegradability_score": 0.90,
    },
}

# Agro-filler effect on thermal conductivity (multiplier).
AGRO_THERMAL_MULT: dict[str, float] = {
    "none": 1.00,
    "rice_husk": 0.85,
    "wheat_straw": 0.88,
    "sugarcane_bagasse": 0.92,
    "coconut_coir": 0.94,
}

# Agro-filler effect on water absorption (additive % per wt% of filler).
AGRO_WATER_ADD_PER_WT: dict[str, float] = {
    "none": 0.0,
    "rice_husk": 0.06,
    "wheat_straw": 0.05,
    "sugarcane_bagasse": 0.04,
    "coconut_coir": 0.03,
}

# Surface-treatment multipliers.
TREATMENT_STRENGTH_MULT: dict[str, float] = {
    "none": 1.00,
    "silane": 1.10,
    "NaOH": 1.05,
    "MAPE_compatibilizer": 1.08,
}
TREATMENT_WATER_MULT: dict[str, float] = {
    "none": 1.00,
    "silane": 0.70,
    "NaOH": 0.85,
    "MAPE_compatibilizer": 0.75,
}

# Nano effect on LOI (% per wt%).
NANO_LOI_PER_WT: dict[str, float] = {
    "none": 0.0,
    "SiO2": 0.5,
    "nanoclay": 0.3,
    "TiO2": 0.2,
    "cellulose_nanocrystal": 0.1,
}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _add_noise(value: float, rng: np.random.Generator) -> float:
    """Add 5% relative Gaussian noise, keeping sign for scores in [0,1]."""
    return float(value * (1.0 + rng.normal(0.0, NOISE_RELATIVE)))


def generate_one(matrix: str, rng: np.random.Generator) -> dict[str, Any]:
    """Generate a single synthetic formulation + predicted properties.

    Equations (all documented; approximate surrogates):

    Tensile strength (MPa):
        base(matrix)
        * (1 - 0.004 * filler_wt_pct)            # filler dilution
        * agro_strength_mult                      # agro type penalty
        * (1 + 0.05 * min(nano_wt_pct, 3)         # nano reinforcement up to 3 wt%
             - 0.08 * max(0, nano_wt_pct - 3))    # agglomeration penalty above 3
        * treatment_strength_mult                 # coupling agent boost
        * (1 + 0.002 * sonication_min)            # better dispersion
        * (1 - 0.001 * recycled_content_pct)      # recycled content slight penalty

    Young's modulus (GPa) = 0.04 * tensile_strength (+ noise).

    Thermal conductivity (W/mK) [Maxwell-Eucken-style]:
        base(matrix) * agro_thermal_mult
        + 0.02 * nano_wt_pct                      # SiO2 raises slightly

    Water absorption (%):
        (base(matrix) + filler_wt_pct * agro_water_add_per_wt)
        * treatment_water_mult

    Limiting oxygen index (%):
        base(matrix) + nano_loi_per_wt * nano_wt_pct

    Density (g/cm3): base(matrix) + small filler/nano contribution.

    GWP (kgCO2e/kg):
        base(matrix) * (1 - 0.004 * recycled_content_pct) * agro_gwp_mult

    Recyclability / Biodegradability (0-1): base(matrix) adjusted by filler.
    """
    base = MATRIX_BASE[matrix]

    # --- sample inputs within config ranges ---
    recycled_content_pct = float(rng.uniform(*NUMERIC_FEATURES["recycled_content_pct"][:2]))
    agro_filler = rng.choice(CATEGORICAL_FEATURES["agro_filler"])
    filler_wt_pct = float(rng.uniform(*NUMERIC_FEATURES["filler_wt_pct"][:2]))
    # Particle size is N/A (0) when there is no agro filler.
    filler_particle_size_um = 0.0 if agro_filler == "none" else float(rng.uniform(*NUMERIC_FEATURES["filler_particle_size_um"][:2]))
    nanoparticle = rng.choice(CATEGORICAL_FEATURES["nanoparticle"])
    # Nano size is N/A (0) when there is no nanoparticle.
    nano_size_nm = 0.0 if nanoparticle == "none" else float(rng.uniform(*NUMERIC_FEATURES["nano_size_nm"][:2]))
    nano_wt_pct = 0.0 if nanoparticle == "none" else float(rng.uniform(*NUMERIC_FEATURES["nano_wt_pct"][:2]))
    surface_treatment = rng.choice(CATEGORICAL_FEATURES["surface_treatment"])
    extrusion_temp_C = float(rng.uniform(*NUMERIC_FEATURES["extrusion_temp_C"][:2]))
    screw_speed_rpm = float(rng.uniform(*NUMERIC_FEATURES["screw_speed_rpm"][:2]))
    sonication_min = float(rng.uniform(*NUMERIC_FEATURES["sonication_min"][:2]))

    # --- tensile strength ---
    agro_strength_mult = {
        "none": 1.00, "rice_husk": 0.90, "wheat_straw": 0.92,
        "sugarcane_bagasse": 0.94, "coconut_coir": 0.93,
    }[agro_filler]
    nano_bonus = 0.05 * min(nano_wt_pct, 3.0) - 0.08 * max(0.0, nano_wt_pct - 3.0)
    tensile = (
        base["tensile_strength_MPa"]
        * (1.0 - 0.004 * filler_wt_pct)
        * agro_strength_mult
        * (1.0 + nano_bonus)
        * TREATMENT_STRENGTH_MULT[surface_treatment]
        * (1.0 + 0.002 * sonication_min)
        * (1.0 - 0.001 * recycled_content_pct)
    )
    tensile = _clip(tensile, 1.0, 80.0)

    # --- youngs modulus ---
    youngs = base["youngs_modulus_GPa"] * (tensile / base["tensile_strength_MPa"])

    # --- thermal conductivity ---
    thermal = (
        base["thermal_conductivity_W_mK"] * AGRO_THERMAL_MULT[agro_filler]
        + 0.02 * nano_wt_pct
    )
    thermal = _clip(thermal, 0.1, 1.5)

    # --- water absorption ---
    water = (
        (base["water_absorption_pct"] + filler_wt_pct * AGRO_WATER_ADD_PER_WT[agro_filler])
        * TREATMENT_WATER_MULT[surface_treatment]
    )
    water = _clip(water, 0.2, 15.0)

    # --- LOI ---
    loi = base["limiting_oxygen_index_pct"] + NANO_LOI_PER_WT[nanoparticle] * nano_wt_pct
    loi = _clip(loi, 15.0, 28.0)

    # --- density ---
    density = base["density_g_cm3"] + 0.001 * filler_wt_pct + 0.005 * nano_wt_pct
    density = _clip(density, 0.8, 1.6)

    # --- GWP ---
    agro_gwp_mult = {"none": 1.0, "rice_husk": 0.85, "wheat_straw": 0.88,
                     "sugarcane_bagasse": 0.90, "coconut_coir": 0.92}[agro_filler]
    gwp = base["gwp_kgCO2e_per_kg"] * (1.0 - 0.004 * recycled_content_pct) * agro_gwp_mult
    gwp = _clip(gwp, 0.2, 5.0)

    # --- recyclability / biodegradability ---
    recyclability = base["recyclability_score"] * (1.0 + 0.001 * recycled_content_pct)
    if agro_filler != "none":
        recyclability *= 0.95
    recyclability = _clip(recyclability, 0.0, 1.0)

    biodegradability = base["biodegradability_score"]
    if agro_filler != "none":
        biodegradability = min(1.0, biodegradability + 0.003 * filler_wt_pct)
    if matrix == "PLA":
        biodegradability = min(1.0, biodegradability + 0.1)
    biodegradability = _clip(biodegradability, 0.0, 1.0)

    # --- apply 5% relative noise to targets ---
    targets = {
        "tensile_strength_MPa": tensile,
        "youngs_modulus_GPa": youngs,
        "thermal_conductivity_W_mK": thermal,
        "water_absorption_pct": water,
        "limiting_oxygen_index_pct": loi,
        "density_g_cm3": density,
        "gwp_kgCO2e_per_kg": gwp,
        "recyclability_score": recyclability,
        "biodegradability_score": biodegradability,
    }
    targets = {k: _add_noise(v, rng) for k, v in targets.items()}
    targets = {k: _clip(v, 0.0, 1.0) if k in ("recyclability_score", "biodegradability_score") else v
               for k, v in targets.items()}

    row: dict[str, Any] = {
        "matrix_polymer": matrix,
        "recycled_content_pct": recycled_content_pct,
        "agro_filler": agro_filler,
        "filler_wt_pct": filler_wt_pct,
        "filler_particle_size_um": filler_particle_size_um,
        "nanoparticle": nanoparticle,
        "nano_wt_pct": nano_wt_pct,
        "nano_size_nm": nano_size_nm,
        "surface_treatment": surface_treatment,
        "extrusion_temp_C": extrusion_temp_C,
        "screw_speed_rpm": screw_speed_rpm,
        "sonication_min": sonication_min,
        "source": "synthetic",
        "reference": "synthetic physics surrogate",
    }
    row.update(targets)
    return row


def generate(n: int = N_SYNTHETIC, seed: int = RNG_SEED) -> pd.DataFrame:
    """Generate ``n`` synthetic rows and return a DataFrame."""
    rng = np.random.default_rng(seed)
    matrices = CATEGORICAL_FEATURES["matrix_polymer"]
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        matrix = rng.choice(matrices)
        rows.append(generate_one(matrix, rng))
    df = pd.DataFrame(rows, columns=INPUT_FEATURES + list(TARGET_PROPERTIES.keys()) + ["source", "reference"])
    # Validate every generated row
    bad = 0
    for _, r in df.iterrows():
        errs = validate_formulation(Formulation(**{k: r[k] for k in Formulation.__dataclass_fields__ if k in r}))
        if errs:
            bad += 1
    if bad:
        raise ValueError(f"{bad} generated rows failed validation")
    return df


def main() -> None:
    """CLI entry: python -m src.data_gen  -> writes data/processed/synthetic.csv"""
    from pathlib import Path  # noqa: PLC0415

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate()
    out_path = out_dir / "synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic rows -> {out_path}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
