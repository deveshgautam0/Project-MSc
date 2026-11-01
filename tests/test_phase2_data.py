"""Phase 2 tests: seed loader, synthetic generator, cleaner."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    INPUT_FEATURES,
    TARGET_PROPERTIES,
    Formulation,
    validate_formulation,
)
from src import clean  # noqa: E402  (runs clean.main)

ALL_COLUMNS = INPUT_FEATURES + list(TARGET_PROPERTIES.keys()) + ["source", "reference"]


def test_seed_has_required_columns():
    df = clean.load_seed()
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    assert not missing, f"missing columns: {missing}"
    assert len(df) >= 15


def test_seed_source_is_literature():
    df = clean.load_seed()
    assert (df["source"] == "literature").all(), "seed data should be source=literature"


def test_synthetic_generation_shape_and_columns():
    from src import data_gen  # noqa: E402

    df = data_gen.generate(n=20, seed=7)
    assert len(df) == 20
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    assert not missing
    assert (df["source"] == "synthetic").all()


def test_synthetic_rows_pass_validation():
    from src import data_gen  # noqa: E402

    df = data_gen.generate(n=50, seed=3)
    bad = 0
    for _, r in df.iterrows():
        rd = {k: r[k] for k in Formulation.__dataclass_fields__ if k in r}
        if validate_formulation(Formulation(**rd)):
            bad += 1
    assert bad == 0


def test_clean_produces_dataset(tmp_path, monkeypatch):
    # Point clean at tmp paths to avoid clobbering real outputs
    import src.clean as clean_mod

    monkeypatch.setattr(clean_mod, "SEED_PATH", tmp_path / "seed.csv")
    monkeypatch.setattr(clean_mod, "SYNTHETIC_PATH", tmp_path / "synthetic.csv")
    monkeypatch.setattr(clean_mod, "OUT_PATH", tmp_path / "dataset.csv")
    monkeypatch.setattr(clean_mod, "REPORT_PATH", tmp_path / "report.md")

    seed = pd.DataFrame([{
        "matrix_polymer": "HDPE", "recycled_content_pct": 50.0, "agro_filler": "none",
        "filler_wt_pct": 0.0, "filler_particle_size_um": 0.0, "nanoparticle": "none",
        "nano_wt_pct": 0.0, "nano_size_nm": 0.0, "surface_treatment": "none",
        "extrusion_temp_C": 180.0, "screw_speed_rpm": 90.0, "sonication_min": 10.0,
        "tensile_strength_MPa": 25.0, "youngs_modulus_GPa": 1.0,
        "thermal_conductivity_W_mK": 0.42, "water_absorption_pct": 2.0,
        "limiting_oxygen_index_pct": 17.5, "density_g_cm3": 0.95,
        "gwp_kgCO2e_per_kg": 2.0, "recyclability_score": 0.8,
        "biodegradability_score": 0.05, "source": "seed_example", "reference": "t",
    }])
    synth = pd.DataFrame([{
        "matrix_polymer": "PLA", "recycled_content_pct": 80.0, "agro_filler": "rice_husk",
        "filler_wt_pct": 15.0, "filler_particle_size_um": 100.0, "nanoparticle": "SiO2",
        "nano_wt_pct": 2.0, "nano_size_nm": 30.0, "surface_treatment": "silane",
        "extrusion_temp_C": 170.0, "screw_speed_rpm": 80.0, "sonication_min": 20.0,
        "tensile_strength_MPa": 40.0, "youngs_modulus_GPa": 2.0,
        "thermal_conductivity_W_mK": 0.33, "water_absorption_pct": 3.0,
        "limiting_oxygen_index_pct": 19.0, "density_g_cm3": 1.2,
        "gwp_kgCO2e_per_kg": 1.0, "recyclability_score": 0.9,
        "biodegradability_score": 0.8, "source": "synthetic", "reference": "s",
    }])
    seed.to_csv(clean_mod.SEED_PATH, index=False)
    synth.to_csv(clean_mod.SYNTHETIC_PATH, index=False)

    clean_mod.main()

    out = pd.read_csv(clean_mod.OUT_PATH)
    assert len(out) == 2
    assert set(out["source"]) == {"seed_example", "synthetic"}
    report = clean_mod.REPORT_PATH.read_text()
    assert "seed_example" in report
    assert "synthetic" in report
