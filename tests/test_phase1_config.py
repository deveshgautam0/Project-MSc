"""Phase 1 tests: config validator + data ranges."""

import sys
from pathlib import Path

import pytest

# Make src/config importable when running from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    Formulation,
    CATEGORICAL_FEATURES,
    FORBIDDEN_ADDITIVES,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    TARGET_PROPERTIES,
    USE_CASES,
    validate_formulation,
    validate_row,
)


def test_input_features_cover_all_configured():
    names = set(INPUT_FEATURES)
    assert names == set(CATEGORICAL_FEATURES.keys()) | set(NUMERIC_FEATURES.keys())


def test_numeric_ranges_sane():
    for name, (lo, hi, unit) in NUMERIC_FEATURES.items():
        assert lo < hi, f"{name} range inverted"
        assert unit, f"{name} missing unit"


def test_targets_have_units():
    for target, unit in TARGET_PROPERTIES.items():
        assert unit, f"{target} missing unit"


def test_use_cases_have_valid_operators():
    for use_case, thresholds in USE_CASES.items():
        for prop, (op, value) in thresholds.items():
            assert op in ("<=", ">="), f"{use_case}/{prop} bad op {op}"
            assert prop in TARGET_PROPERTIES, f"{use_case}/{prop} not a target"


def test_valid_formulation_passes():
    form = Formulation(
        matrix_polymer="HDPE",
        recycled_content_pct=50.0,
        agro_filler="rice_husk",
        filler_wt_pct=20.0,
        filler_particle_size_um=100.0,
        nanoparticle="SiO2",
        nano_wt_pct=2.0,
        nano_size_nm=30.0,
        surface_treatment="silane",
        extrusion_temp_C=190.0,
        screw_speed_rpm=100.0,
        sonication_min=15.0,
        source="synthetic",
        reference="test",
    )
    assert validate_formulation(form) == []


def test_invalid_categorical_rejected():
    form = Formulation(
        matrix_polymer="NOT_A_POLYMER",
        recycled_content_pct=50.0,
        agro_filler="none",
        filler_wt_pct=10.0,
        filler_particle_size_um=100.0,
        nanoparticle="none",
        nano_wt_pct=1.0,
        nano_size_nm=30.0,
        surface_treatment="none",
        extrusion_temp_C=190.0,
        screw_speed_rpm=100.0,
        sonication_min=15.0,
        source="synthetic",
        reference="test",
    )
    errors = validate_formulation(form)
    assert any("matrix_polymer" in e for e in errors)


def test_out_of_range_rejected():
    form = Formulation(
        matrix_polymer="HDPE",
        recycled_content_pct=150.0,  # > 100
        agro_filler="none",
        filler_wt_pct=10.0,
        filler_particle_size_um=100.0,
        nanoparticle="none",
        nano_wt_pct=1.0,
        nano_size_nm=30.0,
        surface_treatment="none",
        extrusion_temp_C=190.0,
        screw_speed_rpm=100.0,
        sonication_min=15.0,
        source="synthetic",
        reference="test",
    )
    errors = validate_formulation(form)
    assert any("recycled_content_pct" in e for e in errors)


def test_forbidden_additive_rejected():
    # Temporarily add a forbidden additive
    import config

    config.FORBIDDEN_ADDITIVES.append("SiO2")
    try:
        form = Formulation(
            matrix_polymer="HDPE",
            recycled_content_pct=50.0,
            agro_filler="none",
            filler_wt_pct=10.0,
            filler_particle_size_um=100.0,
            nanoparticle="SiO2",
            nano_wt_pct=2.0,
            nano_size_nm=30.0,
            surface_treatment="none",
            extrusion_temp_C=190.0,
            screw_speed_rpm=100.0,
            sonication_min=15.0,
            source="synthetic",
            reference="test",
        )
        errors = validate_formulation(form)
        assert any("FORBIDDEN_ADDITIVES" in e for e in errors)
    finally:
        config.FORBIDDEN_ADDITIVES.remove("SiO2")


def test_validate_row_dict():
    row = {
        "matrix_polymer": "PLA",
        "recycled_content_pct": 0.0,
        "agro_filler": "coconut_coir",
        "filler_wt_pct": 30.0,
        "filler_particle_size_um": 200.0,
        "nanoparticle": "nanoclay",
        "nano_wt_pct": 3.0,
        "nano_size_nm": 50.0,
        "surface_treatment": "NaOH",
        "extrusion_temp_C": 170.0,
        "screw_speed_rpm": 80.0,
        "sonication_min": 30.0,
        "source": "literature",
        "reference": "example ref",
    }
    assert validate_row(row) == []
