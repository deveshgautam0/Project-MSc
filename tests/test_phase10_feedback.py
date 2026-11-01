"""Phase 10 tests: feedback loop (submit + retrain)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import feedback  # noqa: E402
from config import Formulation  # noqa: E402

USER_DATA_PATH = Path("data/raw/user_data.csv")


def _sample_form(overrides=None):
    form = {
        "matrix_polymer": "HDPE",
        "recycled_content_pct": 50.0,
        "agro_filler": "rice_husk",
        "filler_wt_pct": 20.0,
        "filler_particle_size_um": 150.0,
        "nanoparticle": "SiO2",
        "nano_wt_pct": 2.0,
        "nano_size_nm": 30.0,
        "surface_treatment": "silane",
        "extrusion_temp_C": 190.0,
        "screw_speed_rpm": 100.0,
        "sonication_min": 20.0,
        "tensile_strength_MPa": 25.0,
        "youngs_modulus_GPa": 1.2,
        "thermal_conductivity_W_mK": 0.4,
        "water_absorption_pct": 3.0,
        "limiting_oxygen_index_pct": 18.0,
        "density_g_cm3": 0.95,
        "gwp_kgCO2e_per_kg": 1.8,
        "recyclability_score": 0.75,
        "biodegradability_score": 0.15,
        "reference": "test",
    }
    if overrides:
        form.update(overrides)
    return form


def _reset_user_data():
    """Remove user_data.csv so tests are isolated."""
    if USER_DATA_PATH.exists():
        USER_DATA_PATH.unlink()
    feedback._ensure_user_data()


def test_submit_sample_saves_with_source_user():
    _reset_user_data()
    form = _sample_form()
    ok, msg = feedback.submit_sample(form)
    assert ok is True, msg
    assert "source=user" in msg or "user" in msg
    df = feedback.list_user_samples()
    assert len(df) == 1
    assert df.iloc[0]["source"] == "user"
    assert df.iloc[0]["matrix_polymer"] == "HDPE"
    assert df.iloc[0]["tensile_strength_MPa"] == 25.0
    _reset_user_data()


def test_submit_sample_rejects_invalid():
    _reset_user_data()
    form = _sample_form({"matrix_polymer": "NOT_A_POLYMER"})
    ok, msg = feedback.submit_sample(form)
    assert ok is False
    assert isinstance(msg, list)
    assert len(msg) > 0
    df = feedback.list_user_samples()
    assert len(df) == 0  # nothing saved
    _reset_user_data()


def test_submit_sample_out_of_range_rejected():
    _reset_user_data()
    form = _sample_form({"recycled_content_pct": 150.0})
    ok, msg = feedback.submit_sample(form)
    assert ok is False
    _reset_user_data()


def test_list_user_samples_empty_when_none():
    _reset_user_data()
    df = feedback.list_user_samples()
    assert len(df) == 0
    _reset_user_data()


def test_formulation_dataclass_has_expected_fields():
    names = set(Formulation.__dataclass_fields__.keys())
    assert "matrix_polymer" in names
    assert "source" in names
    assert "reference" in names
    # targets are NOT dataclass fields (stored separately)
    assert "tensile_strength_MPa" not in names


def test_retrain_returns_structure():
    """Retrain runs the pipeline (slow) and returns before/after + archive."""
    _reset_user_data()
    feedback.submit_sample(_sample_form())
    out = feedback.retrain()
    assert "archive" in out
    assert out["archive"] is not None
    assert Path(out["archive"]).is_dir()
    import glob

    archived = glob.glob(out["archive"] + "/*.joblib")
    assert len(archived) > 0, "archive should contain backed-up models"
    assert "before" in out
    assert "after" in out
    assert out["user_samples"] >= 1
    assert out["after"] is not None
    assert len(out["after"]) > 0
    _reset_user_data()
