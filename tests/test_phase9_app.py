"""Phase 9 tests: Streamlit app (lightweight; app is run via `streamlit run`)."""

import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

APP_PATH = Path("app/app.py")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
KG_VIZ = Path("reports/figures/kg_subgraph.html")
FEATURES_PATH = Path("data/processed/features.csv")

from config import TARGET_PROPERTIES  # noqa: E402


def test_app_compiles():
    import py_compile
    py_compile.compile(str(APP_PATH), doraise=True)


def test_app_exists():
    assert APP_PATH.exists()


def test_models_exist():
    assert any(MODELS_DIR.glob("*.joblib")), "no trained models (run src.train)"


def test_kg_viz_exists():
    assert KG_VIZ.exists(), "kg_subgraph.html missing (run src.kg_build)"


def test_features_exist():
    assert FEATURES_PATH.exists(), "features.csv missing (run src.features)"


@pytest.fixture
def app_module():
    """Import app.py with streamlit stubbed so module-level st.* calls don't fail."""
    with mock.patch.dict(sys.modules):
        # Create a minimal fake streamlit module
        import types

        fake_st = types.ModuleType("streamlit")

        class _DeltaGenerator:
            def __getattr__(self, name):
                return self

            def __call__(self, *args, **kwargs):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake_st.title = lambda *a, **k: None
        fake_st.caption = lambda *a, **k: None
        fake_st.markdown = lambda *a, **k: None
        fake_st.header = lambda *a, **k: None
        fake_st.subheader = lambda *a, **k: None
        class _Sidebar:
            def selectbox(self, label, options, *a, **k):
                return options[0] if options else ""

            def slider(self, label, *a, **k):
                if "value" in k:
                    return k["value"]
                if len(a) >= 3:
                    return a[2]
                return float(a[0]) if a else 0.0

            def __getattr__(self, name):
                def _noop(*a, **k):
                    return self if name in ("selectbox", "slider") else None
                return _noop

        fake_st.sidebar = _Sidebar()
        fake_st.selectbox = lambda *a, **k: a[1] if len(a) > 1 else (a[0] if a else "")
        fake_st.slider = lambda *a, **k: k.get("value", a[2] if len(a) > 2 else 0)
        fake_st.button = lambda *a, **k: False
        fake_st.metric = lambda *a, **k: None
        fake_st.image = lambda *a, **k: None
        fake_st.write = lambda *a, **k: None
        fake_st.error = lambda *a, **k: None
        fake_st.warning = lambda *a, **k: None
        fake_st.info = lambda *a, **k: None
        fake_st.success = lambda *a, **k: None
        fake_st.dataframe = lambda *a, **k: None
        fake_st.download_button = lambda *a, **k: None
        fake_st.components = types.SimpleNamespace(
            v1=types.SimpleNamespace(html=lambda *a, **k: None)
        )
        fake_st.text_area = lambda *a, **k: ""
        fake_st.columns = lambda n: [_DeltaGenerator() for _ in range(n)]
        fake_st.form = lambda *a, **k: _DeltaGenerator()
        fake_st.number_input = lambda *a, **k: k.get("value", 0.0)
        fake_st.text_input = lambda *a, **k: k.get("value", "")
        fake_st.form_submit_button = lambda *a, **k: False
        fake_st.set_page_config = lambda *a, **k: None
        fake_st.cache_data = lambda f: f
        fake_st.cache_resource = lambda f: f
        fake_st.tabs = lambda *args, **k: [_DeltaGenerator() for _ in args[0]]
        fake_st.session_state = {}

        sys.modules["streamlit"] = fake_st
        import importlib
        import app.app as app_mod
        importlib.reload(app_mod)
        yield app_mod


def test_formulation_to_row(app_module):
    fcols = app_module.feature_cols()
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
    }
    row = app_module.formulation_to_row(form, fcols)
    assert isinstance(row, pd.DataFrame)
    assert len(row) == 1
    assert row["recycled_content_pct"].iloc[0] == 50.0
    assert row["matrix_polymer_HDPE"].iloc[0] == 1.0
    assert row["matrix_is_recycled"].iloc[0] == 0.0


def test_predict_properties(app_module):
    if not any(MODELS_DIR.glob("*.joblib")):
        pytest.skip("no models")
    form = {
        "matrix_polymer": "HDPE",
        "recycled_content_pct": 50.0,
        "agro_filler": "none",
        "filler_wt_pct": 0.0,
        "filler_particle_size_um": 0.0,
        "nanoparticle": "none",
        "nano_wt_pct": 0.0,
        "nano_size_nm": 0.0,
        "surface_treatment": "none",
        "extrusion_temp_C": 190.0,
        "screw_speed_rpm": 100.0,
        "sonication_min": 20.0,
    }
    preds = app_module.predict_properties(form)
    assert isinstance(preds, dict)
    assert len(preds) > 0
    for target in TARGET_PROPERTIES:
        if target in preds:
            assert isinstance(preds[target], float)


def test_prediction_uncertainty(app_module):
    if not any(MODELS_DIR.glob("*.joblib")):
        pytest.skip("no models")
    form = {
        "matrix_polymer": "HDPE",
        "recycled_content_pct": 50.0,
        "agro_filler": "none",
        "filler_wt_pct": 0.0,
        "filler_particle_size_um": 0.0,
        "nanoparticle": "none",
        "nano_wt_pct": 0.0,
        "nano_size_nm": 0.0,
        "surface_treatment": "none",
        "extrusion_temp_C": 190.0,
        "screw_speed_rpm": 100.0,
        "sonication_min": 20.0,
    }
    unc = app_module.prediction_uncertainty(form)
    assert isinstance(unc, dict)
    # at least some targets should have None or a float
    for v in unc.values():
        assert v is None or isinstance(v, float)
