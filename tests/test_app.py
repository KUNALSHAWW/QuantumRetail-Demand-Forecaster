"""Smoke test: the Streamlit app renders end to end against the committed demo artefacts."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit-app" / "app.py"

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "qr_v2" / "meta.json").exists() or not (ROOT / "data" / "demo" / "demo_panel.npz").exists(),
    reason="demo artefacts not built (run `python -m quantumretail build-demo`)",
)


def test_app_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("Demand outlook")
    assert len(at.tabs) == 5
    assert len(at.metric) == 4


def test_app_handles_a_what_if_promotion():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    base = at.metric[0].value
    at.sidebar.checkbox[0].check().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].delta is not None and at.metric[0].value != base
