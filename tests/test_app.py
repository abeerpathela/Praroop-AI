"""Streamlit AppTest validation for the Praroop-AI investigation UI."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from utils.generate_test_assets import FORGED_PATH, SAMPLE_PATH, generate_assets

APP_PATH = ROOT / "app" / "main.py"


@pytest.fixture(scope="module", autouse=True)
def ensure_assets() -> None:
    if not SAMPLE_PATH.is_file() or not FORGED_PATH.is_file():
        generate_assets()


def _mock_engine() -> MagicMock:
    engine = MagicMock()
    h, w = 64, 64
    ela = np.zeros((h, w, 3), dtype=np.uint8)
    ela[10:30, 10:30] = 200
    noise = np.full((h, w), 40, dtype=np.uint8)
    noise[10:30, 10:30] = 180
    engine.analyze.return_value = {
        "ela": ela,
        "noise": noise,
        "tamper_density": 14.0,
        "anomaly_score": 14.0,
        "mean_brightness": 0.17,
        "ela_metrics": {
            "tamper_density": 14.0,
            "anomaly_score": 14.0,
            "mean_brightness": 0.17,
        },
    }
    engine.error_level_analysis.return_value = ela
    engine.noise_variance_analysis.return_value = noise
    return engine


def test_generate_report_button_triggers_engine() -> None:
    """Simulate JPG upload and confirm Generate Report invokes ForensicEngine."""
    assert APP_PATH.is_file()
    image_bytes = FORGED_PATH.read_bytes()

    mock_engine = _mock_engine()
    mock_cls = MagicMock(return_value=mock_engine)

    with (
        patch("core.forensics.ForensicEngine", mock_cls),
        patch(
            "app.main.run_template_alignment",
            return_value={
                "template": "Praroop",
                "aligned": False,
                "issues": ["Misaligned Photo"],
                "elements": {},
                "detections": [],
                "skipped": False,
            },
        ),
    ):
        # Also patch the name bound inside the Streamlit script module after load.
        at = AppTest.from_file(str(APP_PATH), default_timeout=30)
        at.run()
        assert not at.exception, f"App failed on load: {at.exception}"

        assert len(at.file_uploader) >= 1
        at.file_uploader[0].set_value(
            ("sample_forged.jpg", image_bytes, "image/jpeg")
        )
        at.run()
        assert not at.exception, f"App failed after upload: {at.exception}"

        report_buttons = [b for b in at.button if b.label == "Generate Report"]
        assert report_buttons, "Generate Report button not found"
        report_buttons[0].click().run()

        assert not at.exception, f"App failed after Generate Report: {at.exception}"
        assert mock_engine.analyze.called, "ForensicEngine.analyze was not triggered"
        last_report = at.session_state["last_report"]
        assert last_report["triggered"] is True
        assert "trust_score" in last_report


def test_compute_trust_score_penalizes_ela_glow() -> None:
    from app.main import compute_trust_score

    high, _ = compute_trust_score(
        forensic_metrics={"tamper_density": 3.0, "anomaly_score": 3.0, "mean_brightness": 0.09},
        template_report={"skipped": True},
    )
    low, _ = compute_trust_score(
        forensic_metrics={"tamper_density": 14.0, "anomaly_score": 14.0, "mean_brightness": 0.17},
        template_report={"skipped": True},
    )
    assert high > low
    assert high > 85
    assert low < 50
    assert 0.0 <= low <= 100.0
    assert 0.0 <= high <= 100.0


def test_load_document_jpg() -> None:
    from app.main import load_document

    img = load_document("sample.jpg", SAMPLE_PATH.read_bytes())
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0
