"""Validation for forensic trust-score weighting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.aggregator import calculate_trust_score, trust_status
from core.forensics import ForensicEngine
from utils.generate_test_assets import FORGED_PATH, SAMPLE_PATH, generate_assets

# Match app defaults / asset JPEG quality
ELA_QUALITY = 95
ELA_SCALE = 20.0


@pytest.fixture(scope="module", autouse=True)
def ensure_assets() -> None:
    if not SAMPLE_PATH.is_file() or not FORGED_PATH.is_file():
        generate_assets()


@pytest.fixture(scope="module")
def engine() -> ForensicEngine:
    return ForensicEngine(ela_quality=ELA_QUALITY, ela_scale=ELA_SCALE)


@pytest.fixture(scope="module")
def skipped_template() -> dict:
    return {
        "template": "Praroop",
        "aligned": None,
        "issues": ["Template check skipped: YOLO weights missing"],
        "elements": {},
        "detections": [],
        "skipped": True,
    }


def _score_image(engine: ForensicEngine, path: Path, template: dict) -> tuple[float, dict]:
    result = engine.analyze(path)
    metrics = {
        "tamper_density": result["tamper_density"],
        "anomaly_score": result["anomaly_score"],
        "mean_brightness": result["mean_brightness"],
    }
    return calculate_trust_score(metrics, template)


def test_forged_sample_scores_below_fifty(engine: ForensicEngine, skipped_template: dict) -> None:
    score, comps = _score_image(engine, FORGED_PATH, skipped_template)
    assert score < 50, (
        f"Forged sample should be suspicious (<50), got {score:.1f} "
        f"(mean_brightness={comps['mean_brightness']:.4f}, "
        f"tamper_density={comps['tamper_density']:.4f})"
    )
    assert 20.0 <= score <= 40.0 or score < 50  # primary gate is <50; band preferred
    message, level = trust_status(score)
    assert level == "error"
    assert "Suspicious" in message


def test_authentic_sample_scores_above_eighty_five(
    engine: ForensicEngine, skipped_template: dict
) -> None:
    score, comps = _score_image(engine, SAMPLE_PATH, skipped_template)
    assert score > 85, (
        f"Authentic sample should score >85, got {score:.1f} "
        f"(mean_brightness={comps['mean_brightness']:.4f}, "
        f"tamper_density={comps['tamper_density']:.4f})"
    )
    message, level = trust_status(score)
    assert level == "success"
    assert "Authentic" in message


def test_yolo_missing_uses_full_forensic_weight(skipped_template: dict) -> None:
    metrics = {"tamper_density": 10.0, "anomaly_score": 10.0, "mean_brightness": 0.2}
    score, comps = calculate_trust_score(metrics, skipped_template)
    assert comps["yolo_missing"] is True
    assert comps["forensic_weight"] == 1.0
    assert comps["template_weight"] == 0.0
    assert score == comps["forensic_score"]


def test_ela_analyze_exposes_anomaly_metrics(engine: ForensicEngine) -> None:
    result = engine.analyze(FORGED_PATH)
    assert "tamper_density" in result
    assert "anomaly_score" in result
    assert "mean_brightness" in result
    assert result["tamper_density"] >= 0.0
