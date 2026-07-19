"""Validation tests for Praroop YOLOv8 template matching."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.template_matcher import TemplateMatcher

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "templates" / "praroop_template.json"
)


@pytest.fixture
def matcher() -> TemplateMatcher:
    """Matcher with an injected stub model (no weight download / load)."""
    return TemplateMatcher(template_path=TEMPLATE_PATH, model=object())


def test_template_file_exists() -> None:
    assert TEMPLATE_PATH.is_file()


def test_misaligned_photo_is_flagged(matcher: TemplateMatcher) -> None:
    """Mock YOLO output: Photo far from template → 'Misaligned Photo'."""
    expected_photo = matcher.template["elements"]["Photo"]["bbox"]
    # Shift photo well outside tolerance / IoU range
    misaligned_photo = [
        expected_photo[0] - 300,
        expected_photo[1] + 250,
        expected_photo[2] - 300,
        expected_photo[3] + 250,
    ]

    signature = matcher.template["elements"]["Signature"]["bbox"]
    qr_code = matcher.template["elements"]["QR Code"]["bbox"]

    mock_detections = [
        {"label": "Photo", "confidence": 0.93, "bbox": misaligned_photo},
        {"label": "Signature", "confidence": 0.88, "bbox": list(signature)},
        {"label": "QR Code", "confidence": 0.91, "bbox": list(qr_code)},
    ]

    report = matcher.compare_to_template(mock_detections)

    assert "Misaligned Photo" in report["issues"]
    assert report["elements"]["Photo"]["status"] == "misaligned"
    assert report["elements"]["Signature"]["status"] == "aligned"
    assert report["elements"]["QR Code"]["status"] == "aligned"
    assert report["aligned"] is False


def test_aligned_layout_has_no_issues(matcher: TemplateMatcher) -> None:
    elements = matcher.template["elements"]
    mock_detections = [
        {"label": label, "confidence": 0.95, "bbox": list(meta["bbox"])}
        for label, meta in elements.items()
    ]
    report = matcher.compare_to_template(mock_detections)
    assert report["issues"] == []
    assert report["aligned"] is True


def test_missing_weights_reports_path_error() -> None:
    missing = Path("models/does_not_exist_praroop.pt")
    with pytest.raises(FileNotFoundError, match="YOLO weights not found") as exc_info:
        TemplateMatcher(template_path=TEMPLATE_PATH, model_path=missing, model=None)
    assert "does_not_exist_praroop.pt" in str(exc_info.value)
    assert "Auto-download is disabled" in str(exc_info.value)


def test_missing_element_is_reported(matcher: TemplateMatcher) -> None:
    elements = matcher.template["elements"]
    mock_detections = [
        {"label": "Photo", "confidence": 0.9, "bbox": list(elements["Photo"]["bbox"])},
        {"label": "QR Code", "confidence": 0.9, "bbox": list(elements["QR Code"]["bbox"])},
    ]
    report = matcher.compare_to_template(mock_detections)
    assert "Missing Signature" in report["issues"]
