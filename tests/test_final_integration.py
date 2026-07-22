"""End-to-end integration: image → ELA + Template + OCR → Trust Score."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.aggregator import calculate_trust_score, evaluate_document, trust_status
from core.ocr_checker import (
    OCRChecker,
    compare_ocr_to_metadata,
    parse_ocr_fields,
    read_file_metadata,
)
from utils.generate_test_assets import (
    DOCUMENT_ID,
    DOCUMENT_NAME,
    FORGED_PATH,
    SAMPLE_PATH,
    generate_assets,
)


@pytest.fixture(scope="module", autouse=True)
def ensure_assets() -> None:
    generate_assets()


class _FakeEasyOCRReader:
    """Stand-in EasyOCR reader that returns deterministic ID-card text."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def readtext(self, _image, detail=0, paragraph=False):  # noqa: ANN001
        return list(self.lines)


def _matching_ocr_checker() -> OCRChecker:
    reader = _FakeEasyOCRReader(
        [
            "GOVERNMENT OF INDIA",
            f"Name: {DOCUMENT_NAME}",
            "DOB: 01-01-1990",
            f"ID: {DOCUMENT_ID}",
        ]
    )
    return OCRChecker(reader=reader, gpu=False)


def test_metadata_embedded_in_sample_assets() -> None:
    meta = read_file_metadata(SAMPLE_PATH)
    assert meta.get("name") == DOCUMENT_NAME
    assert meta.get("id_number") == DOCUMENT_ID
    forged_meta = read_file_metadata(FORGED_PATH)
    assert forged_meta.get("name") == DOCUMENT_NAME
    assert forged_meta.get("id_number") == DOCUMENT_ID


def test_ocr_field_parse_and_metadata_match() -> None:
    text = f"Name: {DOCUMENT_NAME}\nID: {DOCUMENT_ID}\nDOB: 01-01-1990"
    fields = parse_ocr_fields(text)
    comparison = compare_ocr_to_metadata(fields, {"name": DOCUMENT_NAME, "id_number": DOCUMENT_ID})
    assert comparison["all_match"] is True
    assert comparison["ocr_score"] == 100.0


def test_end_to_end_authentic_trust_score() -> None:
    report = evaluate_document(SAMPLE_PATH, ocr_checker=_matching_ocr_checker())
    assert "trust_score" in report
    assert 0.0 <= report["trust_score"] <= 100.0
    assert report["ocr"]["skipped"] is False
    assert report["ocr"]["ocr_score"] == 100.0
    assert report["components"]["ela_score"] > 85
    # With YOLO typically missing: 70% ELA + 30% OCR ≈ high trust
    assert report["trust_score"] > 85
    message, level = trust_status(report["trust_score"])
    assert level == "success"
    assert "Authentic" in message


def test_end_to_end_forged_trust_score_suspicious() -> None:
    report = evaluate_document(FORGED_PATH, ocr_checker=_matching_ocr_checker())
    assert report["ocr"]["ocr_score"] == 100.0  # Name/ID still match metadata
    assert report["components"]["ela_score"] < 50
    # ELA drag should keep overall trust in suspicious band despite OCR match
    assert report["trust_score"] < 50
    _, level = trust_status(report["trust_score"])
    assert level == "error"


def test_aggregator_combines_three_signals() -> None:
    trust, comps = calculate_trust_score(
        {"tamper_density": 3.4, "anomaly_score": 3.4, "mean_brightness": 0.09},
        template_report={"aligned": True, "issues": [], "skipped": False},
        ocr_report={"ocr_score": 100.0, "skipped": False},
    )
    assert comps["forensic_weight"] == pytest.approx(0.50)
    assert comps["template_weight"] == pytest.approx(0.25)
    assert comps["ocr_weight"] == pytest.approx(0.25)
    assert comps["ocr_score"] == 100.0
    assert trust > 85


def test_ocr_cpu_fallback_on_gpu_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU init failure must fall back to CPU once; second failure stops."""
    calls: list[bool] = []

    class _BoomThenOk:
        def __init__(self, _langs, gpu=True):  # noqa: ANN001
            calls.append(gpu)
            if gpu:
                raise RuntimeError("CUDA driver conflict")
            self._ok = True

        def readtext(self, *_a, **_k):  # noqa: ANN001
            return ["Name: Rahul Kumar", "ID: IND-77821"]

    fake_mod = MagicMock()
    fake_mod.Reader = _BoomThenOk
    monkeypatch.setitem(sys.modules, "easyocr", fake_mod)

    checker = OCRChecker(gpu=True, reader=None)
    assert checker.used_cpu_fallback is True
    assert calls == [True, False]
    fields = checker.extract_fields(SAMPLE_PATH)
    assert fields.get("name") == DOCUMENT_NAME
