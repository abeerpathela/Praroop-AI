"""Trust-score aggregation: ELA + Template + OCR → single Trust Percentage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.forensics import ForensicEngine
from core.ocr_checker import OCRChecker
from core.template_matcher import TemplateMatcher

# Non-linear ELA penalty: small compression noise stays high-trust;
# localized high-intensity glow collapses the score quickly.
DEFAULT_PENALTY_FACTOR = 3.0
DEFAULT_PENALTY_EXPONENT = 1.2

# Blend weights when all signals are present
WEIGHT_ELA = 0.50
WEIGHT_TEMPLATE = 0.25
WEIGHT_OCR = 0.25


def _template_component(template_report: dict[str, Any] | None) -> tuple[float, bool]:
    """Return (score 0–100, skipped)."""
    if template_report is None or bool(template_report.get("skipped")):
        return 0.0, True
    if template_report.get("aligned"):
        return 100.0, False
    issues = template_report.get("issues") or []
    return float(max(0.0, 100.0 - 30.0 * len(issues))), False


def _ocr_component(ocr_report: dict[str, Any] | None) -> tuple[float, bool]:
    """Return (score 0–100, skipped)."""
    if ocr_report is None or bool(ocr_report.get("skipped")):
        return 0.0, True
    return float(ocr_report.get("ocr_score", 0.0)), False


def calculate_trust_score(
    forensic_metrics: dict[str, Any],
    template_report: dict[str, Any] | None = None,
    ocr_report: dict[str, Any] | None = None,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
    penalty_exponent: float = DEFAULT_PENALTY_EXPONENT,
) -> tuple[float, dict[str, float | bool | str]]:
    """
    Combine ELA (forensic) score + Template score + OCR match into one trust %.

    Weighting:
    - All available: ELA 50% + Template 25% + OCR 25%
    - YOLO skipped:  redistribute template weight across ELA + OCR
    - OCR skipped:   redistribute OCR weight across ELA + Template
    - Both skipped:  ELA 100%
    """
    tamper_density = float(forensic_metrics.get("tamper_density", 0.0))
    anomaly_score = float(forensic_metrics.get("anomaly_score", tamper_density))
    mean_brightness = float(forensic_metrics.get("mean_brightness", 0.0))

    ela_score = float(
        max(0.0, 100.0 - ((tamper_density**penalty_exponent) * penalty_factor))
    )
    # Alias kept for older callers / UI labels
    forensic_score = ela_score

    template_score, template_skipped = _template_component(template_report)
    ocr_score, ocr_skipped = _ocr_component(ocr_report)

    w_ela, w_tmpl, w_ocr = WEIGHT_ELA, WEIGHT_TEMPLATE, WEIGHT_OCR
    if template_skipped and ocr_skipped:
        w_ela, w_tmpl, w_ocr = 1.0, 0.0, 0.0
    elif template_skipped:
        # Forensic-dominant when layout model is unavailable
        w_ela, w_tmpl, w_ocr = 0.80, 0.0, 0.20
    elif ocr_skipped:
        w_ela, w_tmpl, w_ocr = 0.70, 0.30, 0.0

    trust = w_ela * ela_score + w_tmpl * template_score + w_ocr * ocr_score
    trust = float(max(0.0, min(100.0, trust)))

    return trust, {
        "trust_score": trust,
        "ela_score": ela_score,
        "forensic_score": forensic_score,
        "template_score": template_score,
        "template_component": template_score,
        "ocr_score": ocr_score,
        "forensic_weight": w_ela,
        "template_weight": w_tmpl,
        "ocr_weight": w_ocr,
        "tamper_density": tamper_density,
        "anomaly_score": anomaly_score,
        "mean_brightness": mean_brightness,
        "yolo_missing": template_skipped,
        "ocr_skipped": ocr_skipped,
        "penalty_factor": float(penalty_factor),
    }


def trust_status(score: float) -> tuple[str, str]:
    """Return (message, level) for UI coloring: success | warning | error."""
    if score >= 85:
        return "✅ Authentic Document", "success"
    if score >= 50:
        return "⚠️ Potential Format Mismatch", "warning"
    return "🚫 Suspicious Document: Digital Manipulation Detected", "error"


def _run_template(image: Image.Image | np.ndarray | str | Path) -> dict[str, Any]:
    try:
        matcher = TemplateMatcher()
        if isinstance(image, Image.Image):
            return matcher.verify(np.asarray(image.convert("RGB")))
        return matcher.verify(image)
    except FileNotFoundError as exc:
        return {
            "template": "Praroop",
            "aligned": None,
            "issues": [f"Template check skipped: {exc}"],
            "elements": {},
            "detections": [],
            "skipped": True,
        }


def _run_ocr(
    image: Image.Image | np.ndarray | str | Path,
    metadata_source: str | Path | Image.Image | dict[str, str] | None,
    ocr_checker: OCRChecker | None,
) -> dict[str, Any]:
    try:
        checker = ocr_checker or OCRChecker(gpu=True)
        return checker.cross_check(image, metadata_source=metadata_source)
    except Exception as exc:  # noqa: BLE001 — soft-skip OCR in aggregate pipeline
        return {
            "ocr_score": 0.0,
            "matched": {},
            "fields": {},
            "all_match": False,
            "text_fields": {},
            "metadata": {},
            "skipped": True,
            "error": str(exc),
        }


def evaluate_document(
    image_path: str | Path,
    *,
    engine: ForensicEngine | None = None,
    ocr_checker: OCRChecker | None = None,
    ela_quality: int = 95,
    ela_scale: float = 20.0,
) -> dict[str, Any]:
    """
    End-to-end investigation: image → ELA + Template + OCR → Trust Percentage.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path.resolve()}")

    image = Image.open(path).convert("RGB")
    engine = engine or ForensicEngine(ela_quality=ela_quality, ela_scale=ela_scale)

    forensic = engine.analyze(image)
    template_report = _run_template(image)
    ocr_report = _run_ocr(image, metadata_source=path, ocr_checker=ocr_checker)

    metrics = {
        "tamper_density": forensic["tamper_density"],
        "anomaly_score": forensic["anomaly_score"],
        "mean_brightness": forensic["mean_brightness"],
    }
    trust, components = calculate_trust_score(metrics, template_report, ocr_report)
    status_message, status_level = trust_status(trust)

    return {
        "path": str(path),
        "original": image,
        "ela": forensic["ela"],
        "noise": forensic["noise"],
        "template": template_report,
        "ocr": ocr_report,
        "forensic_metrics": metrics,
        "trust_score": trust,
        "components": components,
        "status_message": status_message,
        "status_level": status_level,
    }
