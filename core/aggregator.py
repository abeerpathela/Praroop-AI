"""Trust-score aggregation from forensic and template signals."""

from __future__ import annotations

from typing import Any

# Non-linear ELA penalty: small compression noise stays high-trust;
# localized high-intensity glow collapses the score quickly.
DEFAULT_PENALTY_FACTOR = 3.0
DEFAULT_PENALTY_EXPONENT = 1.2


def calculate_trust_score(
    forensic_metrics: dict[str, Any],
    template_report: dict[str, Any] | None = None,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
    penalty_exponent: float = DEFAULT_PENALTY_EXPONENT,
) -> tuple[float, dict[str, float | bool | str]]:
    """
    Compute a 0–100 trust score.

    When YOLO / template verification is unavailable (``skipped``), forensics
    account for 100% of the score:

        score = max(0, 100 - (tamper_density ** penalty_exponent) * penalty_factor)

    Otherwise forensics and template alignment are blended 70/30.
    """
    tamper_density = float(forensic_metrics.get("tamper_density", 0.0))
    anomaly_score = float(
        forensic_metrics.get("anomaly_score", tamper_density)
    )
    mean_brightness = float(forensic_metrics.get("mean_brightness", 0.0))

    forensic_score = float(
        max(0.0, 100.0 - ((tamper_density**penalty_exponent) * penalty_factor))
    )

    yolo_missing = template_report is None or bool(template_report.get("skipped"))

    if yolo_missing:
        trust = forensic_score
        template_component = 0.0
        forensic_weight = 1.0
        template_weight = 0.0
    else:
        if template_report.get("aligned"):
            template_component = 100.0
        else:
            issues = template_report.get("issues") or []
            template_component = float(max(0.0, 100.0 - 30.0 * len(issues)))
        forensic_weight = 0.70
        template_weight = 0.30
        trust = forensic_weight * forensic_score + template_weight * template_component

    trust = float(max(0.0, min(100.0, trust)))
    return trust, {
        "trust_score": trust,
        "forensic_score": forensic_score,
        "template_component": template_component,
        "forensic_weight": forensic_weight,
        "template_weight": template_weight,
        "tamper_density": tamper_density,
        "anomaly_score": anomaly_score,
        "mean_brightness": mean_brightness,
        "yolo_missing": yolo_missing,
        "penalty_factor": float(penalty_factor),
    }


def trust_status(score: float) -> tuple[str, str]:
    """Return (message, level) for UI coloring: success | warning | error."""
    if score >= 85:
        return "✅ Authentic Document", "success"
    if score >= 50:
        return "⚠️ Potential Format Mismatch", "warning"
    return "🚫 Suspicious Document: Digital Manipulation Detected", "error"
