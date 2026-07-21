"""Praroop-AI Streamlit investigation interface."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import forensics as forensics_module
from core.aggregator import calculate_trust_score, trust_status
from core.template_matcher import DEFAULT_WEIGHTS, TemplateMatcher

SUPPORTED_TYPES = ["jpg", "jpeg", "png", "pdf"]
PAGE_TITLE = "Praroop-AI Document Investigation"

# Match synthetic asset JPEG quality so authentic residuals stay low.
DEFAULT_ELA_QUALITY = 95
DEFAULT_ELA_SCALE = 20.0


def load_document(file_name: str, file_bytes: bytes) -> Image.Image:
    """Load an uploaded JPG/PNG/PDF into an RGB PIL image (first PDF page)."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ImportError(
                "PDF support requires pypdfium2. Install project requirements."
            ) from exc
        pdf = pdfium.PdfDocument(file_bytes)
        if len(pdf) < 1:
            raise ValueError("PDF has no pages")
        page = pdf[0]
        bitmap = page.render(scale=2.0)
        return bitmap.to_pil().convert("RGB")

    image = Image.open(BytesIO(file_bytes))
    return image.convert("RGB")


# Backwards-compatible alias used by older tests
def compute_trust_score(
    noise_map: np.ndarray | None = None,
    template_report: dict[str, Any] | None = None,
    forensic_metrics: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Delegate to :func:`core.aggregator.calculate_trust_score`.

    Prefer passing ``forensic_metrics``. If only a noise map is provided
    (legacy), derive a weak stand-in density from its mean.
    """
    if forensic_metrics is None:
        if noise_map is None:
            forensic_metrics = {"tamper_density": 0.0, "mean_brightness": 0.0, "anomaly_score": 0.0}
        else:
            mean = float(np.mean(noise_map.astype(np.float64)))
            forensic_metrics = {
                "tamper_density": mean / 10.0,
                "mean_brightness": mean,
                "anomaly_score": mean / 10.0,
            }
    return calculate_trust_score(forensic_metrics, template_report)


def run_template_alignment(image: Image.Image) -> dict[str, Any]:
    """Run template matching; soft-skip when YOLO weights are missing."""
    try:
        matcher = TemplateMatcher()
        arr = np.asarray(image)
        return matcher.verify(arr)
    except FileNotFoundError as exc:
        return {
            "template": "Praroop",
            "aligned": None,
            "issues": [f"Template check skipped: {exc}"],
            "elements": {},
            "detections": [],
            "skipped": True,
        }


def run_investigation(
    image: Image.Image,
    engine: forensics_module.ForensicEngine | None = None,
) -> dict[str, Any]:
    """Execute forensic + template checks and compute the trust score."""
    engine = engine or forensics_module.ForensicEngine(
        ela_quality=DEFAULT_ELA_QUALITY, ela_scale=DEFAULT_ELA_SCALE
    )
    forensic = engine.analyze(image)
    template_report = run_template_alignment(image)
    metrics = {
        "tamper_density": forensic["tamper_density"],
        "anomaly_score": forensic["anomaly_score"],
        "mean_brightness": forensic["mean_brightness"],
    }
    trust, components = calculate_trust_score(metrics, template_report)
    return {
        "original": image,
        "ela": forensic["ela"],
        "noise": forensic["noise"],
        "template": template_report,
        "trust_score": trust,
        "components": components,
        "forensic_metrics": metrics,
        "engine": engine,
    }


def _ela_preview(ela: np.ndarray) -> Image.Image:
    """Convert ELA ndarray to a displayable RGB image."""
    if ela.ndim == 2:
        rgb = np.stack([ela, ela, ela], axis=-1)
    elif ela.ndim == 3 and ela.shape[2] == 3:
        rgb = ela[:, :, ::-1]
    else:
        rgb = ela
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def render_trust_gauge(score: float, yolo_missing: bool = False) -> None:
    """Display trust score with status band messaging."""
    message, level = trust_status(score)
    st.subheader("Trust Score")
    st.metric(label="Overall trust", value=f"{score:.1f} / 100")
    st.progress(min(max(score / 100.0, 0.0), 1.0))

    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)

    if yolo_missing:
        st.warning(
            "Template Verification skipped: YOLO weights missing. "
            "Score based on Forensic signals only."
        )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title("Praroop-AI")
    st.caption("Document forgery investigation — ELA, noise analysis, and template alignment.")

    uploaded = st.file_uploader(
        "Upload a document (JPG, PNG, or PDF)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        key="document_uploader",
    )

    generate = st.button("Generate Report", type="primary", key="generate_report")

    if generate:
        if uploaded is None:
            st.warning("Please upload a document before generating a report.")
            return

        try:
            image = load_document(uploaded.name, uploaded.getvalue())
        except Exception as exc:  # noqa: BLE001 — surface upload/decode errors in UI
            st.error(f"Could not read document: {exc}")
            return

        with st.spinner("Running forensic and template checks..."):
            report = run_investigation(image)

        comps = report["components"]
        st.session_state["last_report"] = {
            "trust_score": report["trust_score"],
            "components": comps,
            "template": report["template"],
            "forensic_metrics": report["forensic_metrics"],
            "triggered": True,
        }

        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            st.image(report["original"], width="stretch")
        with right:
            st.subheader("ELA Heatmap")
            st.image(_ela_preview(report["ela"]), width="stretch")

        render_trust_gauge(
            report["trust_score"],
            yolo_missing=bool(comps.get("yolo_missing")),
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Tamper density", f"{comps['tamper_density']:.2f}")
        c2.metric("Forensic score", f"{comps['forensic_score']:.1f}")
        c3.metric(
            "Template weight",
            "0% (skipped)" if comps.get("yolo_missing") else f"{100 * comps['template_weight']:.0f}%",
        )

        template = report["template"]
        st.subheader("Template alignment")
        if template.get("skipped"):
            st.info(f"Expected YOLO weights at: `{DEFAULT_WEIGHTS}`")
        elif template.get("aligned"):
            st.success("Document layout matches the Praroop template.")
        else:
            issues = template.get("issues") or []
            st.warning("Layout issues detected: " + ", ".join(issues) if issues else "Misaligned")


if __name__ == "__main__":
    main()
