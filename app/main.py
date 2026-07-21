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
from core.template_matcher import DEFAULT_WEIGHTS, TemplateMatcher

SUPPORTED_TYPES = ["jpg", "jpeg", "png", "pdf"]
PAGE_TITLE = "Praroop-AI Document Investigation"


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


def compute_trust_score(
    noise_map: np.ndarray,
    template_report: dict[str, Any] | None,
) -> tuple[float, dict[str, float]]:
    """
    Combine noise-variance intensity and template alignment into a 0–100 score.

    Higher residual noise and more template issues lower trust.
    """
    noise_mean = float(np.mean(noise_map.astype(np.float64)))
    # Map 0..255 mean residual → 100..0 contribution
    noise_component = float(np.clip(100.0 - (noise_mean / 255.0) * 100.0, 0.0, 100.0))

    if template_report is None or template_report.get("skipped"):
        template_component = 50.0
    elif template_report.get("aligned"):
        template_component = 100.0
    else:
        issues = template_report.get("issues") or []
        template_component = float(np.clip(100.0 - 30.0 * len(issues), 0.0, 100.0))

    trust = float(np.clip(0.55 * noise_component + 0.45 * template_component, 0.0, 100.0))
    return trust, {
        "noise_component": noise_component,
        "template_component": template_component,
    }


def run_template_alignment(image: Image.Image) -> dict[str, Any]:
    """Run template matching; soft-skip when YOLO weights are missing."""
    try:
        matcher = TemplateMatcher()
        # Ultralytics accepts numpy RGB arrays
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
    engine = engine or forensics_module.ForensicEngine(ela_quality=90, ela_scale=15.0)
    forensic = engine.analyze(image)
    ela = forensic["ela"]
    noise = forensic["noise"]
    template_report = run_template_alignment(image)
    trust, components = compute_trust_score(noise, template_report)
    return {
        "original": image,
        "ela": ela,
        "noise": noise,
        "template": template_report,
        "trust_score": trust,
        "components": components,
        "engine": engine,
    }


def _ela_preview(ela: np.ndarray) -> Image.Image:
    """Convert ELA ndarray to a displayable RGB image."""
    if ela.ndim == 2:
        rgb = np.stack([ela, ela, ela], axis=-1)
    elif ela.ndim == 3 and ela.shape[2] == 3:
        # ForensicEngine stores OpenCV BGR — convert for Streamlit
        rgb = ela[:, :, ::-1]
    else:
        rgb = ela
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _trust_label(score: float) -> str:
    if score >= 75:
        return "Likely authentic"
    if score >= 45:
        return "Needs review"
    return "High risk of tampering"


def render_trust_gauge(score: float) -> None:
    """Display a simple trust-score gauge using metric + progress."""
    st.subheader("Trust Score")
    st.metric(label="Overall trust", value=f"{score:.1f} / 100", delta=_trust_label(score))
    st.progress(min(max(score / 100.0, 0.0), 1.0))


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

        st.session_state["last_report"] = {
            "trust_score": report["trust_score"],
            "components": report["components"],
            "template": report["template"],
            "triggered": True,
        }

        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            st.image(report["original"], width="stretch")
        with right:
            st.subheader("ELA Heatmap")
            st.image(_ela_preview(report["ela"]), width="stretch")

        render_trust_gauge(report["trust_score"])
        comps = report["components"]
        c1, c2 = st.columns(2)
        c1.metric("Noise integrity", f"{comps['noise_component']:.1f}")
        c2.metric("Template alignment", f"{comps['template_component']:.1f}")

        template = report["template"]
        st.subheader("Template alignment")
        if template.get("skipped"):
            st.info(
                "YOLO weights not found — template alignment skipped. "
                f"Expected: `{DEFAULT_WEIGHTS}`"
            )
        elif template.get("aligned"):
            st.success("Document layout matches the Praroop template.")
        else:
            issues = template.get("issues") or []
            st.warning("Layout issues detected: " + ", ".join(issues) if issues else "Misaligned")


if __name__ == "__main__":
    main()
