"""Validation tests for ForensicEngine ELA detection."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from core.forensics import ForensicEngine


def _jpeg_roundtrip(rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """Compress and reload an RGB image to establish JPEG history."""
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)


@pytest.fixture
def tampered_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Build a JPEG-authentic canvas with a pasted white patch.

    Returns (BGR image, patch box as y0, y1, x0, x1).
    """
    rng = np.random.default_rng(42)
    # Mild texture so JPEG has something to encode (not a flat field).
    base = np.full((240, 240, 3), 110, dtype=np.uint8)
    noise = rng.integers(-12, 13, size=base.shape, dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    authentic = _jpeg_roundtrip(base, quality=90)

    y0, y1, x0, x1 = 80, 140, 80, 140
    tampered = authentic.copy()
    tampered[y0:y1, x0:x1] = 255  # fake white patch

    # ForensicEngine expects BGR for ndarray inputs.
    bgr = tampered[:, :, ::-1].copy()
    return bgr, (y0, y1, x0, x1)


def test_ela_detects_higher_intensity_in_white_patch(tampered_image) -> None:
    image, (y0, y1, x0, x1) = tampered_image
    engine = ForensicEngine(ela_quality=90, ela_scale=20.0)

    ela = engine.error_level_analysis(image)

    assert ela.shape[:2] == image.shape[:2]
    assert ela.dtype == np.uint8

    if ela.ndim == 3:
        ela_gray = ela.mean(axis=2)
    else:
        ela_gray = ela.astype(np.float64)

    patch = ela_gray[y0:y1, x0:x1]
    mask = np.ones(ela_gray.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    outside = ela_gray[mask]

    patch_mean = float(np.mean(patch))
    outside_mean = float(np.mean(outside))

    assert patch_mean > outside_mean, (
        f"ELA should be stronger in the white patch "
        f"(patch={patch_mean:.2f}, outside={outside_mean:.2f})"
    )


def test_noise_variance_analysis_returns_map(tampered_image) -> None:
    image, _ = tampered_image
    engine = ForensicEngine()
    noise_map = engine.noise_variance_analysis(image)

    assert noise_map.shape == image.shape[:2]
    assert noise_map.dtype == np.uint8
    assert int(noise_map.max()) > 0


def test_analyze_returns_both_maps(tampered_image) -> None:
    image, _ = tampered_image
    result = ForensicEngine().analyze(image)
    assert set(result) == {"ela", "noise_variance"}
    assert result["ela"].shape[:2] == image.shape[:2]
    assert result["noise_variance"].shape == image.shape[:2]
