"""Validation tests for ForensicEngine using synthetic authentic/forged assets."""

from __future__ import annotations

import numpy as np
import pytest

from core.forensics import ForensicEngine
from utils.generate_test_assets import (
    FORGED_PATH,
    SAMPLE_PATH,
    generate_assets,
    load_year_box,
)


@pytest.fixture(scope="module", autouse=True)
def ensure_test_assets() -> None:
    """Always regenerate assets so forge geometry matches current fonts/layout."""
    generate_assets()


@pytest.fixture
def engine() -> ForensicEngine:
    # Match asset JPEG quality so authentic regions stay low-ELA.
    return ForensicEngine(ela_quality=95, ela_scale=20.0)


def _ela_mean(ela: np.ndarray) -> float:
    return float(np.mean(ela.astype(np.float64)))


def _year_slices() -> tuple[slice, slice]:
    left, top, right, bottom = load_year_box()
    return slice(top, bottom), slice(left, right)


def test_assets_exist() -> None:
    assert SAMPLE_PATH.is_file(), f"Missing authentic asset: {SAMPLE_PATH}"
    assert FORGED_PATH.is_file(), f"Missing forged asset: {FORGED_PATH}"


def test_ela_forged_has_higher_average_intensity(engine: ForensicEngine) -> None:
    """Forged asset must show higher mean ELA intensity than authentic."""
    ela_auth = engine.perform_ela(SAMPLE_PATH, normalize=False)
    ela_forg = engine.perform_ela(FORGED_PATH, normalize=False)

    mean_auth = _ela_mean(ela_auth)
    mean_forg = _ela_mean(ela_forg)

    assert mean_forg > mean_auth, (
        f"Expected forged ELA mean > authentic "
        f"(forged={mean_forg:.4f}, authentic={mean_auth:.4f})"
    )


def test_noise_variance_anomaly_in_forged_region(engine: ForensicEngine) -> None:
    """Noise variance should flag a statistical anomaly in the year edit zone."""
    noise_auth = engine.noise_variance_analysis(SAMPLE_PATH, normalize=False)
    noise_forg = engine.noise_variance_analysis(FORGED_PATH, normalize=False)

    ys, xs = _year_slices()
    region_auth = noise_auth[ys, xs].astype(np.float64)
    region_forg = noise_forg[ys, xs].astype(np.float64)

    left, top, right, bottom = load_year_box()
    pad = 50
    y0 = max(0, top - pad)
    y1 = min(noise_forg.shape[0], bottom + pad)
    x0 = max(0, left - pad)
    x1 = min(noise_forg.shape[1], right + pad)

    local = noise_forg[y0:y1, x0:x1].astype(np.float64).copy()
    ly0, ly1 = top - y0, bottom - y0
    lx0, lx1 = left - x0, right - x0
    local[ly0:ly1, lx0:lx1] = np.nan
    outside = local[~np.isnan(local)]

    forg_mean = float(np.mean(region_forg))
    auth_mean = float(np.mean(region_auth))
    outside_median = float(np.median(outside)) if outside.size else 0.0

    # Anomaly: edit zone is louder than the authentic counterpart and above
    # the local background median (robust to edge outliers).
    assert forg_mean > auth_mean, (
        f"Forged-region noise should exceed authentic "
        f"(forged={forg_mean:.2f}, authentic={auth_mean:.2f})"
    )
    assert forg_mean > outside_median, (
        f"Forged-region noise should exceed local background median "
        f"(forged={forg_mean:.2f}, outside_median={outside_median:.2f})"
    )


def test_analyze_returns_both_maps(engine: ForensicEngine) -> None:
    result = engine.analyze(FORGED_PATH)
    assert "ela" in result and "noise" in result
    assert result["ela"].ndim in (2, 3)
    assert result["noise"].ndim == 2
    assert "tamper_density" in result
    assert "anomaly_score" in result
