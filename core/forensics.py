"""Core forensic analysis: Error Level Analysis and noise variance."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, Image.Image]


class ForensicEngine:
    """Image forensic utilities for tampering / inconsistency detection."""

    def __init__(self, ela_quality: int = 90, ela_scale: float = 15.0) -> None:
        if not 1 <= ela_quality <= 95:
            raise ValueError("ela_quality must be between 1 and 95")
        if ela_scale <= 0:
            raise ValueError("ela_scale must be positive")
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

    @staticmethod
    def _to_bgr_uint8(image: ImageInput) -> np.ndarray:
        """Normalize supported inputs to a BGR uint8 ndarray."""
        if isinstance(image, (str, Path)):
            arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if arr is None:
                raise FileNotFoundError(f"Unable to read image: {image}")
            return arr

        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Empty image array")
            arr = image
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return arr.copy()
            if arr.ndim == 3 and arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            raise ValueError(f"Unsupported array shape: {arr.shape}")

        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def error_level_analysis(
        self,
        image: ImageInput,
        quality: int | None = None,
        scale: float | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute Error Level Analysis (ELA) of an image.

        Re-encodes the image as JPEG at a fixed quality, then returns the
        amplified absolute difference between the original and recompressed
        versions. Edited regions typically show higher residual intensity.

        When ``normalize`` is False, residuals are only scaled (not peak-mapped),
        which is better for comparing average intensity across images.
        """
        quality = self.ela_quality if quality is None else quality
        scale = self.ela_scale if scale is None else scale
        if not 1 <= quality <= 95:
            raise ValueError("quality must be between 1 and 95")
        if scale <= 0:
            raise ValueError("scale must be positive")

        original = self._to_bgr_uint8(image)
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
        recompressed_bgr = cv2.cvtColor(recompressed, cv2.COLOR_RGB2BGR)

        diff = cv2.absdiff(original, recompressed_bgr).astype(np.float32)
        amplified = diff * float(scale)

        if normalize:
            # Avoid ZeroDivisionError / all-zero collapse when normalizing.
            peak = float(np.max(amplified))
            if peak > 0.0:
                ela = (amplified / peak) * 255.0
            else:
                ela = amplified
        else:
            ela = amplified

        return np.clip(ela, 0, 255).astype(np.uint8)

    def perform_ela(
        self,
        image: ImageInput,
        quality: int | None = None,
        scale: float | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Alias for :meth:`error_level_analysis`."""
        return self.error_level_analysis(
            image, quality=quality, scale=scale, normalize=normalize
        )

    def noise_variance_analysis(
        self,
        image: ImageInput,
        ksize: int = 3,
        window: int = 7,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Detect pixel inconsistencies via Laplacian-based local noise variance.

        Applies a Laplacian filter, then estimates local variance of the
        residual. Abrupt variance changes often indicate spliced / edited areas.
        """
        if ksize not in (1, 3, 5, 7):
            raise ValueError("ksize must be 1, 3, 5, or 7")
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be an odd integer >= 3")

        bgr = self._to_bgr_uint8(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
        residual = np.abs(laplacian)

        # Local variance of residual: E[x^2] - (E[x])^2
        kernel = (window, window)
        mean = cv2.blur(residual, kernel)
        mean_sq = cv2.blur(residual * residual, kernel)
        variance = np.maximum(mean_sq - mean * mean, 0.0)

        if normalize:
            peak = float(np.max(variance))
            if peak > 0.0:
                noise_map = (variance / peak) * 255.0
            else:
                noise_map = variance
            return np.clip(noise_map, 0, 255).astype(np.uint8)

        # Unnormalized map kept as float32 for statistical comparisons.
        return variance.astype(np.float32)

    def analyze(self, image: ImageInput) -> dict[str, np.ndarray]:
        """Run ELA and noise-variance analysis together."""
        return {
            "ela": self.error_level_analysis(image),
            "noise_variance": self.noise_variance_analysis(image),
        }
