"""OCR extraction and metadata cross-checking for Praroop documents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, Image.Image, bytes]

# EXIF tag for ImageDescription
_EXIF_IMAGE_DESCRIPTION = 0x010E

NAME_RE = re.compile(r"Name\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
ID_RE = re.compile(r"ID\s*:\s*([A-Za-z0-9\-]+)", re.IGNORECASE)


def _to_rgb_array(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
        return np.asarray(img)
    if isinstance(image, bytes):
        from io import BytesIO

        img = Image.open(BytesIO(image)).convert("RGB")
        return np.asarray(img)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def read_file_metadata(source: str | Path | Image.Image | bytes) -> dict[str, str]:
    """
    Read Name / ID Number from image metadata (EXIF ImageDescription JSON).

    Expected payload: ``{"name": "...", "id_number": "..."}``.
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, bytes):
        from io import BytesIO

        img = Image.open(BytesIO(source))
    else:
        img = Image.open(source)

    exif = img.getexif()
    raw = exif.get(_EXIF_IMAGE_DESCRIPTION)
    if not raw:
        return {}

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        # Fallback free-text: "Name: X; ID: Y"
        name_m = NAME_RE.search(str(raw))
        id_m = ID_RE.search(str(raw))
        out: dict[str, str] = {}
        if name_m:
            out["name"] = name_m.group(1).strip()
        if id_m:
            out["id_number"] = id_m.group(1).strip()
        return out

    result: dict[str, str] = {}
    if data.get("name"):
        result["name"] = str(data["name"]).strip()
    if data.get("id_number"):
        result["id_number"] = str(data["id_number"]).strip()
    elif data.get("id"):
        result["id_number"] = str(data["id"]).strip()
    return result


def embed_file_metadata(image: Image.Image, name: str, id_number: str) -> Image.Exif:
    """Build EXIF block embedding Name and ID Number as JSON."""
    exif = image.getexif()
    exif[_EXIF_IMAGE_DESCRIPTION] = json.dumps(
        {"name": name, "id_number": id_number},
        ensure_ascii=True,
    )
    return exif


def parse_ocr_fields(text: str) -> dict[str, str]:
    """Pull Name and ID Number from raw OCR text."""
    fields: dict[str, str] = {}
    name_m = NAME_RE.search(text)
    if name_m:
        fields["name"] = name_m.group(1).strip()
    id_m = ID_RE.search(text)
    if id_m:
        fields["id_number"] = id_m.group(1).strip().upper()
    return fields


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def compare_ocr_to_metadata(
    ocr_fields: dict[str, str],
    metadata: dict[str, str],
) -> dict[str, Any]:
    """
    Cross-check OCR Name / ID against file metadata.

    Returns match flags and an ``ocr_score`` in 0–100.
    """
    checks: dict[str, bool] = {}
    details: dict[str, dict[str, str | None]] = {}

    for key in ("name", "id_number"):
        ocr_val = ocr_fields.get(key)
        meta_val = metadata.get(key)
        details[key] = {"ocr": ocr_val, "metadata": meta_val}
        if not ocr_val or not meta_val:
            checks[key] = False
        else:
            checks[key] = _normalize(ocr_val) == _normalize(meta_val)

    matched = sum(1 for ok in checks.values() if ok)
    total = len(checks)
    ocr_score = float(100.0 * matched / total) if total else 0.0

    return {
        "ocr_score": ocr_score,
        "matched": checks,
        "fields": details,
        "all_match": matched == total and total > 0,
    }


class OCRChecker:
    """
    EasyOCR-backed text extractor with one-shot GPU→CPU fallback.

    If GPU initialization fails, immediately retries with ``gpu=False``.
    A second failure is raised (no further retries).
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = True,
        reader: Any | None = None,
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = reader
        self.used_cpu_fallback = False
        if self._reader is None:
            self._reader = self._create_reader(prefer_gpu=gpu)

    def _create_reader(self, prefer_gpu: bool) -> Any:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "easyocr is required for OCR checks. Install project requirements."
            ) from exc

        try:
            return easyocr.Reader(self.languages, gpu=prefer_gpu)
        except Exception as gpu_exc:  # noqa: BLE001 — driver/CUDA conflicts vary
            if not prefer_gpu:
                # Already on CPU — stop per no-loop rule
                raise RuntimeError(
                    f"EasyOCR failed in CPU mode: {gpu_exc}"
                ) from gpu_exc
            # Immediate CPU fallback (single retry)
            self.used_cpu_fallback = True
            try:
                return easyocr.Reader(self.languages, gpu=False)
            except Exception as cpu_exc:  # noqa: BLE001
                raise RuntimeError(
                    "EasyOCR failed on GPU and again on CPU. Stopping. "
                    f"GPU error: {gpu_exc}; CPU error: {cpu_exc}"
                ) from cpu_exc

    def extract_text(self, image: ImageInput) -> str:
        """Return concatenated OCR text lines."""
        arr = _to_rgb_array(image)
        results = self._reader.readtext(arr, detail=0, paragraph=False)
        if isinstance(results, list):
            return "\n".join(str(line) for line in results)
        return str(results)

    def extract_fields(self, image: ImageInput) -> dict[str, str]:
        """Extract Name and ID Number fields from the document image."""
        return parse_ocr_fields(self.extract_text(image))

    def cross_check(
        self,
        image: ImageInput,
        metadata_source: str | Path | Image.Image | bytes | dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        OCR the image and compare Name / ID to file metadata.

        ``metadata_source`` may be a path, image, raw dict, or omitted
        (then metadata is read from ``image`` when it is a path/PIL image).
        """
        ocr_fields = self.extract_fields(image)

        if isinstance(metadata_source, dict):
            metadata = {
                k: str(v)
                for k, v in metadata_source.items()
                if k in ("name", "id_number") and v is not None
            }
        elif metadata_source is not None:
            metadata = read_file_metadata(metadata_source)
        elif isinstance(image, (str, Path, Image.Image, bytes)):
            metadata = read_file_metadata(image)
        else:
            metadata = {}

        comparison = compare_ocr_to_metadata(ocr_fields, metadata)
        return {
            "text_fields": ocr_fields,
            "metadata": metadata,
            "ocr_score": comparison["ocr_score"],
            "matched": comparison["matched"],
            "fields": comparison["fields"],
            "all_match": comparison["all_match"],
            "skipped": False,
            "cpu_fallback": self.used_cpu_fallback,
        }
