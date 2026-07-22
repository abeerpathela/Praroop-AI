"""Generate synthetic authentic and forged ID-style test images."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from core.ocr_checker import embed_file_metadata

# Canvas and layout (1000x1000 sample ID card)
WIDTH, HEIGHT = 1000, 1000
JPEG_QUALITY = 95

NAME_POS = (80, 120)
DOB_POS = (80, 200)
PHOTO_BOX = (650, 100, 900, 400)  # left, top, right, bottom
DOB_LABEL = "DOB: 01-01-"
DOB_YEAR = "1990"
FORGED_YEAR = "1985"

# Canonical identity fields (also embedded in JPEG EXIF metadata)
DOCUMENT_NAME = "Rahul Kumar"
DOCUMENT_ID = "IND-77821"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SAMPLE_PATH = RAW_DIR / "sample.jpg"
FORGED_PATH = RAW_DIR / "sample_forged.jpg"
META_PATH = RAW_DIR / "forge_meta.json"

# Updated at generation time; tests may also read forge_meta.json
YEAR_BOX = (268, 198, 360, 245)
FORGED_YEAR_POS = (272, 205)


def _try_load_font(size: int, bold: bool = False) -> Optional[ImageFont.ImageFont]:
    """
    Attempt to load a TrueType font (max two path attempts).

    Returns None if both attempts fail so callers can use the rectangle fallback.
    """
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates[:2]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return None


def _can_render_text() -> bool:
    return _try_load_font(28) is not None


def _jpeg_bytes(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG and return the decoded RGB image."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _compute_year_box(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Return (YEAR_BOX ltrt, forged text position) aligned to the DOB year."""
    x, y = DOB_POS
    prefix_bbox = draw.textbbox((x, y), DOB_LABEL, font=font)
    full_bbox = draw.textbbox((x, y), DOB_LABEL + DOB_YEAR, font=font)
    # Generous pad so the white-out / paste fully covers glyph edges
    left = prefix_bbox[2] - 4
    top = full_bbox[1] - 6
    right = full_bbox[2] + 8
    bottom = full_bbox[3] + 6
    text_pos = (left + 4, y)
    return (left, top, right, bottom), text_pos


def _draw_authentic_content(
    draw: ImageDraw.ImageDraw,
    font_mode: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Draw ID fields. Returns year box + forged text pos for later editing."""
    global YEAR_BOX, FORGED_YEAR_POS

    draw.rectangle(PHOTO_BOX, fill=(160, 160, 160), outline=(80, 80, 80), width=3)

    if font_mode == "text":
        title_font = _try_load_font(36, bold=True) or ImageFont.load_default()
        body_font = _try_load_font(28) or ImageFont.load_default()
        draw.text((80, 40), "GOVERNMENT OF INDIA", fill=(0, 0, 0), font=title_font)
        draw.text(NAME_POS, "Name: Rahul Kumar", fill=(0, 0, 0), font=body_font)
        draw.text(DOB_POS, DOB_LABEL + DOB_YEAR, fill=(0, 0, 0), font=body_font)
        draw.text((80, 280), "ID: IND-77821", fill=(0, 0, 0), font=body_font)
        year_box, forged_pos = _compute_year_box(draw, body_font)
    else:
        # Fallback: blue bars simulate authentic printed fields (no fonts).
        draw.rectangle([80, 40, 520, 80], fill=(0, 0, 180))
        draw.rectangle([80, 120, 480, 165], fill=(0, 0, 180))
        draw.rectangle([80, 200, 420, 245], fill=(0, 0, 180))
        draw.rectangle([80, 280, 360, 325], fill=(0, 0, 180))
        year_box = (300, 195, 430, 250)
        forged_pos = (308, 205)

    YEAR_BOX = year_box
    FORGED_YEAR_POS = forged_pos
    return year_box, forged_pos


def _build_foreign_year_patch(
    width: int,
    height: int,
    font_mode: str,
) -> Image.Image:
    """
    Build a replacement tile compressed at a mismatched JPEG quality.

    Pasting a low-quality, high-detail tile into a q=95 canvas creates a
    strong ELA / noise signal that raises the forged image's mean residual.
    """
    # High-frequency base (checker + noise) so JPEG q-mismatch is measurable
    rng = np.random.default_rng(7)
    yy, xx = np.indices((height, width))
    checker = ((xx // 4 + yy // 4) % 2) * 40 + 200
    noise = rng.integers(-25, 26, size=(height, width), dtype=np.int16)
    gray = np.clip(checker.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, np.clip(gray.astype(np.int16) - 15, 0, 255).astype(np.uint8)], axis=-1)
    tile = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(tile)

    if font_mode == "text":
        font = _try_load_font(32, bold=True) or _try_load_font(30)
        if font is not None:
            draw.text((4, max(0, (height - 28) // 2)), FORGED_YEAR, fill=(20, 20, 40), font=font)
        else:
            draw.rectangle([2, 2, width - 3, height - 3], outline=(220, 30, 30), width=3)
    else:
        draw.rectangle([2, 2, width - 3, height - 3], fill=(220, 30, 30))

    return _jpeg_bytes(tile, quality=25)


def create_authentic_image(path: Path = SAMPLE_PATH) -> Path:
    """Create a 1000x1000 white RGB ID-style image and save as JPEG q=95."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_mode = "text" if _can_render_text() else "rects"
    year_box, forged_pos = _draw_authentic_content(draw, font_mode=font_mode)

    # Double round-trip at the same quality stabilizes authentic residuals.
    stable = _jpeg_bytes(_jpeg_bytes(img, JPEG_QUALITY), JPEG_QUALITY)
    exif = embed_file_metadata(stable, DOCUMENT_NAME, DOCUMENT_ID)
    stable.save(path, format="JPEG", quality=JPEG_QUALITY, exif=exif)

    META_PATH.write_text(
        json.dumps(
            {
                "year_box": list(year_box),
                "forged_year_pos": list(forged_pos),
                "font_mode": font_mode,
                "name": DOCUMENT_NAME,
                "id_number": DOCUMENT_ID,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def create_forged_image(
    authentic_path: Path = SAMPLE_PATH,
    forged_path: Path = FORGED_PATH,
) -> Path:
    """
    Forge the DOB year: white-out '1990' and paste a mismatched '1985' tile.

    Resaving at quality 95 embeds the foreign JPEG history and creates ELA glow.
    """
    global YEAR_BOX, FORGED_YEAR_POS

    forged_path.parent.mkdir(parents=True, exist_ok=True)
    if META_PATH.is_file():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        YEAR_BOX = tuple(meta["year_box"])  # type: ignore[assignment]
        FORGED_YEAR_POS = tuple(meta["forged_year_pos"])  # type: ignore[assignment]
        font_mode = meta.get("font_mode", "text")
    else:
        font_mode = "text" if _can_render_text() else "rects"

    img = Image.open(authentic_path).convert("RGB")
    left, top, right, bottom = YEAR_BOX
    # Expand the edit zone so the foreign patch dominates global ELA mean.
    left = max(0, left - 20)
    top = max(0, top - 15)
    right = min(img.width, right + 120)
    bottom = min(img.height, bottom + 40)
    YEAR_BOX = (left, top, right, bottom)
    width, height = right - left, bottom - top

    # Persist expanded box so tests inspect the true forged region
    if META_PATH.is_file():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    else:
        meta = {"font_mode": font_mode, "forged_year_pos": list(FORGED_YEAR_POS)}
    meta["year_box"] = list(YEAR_BOX)
    meta["name"] = DOCUMENT_NAME
    meta["id_number"] = DOCUMENT_ID
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # White-out original year area, then paste foreign-history patch
    draw = ImageDraw.Draw(img)
    draw.rectangle(YEAR_BOX, fill=(255, 255, 255))
    patch = _build_foreign_year_patch(width, height, font_mode=font_mode)
    img.paste(patch, (left, top))

    # Extra obvious overpaint of the forged year string
    if font_mode == "text":
        alt = _try_load_font(34, bold=True) or _try_load_font(30)
        if alt is not None:
            draw.text((left + 8, top + 8), FORGED_YEAR, fill=(10, 10, 30), font=alt)

    # Preserve identity metadata so OCR can cross-check Name / ID
    exif = embed_file_metadata(img, DOCUMENT_NAME, DOCUMENT_ID)
    img.save(forged_path, format="JPEG", quality=JPEG_QUALITY, exif=exif)
    return forged_path


def load_year_box() -> tuple[int, int, int, int]:
    """Load forged-region box from meta (preferred) or module default."""
    if META_PATH.is_file():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        box = meta["year_box"]
        return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    return YEAR_BOX  # type: ignore[return-value]


def generate_assets() -> tuple[Path, Path]:
    """Generate authentic and forged test assets under data/raw/."""
    sample = create_authentic_image()
    forged = create_forged_image(sample)
    return sample, forged


if __name__ == "__main__":
    sample, forged = generate_assets()
    print(f"Wrote authentic: {sample}")
    print(f"Wrote forged:    {forged}")
    print(f"Forged region (l,t,r,b): {load_year_box()}")
