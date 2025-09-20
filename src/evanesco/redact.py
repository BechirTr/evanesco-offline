"""Redaction routines.

Provides utilities to draw opaque rectangles over detected PII regions and to
export a set of page images back to a compact PDF while preserving layout.
"""

from typing import List, Tuple
from PIL import Image, ImageDraw
import img2pdf
import io
from PIL import ImageFont


def _inflate(
    box: Tuple[int, int, int, int], px: int, W: int, H: int
) -> Tuple[int, int, int, int]:
    """Inflate a rectangle while clamping to image bounds.

    Parameters
    ----------
    box:
        Rectangle as ``(x, y, w, h)``.
    px:
        Pixels to inflate on all sides.
    W:
        Image width.
    H:
        Image height.

    Returns
    -------
    tuple
        Clamped rectangle ``(x, y, w, h)`` after inflation.
    """
    x, y, w, h = box
    x2 = max(0, x - px)
    y2 = max(0, y - px)
    w2 = min(W - x2, w + 2 * px)
    h2 = min(H - y2, h + 2 * px)
    return (x2, y2, w2, h2)


def redact_page(
    img: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    fill_rgb=(0, 0, 0),
    inflate_px: int = 1,
) -> Image.Image:
    """Draw filled rectangles over PII regions on a copy of the image.

    Parameters
    ----------
    img:
        Source page image.
    boxes:
        List of rectangles ``(x, y, w, h)`` to fill.
    fill_rgb:
        Fill color as an RGB tuple.
    inflate_px:
        Pixels to inflate each rectangle for safer coverage.

    Returns
    -------
    PIL.Image.Image
        Redacted image.
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    for x, y, w, h in boxes:
        x2, y2, w2, h2 = _inflate((x, y, w, h), inflate_px, W, H)
        draw.rectangle([x2, y2, x2 + w2, y2 + h2], fill=tuple(fill_rgb))
    return img


def save_pdf(images: List[Image.Image], out_path: str) -> None:
    """Save a list of PIL images as a compact PDF.

    Parameters
    ----------
    images:
        Ordered page images to export.
    out_path:
        Output PDF file path.
    """
    tmp_files = []
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        tmp = io.BytesIO()
        im.save(tmp, format="JPEG", quality=95)
        tmp.seek(0)
        tmp_files.append(tmp.read())
    with open(out_path, "wb") as f:
        f.write(img2pdf.convert(tmp_files))


def redact_page_with_labels(
    img: Image.Image,
    box_infos: List[dict],
    fill_rgb=(0, 0, 0),
    text_rgb=(255, 255, 255),
    inflate_px: int = 1,
    font_size: int = 14,
) -> Image.Image:
    """Redact with solid boxes and overlay pseudonym labels.

    Draws filled rectangles and writes a short label (e.g., ``PERSON_1``) atop
    each redaction area.
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, info in enumerate(box_infos, start=1):
        x, y, w, h = info.get("box", (0, 0, 0, 0))
        x2, y2, w2, h2 = _inflate((x, y, w, h), inflate_px, W, H)
        draw.rectangle([x2, y2, x2 + w2, y2 + h2], fill=tuple(fill_rgb))
        cat = (info.get("category") or "PII").upper()
        label = f"{cat}_{idx}"
        if font is not None:
            tw, th = draw.textsize(label, font=font)
            tx = min(max(x2 + 2, 0), max(W - tw - 2, 0))
            ty = min(max(y2 + 2, 0), max(H - th - 2, 0))
            draw.text((tx, ty), label, fill=tuple(text_rgb), font=font)
    return img


def draw_preview(
    img: Image.Image,
    kept: List[Tuple[int, int, int, int]],
    rejected: List[Tuple[int, int, int, int]],
    *,
    kept_color=(0, 255, 0),
    rejected_color=(255, 0, 0),
    width: int = 3,
) -> Image.Image:
    """Draw outline boxes for QA preview: green=kept, red=rejected."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for x, y, w, h in kept:
        draw.rectangle([x, y, x + w, y + h], outline=kept_color, width=width)
    for x, y, w, h in rejected:
        draw.rectangle([x, y, x + w, y + h], outline=rejected_color, width=width)
    return out
