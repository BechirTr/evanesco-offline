"""OCR utilities.

Functions in this module rasterize PDFs into images and extract word-level TSV
with character confidence and bounding boxes using Tesseract.

Enhancements for difficult documents:
- Optional preprocessing (grayscale, denoise, binarize) using OpenCV
- Optional deskew to correct small rotation angles
- Optional auto-PSM retry to maximize token recovery on noisy pages
"""

from typing import List, Dict, Any, Optional
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import pytesseract
from PIL import Image
import os
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is optional at runtime
    cv2 = None  # type: ignore


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """Convert a PDF into a list of PIL images (one per page).

    Attempts to use ``pdf2image`` first (requires Poppler). If Poppler is
    unavailable and ``PyMuPDF`` (``pymupdf``) is installed, falls back to
    rasterization via PyMuPDF.

    Parameters
    ----------
    pdf_path:
        Path to a PDF file on disk.
    dpi:
        Rasterization resolution in dots per inch.

    Environment
    -----------
    POPPLER_PATH:
        Optional explicit path to the Poppler binaries for pdf2image.

    Returns
    -------
    list[PIL.Image.Image]
        Rasterized page images in order.
    """
    poppler_path = os.environ.get("POPPLER_PATH")
    try:
        if poppler_path:
            return convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        return convert_from_path(pdf_path, dpi=dpi)
    except PDFInfoNotInstalledError as e:
        # Optional fallback: PyMuPDF
        try:
            import fitz  # PyMuPDF
        except Exception:
            raise RuntimeError(
                "Poppler not found for pdf2image. Install Poppler (e.g., 'brew install poppler' on macOS, 'apt-get install poppler-utils' on Debian/Ubuntu), set POPPLER_PATH, or install PyMuPDF for fallback."
            ) from e
        # Use PyMuPDF to rasterize
        doc = fitz.open(pdf_path)
        images: List[Image.Image] = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            mode = "RGB" if pix.n < 4 else "RGBA"
            im = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode == "RGBA":
                im = im.convert("RGB")
            images.append(im)
    return images


def _preprocess_image(
    img: Image.Image,
    *,
    deskew: bool = True,
    binarize: bool = True,
) -> Image.Image:
    """Apply simple preprocessing to improve OCR robustness.

    - Convert to grayscale
    - Optional deskew using a Hough-based heuristic
    - Optional binarization with adaptive threshold
    """
    if cv2 is None:
        return img
    arr = np.array(img)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    work = gray
    # Deskew (small-angle) via Hough lines median angle
    if deskew:
        edges = cv2.Canny(work, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
        if lines is not None and len(lines) > 0:
            angles = []
            for rho_theta in lines[:200]:
                rho, theta = rho_theta[0]
                angle = (
                    theta * 180 / np.pi
                ) - 90  # convert to degrees, near 0 for horizontal text
                # Normalize to [-45, 45]
                if angle > 45:
                    angle -= 90
                if angle < -45:
                    angle += 90
                angles.append(angle)
            if angles:
                med = float(np.median(angles))
                if abs(med) > 0.3 and abs(med) < 8.0:  # avoid over-rotation
                    h, w = work.shape[:2]
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), med, 1.0)
                    work = cv2.warpAffine(
                        work,
                        M,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
    if binarize:
        work = cv2.adaptiveThreshold(
            work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
        )
    return Image.fromarray(work)


def image_ocr_tsv(
    img: Image.Image,
    lang: str = "eng",
    psm: int = 3,
    whitelist: str = "",
    blacklist: str = "",
    *,
    preprocess: bool = True,
    deskew: bool = True,
    binarize: bool = True,
    auto_psm: bool = True,
    tess_configs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Tesseract OCR on an image and return word-level TSV.

    Parameters
    ----------
    img:
        Input image to OCR.
    lang:
        Tesseract language code.
    psm:
        Page segmentation mode (0–13).
    whitelist:
        Optional restricted character whitelist.
    blacklist:
        Optional character blacklist.

    Returns
    -------
    dict
        Dict with key ``tsv`` mapping to a pandas DataFrame with columns:
        ``level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text``.
    """
    # Optional image preprocessing
    if preprocess:
        img = _preprocess_image(img, deskew=deskew, binarize=binarize)

    custom_oem_psm_config = f"--oem 1 --psm {psm}".strip()
    if whitelist:
        custom_oem_psm_config += f' -c tessedit_char_whitelist="{whitelist}"'
    if blacklist:
        custom_oem_psm_config += f' -c tessedit_char_blacklist="{blacklist}"'
    # Preserve spaces to help span alignment
    cfg = {"preserve_interword_spaces": 1}
    if tess_configs:
        cfg.update(tess_configs)
    for k, v in cfg.items():
        custom_oem_psm_config += f" -c {k}={v}"

    def run(psm_value: int):
        cfg_str = custom_oem_psm_config.replace(f"--psm {psm}", f"--psm {psm_value}")
        return pytesseract.image_to_data(
            img,
            lang=lang,
            config=cfg_str,
            output_type=pytesseract.Output.DATAFRAME,
        )

    tsv = run(psm)
    # Drop NaNs and reset
    tsv = tsv.dropna(subset=["text"]).reset_index(drop=True)
    # Auto-PSM retry: if too few tokens, re-run with alternates and pick best
    if auto_psm and len(tsv) < 5:
        best = tsv
        best_len = len(tsv)
        for alt in (6, 4, 11):
            try:
                alt_df = run(alt).dropna(subset=["text"]).reset_index(drop=True)
                if len(alt_df) > best_len:
                    best, best_len = alt_df, len(alt_df)
            except Exception:
                pass
        tsv = best
    return {"tsv": tsv}
