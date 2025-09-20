"""Align character spans to OCR word boxes to produce redaction rectangles.

The alignment heuristically reconstructs a linear text stream from OCR tokens
and selects words that overlap a requested character span, then merges their
geometries into a single rectangle.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd


def spans_to_boxes(
    tsv_df: pd.DataFrame, text: str, spans: List[Dict[str, Any]]
) -> List[Tuple[int, int, int, int]]:
    """Map text spans onto OCR word boxes and return rectangles.

    Parameters
    ----------
    tsv_df:
        Pandas DataFrame from :func:`evanesco.ocr.image_ocr_tsv`.
    text:
        Reconstructed text stream aligned to the OCR tokens.
    spans:
        List of span dicts with keys ``start`` and ``end``.

    Returns
    -------
    list[tuple]
        Rectangles as ``(x, y, w, h)`` ready for redaction.

    Notes
    -----
    This is a heuristic approach (space-joined). For character-perfect mapping,
    consider hOCR/ALTO with per-character geometry.
    """
    boxes: List[Tuple[int, int, int, int]] = []
    words = tsv_df[["text", "left", "top", "width", "height"]].reset_index(drop=True)
    # Build linear offsets for each OCR word
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for _, row in words.iterrows():
        t = str(row["text"]) if row["text"] is not None else ""
        start = cursor
        end = cursor + len(t)
        offsets.append((start, end))
        cursor = end + 1  # space

    for sp in spans:
        s, e = int(sp["start"]), int(sp["end"])
        sel_idx: List[int] = []
        for i, (ws, we) in enumerate(offsets):
            if not (e <= ws or s >= we):  # overlap
                sel_idx.append(i)
        if sel_idx:
            lefts = words.iloc[sel_idx]["left"]
            tops = words.iloc[sel_idx]["top"]
            rights = words.iloc[sel_idx]["left"] + words.iloc[sel_idx]["width"]
            bottoms = words.iloc[sel_idx]["top"] + words.iloc[sel_idx]["height"]
            x = int(lefts.min())
            y = int(tops.min())
            w = int(rights.max() - x)
            h = int(bottoms.max() - y)
            boxes.append((x, y, w, h))
    return boxes


def spans_to_box_info(
    tsv_df: pd.DataFrame, text: str, spans: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return per-span box info dicts with geometry and provenance.

    Each item has keys: ``box`` (x,y,w,h), ``start``, ``end``, ``text``,
    optional ``label`` and ``source``.
    """
    words = tsv_df[["text", "left", "top", "width", "height"]].reset_index(drop=True)
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for _, row in words.iterrows():
        t = str(row["text"]) if row["text"] is not None else ""
        start = cursor
        end = cursor + len(t)
        offsets.append((start, end))
        cursor = end + 1

    infos: List[Dict[str, Any]] = []
    for sp in spans:
        s, e = int(sp["start"]), int(sp["end"])
        sel_idx: List[int] = []
        for i, (ws, we) in enumerate(offsets):
            if not (e <= ws or s >= we):
                sel_idx.append(i)
        if not sel_idx:
            continue
        lefts = words.iloc[sel_idx]["left"]
        tops = words.iloc[sel_idx]["top"]
        rights = words.iloc[sel_idx]["left"] + words.iloc[sel_idx]["width"]
        bottoms = words.iloc[sel_idx]["top"] + words.iloc[sel_idx]["height"]
        x = int(lefts.min())
        y = int(tops.min())
        w = int(rights.max() - x)
        h = int(bottoms.max() - y)
        item = {
            "box": (x, y, w, h),
            "start": s,
            "end": e,
            "text": sp.get("text", text[s:e]),
        }
        if "label" in sp:
            item["category"] = sp["label"]
        if "source" in sp:
            item["source"] = sp["source"]
        infos.append(item)
    return infos
