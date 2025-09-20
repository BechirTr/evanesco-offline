"""Deterministic regex-based PII detectors.

This module uses the third-party ``regex`` package for robust patterns and
returns span dictionaries that are compatible with alignment routines.
"""

import regex as re
from typing import List, Dict, Any


EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}")
IBAN_RE = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", re.I)
CREDIT_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def regex_findall(text: str) -> List[Dict[str, Any]]:
    """Find PII-like spans using regular expressions.

    Parameters
    ----------
    text:
        Input text to scan.

    Returns
    -------
    list[dict]
        Span dictionaries with keys: ``label``, ``start``, ``end``, ``text``.
    """
    out = []
    for name, pat in [
        ("EMAIL", EMAIL_RE),
        ("PHONE", PHONE_RE),
        ("IBAN", IBAN_RE),
        ("CREDIT_CARD", CREDIT_RE),
    ]:
        for m in pat.finditer(text):
            out.append({
                "label": name,
                "start": m.start(),
                "end": m.end(),
                "text": text[m.start(): m.end()],
                "source": "REGEX",
            })
    return out
