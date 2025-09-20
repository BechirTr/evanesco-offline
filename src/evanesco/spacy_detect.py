"""spaCy-based named entity detection helpers.

Graceful fallbacks:
- Try the requested model name.
- If unavailable, try ``en_core_web_sm``.
- If still unavailable, fall back to ``spacy.blank('en')`` (no NER) and return
  no entities so upstream logic can proceed deterministically.

This module exposes thin wrappers that return simple span dictionaries suitable
for alignment with OCR word boxes.
"""

from typing import List, Dict, Any
from functools import lru_cache
from importlib import import_module
import warnings
import spacy
from pathlib import Path
import os


@lru_cache(maxsize=4)
def _load_spacy(nlp_name: str):
    """Load and cache a spaCy pipeline.

    More robust loading strategy:
    - Strip whitespace from the name
    - If name is a valid path, load from path
    - Try spacy.load(name)
    - Try importing the package and calling its .load()
    - Fall back to en_core_web_sm
    - Fall back to blank('en')
    """
    name = (nlp_name or "").strip()
    debug = bool(os.environ.get("EVANESCO_DEBUG"))
    errors = []
    # Load from model directory path if provided
    p = Path(name)
    if name and p.exists():
        try:
            return spacy.load(str(p))
        except Exception as e:
            errors.append(f"path load failed: {e}")
    # Direct load by name
    if name:
        try:
            return spacy.load(name)
        except Exception as e:
            errors.append(f"spacy.load failed: {e}")
        # If the wheel is installed but not registered, try module import (pkg.load())
        try:
            pkg = import_module(name)
            if hasattr(pkg, "load"):
                return pkg.load()
        except Exception as e:
            errors.append(f"import_module failed: {e}")
    # Language-aware small model fallbacks
    lang_hint = None
    for token in [name, os.environ.get("EVANESCO_LANG", ""), os.environ.get("LANG", "")]:
        t = (token or "").lower()
        if t.startswith("en"): lang_hint = "en"; break
        if t.startswith("fr") or t.startswith("fra"): lang_hint = "fr"; break
    fallback_models = []
    if lang_hint == "fr":
        fallback_models = ["fr_core_news_sm", "en_core_web_sm"]
    else:
        fallback_models = ["en_core_web_sm", "fr_core_news_sm"]
    for fb in fallback_models:
        try:
            return spacy.load(fb)
        except Exception as e:
            errors.append(f"{fb} load failed: {e}")
            try:
                pkg = import_module(fb)
                if hasattr(pkg, "load"):
                    return pkg.load()
            except Exception as e2:
                errors.append(f"{fb} import failed: {e2}")
    msg = (
        f"spaCy model '{name}' not found and 'en_core_web_sm' not available; "
        f"falling back to blank('en') without NER."
    )
    if debug:
        msg += f" details: exec={os.sys.executable} errors={errors}" # pyright: ignore[reportAttributeAccessIssue]
    warnings.warn(msg)
    return spacy.blank("en")


def spacy_ents(nlp_name: str, text: str) -> List[Dict[str, Any]]:
    """Extract entities using a spaCy pipeline.

    Parameters
    ----------
    nlp_name:
        Name of the spaCy model to use for NER.
    text:
        Input text to analyze.

    Returns
    -------
    list[dict]
        List of span dictionaries with keys: ``label``, ``start``, ``end``,
        ``text``. If the pipeline has no NER component, an empty list is
        returned.
    """
    nlp = _load_spacy(nlp_name)
    if "ner" not in nlp.pipe_names:
        return []
    doc = nlp(text)
    return [
        {
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "text": ent.text,
            "source": "SPACY",
        }
        for ent in doc.ents
    ]
