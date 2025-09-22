"""Sphinx configuration for Evanesco Offline documentation.

Build with:
    sphinx-build -b html docs docs/_build
"""

from __future__ import annotations

import os
import sys
from datetime import date

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

project = "Evanesco Offline"
author = "Evanesco contributors"
copyright = f"{date.today().year}, {author}"

try:
    from evanesco import __version__ as version
except Exception:
    version = "0.0.0"
release = version
source_suffix = [
    ".md",
]
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

