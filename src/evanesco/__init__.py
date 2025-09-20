"""Evanesco Offline

Local OCR + PII detection + redaction toolkit with optional Ollama LLM
confirmation. See the ``evanesco.core`` module for the composable pipeline
APIs and ``evanesco.cli`` / ``evanesco.ui`` for user entrypoints.
"""

__all__ = [
    "core",
    "ocr",
    "spacy_detect",
    "regex_detect",
    "align",
    "redact",
    "ollama_client",
    "policy",
    "audit",
    "batch",
    "api",
    "llm",
    "logging",
    "settings",
    "health",
]

__version__ = "0.1.0"
