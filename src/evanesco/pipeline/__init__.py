"""Composable building blocks for the Evanesco redaction pipeline."""

from .config import RunConfig, PageResult
from .detection import detect_candidates, resolve_final_spans
from .llm_confirmation import llm_filter
from .llm_ner import llm_ner_candidates
from .orchestration import process_pdf, process_path

__all__ = [
    "RunConfig",
    "PageResult",
    "detect_candidates",
    "resolve_final_spans",
    "llm_filter",
    "llm_ner_candidates",
    "process_pdf",
    "process_path",
]
