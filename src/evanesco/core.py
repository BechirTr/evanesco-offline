"""Backward-compatible entry points for the Evanesco pipeline.

The implementation now lives in ``evanesco.pipeline`` modules split by
responsibility (detection, LLM integration, rendering, orchestration). This
module re-exports the public surface area expected by downstream callers.
"""

from __future__ import annotations

from .pipeline import (
    RunConfig,
    PageResult,
    detect_candidates,
    llm_filter,
    llm_ner_candidates,
    process_pdf,
    process_path,
    resolve_final_spans,
)

__all__ = [
    "RunConfig",
    "PageResult",
    "detect_candidates",
    "llm_filter",
    "llm_ner_candidates",
    "process_pdf",
    "process_path",
    "resolve_final_spans",
]
