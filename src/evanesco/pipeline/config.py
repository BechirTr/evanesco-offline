"""Configuration primitives for the Evanesco pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel


@dataclass
class RunConfig:
    """Runtime configuration for OCR, detection, and redaction."""

    lang: str = "eng"
    psm: int = 3
    dpi: int = 400
    use_spacy: bool = True
    spacy_model: str = "en_core_web_lg"
    use_llm_ner: bool = False
    llm_ner_model: Optional[str] = "gemma3:4b"
    llm_ner_prompt_path: Optional[str] = None
    use_regex: bool = True
    use_llm: bool = True
    llm_model: str = "gpt-oss:20b"
    llm_url: str = "http://localhost:11434/api/generate"
    llm_timeout: int = 300
    llm_ner_chunk_chars: int = 400
    categories: Optional[List[str]] = None
    box_inflation_px: int = 2
    fill_rgb: Tuple[int, int, int] = (0, 0, 0)
    prompt_path: Optional[str] = None
    policy_path: Optional[str] = None
    mode: str = "redact"  # 'redact' or 'label'
    safe_pdf_mode: bool = True
    track_reasons: bool = True
    preprocess: bool = True
    deskew: bool = True
    binarize: bool = True
    auto_psm: bool = True
    tess_configs: Optional[Dict[str, Any]] = None
    workers: int = 1
    instrument: bool = True
    generate_previews: bool = True
    explain_traces: bool = False
    export_ocr_debug: bool = False


class PageResult(BaseModel):
    """Per-page output payload."""

    page_index: int
    text: str
    candidates: List[Dict[str, Any]]
    llm_json: Optional[Dict[str, Any]] = None
    boxes_applied: int = 0
    boxes: Optional[List[Dict[str, Any]]] = None
    timings: Optional[Dict[str, float]] = None
    preview_path: Optional[str] = None


__all__ = ["RunConfig", "PageResult"]
