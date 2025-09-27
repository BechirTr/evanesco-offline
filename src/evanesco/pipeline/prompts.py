"""Prompt loading helpers shared by LLM integrations."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Optional

NER_PROMPT_FILENAME = "ner_fewshot.jsonl"
CONFIRM_PROMPT_FILENAME = "pii_audit.jsonl"

NER_PROMPT_FALLBACK = (
    "SYSTEM: You are a precise NER extractor.\n"
    "Return STRICT JSON: {items:[{text,start,end,label}]}. Use character offsets into the provided text.\n"
    "Labels: PERSON, ORG, GPE, LOC, DATE, EMAIL, PHONE, IBAN, CREDIT_CARD, OTHER."
)

CONFIRM_PROMPT_FALLBACK = (
    "SYSTEM: You are a strict JSON generator for PII redaction.\n"
    "Return only JSON with keys: items (list of {text,start,end,redact,category,why,score}), notes.\n"
    "Categories may include PERSON, ORG, GPE, LOC, DATE, EMAIL, PHONE, IBAN, CREDIT_CARD, OTHER.\n"
)


@lru_cache(maxsize=16)
def _read_text_cached(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


@lru_cache(maxsize=4)
def _load_packaged_prompt(filename: str) -> Optional[str]:
    try:
        ref = resources.files("evanesco.data").joinpath("prompts", filename)
        if ref.is_file():
            return ref.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


@lru_cache(maxsize=4)
def _search_repo_prompt(filename: str) -> Optional[str]:
    for parent in Path(__file__).resolve().parents:
        cand = parent / "prompts" / filename
        if cand.exists():
            return _read_text_cached(str(cand.resolve()))
    return None


def load_prompt(
    explicit_path: Optional[str], packaged_filename: str, fallback: str
) -> str:
    """Resolve a prompt file using explicit path, repo prompts, or packaged fallback."""

    if explicit_path:
        path_obj = Path(explicit_path)
        if path_obj.exists():
            try:
                return _read_text_cached(str(path_obj.resolve()))
            except Exception:
                pass
    repo_prompt = _search_repo_prompt(packaged_filename)
    if repo_prompt:
        return repo_prompt
    packaged_prompt = _load_packaged_prompt(packaged_filename)
    if packaged_prompt:
        return packaged_prompt
    return fallback


__all__ = [
    "load_prompt",
    "NER_PROMPT_FILENAME",
    "CONFIRM_PROMPT_FILENAME",
    "NER_PROMPT_FALLBACK",
    "CONFIRM_PROMPT_FALLBACK",
]
