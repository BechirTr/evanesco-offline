"""Detection utilities spanning spaCy, regex, and LLM-based extractors."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from evanesco.policy import Policy
from evanesco.regex_detect import regex_findall
from evanesco.spacy_detect import spacy_ents

from .config import RunConfig
from .llm_confirmation import llm_filter
from .llm_ner import llm_ner_candidates
from evanesco.logging import get_logger

logger = get_logger(__name__)


def detect_candidates(text: str, cfg: RunConfig) -> List[Dict[str, Any]]:
    """Run configured detectors and return unique span candidates."""

    candidates: List[Dict[str, Any]] = []
    if cfg.use_spacy:
        for entity in spacy_ents(cfg.spacy_model, text):
            candidates.append(
                {
                    "label": entity["label"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "text": entity["text"],
                }
            )
    if cfg.use_llm_ner:
        candidates.extend(llm_ner_candidates(text, cfg))
    if cfg.use_regex:
        candidates.extend(regex_findall(text))
    # logger.info(f"Detection candidates found {candidates} "f"using cfg={cfg}")
    seen = set()
    unique: List[Dict[str, Any]] = []
    for cand in candidates:
        key = (cand["start"], cand["end"], cand.get("label", ""))
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def resolve_final_spans(
    text: str,
    candidates: List[Dict[str, Any]],
    cfg: RunConfig,
    policy: Optional[Policy],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[float]]:
    """Apply policy and LLM confirmation to produce final spans."""

    final_spans: List[Dict[str, Any]] = []
    llm_json: Optional[Dict[str, Any]] = None
    llm_duration: Optional[float] = None

    if cfg.use_llm:
        start = time.perf_counter()
        llm_json = llm_filter(text, candidates, cfg)
        llm_duration = time.perf_counter() - start
        for item in llm_json.get("items", []):
            category = item.get("category") or item.get("label") or "OTHER"
            decision = bool(item.get("redact", True))
            if policy is not None:
                decision = decision and policy.should_redact(category)
            if decision:
                final_spans.append(
                    {
                        "start": item["start"],
                        "end": item["end"],
                        "text": item.get("text", text[item["start"] : item["end"]]),
                        "label": category,
                        "source": "LLM",
                    }
                )
    else:
        for cand in candidates:
            category = cand.get("label", "OTHER")
            decision = True if policy is None else policy.should_redact(category)
            if decision:
                final_spans.append(
                    {
                        "start": cand["start"],
                        "end": cand["end"],
                        "text": cand.get("text", text[cand["start"] : cand["end"]]),
                        "label": category,
                        "source": cand.get("source", "DETECTOR"),
                    }
                )

    return final_spans, llm_json, llm_duration


__all__ = ["detect_candidates", "resolve_final_spans"]
