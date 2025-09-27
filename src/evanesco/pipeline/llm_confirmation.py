"""LLM confirmation step for redaction decisions."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .config import RunConfig
from .llm_utils import get_ollama_client
from .prompts import (
    CONFIRM_PROMPT_FILENAME,
    CONFIRM_PROMPT_FALLBACK,
    load_prompt,
)


def normalize_llm_items(
    candidates: List[Dict[str, Any]], raw_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Ensure every candidate receives an explicit LLM decision."""

    cand_lookup: Dict[tuple[int, int], Dict[str, Any]] = {
        (int(c["start"]), int(c["end"])): c for c in candidates
    }
    normalized: List[Dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    for raw in raw_items:
        try:
            start = int(raw.get("start"))
            end = int(raw.get("end"))
        except Exception:
            continue
        if start >= end:
            continue
        key = (start, end)
        cand = cand_lookup.get(key, {})
        text = raw.get("text") or cand.get("text")
        if text is None:
            continue
        category = (
            raw.get("category") or raw.get("label") or cand.get("label") or "OTHER"
        ).upper()
        redact = bool(raw.get("redact", True))
        why = raw.get("why") or ""
        score_val = raw.get("score")
        try:
            score = float(score_val) if score_val is not None else 0.0
        except Exception:
            score = 0.0
        normalized.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "category": category,
                "redact": redact,
                "why": why,
                "score": score,
                "source": raw.get("source") or cand.get("source"),
            }
        )
        seen.add(key)

    for cand_key, cand in cand_lookup.items():
        if cand_key in seen:
            continue
        normalized.append(
            {
                "text": cand.get("text"),
                "start": cand_key[0],
                "end": cand_key[1],
                "category": (cand.get("label") or "OTHER").upper(),
                "redact": False,
                "why": "llm_missing_default_keep",
                "score": 0.0,
                "source": cand.get("source"),
            }
        )

    normalized.sort(key=lambda item: (item["start"], item["end"], item["category"]))
    return normalized


def llm_filter(
    text: str, candidates: List[Dict[str, Any]], cfg: RunConfig
) -> Dict[str, Any]:
    """Ask a local LLM (Ollama) to confirm redactions and assign categories."""

    if not candidates:
        return {"items": [], "notes": "no candidates"}

    items = [
        {
            "text": c["text"],
            "start": c["start"],
            "end": c["end"],
            "label": c.get("label", "OTHER"),
            "source": c.get("source"),
        }
        for c in candidates
    ]

    prompt = load_prompt(
        cfg.prompt_path, CONFIRM_PROMPT_FILENAME, CONFIRM_PROMPT_FALLBACK
    )
    user = {
        "page_text": text,
        "candidates": items,
        "instructions": (
            "Return STRICT JSON with keys: items (list of {text,start,end,redact,category,why,score}), notes. "
            "Include a short 'why' rationale per item and an optional numeric 'score' 0-1."
        ),
    }
    full = f"{prompt}\n\nUSER:\n{json.dumps(user)}"
    client = get_ollama_client(cfg.llm_url)
    raw = client.generate_raw(full, model=cfg.llm_model, timeout=cfg.llm_timeout)
    response = raw.get("response", "")
    try:
        parsed = json.loads(response)
    except Exception:
        import re

        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = {
                "items": [
                    {
                        "text": item["text"],
                        "start": item["start"],
                        "end": item["end"],
                        "redact": True,
                        "category": item.get("label", "OTHER"),
                        "why": "fallback redaction (parser)",
                        "score": None,
                        "source": item.get("source"),
                    }
                    for item in items
                ],
                "notes": "fallback all redact",
            }
    if isinstance(parsed, dict):
        items = parsed.get("items", [])
    else:
        items = []
    normalized_items = normalize_llm_items(candidates, items)
    parsed = parsed if isinstance(parsed, dict) else {"notes": "unexpected response"}
    parsed["items"] = normalized_items
    parsed["raw"] = raw
    parsed["prompt"] = full
    parsed["response"] = response

    meta = {
        "model": raw.get("model"),
        "created_at": raw.get("created_at"),
        "eval_count": raw.get("eval_count"),
        "eval_duration": raw.get("eval_duration"),
        "prompt_eval_count": raw.get("prompt_eval_count"),
        "prompt_eval_duration": raw.get("prompt_eval_duration"),
        "total_duration": raw.get("total_duration"),
    }
    parsed["meta"] = meta

    if cfg.explain_traces:
        parsed["trace"] = {"prompt": full, "raw": raw}
    return parsed


__all__ = ["llm_filter", "normalize_llm_items"]
