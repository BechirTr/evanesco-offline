"""Few-shot NER extraction using local LLMs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .config import RunConfig
from .llm_utils import get_ollama_client
from .prompts import (
    NER_PROMPT_FILENAME,
    NER_PROMPT_FALLBACK,
    load_prompt,
)


def _normalize_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            return payload.get("items", [])
        if "text" in payload:
            return _normalize_items(payload["text"])
    if isinstance(payload, list):
        return payload  # type: ignore[return-value]
    if isinstance(payload, str):
        trimmed = payload.strip()
        if trimmed.startswith("```"):
            body = trimmed[3:]
            if body.lower().startswith("json"):
                body = body[4:]
            closing = body.rfind("```")
            if closing != -1:
                body = body[:closing]
            trimmed = body.strip()
        try:
            return _normalize_items(json.loads(trimmed))
        except Exception:
            if trimmed.startswith("["):
                closing = trimmed.rfind("]")
                if closing == -1:
                    closing = trimmed.rfind("}")
                if closing != -1:
                    candidate = trimmed[: closing + 1]
                    try:
                        return _normalize_items(json.loads(candidate))
                    except Exception:
                        pass
                try:
                    fixed = trimmed if trimmed.endswith("]") else trimmed + "]"
                    return _normalize_items(json.loads(fixed))
                except Exception:
                    pass
            objs: List[Dict[str, Any]] = []
            depth = 0
            start_idx: int | None = None
            for idx, ch in enumerate(trimmed):
                if ch == "{":
                    if depth == 0:
                        start_idx = idx
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        snippet = trimmed[start_idx : idx + 1]
                        try:
                            obj = json.loads(snippet)
                            objs.append(obj)
                        except Exception:
                            pass
            return objs
    return []


def llm_ner_candidates(text: str, cfg: RunConfig) -> List[Dict[str, Any]]:
    """Extract NER candidates using an LLM with a few-shot prompt."""

    prompt = load_prompt(
        cfg.llm_ner_prompt_path, NER_PROMPT_FILENAME, NER_PROMPT_FALLBACK
    )
    max_chunk = max(150, int(getattr(cfg, "llm_ner_chunk_chars", 400) or 400))
    raw_chunks: List[Tuple[int, str]] = []
    cursor = 0
    text_len = len(text)
    while cursor < text_len:
        segment = text[cursor : cursor + max_chunk]
        if segment:
            raw_chunks.append((cursor, segment))
        cursor += max_chunk
    if not raw_chunks:
        return []
    chunk_total = len(raw_chunks)

    model = cfg.llm_ner_model or cfg.llm_model
    client = get_ollama_client(cfg.llm_url)

    store_trace = bool(getattr(cfg, "explain_traces", False))

    def _process_chunk(data: Tuple[int, int, int, str]) -> List[Dict[str, Any]]:
        index, total, offset, segment = data
        local_user = {
            "text": segment,
            "instructions": (
                "Extract all personally identifiable information using character offsets (start,end)."
                " Return STRICT JSON with key 'items' only. Each item must include text,start,end,label."
                " Valid labels: PERSON, ORG, GPE, LOC, DATE, EMAIL, PHONE, IBAN, CREDIT_CARD, AGE, GENDER, OTHER."
                " Do not invent offsets. If unsure, omit the item."
                f" This is chunk {index} of {total}."
            ),
        }
        full = f"{prompt}\n\nUSER:\n{json.dumps(local_user)}"
        resp = client.generate(full, model=model, timeout=cfg.llm_timeout)
        try:
            js = json.loads(resp)
        except Exception:
            js = resp
        items = _normalize_items(js)
        chunk_out: List[Dict[str, Any]] = []
        used_ranges: List[Tuple[int, int]] = []

        def ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
            return not (a[1] <= b[0] or a[0] >= b[1])

        def reserve_range(start: int, end: int) -> Tuple[int, int]:
            used_ranges.append((start, end))
            return start, end

        def next_unclaimed_range(text_val: str) -> Optional[Tuple[int, int]]:
            if not text_val:
                return None
            search_pos = 0
            value_len = len(text_val)
            max_pos = len(segment) - value_len
            while search_pos <= max_pos:
                idx = segment.find(text_val, search_pos)
                if idx == -1:
                    return None
                candidate = (idx, idx + value_len)
                if not any(
                    ranges_overlap(candidate, existing) for existing in used_ranges
                ):
                    used_ranges.append(candidate)
                    return candidate
                search_pos = idx + 1
            return None

        for it in items:
            if isinstance(it, str):
                try:
                    it = json.loads(it)
                except Exception:
                    continue
            if not isinstance(it, dict):
                continue
            text_val = it.get("text")
            start = it.get("start")
            end = it.get("end")
            label = (it.get("label") or "OTHER").upper()
            if text_val is None:
                continue
            valid_range = (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(segment)
                and not any(
                    ranges_overlap((start, end), existing) for existing in used_ranges
                )
            )
            if not valid_range:
                allocated = next_unclaimed_range(text_val)
                if allocated is None:
                    continue
                start, end = allocated
            else:
                start, end = reserve_range(int(start), int(end))
            chunk_out.append(
                {
                    "label": label,
                    "start": offset + int(start),
                    "end": offset + int(end),
                    "text": text[offset + int(start) : offset + int(end)],
                    "source": "LLM_NER",
                    **(
                        {"trace": {"prompt": full, "response": resp}}
                        if store_trace
                        else {}
                    ),
                }
            )
        return chunk_out

    all_results: List[Dict[str, Any]] = []
    for idx, (offset, segment) in enumerate(raw_chunks, start=1):
        all_results.extend(_process_chunk((idx, chunk_total, offset, segment)))

    seen = set()
    uniq: List[Dict[str, Any]] = []
    for candidate in all_results:
        key = (candidate["start"], candidate["end"], candidate.get("label", ""))
        if key not in seen:
            seen.add(key)
            uniq.append(candidate)
    return uniq


__all__ = ["llm_ner_candidates"]
