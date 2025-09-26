"""Core pipeline orchestration for OCR, detection, LLM filtering, and redaction.

This module wires together OCR, detection (spaCy and regex), optional LLM
confirmation via Ollama, geometric alignment and final redaction.
"""

from typing import Optional, List, Dict, Any, Tuple
from importlib import resources
from functools import lru_cache
from dataclasses import dataclass
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import time
import shutil
import json
from tqdm import tqdm
from .ocr import pdf_to_images, image_ocr_tsv
from .spacy_detect import spacy_ents
from .regex_detect import regex_findall
from .llm import OllamaClient
from .redact import redact_page, save_pdf, redact_page_with_labels, draw_preview
from .align import spans_to_box_info
from .policy import Policy, find_builtin_policy
from .audit import write_audit
import mimetypes

from evanesco.logging import get_logger

logger = get_logger("evanesco")

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


def _load_prompt(
    explicit_path: Optional[str], packaged_filename: str, fallback: str
) -> str:
    if explicit_path:
        pp = Path(explicit_path)
        if pp.exists():
            try:
                return _read_text_cached(str(pp.resolve()))
            except Exception:
                pass
    repo_prompt = _search_repo_prompt(packaged_filename)
    if repo_prompt:
        return repo_prompt
    packaged_prompt = _load_packaged_prompt(packaged_filename)
    if packaged_prompt:
        return packaged_prompt
    return fallback


@lru_cache(maxsize=4)
def _get_ollama_client(url: str) -> OllamaClient:
    return OllamaClient(url=url)


@dataclass
class RunConfig:
    """Runtime configuration for OCR + detection + redaction.

    Attributes
    ----------
    lang:
        Tesseract language code (e.g., ``eng``).
    psm:
        Tesseract page segmentation mode (0–13).
    dpi:
        Rasterization DPI used for PDFs.
    use_spacy:
        Enable spaCy NER candidate detection.
    spacy_model:
        spaCy pipeline to load for NER.
    use_regex:
        Enable deterministic regex detectors.
    use_llm:
        Enable Ollama LLM confirmation for final redactions.
    llm_model:
        Ollama model identifier to call.
    llm_url:
        Base URL to Ollama ``/api/generate``.
    categories:
        Optional category allowlist for LLM prompts.
    box_inflation_px:
        Inflate redaction rectangles by N pixels for safety.
    fill_rgb:
        Fill color for redaction rectangles.
    prompt_path:
        Optional path to a JSONL/system prompt file for the LLM.
    """

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
    fill_rgb: tuple = (0, 0, 0)
    prompt_path: Optional[str] = None
    policy_path: Optional[str] = None
    mode: str = "redact"  # 'redact' or 'label' (pseudonymize overlay)
    safe_pdf_mode: bool = True
    track_reasons: bool = True
    # OCR enhancements
    preprocess: bool = True
    deskew: bool = True
    binarize: bool = True
    auto_psm: bool = True
    tess_configs: Optional[Dict[str, Any]] = None
    # Performance & previews
    workers: int = 1
    instrument: bool = True
    generate_previews: bool = True
    explain_traces: bool = False
    export_ocr_debug: bool = False


class PageResult(BaseModel):
    """Per-page output, including raw text, candidates, LLM decision, and stats.

    Attributes
    ----------
    page_index:
        Zero-based page index.
    text:
        Reconstructed text stream from OCR tokens.
    candidates:
        Candidate spans before LLM confirmation.
    llm_json:
        Raw parsed LLM response if used.
    boxes_applied:
        Number of redaction boxes applied on this page.
    """

    page_index: int
    text: str
    candidates: List[Dict[str, Any]]
    llm_json: Optional[Dict[str, Any]] = None
    boxes_applied: int = 0
    boxes: Optional[List[Dict[str, Any]]] = None
    timings: Optional[Dict[str, float]] = None
    preview_path: Optional[str] = None


def _page_text_from_tsv(tsv_df):
    """Reconstruct a naive text stream from OCR word rows.

    Parameters
    ----------
    tsv_df:
        Pandas DataFrame as returned by ``image_ocr_tsv``.

    Returns
    -------
    str
        Space-joined text.
    """
    tokens = [str(x) for x in tsv_df["text"].tolist()]
    return " ".join(tokens)


def detect_candidates(text: str, cfg: RunConfig) -> List[Dict[str, Any]]:
    """Run configured detectors and return unique span candidates.

    Parameters
    ----------
    text:
        Input text to analyze.
    cfg:
        Active runtime configuration.

    Returns
    -------
    list[dict]
        Unique spans with ``label``, ``start``, ``end``, ``text``.
    """
    cands: List[Dict[str, Any]] = []
    if cfg.use_spacy:
        for e in spacy_ents(cfg.spacy_model, text):
            cands.append(
                {
                    "label": e["label"],
                    "start": e["start"],
                    "end": e["end"],
                    "text": e["text"],
                }
            )
    if cfg.use_llm_ner:
        for e in llm_ner_candidates(text, cfg):
            cands.append(e)
    if cfg.use_regex:
        for e in regex_findall(text):
            cands.append(e)
    # dedupe by (start,end,label)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in cands:
        key = (c["start"], c["end"], c.get("label", ""))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _normalize_llm_items(
    candidates: List[Dict[str, Any]], raw_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Ensure every candidate receives an explicit LLM decision."""

    cand_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {
        (int(c["start"]), int(c["end"])): c for c in candidates
    }
    normalized: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int]] = set()

    for raw in raw_items:
        try:
            start = int(raw.get("start"))  # pyright: ignore[reportArgumentType]
            end = int(raw.get("end"))  # pyright: ignore[reportArgumentType]
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
        score = raw.get("score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
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
                "score": None,
                "source": cand.get("source"),
            }
        )

    normalized.sort(key=lambda item: (item["start"], item["end"], item["category"]))
    return normalized


def llm_ner_candidates(text: str, cfg: RunConfig) -> List[Dict[str, Any]]:
    """Extract NER candidates using an LLM with few-shot prompt.

    Returns list of span dicts: {label, start, end, text, source="LLM_NER"}.
    """

    prompt = _load_prompt(
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
    client = _get_ollama_client(cfg.llm_url)

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
                # Fallback: extract individual JSON objects
                objs: List[Dict[str, Any]] = []
                depth = 0
                start_idx: Optional[int] = None
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
        # logger.info(f"LLM RESP: {resp}")
        try:
            js = json.loads(resp)
        except Exception:
            js = resp
        # logger.info(f"LLM PARSED: {js}")
        items = _normalize_items(js)
        # logger.info(f"LLM items parsed: {items}")
        chunk_out: List[Dict[str, Any]] = []
        for it in items:
            if isinstance(it, str):
                try:
                    it = json.loads(it)
                except Exception:
                    continue
            if not isinstance(it, dict):
                continue
            text_val = it.get("text")
            s = it.get("start")
            e = it.get("end")
            label = (it.get("label") or "OTHER").upper()
            if text_val is None:
                continue
            if (
                not isinstance(s, int)
                or not isinstance(e, int)
                or not (0 <= s < e <= len(segment))
            ):
                idx = segment.find(text_val)
                if idx < 0:
                    continue
                s, e = idx, idx + len(text_val)
            chunk_out.append(
                {
                    "label": label,
                    "start": offset + int(s),
                    "end": offset + int(e),
                    "text": text[offset + int(s) : offset + int(e)],
                    "source": "LLM_NER",
                    "prompt": full,
                    "response": resp,
                }
            )
            # logger.info(f"LLM NER chunk item: {chunk_out[-1]}")
        return chunk_out

    all_results: List[Dict[str, Any]] = []
    for idx, (offset, segment) in enumerate(raw_chunks, start=1):
        all_results.extend(_process_chunk((idx, chunk_total, offset, segment)))

    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in all_results:
        key = (c["start"], c["end"], c.get("label", ""))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    # logger.info(f"LLM uniq: {uniq}")
    return uniq


def llm_filter(
    text: str, cands: List[Dict[str, Any]], cfg: RunConfig
) -> Dict[str, Any]:
    """Ask a local LLM (Ollama) to confirm redactions and assign categories.

    Parameters
    ----------
    text:
        Page text to provide for context.
    cands:
        Candidate spans to be confirmed or rejected.
    cfg:
        Active runtime configuration (model, URL, prompt path, etc.).

    Returns
    -------
    dict
        Parsed LLM JSON-like output with keys ``items`` and ``notes``. Each item
        is expected to have ``{text,start,end,redact,category}``.
    """
    if not cands:
        return {"items": [], "notes": "no candidates"}
    # compact representation for the LLM
    items = [
        {
            "text": c["text"],
            "start": c["start"],
            "end": c["end"],
            "label": c.get("label", "OTHER"),
            "source": c.get("source"),
        }
        for c in cands
    ]

    prompt = _load_prompt(
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
    client = _get_ollama_client(cfg.llm_url)
    raw = client.generate_raw(full, model=cfg.llm_model, timeout=cfg.llm_timeout)
    resp = raw.get("response", "")
    # Best-effort JSON parse:
    try:
        js = json.loads(resp)
    except Exception:
        # try to extract JSON block
        import re

        m = re.search(r"\{[\s\S]*\}", resp)
        if m:
            js = json.loads(m.group(0))
        else:
            js = {
                "items": [
                    {
                        "text": i["text"],
                        "start": i["start"],
                        "end": i["end"],
                        "redact": True,
                        "category": i.get("label", "OTHER"),
                        "why": "fallback redaction (parser)",
                        "score": None,
                        "source": i.get("source"),
                    }
                    for i in items
                ],
                "notes": "fallback all redact",
            }

    js["items"] = _normalize_llm_items(cands, js.get("items", []))
    # Attach explainability metadata and excerpts
    meta = {
        "model": raw.get("model"),
        "created_at": raw.get("created_at"),
        "eval_count": raw.get("eval_count"),
        "eval_duration": raw.get("eval_duration"),
        "prompt_eval_count": raw.get("prompt_eval_count"),
        "prompt_eval_duration": raw.get("prompt_eval_duration"),
        "total_duration": raw.get("total_duration"),
    }
    js["meta"] = meta
    js["prompt"] = full
    js["response"] = resp
    if cfg.explain_traces:
        js["trace"] = {"prompt": full, "raw": raw}
    return js


def _resolve_policy(cfg: RunConfig) -> Optional[Policy]:
    if cfg.policy_path:
        p = Path(cfg.policy_path)
        if p.exists():
            from .policy import Policy as _P

            return _P.from_file(p)
        # try builtin name
        found = find_builtin_policy(cfg.policy_path)
        if found:
            from .policy import Policy as _P

            return _P.from_file(found)
    return None


def _process_images(
    images: List[Image.Image], output_path: str, cfg: RunConfig
) -> Dict[str, Any]:
    """Run OCR → detect → (optional LLM) → redact on images and export a PDF.

    Parameters
    ----------
    images:
        List of page images in order.
    output_path:
        Destination PDF path.
    cfg:
        Runtime configuration.

    Returns
    -------
    dict
        Summary with keys ``pages`` (list of :class:`PageResult` dicts) and
        ``out`` (PDF path).
    """

    out_base = Path(output_path)
    preview_dir = out_base.parent / f"{out_base.stem}.previews"
    trace_dir = out_base.parent / f"{out_base.stem}.traces"
    if cfg.generate_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
    if cfg.explain_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)
    ocr_debug_dir = None
    cfg.export_ocr_debug = True
    if cfg.export_ocr_debug:
        ocr_debug_dir = out_base.parent / f"{out_base.stem}.ocr"
        ocr_debug_dir.mkdir(parents=True, exist_ok=True)
        # #logger.info(f"Exporting OCR debug to {ocr_debug_dir}")
    results: List[Dict[str, Any]] = []
    redacted_imgs: List[Image.Image] = []
    ocr_debug_paths: Dict[int, Dict[str, str]] = {}
    all_texts: Dict[int, str] = {}
    policy = _resolve_policy(cfg)

    for idx, img in enumerate(tqdm(images, desc="OCR+Detect+Redact")):
        t0 = time.perf_counter()
        ocr_res = image_ocr_tsv(
            img,
            lang=cfg.lang,
            psm=cfg.psm,
            preprocess=cfg.preprocess,
            deskew=cfg.deskew,
            binarize=cfg.binarize,
            auto_psm=cfg.auto_psm,
            tess_configs=cfg.tess_configs,
        )
        tsv = ocr_res["tsv"]
        ocr_meta = {k: v for k, v in ocr_res.items() if k != "tsv"}
        # logger.info("OCR completed: {} words, {} chars".format(len(tsv), sum(len(str(x)) for x in tsv["text"])))
        # #logger.info(f"debug directory: {ocr_debug_dir}")
        if cfg.export_ocr_debug and ocr_debug_dir is not None:
            raw_path = ocr_debug_dir / f"page_{idx:04d}_words.tsv"
            text_path = ocr_debug_dir / f"page_{idx:04d}_text.txt"
            try:
                tsv.to_csv(raw_path, sep="\t", index=False)
                text_content = ocr_meta.get("text") or _page_text_from_tsv(tsv)
                text_path.write_text(text_content, encoding="utf-8")
                ocr_debug_paths[idx] = {
                    "tsv": str(raw_path),
                    "text": str(text_path),
                }
                all_texts[idx] = text_content
                # #logger.info(f"OCR debug for page {idx} at {raw_path} and {text_path}")
            except Exception as e:
                # logger.info(f"Failed to export OCR debug files error {e}", exc_info=True)
                pass
        t_ocr = time.perf_counter()
        page_text = ocr_meta.get("text") or _page_text_from_tsv(tsv)
        t_det0 = time.perf_counter()
        candidates = detect_candidates(page_text, cfg)
        t_det = time.perf_counter()
        final_spans: List[Dict[str, Any]] = []
        llm_json: Optional[Dict[str, Any]] = None
        t_llm0 = None
        t_llm = None
        if cfg.use_llm:
            t_llm0 = time.perf_counter()
            llm_json = llm_filter(page_text, candidates, cfg)
            t_llm = time.perf_counter()
            for it in llm_json.get("items", []):
                cat = it.get("category") or it.get("label") or "OTHER"
                decision = bool(it.get("redact", True))
                if policy is not None:
                    decision = decision and policy.should_redact(cat)
                if decision:
                    final_spans.append(
                        {
                            "start": it["start"],
                            "end": it["end"],
                            "text": it.get("text", page_text[it["start"] : it["end"]]),
                            "label": cat,
                            "source": "LLM",
                        }
                    )
        else:
            # redact everything detected deterministically
            for c in candidates:
                cat = c.get("label", "OTHER")
                decision = True if policy is None else policy.should_redact(cat)
                if decision:
                    final_spans.append(
                        {
                            "start": c["start"],
                            "end": c["end"],
                            "text": c.get("text", page_text[c["start"] : c["end"]]),
                            "label": cat,
                            "source": c.get("source", "DETECTOR"),
                        }
                    )

        # kept vs rejected boxes
        t_align0 = time.perf_counter()
        box_infos = spans_to_box_info(tsv, page_text, final_spans)
        boxes = [b["box"] for b in box_infos]
        kept_keys = {(s["start"], s["end"]) for s in final_spans}
        rejected_spans = [
            c for c in candidates if (c["start"], c["end"]) not in kept_keys
        ]
        rejected_infos = spans_to_box_info(tsv, page_text, rejected_spans)
        rejected_boxes = [b["box"] for b in rejected_infos]
        t_align = time.perf_counter()

        if cfg.mode == "label":
            red_img = redact_page_with_labels(
                img, box_infos, fill_rgb=cfg.fill_rgb, inflate_px=cfg.box_inflation_px
            )
        else:
            red_img = redact_page(
                img, boxes, fill_rgb=cfg.fill_rgb, inflate_px=cfg.box_inflation_px
            )
        redacted_img = red_img
        t_redact = time.perf_counter()

        # Save preview overlay if requested
        preview_path = None
        if cfg.generate_previews:
            prev = draw_preview(img, boxes, rejected_boxes)
            preview_path = str(preview_dir / f"page_{idx:04d}.png")
            prev.save(preview_path)

        # Save explain trace if requested
        if cfg.explain_traces and llm_json and llm_json.get("trace"):
            trace = llm_json.get("trace")
            (trace_dir / f"page_{idx:04d}.llm.json").write_text(
                json.dumps(trace), encoding="utf-8"
            )

        timings = None
        if cfg.instrument:
            timings = {
                "ocr": t_ocr - t0,
                "detect": t_det - t_det0,
                **(
                    {"llm": (t_llm - t_llm0)}
                    if (t_llm0 is not None and t_llm is not None)
                    else {}
                ),
                "align": t_align - t_align0,
                "redact": t_redact - t_align,
                "total": t_redact - t0,
            }

        page_result = PageResult(
            page_index=idx,
            text=page_text,
            candidates=candidates,
            llm_json=llm_json,
            boxes_applied=len(boxes),
            boxes=box_infos if cfg.track_reasons else None,
            timings=timings,
            preview_path=preview_path,
        ).dict()
        results.append(page_result)
        redacted_imgs.append(redacted_img)

    for res in results:
        page_idx = res.get("page_index")
        if page_idx is not None and page_idx in ocr_debug_paths:
            res.setdefault("debug", {})
            res["debug"]["ocr"] = ocr_debug_paths[page_idx]

    debug_payload: Dict[str, Any] = {}
    if cfg.export_ocr_debug and ocr_debug_dir is not None:
        try:
            aggregated_path = None
            if all_texts:
                aggregated_path = ocr_debug_dir / "all_text.txt"
                ordered = [all_texts[idx] for idx in sorted(all_texts.keys())]
                aggregated_path.write_text("\n\n".join(ordered), encoding="utf-8")
            zip_path = Path(
                shutil.make_archive(str(ocr_debug_dir), "zip", root_dir=ocr_debug_dir)
            )
            artifacts_dir = Path.cwd() / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            zip_dest = artifacts_dir / f"{out_base.stem}.ocr.zip"
            shutil.copy2(zip_path, zip_dest)
            debug_payload["ocr_tsv_zip"] = str(zip_dest)
            if aggregated_path and aggregated_path.exists():
                text_dest = artifacts_dir / f"{out_base.stem}.all_text.txt"
                shutil.copy2(aggregated_path, text_dest)
                debug_payload["ocr_text"] = str(text_dest)
                # #logger.info(f"All page text at {text_dest}")
            # #logger.info(f"OCR debug ZIP at {zip_dest}")
        except Exception as e:
            # logger.info(f"Failed to export OCR debug ZIP with exception {e}", exc_info=True)
            pass
    save_pdf([img for img in redacted_imgs if img is not None], output_path)
    return {"pages": results, "out": output_path, "debug": debug_payload}


def process_pdf(input_path: str, output_path: str, cfg: RunConfig) -> Dict[str, Any]:
    """Backwards-compatible handler: process a PDF path to a redacted PDF.

    Parameters
    ----------
    input_path:
        Input PDF path.
    output_path:
        Output redacted PDF path.
    cfg:
        Runtime configuration.

    Returns
    -------
    dict
        Pipeline results as returned by :func:`_process_images`.
    """
    pages: List[Image.Image] = pdf_to_images(input_path, dpi=cfg.dpi)
    return _process_images(pages, output_path, cfg)


def process_path(input_path: str, output_path: str, cfg: RunConfig) -> Dict[str, Any]:
    """Process a path that can be a PDF, image, or directory of images.

    Behavior
    --------
    - PDF: rasterize all pages and redact into a PDF.
    - Image (png/jpeg/tiff): single-image pipeline, writes a single-page PDF.
    - Directory: all supported images sorted by name into a multi-page PDF.

    Parameters
    ----------
    input_path:
        File or directory to process.
    output_path:
        Destination PDF path.
    cfg:
        Runtime configuration.

    Returns
    -------
    dict
        Pipeline results as returned by :func:`_process_images`.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if p.is_dir():
        imgs: List[Image.Image] = []
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        for fp in sorted(p.iterdir()):
            if fp.suffix.lower() in exts:
                imgs.append(Image.open(fp))
        if not imgs:
            raise ValueError(f"No supported images in directory: {input_path}")
        res = _process_images(imgs, output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(
                str(p),
                output_path,
                res,
                cfg.__dict__,
                policy=pol.to_dict() if pol else None,
            )
        except Exception:
            pass
        return res
    # file: check type
    ext = p.suffix.lower()
    if ext == ".pdf":
        res = process_pdf(str(p), output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(
                str(p),
                output_path,
                res,
                cfg.__dict__,
                policy=pol.to_dict() if pol else None,
            )
        except Exception:
            pass
        return res
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        img = Image.open(p)
        res = _process_images([img], output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(
                str(p),
                output_path,
                res,
                cfg.__dict__,
                policy=pol.to_dict() if pol else None,
            )
        except Exception:
            pass
        return res
    # heuristic via mimetype as fallback
    mt = mimetypes.guess_type(str(p))[0] or ""
    if "pdf" in mt:
        res = process_pdf(str(p), output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(
                str(p),
                output_path,
                res,
                cfg.__dict__,
                policy=pol.to_dict() if pol else None,
            )
        except Exception:
            pass
        return res
    if mt.startswith("image/"):
        img = Image.open(p)
        res = _process_images([img], output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(
                str(p),
                output_path,
                res,
                cfg.__dict__,
                policy=pol.to_dict() if pol else None,
            )
        except Exception:
            pass
        return res
    raise ValueError(f"Unsupported input type: {input_path}")
