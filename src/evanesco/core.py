"""Core pipeline orchestration for OCR, detection, LLM filtering, and redaction.

This module wires together OCR, detection (spaCy and regex), optional LLM
confirmation via Ollama, geometric alignment and final redaction.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import orjson, json
from tqdm import tqdm
from .ocr import pdf_to_images, image_ocr_tsv
from .spacy_detect import spacy_ents
from .regex_detect import regex_findall
from .llm import OllamaClient
from .redact import redact_page, save_pdf, redact_page_with_labels, draw_preview
from .align import spans_to_boxes, spans_to_box_info
from .policy import Policy, find_builtin_policy
from .audit import write_audit
import mimetypes


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
    llm_ner_model: Optional[str] = None
    llm_ner_prompt_path: Optional[str] = None
    use_regex: bool = True
    use_llm: bool = True
    llm_model: str = "gpt-oss:20b"
    llm_url: str = "http://localhost:11434/api/generate"
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
            cands.append({"label": e["label"], "start": e["start"], "end": e["end"], "text": e["text"]})
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


def llm_ner_candidates(text: str, cfg: RunConfig) -> List[Dict[str, Any]]:
    """Extract NER candidates using an LLM with few-shot prompt.

    Returns list of span dicts: {label, start, end, text, source="LLM_NER"}.
    """
    def _read_prompt() -> str:
        # 1) explicit path if provided
        if cfg.llm_ner_prompt_path:
            pp = Path(cfg.llm_ner_prompt_path)
            if pp.exists():
                return pp.read_text(encoding="utf-8")
        # 2) search upwards for prompts/ner_fewshot.jsonl
        for parent in Path(__file__).resolve().parents:
            cand = parent / "prompts" / "ner_fewshot.jsonl"
            if cand.exists():
                return cand.read_text(encoding="utf-8")
        # 3) fallback minimal system prompt
        return (
            "SYSTEM: You are a precise NER extractor.\n"
            "Return STRICT JSON: {items:[{text,start,end,label}]}. Use character offsets into the provided text.\n"
            "Labels: PERSON, ORG, GPE, LOC, DATE, EMAIL, PHONE, IBAN, CREDIT_CARD, OTHER."
        )

    prompt = _read_prompt()
    user = {
        "text": text,
        "instructions": (
            "Extract entities with character offsets (start,end). Labels among: "
            "PERSON, ORG, GPE, LOC, DATE, EMAIL, PHONE, IBAN, CREDIT_CARD, OTHER."
        ),
    }
    full = f"{prompt}\n\nUSER:\n{json.dumps(user)}"
    model = cfg.llm_ner_model or cfg.llm_model
    client = OllamaClient(url=cfg.llm_url)
    resp = client.generate(full, model=model)
    # Parse JSON
    js = {}
    try:
        js = json.loads(resp)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", resp)
        if m:
            js = json.loads(m.group(0))
        else:
            js = {"items": []}
    items = js.get("items", [])
    out: List[Dict[str, Any]] = []
    for it in items:
        t = it.get("text")
        s = it.get("start")
        e = it.get("end")
        label = (it.get("label") or "OTHER").upper()
        if t is None:
            continue
        if not isinstance(s, int) or not isinstance(e, int) or not (0 <= s < e <= len(text)):
            # fallback: locate substring
            idx = text.find(t)
            if idx >= 0:
                s, e = idx, idx + len(t)
            else:
                continue
        out.append({
            "label": label,
            "start": int(s),
            "end": int(e),
            "text": text[int(s): int(e)],
            "source": "LLM_NER",
        })
    # Deduplicate within LLM results
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in out:
        key = (c["start"], c["end"], c.get("label", ""))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def llm_filter(text: str, cands: List[Dict[str, Any]], cfg: RunConfig) -> Dict[str, Any]:
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
        {"text": c["text"], "start": c["start"], "end": c["end"], "label": c.get("label", "OTHER")}
        for c in cands
    ]
    def _read_prompt() -> str:
        # 1) explicit path if provided
        if cfg.prompt_path:
            pp = Path(cfg.prompt_path)
            if pp.exists():
                return pp.read_text(encoding="utf-8")
        # 2) search upwards for prompts/pii_audit.jsonl
        for parent in Path(__file__).resolve().parents:
            cand = parent / "prompts" / "pii_audit.jsonl"
            if cand.exists():
                return cand.read_text(encoding="utf-8")
        # 3) fallback minimal system prompt
        return (
            "SYSTEM: You are a strict JSON generator for PII redaction.\n"
            "Return only JSON with keys: items (list of {text,start,end,redact,category}), notes.\n"
            "Categories may include PERSON, ORG, GPE, LOC, DATE, EMAIL, PHONE, IBAN, CREDIT_CARD, OTHER.\n"
        )
    prompt = _read_prompt()
    user = {
        "page_text": text,
        "candidates": items,
        "instructions": (
            "Return STRICT JSON with keys: items (list of {text,start,end,redact,category,why,score}), notes. "
            "Include a short 'why' rationale per item and an optional numeric 'score' 0-1."
        ),
    }
    full = f"{prompt}\n\nUSER:\n{json.dumps(user)}"
    client = OllamaClient(url=cfg.llm_url)
    raw = client.generate_raw(full, model=cfg.llm_model)
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
                    }
                    for i in items
                ],
                "notes": "fallback all redact",
            }
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
    js["prompt_excerpt"] = full[:800]
    js["response_excerpt"] = resp[:800]
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


def _process_images(images: List[Image.Image], output_path: str, cfg: RunConfig) -> Dict[str, Any]:
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
    import time
    out_base = Path(output_path)
    preview_dir = out_base.parent / f"{out_base.stem}.previews"
    trace_dir = out_base.parent / f"{out_base.stem}.traces"
    if cfg.generate_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
    if cfg.explain_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    redacted_imgs: List[Image.Image] = []
    policy = _resolve_policy(cfg)
    for idx, img in enumerate(tqdm(images, desc="OCR+Detect+Redact")):
        t0 = time.perf_counter()
        tsv = image_ocr_tsv(
            img,
            lang=cfg.lang,
            psm=cfg.psm,
            preprocess=cfg.preprocess,
            deskew=cfg.deskew,
            binarize=cfg.binarize,
            auto_psm=cfg.auto_psm,
            tess_configs=cfg.tess_configs,
        )["tsv"]
        t_ocr = time.perf_counter()
        page_text = _page_text_from_tsv(tsv)
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
                    final_spans.append({
                        "start": it["start"],
                        "end": it["end"],
                        "text": it.get("text", page_text[it["start"]:it["end"]]),
                        "label": cat,
                        "source": "LLM",
                    })
        else:
            # redact everything detected deterministically
            for c in candidates:
                cat = c.get("label", "OTHER")
                decision = True if policy is None else policy.should_redact(cat)
                if decision:
                    final_spans.append({
                        "start": c["start"],
                        "end": c["end"],
                        "text": c.get("text", page_text[c["start"]:c["end"]]),
                        "label": cat,
                        "source": c.get("source", "DETECTOR"),
                    })

        # kept vs rejected boxes
        t_align0 = time.perf_counter()
        box_infos = spans_to_box_info(tsv, page_text, final_spans)
        boxes = [b["box"] for b in box_infos]
        kept_keys = {(s["start"], s["end"]) for s in final_spans}
        rejected_spans = [c for c in candidates if (c["start"], c["end"]) not in kept_keys]
        rejected_infos = spans_to_box_info(tsv, page_text, rejected_spans)
        rejected_boxes = [b["box"] for b in rejected_infos]
        t_align = time.perf_counter()

        if cfg.mode == "label":
            red_img = redact_page_with_labels(img, box_infos, fill_rgb=cfg.fill_rgb, inflate_px=cfg.box_inflation_px)
        else:
            red_img = redact_page(img, boxes, fill_rgb=cfg.fill_rgb, inflate_px=cfg.box_inflation_px)
        redacted_imgs.append(red_img)
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
            (trace_dir / f"page_{idx:04d}.llm.json").write_text(json.dumps(trace), encoding="utf-8")

        timings = None
        if cfg.instrument:
            timings = {
                "ocr": t_ocr - t0,
                "detect": t_det - t_det0,
                **({"llm": (t_llm - t_llm0)} if (t_llm0 is not None and t_llm is not None) else {}),
                "align": t_align - t_align0,
                "redact": t_redact - t_align,
                "total": t_redact - t0,
            }

        results.append(
            PageResult(
                page_index=idx,
                text=page_text,
                candidates=candidates,
                llm_json=llm_json,
                boxes_applied=len(boxes),
                boxes=box_infos if cfg.track_reasons else None,
                timings=timings,
                preview_path=preview_path,
            ).dict()
        )
    save_pdf(redacted_imgs, output_path)
    return {"pages": results, "out": output_path}


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
            write_audit(str(p), output_path, res, cfg.__dict__, policy=pol.to_dict() if pol else None)
        except Exception:
            pass
        return res
    # file: check type
    ext = p.suffix.lower()
    if ext == ".pdf":
        res = process_pdf(str(p), output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(str(p), output_path, res, cfg.__dict__, policy=pol.to_dict() if pol else None)
        except Exception:
            pass
        return res
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        img = Image.open(p)
        res = _process_images([img], output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(str(p), output_path, res, cfg.__dict__, policy=pol.to_dict() if pol else None)
        except Exception:
            pass
        return res
    # heuristic via mimetype as fallback
    mt = mimetypes.guess_type(str(p))[0] or ""
    if "pdf" in mt:
        res = process_pdf(str(p), output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(str(p), output_path, res, cfg.__dict__, policy=pol.to_dict() if pol else None)
        except Exception:
            pass
        return res
    if mt.startswith("image/"):
        img = Image.open(p)
        res = _process_images([img], output_path, cfg)
        try:
            pol = _resolve_policy(cfg)
            write_audit(str(p), output_path, res, cfg.__dict__, policy=pol.to_dict() if pol else None)
        except Exception:
            pass
        return res
    raise ValueError(f"Unsupported input type: {input_path}")
