"""High-level orchestration for Evanesco redaction runs."""

from __future__ import annotations

import json
import mimetypes
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image
from tqdm import tqdm

from evanesco.audit import write_audit
from evanesco.logging import get_logger
from evanesco.ocr import image_ocr_tsv, pdf_to_images
from evanesco.policy import Policy, find_builtin_policy
from evanesco.redact import save_pdf

from .config import RunConfig
from .detection import detect_candidates, resolve_final_spans
from .rendering import RenderResult, render_page

logger = get_logger("evanesco")


def _page_text_from_tsv(tsv_df) -> str:
    tokens = [str(value) for value in tsv_df["text"].tolist()]
    return " ".join(tokens)


def _resolve_policy(cfg: RunConfig) -> Optional[Policy]:
    if cfg.policy_path:
        path = Path(cfg.policy_path)
        if path.exists():
            from evanesco.policy import Policy as _Policy

            return _Policy.from_file(path)
        found = find_builtin_policy(cfg.policy_path)
        if found:
            from evanesco.policy import Policy as _Policy

            return _Policy.from_file(found)
    return None


def _prepare_dirs(
    base: Path, cfg: RunConfig
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    preview_dir: Optional[Path] = None
    trace_dir: Optional[Path] = None
    ocr_debug_dir: Optional[Path] = None

    if cfg.generate_previews:
        preview_dir = base.parent / f"{base.stem}.previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
    if cfg.explain_traces:
        trace_dir = base.parent / f"{base.stem}.traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
    if cfg.export_ocr_debug:
        ocr_debug_dir = base.parent / f"{base.stem}.ocr"
        ocr_debug_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir, trace_dir, ocr_debug_dir


def _write_ocr_debug(
    tsv_df,
    page_text: str,
    idx: int,
    ocr_dir: Path,
    store: Dict[int, Dict[str, str]],
    texts: Dict[int, str],
) -> None:
    raw_path = ocr_dir / f"page_{idx:04d}_words.tsv"
    text_path = ocr_dir / f"page_{idx:04d}_text.txt"
    try:
        tsv_df.to_csv(raw_path, sep="\t", index=False)
        text_path.write_text(page_text, encoding="utf-8")
        store[idx] = {"tsv": str(raw_path), "text": str(text_path)}
        texts[idx] = page_text
    except Exception as exc:  # pragma: no cover - logging only
        logger.debug("Failed to export OCR debug files", exc_info=exc)


def _process_images(
    images: List[Image.Image], output_path: str, cfg: RunConfig
) -> Dict[str, Any]:
    out_path = Path(output_path)
    # Maintain legacy behavior: exporting OCR debug by default inside pipeline
    if not getattr(cfg, "export_ocr_debug", False):
        cfg.export_ocr_debug = True
    preview_dir, trace_dir, ocr_debug_dir = _prepare_dirs(out_path, cfg)

    results: List[Dict[str, Any]] = []
    redacted_images: List[Image.Image] = []
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
        page_text = ocr_meta.get("text") or _page_text_from_tsv(tsv)

        if cfg.export_ocr_debug and ocr_debug_dir is not None:
            _write_ocr_debug(
                tsv, page_text, idx, ocr_debug_dir, ocr_debug_paths, all_texts
            )

        t_ocr = time.perf_counter()
        detect_start = time.perf_counter()
        candidates = detect_candidates(page_text, cfg)
        detect_end = time.perf_counter()

        final_spans, llm_json, llm_duration = resolve_final_spans(
            page_text, candidates, cfg, policy
        )

        render_result: RenderResult = render_page(
            img,
            tsv,
            page_text,
            final_spans,
            candidates,
            cfg,
            idx,
            preview_dir,
            trace_dir,
            llm_json,
        )

        total_end = time.perf_counter()

        timings: Optional[Dict[str, float]] = None
        if cfg.instrument:
            timings = {
                "ocr": t_ocr - t0,
                "detect": detect_end - detect_start,
                "align": render_result.align_duration,
                "redact": render_result.redact_duration,
                "total": total_end - t0,
            }
            if llm_duration is not None:
                timings["llm"] = llm_duration
            render_result.page["timings"] = timings

        results.append(render_result.page)
        redacted_images.append(render_result.redacted_image)

    for page in results:
        page_idx = page.get("page_index")
        if page_idx is not None and page_idx in ocr_debug_paths:
            page.setdefault("debug", {})
            page["debug"]["ocr"] = ocr_debug_paths[page_idx]

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
            zip_dest = artifacts_dir / f"{out_path.stem}.ocr.zip"
            shutil.copy2(zip_path, zip_dest)
            debug_payload["ocr_tsv_zip"] = str(zip_dest)
            if aggregated_path and aggregated_path.exists():
                text_dest = artifacts_dir / f"{out_path.stem}.all_text.txt"
                shutil.copy2(aggregated_path, text_dest)
                debug_payload["ocr_text"] = str(text_dest)
        except Exception as exc:  # pragma: no cover - logging only
            logger.debug("Failed to export OCR debug ZIP", exc_info=exc)

    save_pdf([img for img in redacted_images if img is not None], output_path)
    return {"pages": results, "out": output_path, "debug": debug_payload}


def process_pdf(input_path: str, output_path: str, cfg: RunConfig) -> Dict[str, Any]:
    pages: List[Image.Image] = pdf_to_images(input_path, dpi=cfg.dpi)
    return _process_images(pages, output_path, cfg)


def process_path(input_path: str, output_path: str, cfg: RunConfig) -> Dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    def _write_audit(input_ref: str, result: Dict[str, Any]) -> None:
        try:
            pol = _resolve_policy(cfg)
            write_audit(
                input_ref,
                output_path,
                result,
                cfg.__dict__,
                policy=pol.to_dict() if pol else None,
            )
        except Exception:  # pragma: no cover - best effort
            pass

    if path.is_dir():
        images: List[Image.Image] = []
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        for fp in sorted(path.iterdir()):
            if fp.suffix.lower() in exts:
                images.append(Image.open(fp))
        if not images:
            raise ValueError(f"No supported images in directory: {input_path}")
        result = _process_images(images, output_path, cfg)
        _write_audit(str(path), result)
        return result

    ext = path.suffix.lower()
    if ext == ".pdf":
        result = process_pdf(str(path), output_path, cfg)
        _write_audit(str(path), result)
        return result
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        result = _process_images([Image.open(path)], output_path, cfg)
        _write_audit(str(path), result)
        return result

    mimetype = mimetypes.guess_type(str(path))[0] or ""
    if "pdf" in mimetype:
        result = process_pdf(str(path), output_path, cfg)
        _write_audit(str(path), result)
        return result
    if mimetype.startswith("image/"):
        result = _process_images([Image.open(path)], output_path, cfg)
        _write_audit(str(path), result)
        return result

    raise ValueError(f"Unsupported input type: {input_path}")


__all__ = ["process_pdf", "process_path"]
