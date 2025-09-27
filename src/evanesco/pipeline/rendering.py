"""Rendering helpers for applying redactions and building page payloads."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

from PIL import Image

from evanesco.align import spans_to_box_info
from evanesco.redact import redact_page, redact_page_with_labels, draw_preview

from .config import RunConfig, PageResult


@dataclass
class RenderResult:
    page: Dict[str, Any]
    redacted_image: Image.Image
    align_duration: float
    redact_duration: float


def render_page(
    img: Image.Image,
    tsv_df,
    page_text: str,
    final_spans: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    cfg: RunConfig,
    page_index: int,
    preview_dir: Optional[Path],
    trace_dir: Optional[Path],
    llm_json: Optional[Dict[str, Any]],
) -> RenderResult:
    """Produce redacted imagery and per-page metadata."""

    align_start = time.perf_counter()
    box_infos = spans_to_box_info(tsv_df, page_text, final_spans)
    boxes = [info["box"] for info in box_infos]
    kept_keys = {(span["start"], span["end"]) for span in final_spans}
    rejected_spans = [
        cand for cand in candidates if (cand["start"], cand["end"]) not in kept_keys
    ]
    rejected_infos = spans_to_box_info(tsv_df, page_text, rejected_spans)
    rejected_boxes = [info["box"] for info in rejected_infos]
    align_duration = time.perf_counter() - align_start

    redact_start = time.perf_counter()
    if cfg.mode == "label":
        redacted = redact_page_with_labels(
            img, box_infos, fill_rgb=cfg.fill_rgb, inflate_px=cfg.box_inflation_px
        )
    else:
        redacted = redact_page(
            img, boxes, fill_rgb=cfg.fill_rgb, inflate_px=cfg.box_inflation_px
        )
    redact_duration = time.perf_counter() - redact_start

    preview_path: Optional[str] = None
    if cfg.generate_previews and preview_dir is not None:
        preview = draw_preview(img, boxes, rejected_boxes)
        preview_file = preview_dir / f"page_{page_index:04d}.png"
        preview.save(preview_file)
        preview_path = str(preview_file)

    if (
        cfg.explain_traces
        and trace_dir is not None
        and llm_json
        and llm_json.get("trace")
    ):
        trace_file = trace_dir / f"page_{page_index:04d}.llm.json"
        trace_file.write_text(json.dumps(llm_json.get("trace")), encoding="utf-8")

    page_payload = PageResult(
        page_index=page_index,
        text=page_text,
        candidates=candidates,
        llm_json=llm_json,
        boxes_applied=len(boxes),
        boxes=box_infos if cfg.track_reasons else None,
        timings=None,
        preview_path=preview_path,
    ).dict()

    return RenderResult(
        page=page_payload,
        redacted_image=redacted,
        align_duration=align_duration,
        redact_duration=redact_duration,
    )


__all__ = ["RenderResult", "render_page"]
