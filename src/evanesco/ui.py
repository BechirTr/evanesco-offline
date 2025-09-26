"""Gradio UI for Evanesco Offline document anonymization.

This module exposes a small web UI to run the OCR + detection + redaction
pipeline against PDF and image inputs. It supports tuning key parameters and
optionally using a local Ollama LLM to confirm redactions.

Run from CLI after install:

- `evanesco-ui` to launch directly
- or `evanesco ui` if using the Typer subcommand (added in `cli.py`)

Notes
-----
- PDF rasterization via `pdf2image` requires Poppler. If Poppler is not
  installed, the pipeline falls back to PyMuPDF (`pymupdf`) when available.
- The UI returns a downloadable redacted PDF and a small JSON summary of the
  run (counts, boxes applied, etc.).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # type: ignore

from .core import RunConfig, process_path


def _run_pipeline(
    input_file: Any | None,
    spacy_model: str,
    use_spacy: bool,
    use_regex: bool,
    use_llm: bool,
    llm_model: str,
    llm_url: str,
    prompt_path: str | None,
    use_llm_ner: bool,
    llm_ner_model: str,
    llm_ner_prompt: str | None,
    policy_dropdown: str | None,
    policy_custom: str | None,
    dpi: int,
    psm: int,
    lang: str,
    inflate: int,
    mode: str,
    generate_review: bool,
    generate_previews: bool,
    save_traces: bool,
    enhance: bool,
    deskew_flag: bool,
    binarize_flag: bool,
    auto_psm_flag: bool,
    export_ocr_debug: bool,
    progress: Any = None,
) -> Tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    """Execute the anonymization pipeline and return output artifacts.

    Parameters
    ----------
    input_file:
        The uploaded file (PDF or image) as provided by Gradio. May be `None`.
    spacy_model:
        spaCy model name to load, e.g. `en_core_web_lg`. If unavailable,
        the pipeline falls back to `en_core_web_sm` or `spacy.blank('en')`.
    use_spacy:
        Enable spaCy-based NER candidate detection.
    use_regex:
        Enable deterministic regex-based candidate detection.
    use_llm:
        Enable Ollama LLM confirmation and categorization.
    llm_model:
        Ollama model name (e.g., `gpt-oss:20b`). Used when `use_llm=True`.
    llm_url:
        Base URL to the Ollama `/api/generate` endpoint.
    prompt_path:
        Optional path to a JSONL/system prompt file for the LLM.
    dpi:
        Rasterization DPI for PDFs (higher = slower, sharper).
    psm:
        Tesseract page segmentation mode.
    lang:
        Tesseract language (e.g., `eng`).
    inflate:
        Pixels to inflate each redaction rectangle for safer coverage.
    progress:
        Gradio progress wrapper which tracks `tqdm`.

    Returns
    -------
    output_pdf:
        Path to the redacted PDF to download, or `None` on failure.
    output_meta:
        Path to the JSON metadata about the run, or `None` on failure.
    summary:
        A short summary dict to render in the UI.
    """
    # Lazy import so that importing the package doesn't require Gradio
    if progress is None:
        try:
            import gradio as gr  # type: ignore

            progress = gr.Progress(track_tqdm=True)
        except Exception:

            class _No:
                def __call__(self, *a, **k):
                    return None

            progress = _No()

    if input_file is None:
        return (
            None,
            None,
            None,
            None,
            {"status": "error", "message": "Please upload a PDF or image."},
        )

    in_path = Path(input_file.name)
    # Write output into a temp dir per invocation
    out_dir = Path(tempfile.mkdtemp(prefix="evanesco_ui_"))
    out_pdf = out_dir / f"{in_path.stem}.redacted.pdf"
    out_meta = out_pdf.with_suffix(".meta.json")
    cand_csv = out_dir / f"{in_path.stem}.candidates.csv"
    boxes_csv = out_dir / f"{in_path.stem}.boxes.csv"

    eff_policy = (
        (policy_custom or "").strip() or (policy_dropdown or "").strip() or None
    )
    spacy_model = (spacy_model or "").strip()
    llm_ner_model = (llm_ner_model or "").strip()
    cfg = RunConfig(
        lang=lang,
        psm=psm,
        dpi=dpi,
        use_spacy=use_spacy,
        spacy_model=spacy_model,
        use_regex=use_regex,
        use_llm=use_llm,
        use_llm_ner=use_llm_ner,
        llm_model=llm_model,
        llm_ner_model=llm_ner_model or None,
        llm_url=llm_url,
        prompt_path=prompt_path or None,
        llm_ner_prompt_path=(llm_ner_prompt or None),
        policy_path=eff_policy,
        box_inflation_px=inflate,
        mode=mode,
        preprocess=enhance,
        deskew=deskew_flag,
        binarize=binarize_flag,
        auto_psm=auto_psm_flag,
        generate_previews=generate_previews,
        explain_traces=save_traces,
        export_ocr_debug=export_ocr_debug,
    )

    try:
        progress(0.05, desc="Starting pipeline")
        res = process_path(str(in_path), str(out_pdf), cfg)
        out_meta.write_bytes(
            __import__("orjson").dumps(res, option=__import__("orjson").OPT_INDENT_2)
        )
        debug_info = res.get("debug", {}) or {}
        pages = res.get("pages", [])
        total_boxes = sum(p.get("boxes_applied", 0) for p in pages)
        summary = {
            "status": "ok",
            "pages": len(pages),
            "total_boxes": total_boxes,
            "output_dir": str(out_dir),
        }
        # Build explainability rows from LLM JSON
        explain_rows = []
        for p in pages:
            idx = p.get("page_index")
            lj = p.get("llm_json") or {}
            for it in lj.get("items", []):
                explain_rows.append(
                    {
                        "page": idx,
                        "text": it.get("text"),
                        "start": it.get("start"),
                        "end": it.get("end"),
                        "category": it.get("category"),
                        "redact": it.get("redact"),
                        "score": it.get("score"),
                        "why": it.get("why"),
                    }
                )
        explain = {"items": explain_rows}
        # Previews and performance summaries
        previews = [p.get("preview_path") for p in pages if p.get("preview_path")]
        perf_pages = []
        agg = {
            "ocr": 0.0,
            "detect": 0.0,
            "llm": 0.0,
            "align": 0.0,
            "redact": 0.0,
            "total": 0.0,
        }
        n = 0
        for p in pages:
            tm = p.get("timings") or {}
            if tm:
                row = {"page": p.get("page_index"), **tm}
                perf_pages.append(row)
                for k in list(agg.keys()):
                    if k in tm and tm[k] is not None:
                        agg[k] += float(tm[k])
                n += 1
        perf = {
            "per_page": perf_pages,
            "summary": {k: (v / max(1, n)) for k, v in agg.items()},
            "pages": n,
        }
        # Optionally materialize review CSVs
        out_cand, out_boxes = None, None
        if generate_review:
            try:
                import pandas as pd  # type: ignore

                rows = []
                for p in pages:
                    for c in p.get("candidates", []):
                        rows.append(
                            {
                                "page_index": p.get("page_index"),
                                "start": c.get("start"),
                                "end": c.get("end"),
                                "text": c.get("text"),
                                "label": c.get("label"),
                                "source": c.get("source"),
                            }
                        )
                if rows:
                    pd.DataFrame(rows).to_csv(cand_csv, index=False)
                    out_cand = str(cand_csv)
                # Boxes info
                brow = []
                for p in pages:
                    for b in p.get("boxes") or []:
                        x, y, w, h = b.get("box", (0, 0, 0, 0))
                        brow.append(
                            {
                                "page_index": p.get("page_index"),
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                                "category": b.get("category"),
                                "source": b.get("source"),
                                "text": b.get("text"),
                            }
                        )
                if brow:
                    pd.DataFrame(brow).to_csv(boxes_csv, index=False)
                    out_boxes = str(boxes_csv)
                # LLM explain CSV
                if explain_rows:
                    pd.DataFrame(explain_rows).to_csv(
                        out_dir / f"{in_path.stem}.llm_items.csv", index=False
                    )
            except Exception:
                pass
        ocr_tsv_path = debug_info.get("ocr_tsv_zip") if export_ocr_debug else None
        ocr_text_path = debug_info.get("ocr_text") if export_ocr_debug else None
        return (
            str(out_pdf),
            str(out_meta),
            out_cand,
            out_boxes,
            ocr_tsv_path,
            ocr_text_path,
            summary,
            explain,
            {"previews": previews},
            perf,
        )
    except Exception as e:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            {"status": "error", "message": str(e)},
            {"items": []},
            {"previews": []},
            {"per_page": [], "summary": {}},
        )


def build_interface():
    """Construct and return the Gradio Blocks interface.

    The UI exposes:
    - File upload (PDF/image)
    - Detection toggles (spaCy, regex, LLM)
    - OCR + rasterization parameters (lang, DPI, PSM)
    - LLM parameters (model, URL, prompt path)
    - Redaction tuning (box inflation)
    """
    import gradio as gr  # local import to avoid hard dependency at import time

    with gr.Blocks(title="Evanesco Offline – PII Redaction") as demo:
        gr.Markdown("""
        # Evanesco Offline – PII Redaction
        Upload a PDF or image to anonymize. Runs fully local OCR and detection.
        Optionally confirm candidates with an Ollama LLM.
        """)

        with gr.Row():
            file_in = gr.File(
                label="PDF or image",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            )

        with gr.Accordion("Detection & Policy", open=True):
            with gr.Row():
                use_spacy = gr.Checkbox(value=True, label="Use spaCy NER")
                spacy_model = gr.Textbox(value="en_core_web_lg", label="spaCy model")
                use_regex = gr.Checkbox(value=True, label="Use regex detectors")
            with gr.Row():
                use_llm = gr.Checkbox(value=True, label="Use Ollama LLM confirmation")
                llm_model = gr.Textbox(value="gpt-oss:20b", label="LLM model")
                llm_url = gr.Textbox(
                    value="http://localhost:11434/api/generate", label="Ollama URL"
                )
                prompt_path = gr.Textbox(value="", label="Prompt path (optional)")
            with gr.Row():
                use_llm_ner = gr.Checkbox(
                    value=False, label="Use LLM NER (few-shot instead of spaCy)"
                )
                llm_ner_model = gr.Textbox(value="gemma3:4b", label="LLM NER model")
                llm_ner_prompt = gr.Textbox(value="", label="LLM NER prompt (optional)")
            with gr.Row():
                policy = gr.Dropdown(
                    choices=["", "default", "gdpr", "hipaa", "pci"],
                    value="",
                    label="Policy (or leave blank)",
                )
                custom_policy = gr.Textbox(
                    value="", label="Custom policy path (overrides dropdown)"
                )

        with gr.Accordion("OCR + Redaction", open=False):
            with gr.Row():
                lang = gr.Textbox(value="eng", label="Tesseract language")
                dpi = gr.Slider(
                    value=400,
                    minimum=72,
                    maximum=600,
                    step=12,
                    label="PDF rasterization DPI",
                )
                psm = gr.Slider(
                    value=3, minimum=0, maximum=13, step=1, label="Tesseract PSM"
                )
                inflate = gr.Slider(
                    value=2, minimum=0, maximum=10, step=1, label="Box inflate (px)"
                )
                mode = gr.Radio(
                    choices=["redact", "label"], value="redact", label="Mode"
                )
        with gr.Accordion("OCR Enhancements", open=False):
            with gr.Row():
                enhance = gr.Checkbox(
                    value=True, label="Preprocess (denoise + binarize)"
                )
                deskew = gr.Checkbox(value=True, label="Deskew")
                binarize = gr.Checkbox(value=True, label="Binarize")
                auto_psm = gr.Checkbox(value=True, label="Auto-PSM retry")

        with gr.Accordion("Debugging", open=False):
            export_ocr_debug = gr.Checkbox(
                value=False, label="Export OCR debug artifacts (TSV + text)"
            )
        with gr.Accordion("Review & Artifacts", open=False):
            generate_review = gr.Checkbox(
                value=True, label="Generate review CSVs (candidates, boxes, LLM items)"
            )
            generate_previews = gr.Checkbox(
                value=True, label="Generate preview overlays (kept/rejected)"
            )
            save_traces = gr.Checkbox(
                value=False, label="Save full LLM traces (prompt+response)"
            )

        run_btn = gr.Button("Anonymize", variant="primary")

        with gr.Row():
            out_pdf = gr.File(label="Redacted PDF", interactive=False)
            out_meta = gr.File(label="Run metadata (JSON)", interactive=False)
        with gr.Row():
            cand_out = gr.File(label="Candidates CSV", interactive=False)
            boxes_out = gr.File(label="Boxes CSV", interactive=False)
        with gr.Row():
            ocr_tsv = gr.File(label="OCR TSV", interactive=False)
            ocr_text = gr.File(label="OCR Text", interactive=False)
        summary = gr.JSON(label="Summary")
        with gr.Accordion("LLM Explainability", open=False):
            explain_json = gr.JSON(label="LLM items (with why/score)")
            with gr.Row():
                why_page = gr.Slider(
                    value=-1, minimum=-1, maximum=0, step=1, label="Page (-1=all)"
                )
                why_cat = gr.Textbox(value="", label="Category filter (empty=all)")
                why_dec = gr.Radio(
                    choices=["all", "redact", "reject"], value="all", label="Decision"
                )
            why_table = gr.Dataframe(
                headers=[
                    "page",
                    "text",
                    "start",
                    "end",
                    "category",
                    "redact",
                    "score",
                    "why",
                ],
                interactive=False,
            )
        with gr.Accordion("Previews", open=False):
            previews_json = gr.JSON(visible=False)
            page_slider = gr.Slider(value=0, minimum=0, maximum=0, step=1, label="Page")
            preview_img = gr.Image(label="Preview (kept=green, rejected=red)")
        with gr.Accordion("Performance", open=False):
            perf_json = gr.JSON(label="Aggregated timings")

        run_btn.click(
            _run_pipeline,
            inputs=[
                file_in,
                spacy_model,
                use_spacy,
                use_regex,
                use_llm,
                llm_model,
                llm_url,
                prompt_path,
                use_llm_ner,
                llm_ner_model,
                llm_ner_prompt,
                policy,
                custom_policy,
                dpi,
                psm,
                lang,
                inflate,
                mode,
                generate_review,
                enhance,
                deskew,
                binarize,
                auto_psm,
                export_ocr_debug,
                generate_previews,
                save_traces,
            ],
            outputs=[
                out_pdf,
                out_meta,
                cand_out,
                boxes_out,
                ocr_tsv,
                ocr_text,
                summary,
                explain_json,
                previews_json,
                perf_json,
            ],
        )

        # Wire preview controls
        def _pick_preview(previews: Dict[str, Any], idx: int):
            paths = (previews or {}).get("previews", [])
            if not paths:
                return None
            idx = max(0, min(int(idx), len(paths) - 1))
            return paths[idx]

        def _init_preview(previews: Dict[str, Any]):
            paths = (previews or {}).get("previews", [])
            return gr.update(
                minimum=0, maximum=max(0, len(paths) - 1), value=0
            ), _pick_preview(previews, 0)

        previews_json.change(
            _init_preview, inputs=[previews_json], outputs=[page_slider, preview_img]
        )
        page_slider.change(
            _pick_preview, inputs=[previews_json, page_slider], outputs=[preview_img]
        )

        # Why viewer filtering
        def _filter_why(explain: Dict[str, Any], page: int, cat: str, dec: str):
            items = (explain or {}).get("items", [])
            rows = []
            for it in items:
                if page >= 0 and int(it.get("page", -1)) != int(page):
                    continue
                if cat and (str(it.get("category") or "").upper() != str(cat).upper()):
                    continue
                r = it.get("redact")
                if dec == "redact" and not r:
                    continue
                if dec == "reject" and r:
                    continue
                rows.append(
                    [
                        it.get("page"),
                        it.get("text"),
                        it.get("start"),
                        it.get("end"),
                        it.get("category"),
                        it.get("redact"),
                        it.get("score"),
                        it.get("why"),
                    ]
                )
            return rows

        def _init_why(explain: Dict[str, Any]):
            items = (explain or {}).get("items", [])
            max_page = 0
            for it in items:
                try:
                    max_page = max(max_page, int(it.get("page", 0)))
                except Exception:
                    pass
            return gr.update(minimum=-1, maximum=max_page, value=-1), _filter_why(
                explain, -1, "", "all"
            )

        explain_json.change(
            _init_why, inputs=[explain_json], outputs=[why_page, why_table]
        )
        why_page.change(
            _filter_why,
            inputs=[explain_json, why_page, why_cat, why_dec],
            outputs=[why_table],
        )
        why_cat.change(
            _filter_why,
            inputs=[explain_json, why_page, why_cat, why_dec],
            outputs=[why_table],
        )
        why_dec.change(
            _filter_why,
            inputs=[explain_json, why_page, why_cat, why_dec],
            outputs=[why_table],
        )

        gr.Markdown(
            """
            Tips:
            - If PDF rasterization fails, install Poppler or set `POPPLER_PATH`.
            - For LLM confirmation, install and run Ollama locally with the specified model.
            """
        )

    return demo


def launch(
    server_name: str = "127.0.0.1", server_port: int = 7860, inbrowser: bool = False
) -> None:
    """Launch the Gradio UI.

    Parameters
    ----------
    server_name:
        Host interface for the Gradio server.
    server_port:
        Port for the Gradio server.
    inbrowser:
        Open a browser window on launch when True.
    """
    app = build_interface()
    app.launch(server_name=server_name, server_port=server_port, inbrowser=inbrowser)
