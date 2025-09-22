"""Command-line interface for Evanesco Offline PII redaction.

Provides:
- `run`: Process an input path (PDF/image/dir) to a redacted PDF.
- `eval`: Simple coverage metric given ground truth.
- `ui`: Launch a local Gradio UI for interactive anonymization.
"""

import typer
from pathlib import Path
from typing import Optional
from rich import print
from .core import RunConfig, process_path
from .ui import launch as launch_ui
from .batch import run_batch
import orjson

app = typer.Typer(add_completion=False, help="Evanesco Offline PII Redactor")


@app.command()
def run(
    input: str = typer.Option(..., "--input", "-i", help="Input path (PDF/image/dir)"),
    output: str = typer.Option(..., "--output", "-o", help="Output redacted PDF path"),
    spacy_model: str = typer.Option(
        "en_core_web_lg", help="spaCy model name (or 'auto')"
    ),
    use_spacy: bool = typer.Option(
        True, "--use-spacy/--no-use-spacy", help="Enable spaCy NER detection"
    ),
    use_llm: bool = typer.Option(True, help="Use Ollama LLM confirmation"),
    use_llm_ner: bool = typer.Option(
        False, help="Use LLM few-shot NER (instead of spaCy)"
    ),
    llm_model: str = typer.Option("gpt-oss:20b", help="Ollama model name"),
    llm_ner_model: Optional[str] = typer.Option(
        None, help="Override model name for LLM NER"
    ),
    prompt_path: Optional[str] = typer.Option(
        None, help="Path to prompts/pii_audit.jsonl"
    ),
    llm_ner_prompt: Optional[str] = typer.Option(
        None, help="Path to prompts/ner_fewshot.jsonl for LLM NER"
    ),
    policy: Optional[str] = typer.Option(
        None, help="Policy name or YAML path (e.g., default, gdpr, hipaa, pci)"
    ),
    dpi: int = typer.Option(400, help="Rasterization DPI for PDFs"),
    psm: int = typer.Option(3, help="Tesseract PSM"),
    lang: str = typer.Option("eng", help="Tesseract language"),
    box_inflate: int = typer.Option(2, help="Inflate redaction boxes (px)"),
    mode: str = typer.Option(
        "redact", help="Mode: redact | label (pseudonymize overlay)"
    ),
    preprocess: bool = typer.Option(
        True, "--preprocess/--no-preprocess", help="Enhance OCR via preprocessing"
    ),
    deskew: bool = typer.Option(
        True, "--deskew/--no-deskew", help="Deskew pages before OCR"
    ),
    binarize: bool = typer.Option(
        True, "--binarize/--no-binarize", help="Binarize images for OCR"
    ),
    auto_psm: bool = typer.Option(
        True, "--auto-psm/--no-auto-psm", help="Auto retry PSM to maximize tokens"
    ),
):
    """Redact PII from PDFs or images and write a redacted PDF.

    Parameters
    ----------
    input:
        Input path: PDF, image file, or directory of images.
    output:
        Output PDF path for the redacted document.
    spacy_model:
        spaCy model to use for NER.
    use_spacy:
        Enable/disable spaCy NER detection.
    use_llm:
        Confirm candidates via an Ollama LLM.
    llm_model:
        Ollama model name (e.g., ``gpt-oss:20b``).
    prompt_path:
        Optional system prompt file for the LLM.
    dpi:
        PDF rasterization DPI.
    psm:
        Tesseract page segmentation mode.
    lang:
        Tesseract language code.
    box_inflate:
        Pixels to inflate redaction rectangles.
    """
    # Resolve spaCy model when 'auto' requested based on OCR lang
    _sm = spacy_model
    if (_sm or "").strip().lower() == "auto":
        lang_lower = (lang or "").lower()
        if lang_lower.startswith("fr"):
            _sm = "fr_core_news_lg"
        elif lang_lower.startswith("en"):
            _sm = "en_core_web_lg"
        else:
            _sm = "en_core_web_lg"

    cfg = RunConfig(
        lang=lang,
        psm=psm,
        dpi=dpi,
        use_spacy=use_spacy,
        spacy_model=_sm,
        use_regex=True,
        use_llm=use_llm,
        use_llm_ner=use_llm_ner,
        llm_model=llm_model,
        llm_ner_model=llm_ner_model,
        prompt_path=prompt_path,
        llm_ner_prompt_path=llm_ner_prompt,
        policy_path=policy,
        box_inflation_px=box_inflate,
        mode=mode,
        preprocess=preprocess,
        deskew=deskew,
        binarize=binarize,
        auto_psm=auto_psm,
    )
    res = process_path(input, output, cfg)
    meta_path = Path(output).with_suffix(".meta.json")
    meta_path.write_bytes(orjson.dumps(res, option=orjson.OPT_INDENT_2))
    print(f"[green]Redacted PDF:[/green] {output}")
    print(f"[green]Details:[/green] {str(meta_path)}")


@app.command()
def eval(
    truth_json: str = typer.Option(..., help="Ground-truth JSON path"),
    meta_json: str = typer.Option(..., help="Pipeline output meta.json"),
):
    """Compute simple coverage metric of redactions vs ground-truth spans.

    Parameters
    ----------
    truth_json:
        Path to a JSON file with ground-truth spans (``{"spans":[...]}``).
    meta_json:
        Pipeline output meta.json as produced by the tool.
    """
    import orjson

    truth = orjson.loads(Path(truth_json).read_bytes())
    meta = orjson.loads(Path(meta_json).read_bytes())

    # Simple coverage: count number of truth spans that overlap with any redacted candidate
    def overlap(a, b):
        return not (a["end"] <= b["start"] or a["start"] >= b["end"])

    red_spans = []
    for p in meta["pages"]:
        cands = p.get("candidates", [])
        for c in cands:
            red_spans.append({"start": c["start"], "end": c["end"]})
    tp = 0
    for t in truth.get("spans", []):
        if any(overlap(t, r) for r in red_spans):
            tp += 1
    cov = tp / max(1, len(truth.get("spans", [])))
    print(f"Coverage: {cov:.3f} ({tp}/{len(truth.get('spans', []))})")


@app.command()
def ui(
    host: str = typer.Option(
        "127.0.0.1",
        help="Host to bind the UI server (use 0.0.0.0 only when intentional)",
    ),
    port: int = typer.Option(7860, help="Port for the UI server"),
    inbrowser: bool = typer.Option(False, help="Open browser on launch"),
):
    """Launch the Gradio-based UI for document anonymization.

    Parameters
    ----------
    host:
        Interface to bind the web server.
    port:
        Port number for the web server.
    inbrowser:
        Open the default browser on launch when True.
    """
    launch_ui(server_name=host, server_port=port, inbrowser=inbrowser)


if __name__ == "__main__":
    app()


@app.command()
def batch(
    input_dir: str = typer.Option(..., help="Input directory or glob pattern"),
    output_dir: str = typer.Option(..., help="Output directory for PDFs"),
    workers: int = typer.Option(2, help="Concurrent workers"),
    spacy_model: str = typer.Option(
        "en_core_web_lg", help="spaCy model name (or 'auto')"
    ),
    use_spacy: bool = typer.Option(
        True, "--use-spacy/--no-use-spacy", help="Enable spaCy NER detection"
    ),
    use_llm: bool = typer.Option(False, help="Use Ollama LLM confirmation"),
    use_llm_ner: bool = typer.Option(
        False, help="Use LLM few-shot NER (instead of spaCy)"
    ),
    llm_model: str = typer.Option("gpt-oss:20b", help="Ollama model name"),
    llm_ner_model: Optional[str] = typer.Option(
        None, help="Override model name for LLM NER"
    ),
    prompt_path: Optional[str] = typer.Option(
        None, help="Path to prompts/pii_audit.jsonl"
    ),
    llm_ner_prompt: Optional[str] = typer.Option(
        None, help="Path to prompts/ner_fewshot.jsonl for LLM NER"
    ),
    policy: Optional[str] = typer.Option(None, help="Policy name or YAML path"),
    dpi: int = typer.Option(400, help="Rasterization DPI for PDFs"),
    psm: int = typer.Option(3, help="Tesseract PSM"),
    lang: str = typer.Option("eng", help="Tesseract language"),
    box_inflate: int = typer.Option(2, help="Inflate redaction boxes (px)"),
    mode: str = typer.Option(
        "redact", help="Mode: redact | label (pseudonymize overlay)"
    ),
    preprocess: bool = typer.Option(
        True, "--preprocess/--no-preprocess", help="Enhance OCR via preprocessing"
    ),
    deskew: bool = typer.Option(
        True, "--deskew/--no-deskew", help="Deskew pages before OCR"
    ),
    binarize: bool = typer.Option(
        True, "--binarize/--no-binarize", help="Binarize images for OCR"
    ),
    auto_psm: bool = typer.Option(
        True, "--auto-psm/--no-auto-psm", help="Auto retry PSM to maximize tokens"
    ),
):
    """Batch process multiple inputs concurrently."""
    from glob import glob

    files = []
    p = Path(input_dir)
    if p.exists() and p.is_dir():
        for fp in p.iterdir():
            if fp.suffix.lower() in {
                ".pdf",
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
            }:
                files.append(str(fp))
    else:
        files = glob(input_dir)
    if not files:
        print("[red]No inputs found[/red]")
        raise SystemExit(1)
    # Resolve spaCy model when 'auto'
    _sm = spacy_model
    if (_sm or "").strip().lower() == "auto":
        lang_lower = (lang or "").lower()
        if lang_lower.startswith("fr"):
            _sm = "fr_core_news_lg"
        elif lang_lower.startswith("en"):
            _sm = "en_core_web_lg"
        else:
            _sm = "en_core_web_lg"

    cfg = RunConfig(
        lang=lang,
        psm=psm,
        dpi=dpi,
        use_spacy=use_spacy,
        spacy_model=_sm,
        use_regex=True,
        use_llm=use_llm,
        use_llm_ner=use_llm_ner,
        llm_model=llm_model,
        llm_ner_model=llm_ner_model,
        prompt_path=prompt_path,
        llm_ner_prompt_path=llm_ner_prompt,
        policy_path=policy,
        box_inflation_px=box_inflate,
        mode=mode,
        preprocess=preprocess,
        deskew=deskew,
        binarize=binarize,
        auto_psm=auto_psm,
    )
    pairs = run_batch(files, output_dir, cfg, workers=workers)
    print(f"[green]Completed {len(pairs)} files[/green]")
