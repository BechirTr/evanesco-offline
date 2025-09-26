---
title: Architecture
---

# Architecture

The latest diagram captures the full offline pipeline, including OCR
debugging, policy enforcement, review artifacts, and observability hooks.

- `docs/architecture.mmd` contains the Mermaid source.
- `docs/architecture.png` is the raster export used in the README.
- `docs/architecture.html` is a stand-alone interactive rendering.

Regenerate all formats with `scripts/render_diagram.sh`:

```bash
./scripts/render_diagram.sh docs/architecture.mmd docs/architecture.png
npx @mermaid-js/mermaid-cli@10.9.1 -i docs/architecture.mmd -o docs/architecture.html -t neutral -b transparent -s 1.25
```

```{note}
Sphinx publishes the HTML diagram as a download to keep the page light. Use the
link below when browsing the built docs.
```

- [Download architecture.html](architecture.html)

## End-to-end workflow

### 1. Ingestion & preparation

- **Entry points**: the CLI (`evanesco run`), the batch runner, and the Gradio UI
share the same `RunConfig`. Policies can be selected by name (`gdpr`, `hipaa`,
etc.) or supplied as YAML.
- **Rasterization**: PDFs are rasterized via Poppler (`pdf2image`) with a
PyMuPDF fallback. Directories of images bypass rasterization and stream into the
pipeline in sorted order.
- **Page conditioning**: configurable preprocessing (`preprocess`, `deskew`,
`binarize`, `auto_psm`) is applied before Tesseract. These toggles are exposed
in both the CLI and UI so hard scans can be tuned without code changes.

### 2. OCR & text reconstruction

- **Tesseract TSV**: every page yields word-level bounding boxes and confidence
scores. When `export_ocr_debug` is active (enabled by default for the CLI and
UI), per-page TSV and text dumps are materialized under
`artifacts/<run>.ocr.zip` plus `<run>.all_text.txt` for quick greps.
- **Plain-text stream**: TSV rows are converted into a linear text stream so the
detectors and LLM prompts work in character offsets rather than pixel space.

### 3. Candidate detection

- **spaCy NER**: loaded per language (auto-selects on CLI when `--spacy-model
auto`). Supports switching off entirely for LLM-only runs.
- **Regex detectors**: deterministic patterns catch IDs like email, phone,
IBAN, SSN, credit cards, etc. They always run when enabled in the config.
- **LLM NER (optional)**: few-shot prompting via Ollama can supplement or
replace spaCy. Chunking bounds (`llm_ner_chunk_chars`) keep context sizes under
control for small models.
- The union of all detectors is stored on each page as `candidates`, including
the detector source, character offsets, and extracted text snippet.

### 4. Policy enforcement & confirmation

- **Policies**: the policy module narrows which categories require redaction.
Built-ins include `default`, `gdpr`, `hipaa`, and `pci`. Custom YAML policies
can toggle categories or force redaction for all spans.
- **Ollama confirmation (optional)**: candidates are chunked into prompts that
include detector context. The default confirmation model is `gpt-oss:20b`, but
any local Ollama model works. Responses are parsed into `items[]` with category,
decision, score, and explanation, then merged with the policy decision.
- **Deterministic mode**: when `--no-use-llm` is passed, the pipeline simply
inherits the detector outputs after policy filtering.

### 5. Alignment & rendering

- **Box alignment**: final spans are re-mapped to OCR bounding boxes. Accepted
and rejected spans are tracked separately so preview overlays can show the
difference.
- **Rendering**: redaction can operate in blackout mode or label mode (the
`--mode label` CLI flag / UI toggle). Bounding boxes are optionally inflated by
`box_inflation_px` for safety.
- **Preview overlays**: when enabled (`generate_previews`), the pipeline writes
PNG overlays that highlight approved boxes (green) and rejected candidates
(red). These live under `<run>.previews/` next to the output PDF.

### 6. Outputs & observability

- **Primary outputs**: redacted PDF (`img2pdf`), `<output>.meta.json`,
`<output>.audit.json`, and optional review CSVs (candidates, boxes, and
LLM-decisions) when the UI toggle or CLI automation uses the review helper.
- **LLM traces**: enabling `explain_traces` drops one JSON per page with the full
prompt/response to support audits.
- **Performance metrics**: per-stage timings (`ocr`, `detect`, `llm`, `align`,
`redact`, `total`) are captured on every page. The Gradio UI aggregates them to
drive its Performance panel.
- **Logging**: the pipeline emits structured JSON logs via `evanesco.logging` so
orchestrators can ingest run-level metadata.

The updated diagram (see above) mirrors this flow and explicitly calls out the
review artifacts, LLM trace export, and policy enforcement so operators can see
where each feature hooks into the run.
