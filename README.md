![Evanesco Offline](img/header.png)

# Evanesco (Offline)

Fully local, offline document anonymization / PII redaction pipeline.

- **OCR**: pdf2image + Tesseract (via `pytesseract`) with per-word bounding boxes (TSV).
- **Detection**: hybrid rules - spaCy NER (offline models), deterministic regexes (emails, phones, IBAN, SSN-like), and optional LLM confirmation via **Ollama** (local HTTP, e.g., `gpt-oss:20b`).
  You can also use LLM few-shot NER instead of spaCy.
- **Redaction**: image-based black boxes drawn over the OCR bounding boxes, then re-assembled into a redacted PDF while preserving the document's visual layout (forms included).
- **Reproducible**: config-driven, deterministic by default. LLM step can be disabled.
- **No cloud calls**: stays on your machine.

## System requirements

- Python 3.10+
- **Tesseract** installed and in PATH.
- **poppler** installed for `pdf2image` (for PDF rasterization).
- Optional: **Ollama** with a local model (e.g., `gpt-oss:20b`).
- Optional: spaCy model installed offline, e.g.:
  ```bash
  python -m spacy download en_core_web_lg
  ```

## Quickstart

```bash
# 1) Install
pip install -e .

# 2) Set configs (optional)
cp configs/example.yaml configs/local.yaml

# 3) Run (LLM confirmation on by default)
evanesco run --input data/in/sample.pdf --output data/out/sample_redacted.pdf --spacy-model en_core_web_lg

# Variants
# Single image
evanesco run -i data/in/photo.jpg -o data/out/photo_redacted.pdf

# Directory of images (alphabetical order)
evanesco run -i data/in/batch_images -o data/out/batch_redacted.pdf
```

## Pipeline Overview

The anonymization run goes through deterministic stages so you can reason about every decision:

1. **Ingestion & Rasterization** - PDFs are rasterized with Poppler (fallback to PyMuPDF). Directories of images stream straight into the pipeline in sorted order.
2. **OCR (Tesseract)** - each page is processed with the configured DPI/PSM and optional preprocessing/deskew. Word-level TSV is retained so spans map back to the correct bounding boxes. When `export_ocr_debug` is enabled we persist per-page TSV + text dumps under `artifacts/`.
3. **Detectors** - spaCy NER, deterministic regexes, and optional LLM NER (few-shot via Ollama) run in parallel. Results keep detector provenance so you can audit who flagged each span.
4. **Policy & LLM Confirmation (optional)** - policy rules (`default`, `gdpr`, `hipaa`, `pci`, or custom YAML) gate which categories must redact. Candidates are chunked and sent to an Ollama model (default `gpt-oss:20b`) which confirms/labels spans, returning category, decision, score, and explanation. Full prompts/responses can be saved for audits.
5. **Alignment & Redaction** - confirmed spans are aligned to OCR boxes and rendered on the original page images (blackout or label mode). Preview overlays highlight accepted (green) and rejected (red) boxes for QA.
6. **Outputs & Audit** - we emit the redacted PDF, `.meta.json`, `.audit.json`, optional review CSVs, preview PNGs, OCR debug archives, and (when enabled) full LLM traces under `<output>.traces/`.

The new `--export-ocr-debug` flag (also surfaced in the UI) drops `<output>.ocr.zip` and `<output>.all_text.txt` into `artifacts/` so you can inspect raw OCR tokens without digging through temp folders.

## Visual Example

Real documents rarely fit into contrived samples. Below is a real-world page run through Evanesco, with the left panel showing the original scan and the right panel presenting the redacted output.

![Document before and after redaction](img/beforeVsAfter.png)

## CLI

- `evanesco run` - process a file or directory of PDFs/images.
- `evanesco eval` - evaluate redaction quality given ground-truth JSON.
- `evanesco ui` or `evanesco-ui` - launch the Gradio UI.
  - Optionally enable LLM NER (few-shot) in the UI.
  - Preview overlays (kept=green, rejected=red) and an explainability panel are available in the UI under accordions.

## UI

Launch a local web UI with Gradio to upload documents and tune parameters:

```bash
evanesco ui --host 0.0.0.0 --port 7860
# or
evanesco-ui
# or (guarantees using the venv's Python)
python -m evanesco.cli ui --host 0.0.0.0 --port 7860
```

## Documentation

Sphinx documentation with API reference is available under `docs/`.

```bash
pip install .[docs]
sphinx-build -b html docs docs/_build
open docs/_build/index.html
```

Troubleshooting spaCy model not found:
- Ensure the UI runs from the same interpreter where you installed the model.
- Use: `python -m evanesco.cli ui ...` to force the venv's Python.
- Set `EVANESCO_DEBUG=1` to include interpreter and loader details in warnings.

## Deployment

- Hardened Dockerfile for production builds: `docker build -t ghcr.io/your-org/evanesco-offline:latest .`.
- FastAPI now exposes `/livez` and `/readyz` for Kubernetes probes; `/redact` enforces an optional bearer token via `EVANESCO_API_TOKEN`.
- Opinionated OpenShift/Kubernetes manifests live under `deploy/kubernetes/` (ConfigMap, Secret, ServiceAccount, Deployment, Service, Route).
- Detailed cluster guidance (TLS, probes, observability) is in `docs/deployment.md`.

## Notes

- Redaction is **image-based** for robustness and layout fidelity. We overlay rectangles on detected PII words/lines and export a redacted PDF using `img2pdf`. For exact text redaction in vector PDFs, integrate `PyMuPDF` search on the original PDF text layer when available.
- The Ollama step adds a light-weight "human-like" confirmation (e.g., ambiguous names). It's on by default and can be disabled.
- OCR defaults are tuned for harder documents: DPI 400, preprocessing, deskew, binarization, and auto-PSM retry.
 - LLM NER: `--use-llm-ner --no-use-spacy` with `--llm-ner-prompt src/prompts/ner_fewshot.jsonl` for few-shot extraction.
 - UI includes a Performance panel that shows per-stage timings to guide tuning.
- Default LLM NER model is `mistral:7b`; override via `--llm-ner-model` or the UI when you want something heavier/lighter.
- All OCR debug artifacts live under `artifacts/` once `--export-ocr-debug` is enabled, making it easier to sanity-check TSVs after a run.

## Enterprise Features

- Policies: built-in `default`, `gdpr`, `hipaa`, `pci` or custom YAML via `--policy`.
- Audit: per-run audit JSON with input/output hashes; optional HMAC via `EVANESCO_HMAC_KEY`.
- Pseudonymization: `--mode label` overlays category tags instead of pure blackout.
- Batch: `evanesco batch` processes many files concurrently.
- API: `evanesco-api` (FastAPI) exposes `/redact` and `/health`, with `/metrics` when `prometheus-client` is installed.
