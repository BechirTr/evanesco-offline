---
title: evanesco.core detailed reference
---

# ``evanesco.core`` Detailed Reference

The ``evanesco.core`` module preserves the stable public API for downstream
integrators after the pipeline refactor. It re-exports configuration types and
orchestration helpers from the ``evanesco.pipeline`` package while keeping the
original import paths working.

```{literalinclude} ../../src/evanesco/core.py
:language: python
:linenos:
```

## Line-by-line walkthrough

1. ``"""Backward-compatible entry points..."""`` – module docstring that
   states the compatibility goal and explains that the actual implementation
   lives under ``evanesco.pipeline``.
2. ``from __future__ import annotations`` – enables postponed evaluation of
   type annotations so forward references in the imported symbols stay valid on
   Python 3.10+.
3. ``from .pipeline import (...)`` – single grouped import that pulls the
   public classes and functions from the reorganised pipeline package. Each
   symbol is documented in detail below.
4. ``__all__ = [...]`` – explicit export list to guarantee ``from evanesco.core
   import *`` exposes exactly the compatibility surface and nothing else.

Because the file is deliberately thin, every subsequent section dives into the
re-exported objects, summarising their purpose, inputs, outputs, and linking to
origin modules.

## Public API exports

### ``RunConfig``
- **Origin**: ``evanesco.pipeline.config`` dataclass (see
  ``src/evanesco/pipeline/config.py``).
- **Purpose**: encapsulates runtime knobs for OCR (language, DPI, deskewing),
  detection (spaCy toggle, regex and LLM usage), LLM endpoints, policy paths,
  rendering preferences, and instrumentation settings.
- **Inputs**: constructed directly with keyword arguments; all fields have
  defaults (e.g. ``lang="eng"``, ``dpi=400``).
- **Outputs**: instances act as immutable configuration objects passed into
  orchestrators (`process_pdf`, `process_path`) and detector helpers.
- **Important attributes**: ``use_spacy``, ``use_regex``, ``use_llm``,
  ``llm_model``, ``policy_path``, ``generate_previews``, ``export_ocr_debug``.

### ``PageResult``
- **Origin**: Pydantic model in ``evanesco.pipeline.config``.
- **Purpose**: standardised per-page payload produced by detection and
  redaction steps, containing recognised text, candidate spans, LLM decisions,
  bounding boxes, timings, and preview paths.
- **Inputs**: created internally by the pipeline; individual fields are fully
  typed to assist IDEs and downstream validators.
- **Outputs**: included in the ``process_pdf`` / ``process_path`` return
  structure (``{"pages": [PageResult, ...], "out": path, "debug": {...}}``).

### ``detect_candidates(text: str, cfg: RunConfig) -> List[Dict[str, Any]]``
- **Origin**: ``evanesco.pipeline.detection``.
- **Purpose**: run configured detectors (spaCy, regex, LLM NER) over OCR text to
  produce unique entity span candidates with provenance metadata.
- **Inputs**:
  - ``text``: page-level string from OCR.
  - ``cfg``: ``RunConfig`` controlling which detectors activate, the spaCy
    model to load, and LLN prompt/model choices.
- **Outputs**: list of dictionaries each containing ``label``, ``start``,
  ``end``, ``text`` keys; deduplicated based on span boundaries and label.
- **Reference implementation**: ``src/evanesco/pipeline/detection.py:12``.

### ``llm_filter(text: str, candidates: List[Dict[str, Any]], cfg: RunConfig)``
- **Origin**: ``evanesco.pipeline.llm_confirmation``.
- **Purpose**: send detection candidates to the configured Ollama/LLM endpoint
  for confirmation. Applies policy gating, aggregates explanations, and returns
  JSON serialisable decision payloads.
- **Inputs**: OCR ``text``, raw candidate list (usually from
  ``detect_candidates``), ``RunConfig`` with LLM connection details and policy
  settings.
- **Outputs**: dictionary with ``items`` containing enriched spans, confidence
  scores, redact flags, and textual rationales. ``resolve_final_spans`` consumes
  this structure.

### ``llm_ner_candidates(text: str, cfg: RunConfig)``
- **Origin**: ``evanesco.pipeline.llm_ner``.
- **Purpose**: generate additional span candidates via few-shot prompts when
  ``cfg.use_llm_ner`` is enabled.
- **Inputs**: OCR ``text`` chunk and full ``RunConfig`` (notably the
  ``llm_ner_model`` and ``llm_ner_prompt_path`` fields).
- **Outputs**: list of candidate dictionaries compatible with
  ``detect_candidates`` output, tagged with ``source="LLM_NER"``.

### ``process_pdf(input_path: str, output_path: str, cfg: RunConfig)``
- **Origin**: ``evanesco.pipeline.orchestration``.
- **Purpose**: load a single PDF into rasterised pages, drive the redaction
  pipeline, and emit the standard result map.
- **Inputs**:
  - ``input_path``: filesystem path to a PDF on disk.
  - ``output_path``: destination path where the redacted PDF will be written.
  - ``cfg``: ``RunConfig`` controlling OCR/DPI, detectors, policies, and output
    options.
- **Outputs**: dictionary containing ``pages`` (list of ``PageResult``
  metadata), ``out`` (string path of the redacted PDF), and ``debug`` (paths to
  artefacts like OCR TSV archives when enabled).
- **Side effects**: creates rendered PDF at ``output_path``; may write audit
  artefacts to ``artifacts/`` depending on configuration.

### ``process_path(input_path: str, output_path: str, cfg: RunConfig)``
- **Origin**: ``evanesco.pipeline.orchestration``.
- **Purpose**: high-level helper that accepts directories, PDFs, or image files;
  delegates to ``process_pdf`` or image pipelines, and writes audit metadata.
- **Inputs**: flexible ``input_path`` (dir/image/PDF), plus ``output_path`` and
  ``RunConfig``.
- **Outputs**: same dictionary schema as ``process_pdf``.
- **Error handling**: raises ``FileNotFoundError`` when input is missing and
  ``ValueError`` for unsupported file types.

### ``resolve_final_spans(text, candidates, cfg, policy)``
- **Origin**: ``evanesco.pipeline.detection``.
- **Purpose**: combine policy enforcement with optional LLM confirmation to
  decide which candidates get redacted.
- **Inputs**: OCR ``text``, candidate list, ``RunConfig``, and optional
  ``Policy`` object.
- **Outputs**: tuple ``(final_spans, llm_json, llm_duration)`` where
  ``final_spans`` are ready for rendering; ``llm_json`` mirrors the raw
  confirmation payload; ``llm_duration`` captures elapsed inference time.

## Suggested usage pattern

Most callers only need to import from ``evanesco.core``:

```python
from evanesco.core import RunConfig, process_path

cfg = RunConfig(use_llm=False, export_ocr_debug=True)
result = process_path("./data/in/sample.pdf", "./data/out/sample_redacted.pdf", cfg)
```

The import keeps compatibility with earlier releases while still leveraging the
refactored pipeline modules behind the scenes.
