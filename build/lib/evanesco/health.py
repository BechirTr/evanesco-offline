"""Infrastructure readiness checks for API / Kubernetes probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .settings import ServiceSettings


@dataclass
class HealthCheckResult:
    name: str
    status: str  # "pass" | "fail" | "warn"
    detail: Optional[str] = None
    required: bool = True


def _check_tesseract(langs: List[str]) -> HealthCheckResult:
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:  # pragma: no cover - depends on runtime
        return HealthCheckResult(name="tesseract", status="fail", detail=str(exc))

    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        available = set()
    missing = [lang for lang in langs if lang not in available]
    if missing and available:
        return HealthCheckResult(
            name="tesseract",
            status="warn",
            detail=f"Missing language packs: {', '.join(missing)}",
        )
    if missing and not available:
        return HealthCheckResult(
            name="tesseract",
            status="warn",
            detail="Could not enumerate language packs; ensure tessdata is mounted.",
        )
    return HealthCheckResult(name="tesseract", status="pass")


def _check_spacy_model(model_name: Optional[str]) -> HealthCheckResult:
    if not model_name:
        return HealthCheckResult(
            name="spacy", status="warn", detail="Model not specified", required=False
        )
    try:
        from .spacy_detect import _load_spacy

        nlp = _load_spacy(model_name)
        has_ner = "ner" in getattr(nlp, "pipe_names", [])
        if not has_ner:
            return HealthCheckResult(
                name="spacy", status="warn", detail="NER component missing"
            )
        return HealthCheckResult(name="spacy", status="pass")
    except Exception as exc:  # pragma: no cover - depends on runtime
        return HealthCheckResult(name="spacy", status="fail", detail=str(exc))


def _check_llm_endpoint(url: str, health_url: Optional[str]) -> HealthCheckResult:
    import requests

    target = health_url or url
    try:
        resp = requests.request("HEAD", target, timeout=2)
        if resp.status_code >= 500:
            return HealthCheckResult(
                name="llm", status="fail", detail=f"HTTP {resp.status_code}"
            )
        if resp.status_code == 405:
            return HealthCheckResult(
                name="llm",
                status="warn",
                detail="HEAD not supported, endpoint reachable",
            )
        return HealthCheckResult(name="llm", status="pass")
    except Exception as exc:  # pragma: no cover - network dependent
        return HealthCheckResult(name="llm", status="fail", detail=str(exc))


def run_readiness_checks(settings: ServiceSettings) -> List[HealthCheckResult]:
    checks: List[HealthCheckResult] = []
    if settings.readiness_check_ocr:
        checks.append(_check_tesseract(settings.readiness_tesseract_langs))
    if settings.readiness_check_spacy:
        checks.append(_check_spacy_model(settings.readiness_spacy_model))
    if settings.readiness_check_llm:
        checks.append(
            _check_llm_endpoint(
                settings.llm_generate_url, settings.readiness_llm_health_url
            )
        )
    return checks
