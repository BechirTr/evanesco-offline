"""Service configuration helpers for deployment environments.

This module centralises runtime configuration that previously relied on
scattered environment variable lookups. It is intentionally lightweight so it
can be imported from both CLI tools and FastAPI without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional
import os


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts


@dataclass
class ServiceSettings:
    """Runtime settings tailored for container/Kubernetes deployments."""

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_token: Optional[str] = None
    cors_origins: List[str] = field(default_factory=list)
    readiness_check_ocr: bool = True
    readiness_check_spacy: bool = True
    readiness_check_llm: bool = False
    readiness_spacy_model: Optional[str] = "en_core_web_lg"
    readiness_tesseract_langs: List[str] = field(default_factory=lambda: ["eng"])
    readiness_llm_health_url: Optional[str] = None
    llm_generate_url: str = "http://localhost:11434/api/generate"
    allowance_warn_only_checks: bool = True

    @staticmethod
    def from_env() -> "ServiceSettings":
        cors_raw = os.environ.get("EVANESCO_API_CORS_ORIGINS")
        settings = ServiceSettings(
            api_host=os.environ.get("EVANESCO_API_HOST", "127.0.0.1"),
            api_port=int(os.environ.get("EVANESCO_API_PORT", "8000")),
            api_token=os.environ.get("EVANESCO_API_TOKEN"),
            cors_origins=_split_csv(cors_raw),
            readiness_check_ocr=_parse_bool(
                os.environ.get("EVANESCO_READY_CHECK_OCR"), default=True
            ),
            readiness_check_spacy=_parse_bool(
                os.environ.get("EVANESCO_READY_CHECK_SPACY"), default=True
            ),
            readiness_check_llm=_parse_bool(
                os.environ.get("EVANESCO_READY_CHECK_LLM"), default=False
            ),
            readiness_spacy_model=os.environ.get("EVANESCO_READY_SPACY_MODEL")
            or "en_core_web_lg",
            readiness_tesseract_langs=_split_csv(
                os.environ.get("EVANESCO_READY_TESS_LANGS")
            )
            or ["eng"],
            readiness_llm_health_url=os.environ.get("EVANESCO_LLM_HEALTH_URL"),
            llm_generate_url=os.environ.get(
                "EVANESCO_LLM_URL", "http://localhost:11434/api/generate"
            ),
            allowance_warn_only_checks=_parse_bool(
                os.environ.get("EVANESCO_READY_WARN_ONLY"), default=True
            ),
        )
        return settings


@lru_cache(maxsize=1)
def get_settings() -> ServiceSettings:
    """Return cached service settings."""
    return ServiceSettings.from_env()


def reset_settings_cache() -> None:
    """Reset cached settings (useful in tests)."""
    get_settings.cache_clear()  # type: ignore[attr-defined]
