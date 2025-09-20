"""Pluggable LLM client interface with retry/backoff.

Default implementation targets Ollama's `/api/generate` endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import requests


class LLMClient:
    def generate(self, prompt: str, *, model: str, options: Optional[Dict[str, Any]] = None, timeout: int = 120) -> str:  # noqa: D401
        """Generate a completion. Implement in subclasses."""
        raise NotImplementedError


@dataclass
class OllamaClient(LLMClient):
    url: str = "http://localhost:11434/api/generate"
    retries: int = 2
    backoff: float = 1.5

    def generate(self, prompt: str, *, model: str, options: Optional[Dict[str, Any]] = None, timeout: int = 120) -> str:
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(self.url, json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:  # pragma: no cover - network variability
                last_exc = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
                else:
                    raise
        # Unreachable, but for static analyzers
        if last_exc:
            raise last_exc
        return ""

    def generate_raw(self, prompt: str, *, model: str, options: Optional[Dict[str, Any]] = None, timeout: int = 120) -> Dict[str, Any]:
        """Return full Ollama JSON (includes timings, counts) for explainability."""
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(self.url, json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # pragma: no cover
                last_exc = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
                else:
                    raise
        if last_exc:
            raise last_exc
        return {}
