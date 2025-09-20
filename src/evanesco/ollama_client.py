"""Thin client for calling a local Ollama HTTP server.

This client provides a convenience function :func:`ollama_generate` that issues
non-streaming completion calls and returns the raw ``response`` string from
Ollama.
"""

from typing import Dict, Any, Optional
import requests


def ollama_generate(
    prompt: str,
    model: str = "gpt-oss:20b",
    url: str = "http://localhost:11434/api/generate",
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> str:
    """Call Ollama's ``/api/generate`` endpoint and return the ``response`` field.

    Parameters
    ----------
    prompt:
        Full prompt text to send (system + user if applicable).
    model:
        Local model name, e.g. ``gpt-oss:20b``.
    url:
        Ollama base URL including ``/api/generate``.
    options:
        Optional model options (temperature, num_ctx, etc.).
    timeout:
        Request timeout in seconds.

    Returns
    -------
    str
        Response string (may contain non-JSON content depending on prompt).

    Raises
    ------
    requests.HTTPError
        If the HTTP response is not successful.
    """
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")
