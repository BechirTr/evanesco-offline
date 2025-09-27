"""Shared helpers for interacting with Ollama LLM backends."""

from functools import lru_cache

from evanesco.llm import OllamaClient


@lru_cache(maxsize=4)
def get_ollama_client(url: str) -> OllamaClient:
    return OllamaClient(url=url)


__all__ = ["get_ollama_client"]
