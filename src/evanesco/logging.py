"""Structured logging helpers (JSON).

Use `get_logger(__name__)` to emit JSON logs with all extra fields included.
"""

from __future__ import annotations

import logging
import orjson


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.args and isinstance(record.args, dict):
            payload.update(record.args)
        if hasattr(record, "extra") and isinstance(getattr(record, "extra"), dict):
            payload.update(getattr(record, "extra"))
        return orjson.dumps(payload).decode("utf-8")


def get_logger(name: str = "evanesco") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

