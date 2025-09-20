"""Audit logging for Evanesco runs.

Produces an audit JSON alongside the output PDF including config snapshot,
hashes, version, page summaries, and an optional HMAC signature when
`EVANESCO_HMAC_KEY` is present.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib
import hmac
import os
import socket
import getpass
import time
import orjson


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_audit(
    input_path: str,
    output_path: str,
    result: Dict[str, Any],
    cfg: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
) -> Path:
    """Write an audit JSON next to the output PDF and return its path."""
    out_pdf = Path(output_path)
    inp = Path(input_path)
    audit_path = out_pdf.with_suffix(".audit.json")
    from evanesco import __version__ as version

    pages = result.get("pages", [])
    page_summaries = []
    for p in pages:
        cats = {}
        for bi in p.get("boxes", []):
            cats[bi.get("category", "OTHER")] = cats.get(bi.get("category", "OTHER"), 0) + 1
        page_summaries.append({
            "page_index": p.get("page_index"),
            "boxes": int(p.get("boxes_applied", 0)),
            "by_category": cats,
        })

    # Aggregate LLM meta across pages
    llm_pages = 0
    llm_items = 0
    llm_model = None
    llm_metrics = {"eval_count": 0, "prompt_eval_count": 0, "eval_duration": 0, "prompt_eval_duration": 0, "total_duration": 0}
    for p in result.get("pages", []):
        lj = p.get("llm_json") or {}
        if lj:
            llm_pages += 1
            llm_items += len(lj.get("items", []))
            m = lj.get("meta") or {}
            if m.get("model"):
                llm_model = m.get("model")
            for k in list(llm_metrics.keys()):
                try:
                    llm_metrics[k] += int(m.get(k) or 0)
                except Exception:
                    pass

    record = {
        "version": version,
        "timestamp": int(time.time()),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "input": {
            "path": str(inp),
            "sha256": _sha256_file(inp),
        },
        "output": {
            "path": str(out_pdf),
            "sha256": _sha256_file(out_pdf) if out_pdf.exists() else None,
        },
        "config": cfg,
        "policy": policy,
        "result": {
            "summary": {
                "pages": len(pages),
                "boxes": sum(p.get("boxes_applied", 0) for p in pages),
            },
            "pages": page_summaries,
        },
        "llm": {
            "model": llm_model,
            "pages_used": llm_pages,
            "items": llm_items,
            "metrics": llm_metrics,
        },
        "errors": errors or [],
    }

    # Optional HMAC signature for tamper detection
    key = os.environ.get("EVANESCO_HMAC_KEY")
    data_bytes = orjson.dumps(record)
    if key:
        sig = hmac.new(key.encode("utf-8"), data_bytes, hashlib.sha256).hexdigest()
        record["hmac"] = {"alg": "HMAC-SHA256", "key_hint": "env:EVANESCO_HMAC_KEY", "value": sig}

    audit_path.write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2))
    return audit_path
