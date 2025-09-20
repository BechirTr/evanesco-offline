"""Redaction policy management.

Provides a simple allow/deny policy for PII categories with YAML loaders and a
`should_redact` decision helper. Policies can be packaged as YAML files and
selected at runtime via `RunConfig.policy_path`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import orjson

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional; fallback to JSON
    yaml = None


@dataclass
class Policy:
    """PII redaction policy.

    Attributes
    ----------
    name:
        Human-friendly identifier for the policy.
    allowed_categories:
        If set, only these categories are redacted; all others are ignored.
    denied_categories:
        Categories explicitly not redacted (overrides allowed when both set).
    default_redact:
        Default decision when a category is not found in either list.
    metadata:
        Free-form metadata (e.g., version, source, jurisdiction).
    """

    name: str = "default"
    allowed_categories: Optional[List[str]] = None
    denied_categories: Optional[List[str]] = None
    default_redact: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_redact(self, category: str) -> bool:
        """Return True if the given category should be redacted under this policy."""
        c = (category or "").upper()
        if self.denied_categories and c in set(x.upper() for x in self.denied_categories):
            return False
        if self.allowed_categories is not None:
            return c in set(x.upper() for x in self.allowed_categories)
        return self.default_redact

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "allowed_categories": self.allowed_categories,
            "denied_categories": self.denied_categories,
            "default_redact": self.default_redact,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_file(path: str | Path) -> "Policy":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")
        text = p.read_text(encoding="utf-8")
        data: Dict[str, Any]
        if p.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
            data = yaml.safe_load(text)  # type: ignore[no-redef]
        else:
            data = orjson.loads(text)
        return Policy(
            name=data.get("name", p.stem),
            allowed_categories=data.get("allowed_categories"),
            denied_categories=data.get("denied_categories"),
            default_redact=bool(data.get("default_redact", True)),
            metadata=data.get("metadata", {}),
        )


def find_builtin_policy(name: str) -> Optional[Path]:
    """Search repository for a builtin policy YAML by name and return its path."""
    for parent in Path(__file__).resolve().parents:
        cand = parent / ".." / ".." / "configs" / "policies" / f"{name}.yaml"
        cand = cand.resolve()
        if cand.exists():
            return cand
    return None

