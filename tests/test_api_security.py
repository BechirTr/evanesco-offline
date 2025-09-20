import importlib
import os

import pytest
from fastapi.testclient import TestClient

import evanesco.settings as settings
from evanesco.health import HealthCheckResult


def _make_client():
    os.environ["EVANESCO_API_TOKEN"] = "super-secret"
    os.environ["EVANESCO_READY_CHECK_OCR"] = "false"
    os.environ["EVANESCO_READY_CHECK_SPACY"] = "false"
    os.environ["EVANESCO_READY_CHECK_LLM"] = "false"
    settings.reset_settings_cache()
    import evanesco.api as api  # noqa: F401

    api = importlib.reload(api)
    return TestClient(api.app), api


def test_redact_requires_bearer_token():
    client, _ = _make_client()

    resp = client.post(
        "/redact",
        files={"file": ("dummy.pdf", b"%PDF-1.0\n", "application/pdf")},
    )
    assert resp.status_code == 401

    resp_ok = client.post(
        "/redact",
        headers={"Authorization": "Bearer super-secret"},
        files={"file": ("dummy.pdf", b"%PDF-1.0\n", "application/pdf")},
    )
    assert resp_ok.status_code in {400, 415}


def test_readyz_reflects_health(monkeypatch: pytest.MonkeyPatch):
    client, api = _make_client()

    def fake_checks(_settings):
        return [HealthCheckResult(name="tesseract", status="fail", detail="missing", required=True)]

    monkeypatch.setattr(api, "run_readiness_checks", fake_checks)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    payload = resp.json()
    assert payload["ready"] is False
    assert payload["checks"][0]["name"] == "tesseract"
