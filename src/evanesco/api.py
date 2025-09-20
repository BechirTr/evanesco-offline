"""FastAPI service exposing Evanesco redaction as HTTP endpoints.

Run locally:
    uvicorn evanesco.api:app --host 0.0.0.0 --port 8000

Or via console script:
    evanesco-api --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .core import RunConfig, process_path
from .health import run_readiness_checks
from .settings import ServiceSettings, get_settings

settings: ServiceSettings = get_settings()

app = FastAPI(title="Evanesco Offline API")

if settings.cors_origins:
    allow_origins = ["*"] if "*" in settings.cors_origins else settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

_bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme)) -> None:
    """Simple bearer-token protection for managed cluster deployments."""
    token = settings.api_token
    if token is None:
        return
    if credentials is None or credentials.credentials != token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


# Optional Prometheus metrics mount
try:
    from prometheus_client import Counter, make_asgi_app

    REQUESTS = Counter("evanesco_requests_total", "API requests", ["route"])  # type: ignore
    app.mount("/metrics", make_asgi_app())  # type: ignore
except Exception:  # pragma: no cover
    REQUESTS = None  # type: ignore


@app.get("/health")
def health():
    if REQUESTS:
        REQUESTS.labels(route="health").inc()
    return {"status": "ok"}


@app.get("/livez")
def livez():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    """Kubernetes readiness probe with dependency checks."""
    checks = run_readiness_checks(settings)
    ready = True
    for chk in checks:
        if chk.status == "fail" and chk.required:
            ready = False
            break
        if chk.status == "warn" and chk.required and not settings.allowance_warn_only_checks:
            ready = False
            break
    content = {
        "ready": ready,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "detail": c.detail,
                "required": c.required,
            }
            for c in checks
        ],
    }
    status_code = 200 if ready else 503
    return JSONResponse(status_code=status_code, content=content)


@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    spacy_model: str = Form("en_core_web_lg"),
    use_spacy: bool = Form(True),
    use_regex: bool = Form(True),
    use_llm: bool = Form(False),
    llm_model: str = Form("gpt-oss:20b"),
    policy: Optional[str] = Form(None),
    dpi: int = Form(300),
    psm: int = Form(3),
    lang: str = Form("eng"),
    box_inflate: int = Form(2),
    mode: str = Form("redact"),
    auth: None = Depends(require_auth),
):
    """Redact an uploaded file and return the redacted PDF."""
    if REQUESTS:
        REQUESTS.labels(route="redact").inc()
    # Persist upload to temp file
    tmp_dir = Path(tempfile.mkdtemp(prefix="evanesco_api_"))
    inp = tmp_dir / file.filename
    with open(inp, "wb") as f:
        f.write(await file.read())
    out_pdf = tmp_dir / f"{inp.stem}.redacted.pdf"

    cfg = RunConfig(
        lang=lang,
        psm=psm,
        dpi=dpi,
        use_spacy=use_spacy,
        spacy_model=spacy_model,
        use_regex=use_regex,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_url=settings.llm_generate_url,
        policy_path=policy,
        box_inflation_px=box_inflate,
        mode=mode,
    )
    try:
        process_path(str(inp), str(out_pdf), cfg)
        return FileResponse(str(out_pdf), media_type="application/pdf", filename=out_pdf.name)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


def run(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """Launch the API server via `uvicorn`."""
    import uvicorn

    bind_host = host or settings.api_host
    bind_port = port or settings.api_port
    uvicorn.run(
        "evanesco.api:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        workers=workers,
        log_level=log_level,
    )
