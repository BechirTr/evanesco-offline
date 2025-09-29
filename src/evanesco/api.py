"""FastAPI service exposing Evanesco redaction workflows with structured APIs.

The module follows common backend design patterns:

* Pydantic models capture request/response payloads and enforce validation.
* Routers group health checks and CRUD-style redaction job endpoints.
* A lightweight in-memory repository tracks jobs for local experimentation.
* Swagger UI (``/docs``) and ReDoc (``/redoc``) provide interactive manuals for
  exercising each endpoint locally.

Run locally::

    uvicorn evanesco.api:app --host 0.0.0.0 --port 8000

Or via console script::

    evanesco-api --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError, validator

from .core import PageResult, RunConfig, process_path
from .health import run_readiness_checks
from .settings import ServiceSettings, get_settings

settings: ServiceSettings = get_settings()


auth_scheme = HTTPBearer(auto_error=False)


class JobStatus(str, Enum):
    """Lifecycle states for a redaction job."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class RedactionOptions(BaseModel):
    """User-tunable configuration for a redaction job."""

    lang: str = "eng"
    psm: int = 3
    dpi: int = 400
    use_spacy: bool = True
    spacy_model: str = "en_core_web_lg"
    use_regex: bool = True
    use_llm: bool = True
    llm_model: str = "gpt-oss:20b"
    use_llm_ner: bool = False
    llm_ner_model: Optional[str] = None
    llm_ner_prompt_path: Optional[str] = None
    policy_path: Optional[str] = None
    box_inflation_px: int = 2
    mode: Literal["redact", "label"] = "redact"
    generate_previews: bool = True
    export_ocr_debug: bool = False

    @validator("psm")
    def validate_psm(cls, value: int) -> int:  # noqa: D417 - custom message
        if value < 0:
            raise ValueError("psm must be non-negative")
        return value

    def to_run_config(self, svc_settings: ServiceSettings) -> RunConfig:
        """Translate user options into an internal ``RunConfig`` instance."""

        return RunConfig(
            lang=self.lang,
            psm=self.psm,
            dpi=self.dpi,
            use_spacy=self.use_spacy,
            spacy_model=self.spacy_model,
            use_regex=self.use_regex,
            use_llm=self.use_llm,
            use_llm_ner=self.use_llm_ner,
            llm_ner_model=self.llm_ner_model,
            llm_ner_prompt_path=self.llm_ner_prompt_path,
            llm_model=self.llm_model,
            llm_url=svc_settings.llm_generate_url,
            policy_path=self.policy_path,
            box_inflation_px=self.box_inflation_px,
            mode=self.mode,
            generate_previews=self.generate_previews,
            export_ocr_debug=self.export_ocr_debug,
        )


class RedactionJobResult(BaseModel):
    """Output artefacts emitted by the redaction pipeline."""

    output_path: str
    page_count: int
    debug_artifacts: Dict[str, str] = Field(default_factory=dict)
    pages: List[PageResult] = Field(default_factory=list)


class RedactionJob(BaseModel):
    """Persistent view of a redaction job for CRUD operations."""

    id: str
    input_path: str
    output_path: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    options: RedactionOptions
    detail: Optional[str] = None
    result: Optional[RedactionJobResult] = None


class RedactionJobCreate(BaseModel):
    """Request payload for creating a new redaction job."""

    input_path: str
    output_path: Optional[str] = None
    options: RedactionOptions = Field(default_factory=RedactionOptions)

    @validator("input_path")
    def validate_input_path(cls, value: str) -> str:  # noqa: D417 - concise rule
        if not value:
            raise ValueError("input_path must be provided")
        return value


class RedactionJobUpdate(BaseModel):
    """Request payload for updating/re-running an existing job."""

    output_path: Optional[str] = None
    options: Optional[RedactionOptions] = None


class JobListResponse(BaseModel):
    """Response envelope for listing jobs."""

    items: List[RedactionJob]
    total: int


class HealthResponse(BaseModel):
    """Canonical health endpoint payload."""

    status: Literal["ok"] = "ok"


class ReadinessCheckModel(BaseModel):
    """Single readiness check result."""

    name: str
    status: Literal["pass", "warn", "fail"]
    detail: Optional[str] = None
    required: bool


class ReadyResponse(BaseModel):
    """Aggregated readiness response."""

    ready: bool
    checks: List[ReadinessCheckModel]


class JobNotFoundError(Exception):
    """Raised when a job id is not present in the repository."""


class JobRepository:
    """In-memory persistence layer for redaction jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, RedactionJob] = {}

    def list(self) -> List[RedactionJob]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at)

    def get(self, job_id: str) -> RedactionJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:  # pragma: no cover - trivial
            raise JobNotFoundError(job_id) from exc

    def save(self, job: RedactionJob) -> RedactionJob:
        self._jobs[job.id] = job
        return job

    def delete(self, job_id: str) -> None:
        if job_id not in self._jobs:
            raise JobNotFoundError(job_id)
        del self._jobs[job_id]


job_repository = JobRepository()


# Optional Prometheus metrics mount
try:
    from prometheus_client import Counter, make_asgi_app

    REQUESTS = Counter("evanesco_requests_total", "API requests", ["route"])  # type: ignore
except Exception:  # pragma: no cover - metrics optional
    REQUESTS = None  # type: ignore


app = FastAPI(
    title="Evanesco Offline API",
    description="Redaction microservice built on top of the Evanesco pipeline.",
    version="0.1.3",
    openapi_tags=[
        {"name": "health", "description": "Service health and readiness probes."},
        {
            "name": "redaction",
            "description": "CRUD endpoints for managing redaction jobs and retrieving outputs.",
        },
    ],
)

if settings.cors_origins:
    allow_origins = ["*"] if "*" in settings.cors_origins else settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if REQUESTS:
    app.mount("/metrics", make_asgi_app())  # type: ignore


health_router = APIRouter(tags=["health"])
redaction_router = APIRouter(prefix="/jobs", tags=["redaction"])


def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme),
) -> None:
    """Simple bearer-token protection for managed cluster deployments."""

    token = settings.api_token
    if token is None:
        return
    if credentials is None or credentials.credentials != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


def _increment_metric(route: str) -> None:
    if REQUESTS:
        REQUESTS.labels(route=route).inc()  # type: ignore


def _resolve_output_path(
    input_path: str, job_id: str, explicit_output: Optional[str]
) -> str:
    if explicit_output:
        output = Path(explicit_output)
    else:
        artifacts_dir = Path.cwd() / "artifacts" / "redactions"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem or "document"
        output = artifacts_dir / f"{stem}_{job_id}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    return str(output)


def _parse_pages(raw_pages: List[Any]) -> List[PageResult]:
    pages: List[PageResult] = []
    for page in raw_pages:
        if isinstance(page, PageResult):
            pages.append(page)
        else:
            pages.append(PageResult.model_validate(page))
    return pages


def _execute_job(
    payload: RedactionJobCreate,
    *,
    job_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> RedactionJob:
    """Run the redaction pipeline and persist the resulting job record."""

    resolved_job_id = job_id or uuid4().hex
    now = datetime.utcnow()
    created_ts = created_at or now

    options = payload.options or RedactionOptions()
    input_path = Path(payload.input_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Input path not found"
        )

    output_path = _resolve_output_path(
        str(input_path), resolved_job_id, payload.output_path
    )
    job = RedactionJob(
        id=resolved_job_id,
        input_path=str(input_path),
        output_path=output_path,
        status=JobStatus.PENDING,
        created_at=created_ts,
        updated_at=now,
        options=options,
    )
    job_repository.save(job)

    try:
        cfg = options.to_run_config(settings)
        result_map = process_path(str(input_path), output_path, cfg)
        pages_raw = result_map.get("pages", [])
        pages = _parse_pages(pages_raw)
        debug_payload = result_map.get("debug") or {}
        debug_artifacts = {k: str(v) for k, v in debug_payload.items()}
        job = job.copy(
            update={
                "status": JobStatus.COMPLETED,
                "updated_at": datetime.utcnow(),
                "detail": None,
                "result": RedactionJobResult(
                    output_path=output_path,
                    page_count=len(pages),
                    debug_artifacts=debug_artifacts,
                    pages=pages,
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive path
        job = job.copy(
            update={
                "status": JobStatus.FAILED,
                "updated_at": datetime.utcnow(),
                "detail": str(exc),
            }
        )
    job_repository.save(job)
    return job


def _get_job_or_404(job_id: str) -> RedactionJob:
    try:
        return job_repository.get(job_id)
    except JobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )


@health_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    _increment_metric("health")
    return HealthResponse()


@health_router.get("/livez", response_model=HealthResponse)
def livez() -> HealthResponse:
    return HealthResponse(status="ok")


@health_router.get("/readyz", response_model=ReadyResponse)
def readyz():
    checks = run_readiness_checks(settings)
    ready = True
    payload: List[ReadinessCheckModel] = []
    for check in checks:
        payload.append(
            ReadinessCheckModel(
                name=check.name,
                status=check.status,
                detail=check.detail,
                required=check.required,
            )
        )
        if check.required and check.status == "fail":
            ready = False
        if (
            check.required
            and check.status == "warn"
            and not settings.allowance_warn_only_checks
        ):
            ready = False
    response = ReadyResponse(ready=ready, checks=payload)
    status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=response.model_dump())


@redaction_router.post(
    "", response_model=RedactionJob, status_code=status.HTTP_201_CREATED
)
def create_job(
    payload: RedactionJobCreate,
    auth: None = Depends(require_auth),
) -> RedactionJob:
    _increment_metric("jobs_create")
    return _execute_job(payload)


@redaction_router.get("", response_model=JobListResponse)
def list_jobs(auth: None = Depends(require_auth)) -> JobListResponse:
    jobs = job_repository.list()
    return JobListResponse(items=jobs, total=len(jobs))


@redaction_router.get("/{job_id}", response_model=RedactionJob)
def get_job(job_id: str, auth: None = Depends(require_auth)) -> RedactionJob:
    return _get_job_or_404(job_id)


@redaction_router.put("/{job_id}", response_model=RedactionJob)
def update_job(
    job_id: str,
    payload: RedactionJobUpdate,
    auth: None = Depends(require_auth),
) -> RedactionJob:
    existing = _get_job_or_404(job_id)
    updated_payload = RedactionJobCreate(
        input_path=existing.input_path,
        output_path=payload.output_path or existing.output_path,
        options=payload.options or existing.options,
    )
    return _execute_job(updated_payload, job_id=job_id, created_at=existing.created_at)


@redaction_router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str, auth: None = Depends(require_auth)) -> Response:
    job = _get_job_or_404(job_id)
    job_repository.delete(job_id)
    try:
        output_path = Path(job.output_path)
        if output_path.exists():
            output_path.unlink()
    except OSError:
        pass
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@redaction_router.get("/{job_id}/download")
def download_job(job_id: str, auth: None = Depends(require_auth)) -> FileResponse:
    job = _get_job_or_404(job_id)
    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No redacted output available"
        )
    path = Path(job.result.output_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Redacted file missing on disk",
        )
    return FileResponse(str(path), media_type="application/pdf", filename=path.name)


@redaction_router.post(
    "/upload", response_model=RedactionJob, status_code=status.HTTP_201_CREATED
)
async def create_job_from_upload(
    file: UploadFile = File(...),
    options: Optional[str] = Form(
        None,
        description="JSON-encoded redaction options",
        example='{"use_llm": false}',
    ),
    auth: None = Depends(require_auth),
) -> RedactionJob:
    _increment_metric("jobs_upload")
    raw_options = (options or "").strip()
    parsed_options: RedactionOptions
    if not raw_options or raw_options.lower() in {"null", "none", "string"}:
        parsed_options = RedactionOptions()
    else:
        try:
            parsed_options = RedactionOptions.model_validate_json(raw_options)
        except ValidationError as exc:  # pragma: no cover - validation bubble-up
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=json.loads(exc.json()),
            ) from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail='`options` must be valid JSON (e.g. {"use_llm": false})',
            ) from exc

    job_id = uuid4().hex
    uploads_root = Path.cwd() / "artifacts" / "uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)
    job_dir = uploads_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or "document"
    input_path = job_dir / filename
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    output_path = job_dir / f"{Path(filename).stem}.redacted.pdf"
    payload = RedactionJobCreate(
        input_path=str(input_path),
        output_path=str(output_path),
        options=parsed_options,
    )
    return _execute_job(payload, job_id=job_id)


@app.post(
    "/redact",
    response_model=RedactionJob,
    deprecated=True,
    summary="Legacy single-shot redaction endpoint",
)
async def legacy_redact(
    file: UploadFile = File(...),
    options: Optional[str] = Form(None),
    auth: None = Depends(require_auth),
) -> RedactionJob:
    """Compatibility shim around ``/jobs/upload`` for older clients."""

    return await create_job_from_upload(file=file, options=options, auth=auth)


app.include_router(health_router)
app.include_router(redaction_router)


def run(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """Launch the API server via ``uvicorn``."""

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
