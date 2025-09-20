# syntax=docker/dockerfile:1.4

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip \
    && pip install --no-cache-dir .[server]

# Create a non-root user that still works with OpenShift random UID constraints
RUN groupadd -r evanesco && useradd -r -g evanesco evanesco \
    && mkdir -p /data \
    && chown -R evanesco:evanesco /app /data \
    && chmod -R g+rwX /app /data

USER evanesco

EXPOSE 8000

ENV EVANESCO_DATA_DIR=/data

CMD ["uvicorn", "evanesco.api:app", "--host", "0.0.0.0", "--port", "8000"]
