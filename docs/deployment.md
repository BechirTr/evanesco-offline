---
title: Deployment
---

# Deployment on Kubernetes / OpenShift

Evanesco Offline now ships with container images and manifests that target
managed clusters. This page outlines the recommended approach for production
setups.

## Container Image

Build the image directly from the repository root:

```bash
docker build -t ghcr.io/your-org/evanesco-offline:latest .
```

The Dockerfile installs Poppler and Tesseract, exposes port `8000`, and runs the
FastAPI service under a non-root user. Mount `/data` if you need to persist
preview artefacts or audit logs beyond the container lifecycle.

## Runtime Environment Variables

The API reads configuration from environment variables; the ConfigMap template
(`deploy/kubernetes/configmap.yaml`) covers the common ones:

| Variable | Purpose |
| --- | --- |
| `EVANESCO_API_TOKEN` | Bearer token required for `/redact` requests. Stored in a Kubernetes Secret. |
| `EVANESCO_API_CORS_ORIGINS` | Comma separated allow-list for browsers (e.g. the UI front-end). |
| `EVANESCO_READY_CHECK_*` | Enable readiness validation for OCR, spaCy, or LLM dependencies. |
| `EVANESCO_READY_SPACY_MODEL` | Model name used during readiness probing (`en_core_web_lg` by default). |
| `EVANESCO_READY_TESS_LANGS` | Required Tesseract language packs (comma separated). |
| `EVANESCO_LLM_URL` | Override the Ollama endpoint when the model runs outside the pod. |
| `EVANESCO_READY_WARN_ONLY` | When `true`, warnings during readiness (e.g. missing spaCy NER) still report ready. |

If you rely on the audit HMAC signature, inject `EVANESCO_HMAC_KEY` via a
Secret and mount it alongside the API token.

## Probes

The API exposes:

- `GET /health` – lightweight status (still unauthenticated)
- `GET /livez` – liveness probe endpoint
- `GET /readyz` – readiness probe returning a JSON report; HTTP 503 on failures

Kubernetes manifests wire these endpoints into standard probes to ensure pods
only receive traffic when the runtime dependencies are satisfied.

## Kubernetes Manifests

The manifests under `deploy/kubernetes/` provide an opinionated starting point:

1. Create a namespace and apply the ServiceAccount:
   ```bash
   oc new-project pii-redaction
   oc apply -f deploy/kubernetes/serviceaccount.yaml
   ```
2. Apply the ConfigMap and Secret (replace the base64 placeholders first):
   ```bash
   oc apply -f deploy/kubernetes/configmap.yaml
   oc apply -f deploy/kubernetes/secret.yaml
   ```
3. Deploy the API and expose it:
   ```bash
   oc apply -f deploy/kubernetes/deployment.yaml
   oc apply -f deploy/kubernetes/service.yaml
   oc apply -f deploy/kubernetes/route.yaml
   ```

Adjust the image reference, resource requests, and replicas to match your
cluster sizing. The Deployment mounts an `emptyDir` at `/data`; replace this with
a PersistentVolumeClaim if you need durable artefacts.

## Network & TLS

OpenShift Routes terminate TLS by default. For alternative ingress solutions,
expose the Service through Ingress or a Service Mesh and terminate TLS at that
layer. When the LLM endpoint runs in a separate namespace/cluster, enable
mTLS or network policies to restrict traffic to the authorised service.

## Observability

Enable Prometheus scraping by annotating the Service:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
```

Logs are structured JSON; forward them to your aggregation stack (e.g. Loki or
Elastic) and correlate with `/metrics` counters for redaction throughput.

## Upgrade Workflow

1. Build and push the image.
2. Update the Deployment image tag and apply.
3. Observe `/readyz` until pods report ready and the Route shows success.
4. Optionally run `evanesco run` against a test document via the API to verify.

Rolling updates with `maxUnavailable: 0` ensure zero downtime while new pods
pass readiness checks before receiving traffic.
