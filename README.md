# CogniBrew Cloud Vector Operation

Read-only API exposing the Qdrant face-embedding gallery and drift telemetry produced by the Airflow vector-operation DAG. Consumed by `cognibrew-cloud-edge-sync`.

On startup the service automatically creates the `face_embeddings` collection in Qdrant if it does not already exist.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/vectors/{username}/gallery` | Face-embedding gallery for a user |
| `GET` | `/api/v1/vectors/drift-signals` | Drift telemetry for all (or one) user |
| `GET` | `/api/v1/utils/health-check/` | Health check |

### Get Gallery

```bash
# Without embeddings (metadata only)
curl "http://localhost:8000/api/v1/vectors/alice/gallery"

# With 512-dim embedding arrays
curl "http://localhost:8000/api/v1/vectors/alice/gallery?include_embeddings=true"
```

**Response:**

```json
{
  "username": "alice",
  "total_vectors": 12,
  "baseline_count": 1,
  "secondary_count": 6,
  "temporal_count": 5,
  "vectors": [...]
}
```

### Get Drift Signals

```bash
# All users
curl "http://localhost:8000/api/v1/vectors/drift-signals"

# Single user
curl "http://localhost:8000/api/v1/vectors/drift-signals?username=alice"
```

**Response:**

```json
{
  "signals": [
    {
      "username": "alice",
      "mean_drift": 0.08,
      "max_drift": 0.14,
      "gallery_size": 12,
      "is_drifting": false
    }
  ],
  "global_mean_drift": 0.08
}
```

## Project Structure

```
.github/workflows/
└── ci.yml                  # Lint + Docker build & push
app/
├── api/
│   ├── deps.py             # Shared dependencies (placeholder)
│   ├── main.py             # API router assembly
│   └── routes/
│       ├── vectors.py      # Gallery & drift endpoints
│       └── utils.py        # Health check
├── core/
│   ├── config.py           # Settings (pydantic-settings)
│   ├── engine.py           # Drift detection logic
│   ├── logger.py           # JSON singleton logger
│   ├── qdrant.py           # Qdrant client + collection init
│   └── security.py         # Security notes (placeholder)
├── models/                 # Pydantic schemas (placeholder)
├── main.py                 # FastAPI application entry point
├── pre_start.py            # Pre-startup script
└── utils.py                # Utility functions (placeholder)
scripts/
├── init_qdrant.sh          # Local dev: spin up Qdrant container
├── init_db.sh              # Local dev: spin up Postgres container
└── prestart.sh             # Docker container pre-start hook
```

## Development Setup

### Prerequisites

- Docker
- Python 3.10+
- `curl` (for `init_qdrant.sh` readiness check)

### Run Locally

**1. Start Qdrant:**

```bash
bash scripts/init_qdrant.sh
```

**2. Run the API with Docker:**

```bash
docker build -t cognibrew-cloud-vector-operation .
docker run --name vector-operation \
  -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e ENVIRONMENT=local \
  -e QDRANT_HOST=host.docker.internal \
  cognibrew-cloud-vector-operation
```

**3. Run without Docker:**

```bash
pip install -r requirements.txt
QDRANT_HOST=localhost uvicorn app.main:app --reload --port 8000
```

### Open API Docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger documentation.

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

| Job | Trigger | Description |
|-----|---------|-------------|
| **Lint** | PR to `main`, push to `main`, tags | Runs [Ruff](https://docs.astral.sh/ruff/) linter |
| **Build & Push** | Tags matching `v*` | Builds Docker image and pushes to Docker Hub |

### Image Tags

```
<DOCKERHUB_USERNAME>/actions:cognibrew-cloud-vector-operation-v1.0.0-abc1234
<DOCKERHUB_USERNAME>/actions:cognibrew-cloud-vector-operation-latest
```

### Required Secrets

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

## Environment Variables

See [`.env.example`](.env.example) for all available configuration options.

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `ENVIRONMENT` | `production` | `local`, `staging`, or `production` |
| `API_PREFIX_STR` | `/api/v1` | API route prefix |
| `PROJECT_NAME` | `CogniBrew Vector Operation` | OpenAPI title |
| `QDRANT_HOST` | `qdrant` | Qdrant service hostname |
| `QDRANT_PORT` | `6334` | Qdrant gRPC port |
| `QDRANT_COLLECTION` | `face_embeddings` | Qdrant collection name |
| `EMBEDDING_DIM` | `512` | Embedding vector dimension |
| `DRIFT_THRESHOLD` | `0.15` | Cosine distance threshold for drift detection |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow tracking server URL |
| `MLFLOW_EXPERIMENT_NAME` | `vector-evolution` | MLflow experiment name |
| `OUTLIER_IQR_FACTOR` | `1.5` | IQR multiplier for outlier filtering |
