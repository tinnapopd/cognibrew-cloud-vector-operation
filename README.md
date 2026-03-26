# CogniBrew Cloud Vector Operation

API service for managing face-embedding baselines in Qdrant and calibrating per-device similarity thresholds via MLflow. Consumed by `cognibrew-cloud-edge-sync`.

On startup the service automatically creates the `face_embeddings` collection in Qdrant if it does not already exist.

## API Endpoints

| Method | Path                                    | Description                                          |
| ------ | --------------------------------------- | ---------------------------------------------------- |
| `POST` | `/api/v1/vectors/update/user-baseline`  | Ingest embeddings and update user's baseline vectors |
| `GET`  | `/api/v1/vectors/threshold/{device_id}` | Get optimal similarity threshold for a device        |
| `GET`  | `/api/v1/vectors/{device_id}`           | Get all baseline vectors for a device across users   |
| `GET`  | `/api/v1/utils/health-check/`           | Health check                                         |

### Update User Baseline

```bash
curl -X POST "http://localhost:8000/api/v1/vectors/update/user-baseline" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "device_id": "cam-01",
    "vectors": [
      {"embedding": [0.1, 0.2, ...], "is_correct": true}
    ]
  }'
```

**Response:**

```json
{
  "status": "ok",
  "action": "ema_update",
  "username": "alice",
  "max_similarity": 0.9512
}
```

> `action` is one of: `bootstrap`, `ema_update`, `new_look`, `discard`, or `skipped`.

### Get Device Threshold

```bash
curl "http://localhost:8000/api/v1/vectors/threshold/cam-01"
```

**Response:**

```json
{
  "device_id": "cam-01",
  "optimal_threshold": 0.8734,
  "sample_count": 42
}
```

> Falls back to `FALLBACK_SIMILARITY_THRESHOLD` when fewer than `MIN_CALIBRATION_SAMPLES` runs exist for the device.

### Get Vectors by Device

```bash
curl "http://localhost:8000/api/v1/vectors/cam-01"
```

**Response:**

```json
{
  "device_id": "cam-01",
  "users": [
    {
      "username": "alice",
      "vectors": [[0.1, 0.2, ...], [0.15, 0.25, ...]],
      "vector_count": 2
    }
  ],
  "total_users": 1
}
```

> Returns all baseline vectors collected per device across all users.

## Project Structure

```
.github/workflows/
└── ci.yml                  # Lint + Docker build & push
app/
├── api/
│   ├── deps.py             # Shared dependencies (placeholder)
│   ├── main.py             # API router assembly
│   └── routes/
│       ├── vectors.py      # Vector endpoints + MLflow calibration logic
│       └── utils.py        # Health check
├── core/
│   ├── config.py           # Settings (pydantic-settings) - 16 config parameters
│   ├── logger.py           # JSON singleton logger
│   ├── qdrant.py           # Qdrant client initialization with CRUD operations
│   └── security.py         # Security notes (placeholder - TLS at infrastructure level)
├── models/
│   ├── __init__.py
│   └── schemas.py          # Pydantic request/response schemas
├── __init__.py
├── main.py                 # FastAPI application entry point with lifespan context
├── pre_start.py            # Pre-startup hook (placeholder)
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
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  cognibrew-cloud-vector-operation
```

**3. Run without Docker:**

```bash
pip install -r requirements.txt
QDRANT_HOST=localhost MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn app.main:app --reload --port 8000
```

### Open API Docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger documentation.

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

| Job              | Trigger                            | Description                                      |
| ---------------- | ---------------------------------- | ------------------------------------------------ |
| **Lint**         | PR to `main`, push to `main`, tags | Runs [Ruff](https://docs.astral.sh/ruff/) linter |
| **Build & Push** | Tags matching `v*`                 | Builds Docker image and pushes to Docker Hub     |

### Image Tags

```
<DOCKERHUB_USERNAME>/actions:cognibrew-cloud-vector-operation-v1.0.0-abc1234
<DOCKERHUB_USERNAME>/actions:cognibrew-cloud-vector-operation-latest
```

### Required Secrets

| Secret               | Description             |
| -------------------- | ----------------------- |
| `DOCKERHUB_USERNAME` | Docker Hub username     |
| `DOCKERHUB_TOKEN`    | Docker Hub access token |

## Environment Variables

See [`.env.example`](.env.example) for all available configuration options.

| Variable                        | Default                      | Description                                          |
| ------------------------------- | ---------------------------- | ---------------------------------------------------- |
| `LOG_LEVEL`                     | `INFO`                       | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`   |
| `ENVIRONMENT`                   | `production`                 | `local`, `staging`, or `production`                  |
| `API_PREFIX_STR`                | `/api/v1`                    | API route prefix                                     |
| `PROJECT_NAME`                  | `CogniBrew Vector Operation` | OpenAPI title                                        |
| `QDRANT_HOST`                   | `qdrant`                     | Qdrant service hostname                              |
| `QDRANT_PORT`                   | `6334`                       | Qdrant gRPC port                                     |
| `QDRANT_COLLECTION`             | `face_embeddings`            | Qdrant collection name                               |
| `EMBEDDING_DIM`                 | `512`                        | Embedding vector dimension                           |
| `MLFLOW_TRACKING_URI`           | `http://mlflow:5000`         | MLflow tracking server URL                           |
| `MLFLOW_EXPERIMENT_NAME`        | `vector-evolution`           | MLflow experiment name                               |
| `MAX_VECTORS_PER_USER`          | `10`                         | Maximum baseline vectors stored per user             |
| `UPPER_SIMILARITY_THRESHOLD`    | `0.92`                       | EMA update triggered above this similarity           |
| `LOWER_SIMILARITY_THRESHOLD`    | `0.75`                       | New-look slot filled above this similarity           |
| `EMA_ALPHA`                     | `0.1`                        | EMA smoothing factor (0 = slow, 1 = fast)            |
| `MIN_CALIBRATION_SAMPLES`       | `10`                         | Minimum MLflow runs before ROC calibration           |
| `FALLBACK_SIMILARITY_THRESHOLD` | `0.65`                       | Threshold used when calibration data is insufficient |
