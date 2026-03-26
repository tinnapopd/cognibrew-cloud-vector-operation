#!/usr/bin/env bash

set -x
set -eo pipefail

if ! [ -x "$(command -v docker)" ]; then
    echo >&2 "Error: Docker is not installed."
    exit 1
fi

# MLflow configuration
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_BACKEND_STORE=${MLFLOW_BACKEND_STORE:-/mlflow/mlruns}

# Launch MLflow using Docker
# Allow to skip docker if a dockerized MLflow is already running
# Use: SKIP_DOCKER=1 ./scripts/init_mlflow.sh
if [[ -z "${SKIP_DOCKER}" ]]; then
    # Remove any previous MLflow docker container
    docker rm -f mlflow || true
    docker run \
        --name mlflow \
        -p "${MLFLOW_PORT}":5000 \
        -d ghcr.io/mlflow/mlflow:latest \
        mlflow server \
            --host 0.0.0.0 \
            --port 5000 \
            --backend-store-uri "${MLFLOW_BACKEND_STORE}"
fi

# Keep pinging MLflow until it's ready
until curl -sf "http://localhost:${MLFLOW_PORT}/health" > /dev/null 2>&1; do
    >&2 echo "MLflow is still unavailable - sleeping"
    sleep 1
done

>&2 echo "MLflow is up and running on port ${MLFLOW_PORT}, ready to go!"
