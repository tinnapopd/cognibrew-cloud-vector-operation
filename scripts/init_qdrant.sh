#!/usr/bin/env bash

set -x
set -eo pipefail

if ! [ -x "$(command -v docker)" ]; then
    echo >&2 "Error: Docker is not installed."
    exit 1
fi

# Qdrant configuration
QDRANT_PORT=${QDRANT_PORT:-6333}
QDRANT_GRPC_PORT=${QDRANT_GRPC_PORT:-6334}

# Launch Qdrant using Docker
# Allow to skip docker if a dockerized Qdrant is already running
# Use: SKIP_DOCKER=1 ./scripts/init_qdrant.sh
if [[ -z "${SKIP_DOCKER}" ]]; then
    # Remove any previous Qdrant docker container
    docker rm -f qdrant || true
    docker run \
        --name qdrant \
        -p "${QDRANT_PORT}":6333 \
        -p "${QDRANT_GRPC_PORT}":6334 \
        -d qdrant/qdrant:latest
fi

# Keep pinging Qdrant until it's ready
until curl -sf "http://localhost:${QDRANT_PORT}/readyz" > /dev/null 2>&1; do
    >&2 echo "Qdrant is still unavailable - sleeping"
    sleep 1
done

>&2 echo "Qdrant is up and running on port ${QDRANT_PORT}, ${QDRANT_GRPC_PORT}, ready to go!"
