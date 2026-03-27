from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient

from app.core.config import settings

client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    prefer_grpc=True,
)


def get_qdrant() -> Generator[QdrantClient, None, None]:
    yield client


# Dependency for FastAPI
QdrantDep = Annotated[QdrantClient, Depends(get_qdrant)]
