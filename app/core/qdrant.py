import uuid
from datetime import datetime, timezone

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    PointVectors,
    VectorParams,
)

from app.core.config import settings


def qdrant_client() -> QdrantClient:
    """Return a Qdrant gRPC client."""
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        prefer_grpc=True,
    )


def init_collection() -> None:
    client = qdrant_client()
    existing = {col.name for col in client.get_collections().collections}
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


def get_user_baselines(username: str) -> list[np.ndarray]:
    """Return the baseline vectors for a user, or empty list if not found."""
    client = qdrant_client()
    results = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="username", match=MatchValue(value=username)
                )
            ]
        ),
        with_vectors=True,
        limit=settings.MAX_VECTORS_PER_USER,
    )[0]

    baselines: list[np.ndarray] = []
    for pt in results:
        raw = pt.vector
        if not isinstance(raw, list):
            continue

        baselines.append(np.array(raw, dtype=np.float32))

    return baselines


def update_vector(
    username: str,
    baseline_index: int,
    new_vector: list[float],
) -> None:
    client = qdrant_client()
    results = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="username", match=MatchValue(value=username)
                )
            ]
        ),
        with_vectors=False,
        limit=settings.MAX_VECTORS_PER_USER,
    )[0]

    if baseline_index >= len(results):
        return None

    point_id = results[baseline_index].id
    client.update_vectors(
        collection_name=settings.QDRANT_COLLECTION,
        points=[PointVectors(id=point_id, vector=new_vector)],
    )


def upsert_vector(
    username: str,
    embedding: list[float],
    is_correct: bool,
) -> None:
    client = qdrant_client()
    now = datetime.now(timezone.utc).isoformat()
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "username": username,
                    "is_correct": is_correct,
                    "timestamp": now,
                },
            )
        ],
    )


def get_vectors_by_usernames(
    usernames: list[str],
) -> dict[str, list[list[float]]]:

    client = qdrant_client()
    result = {}
    for username in usernames:
        points = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="username", match=MatchValue(value=username)
                    )
                ]
            ),
            with_vectors=True,
            limit=settings.MAX_VECTORS_PER_USER,
        )[0]

        vectors = []
        for pt in points:
            raw = pt.vector
            if isinstance(raw, list) and raw:
                vectors.append(
                    [float(v) for v in raw if isinstance(v, (int, float))]
                )

        if vectors:
            result[username] = vectors

    return result
