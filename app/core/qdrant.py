import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from app.core.config import settings


def client() -> QdrantClient:
    """Return a Qdrant gRPC client."""
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        prefer_grpc=True,
    )


def init_collection() -> None:
    """Create the face-embeddings collection if it doesn't exist."""
    c = client()
    existing = {col.name for col in c.get_collections().collections}
    if settings.QDRANT_COLLECTION not in existing:
        c.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


def _user_filter(username: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="username", match=MatchValue(value=username))]
    )


def get_user_vectors(
    username: str, *, with_vectors: bool = True, limit: int = 1000
) -> list[dict]:
    """Retrieve all gallery vectors for a user.

    Returns list of dicts: {point_id, username, anchor_type, timestamp, embedding?}.
    """
    results = client().scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=_user_filter(username),
        with_vectors=with_vectors,
        limit=limit,
    )[0]

    out = []
    for pt in results:
        if pt.payload is None:
            continue
        rec: dict = {
            "point_id": str(pt.id),
            "username": pt.payload["username"],
            "anchor_type": pt.payload["anchor_type"],
            "timestamp": pt.payload.get("timestamp", ""),
        }
        if with_vectors and isinstance(pt.vector, list):
            rec["embedding"] = pt.vector
        out.append(rec)

    return out


def get_user_baseline(username: str) -> np.ndarray | None:
    """Return the baseline vector for a user, or None if not found."""
    filt = Filter(
        must=[
            FieldCondition(key="username", match=MatchValue(value=username)),
            FieldCondition(
                key="anchor_type", match=MatchValue(value="baseline")
            ),
        ]
    )
    results = client().scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=filt,
        with_vectors=True,
        limit=1,
    )[0]
    if not results:
        return None
    vec = results[0].vector
    if not isinstance(vec, list):
        return None
    return np.array(vec, dtype=np.float32)


def get_all_usernames() -> list[str]:
    """Return distinct usernames in the gallery."""
    results = client().scroll(
        collection_name=settings.QDRANT_COLLECTION,
        with_vectors=False,
        limit=10_000,
    )[0]
    return list(
        {pt.payload["username"] for pt in results if pt.payload is not None}
    )
