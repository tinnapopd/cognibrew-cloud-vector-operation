import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from fastapi import APIRouter, Query

from app.core.engine import detect_drift
from app.core.qdrant import (
    get_all_usernames,
    get_user_baseline,
    get_user_vectors,
)

router = APIRouter(prefix="/vectors", tags=["vectors"])

# Single shared thread-pool — Qdrant client is synchronous
_executor = ThreadPoolExecutor(max_workers=8)


def _run_sync(fn, *args, **kwargs):
    """Offload a synchronous function to the executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


@router.get("/{username}/gallery")
async def get_gallery(
    username: str,
    include_embeddings: bool = Query(
        False, description="Include 512-dim embedding arrays in response"
    ),
) -> dict:
    """Return the vector gallery for a specific user.

    This is consumed by ``cognibrew-cloud-edge-sync`` to build the sync bundle:
        GET /vectors/{username}/gallery?include_embeddings=true
    """
    raw = await _run_sync(
        get_user_vectors, username, with_vectors=include_embeddings
    )

    baseline_count = sum(1 for v in raw if v["anchor_type"] == "baseline")
    secondary_count = sum(1 for v in raw if v["anchor_type"] == "secondary")
    temporal_count = sum(1 for v in raw if v["anchor_type"] == "temporal")

    return {
        "username": username,
        "total_vectors": len(raw),
        "baseline_count": baseline_count,
        "secondary_count": secondary_count,
        "temporal_count": temporal_count,
        "vectors": raw,
    }


@router.get("/drift-signals")
async def get_drift_signals(
    username: Optional[str] = Query(
        None, description="Compute drift for a single user only"
    ),
) -> dict:
    """Return drift telemetry for all users (or one user).

    Consumed by ``cognibrew-cloud-edge-sync`` to build the user list before
    paginating galleries:
        GET /vectors/drift-signals
    """
    usernames = [username] if username else await _run_sync(get_all_usernames)

    if not usernames:
        return {"signals": [], "global_mean_drift": 0.0}

    async def _drift_for_user(u: str) -> dict | None:
        baseline = await _run_sync(get_user_baseline, u)
        if baseline is None:
            return None
        vecs_raw = await _run_sync(get_user_vectors, u, with_vectors=True)
        if not vecs_raw:
            return None
        embeddings = np.array(
            [v["embedding"] for v in vecs_raw if v.get("embedding")],
            dtype=np.float32,
        )
        if len(embeddings) == 0:
            return None
        result = detect_drift(
            username=u,
            gallery_vectors=embeddings,
            baseline=baseline,
            gallery_size=len(vecs_raw),
        )
        return {
            "username": result.username,
            "mean_drift": result.mean_drift,
            "max_drift": result.max_drift,
            "gallery_size": result.gallery_size,
            "is_drifting": result.is_drifting,
        }

    results = await asyncio.gather(*[_drift_for_user(u) for u in usernames])
    signals = [r for r in results if r is not None]

    global_mean = (
        float(np.mean([s["mean_drift"] for s in signals])) if signals else 0.0
    )

    return {"signals": signals, "global_mean_drift": global_mean}
