import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import mlflow
import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.core.config import settings
from app.core.engine import detect_drift
from app.core.qdrant import (
    ensure_collection,
    get_all_usernames,
    get_user_baseline,
    get_user_vectors,
    upsert_vectors,
)

router = APIRouter(prefix="/vectors", tags=["vectors"])

# Single shared thread-pool — Qdrant client is synchronous
_executor = ThreadPoolExecutor(max_workers=8)

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

_IQR_FACTOR = settings.OUTLIER_IQR_FACTOR


def _run_sync(fn, *args, **kwargs):
    """Offload a synchronous function to the executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


class VectorRecord(BaseModel):
    username: str
    embedding: list[float]
    is_correct: bool = True
    is_fallback: bool = False


class ProcessBatchRequest(BaseModel):
    vectors: list[VectorRecord] = []


class ExpandGalleryRequest(BaseModel):
    vectors: list[VectorRecord] = []


def _filter_outliers_iqr(distances: np.ndarray) -> np.ndarray:
    """Return a boolean mask; True = keep."""
    if len(distances) < 4:
        return np.ones(len(distances), dtype=bool)
    q1 = float(np.percentile(distances, 25))
    q3 = float(np.percentile(distances, 75))
    iqr = q3 - q1
    lower = q1 - _IQR_FACTOR * iqr
    upper = q3 + _IQR_FACTOR * iqr
    return (distances >= lower) & (distances <= upper)


def _cosine_distances(
    vectors: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    ref_norm = reference / (np.linalg.norm(reference) + 1e-12)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return 1.0 - (vectors / norms) @ ref_norm


@router.post("/process-batch")
async def process_batch(body: ProcessBatchRequest) -> dict:
    """Analyse incoming embeddings: drift detection +
    outlier filtering + gallery upsert.

    Called by the Airflow ``process_vectors`` task:
        POST /vectors/process-batch
    """
    await _run_sync(ensure_collection)

    by_user: dict[str, list[VectorRecord]] = defaultdict(list)
    for v in body.vectors:
        by_user[v.username].append(v)

    total_accepted = 0
    total_rejected = 0
    drift_users: list[str] = []

    for username, records in by_user.items():
        embeddings = np.array([r.embedding for r in records], dtype=np.float32)
        baseline = await _run_sync(get_user_baseline, username)
        existing = await _run_sync(
            get_user_vectors, username, with_vectors=False
        )
        gallery_size = len(existing)

        if baseline is None:
            # First-ever vectors — first one becomes baseline
            to_upsert = [
                (
                    username,
                    emb.tolist(),
                    "baseline" if i == 0 else "temporal",
                    records[i].is_correct,
                    records[i].is_fallback,
                )
                for i, emb in enumerate(embeddings)
            ]
            await _run_sync(upsert_vectors, to_upsert)
            total_accepted += len(to_upsert)
            mean_drift, max_drift, is_drifting = 0.0, 0.0, False
        else:
            distances = _cosine_distances(embeddings, baseline)
            mask = _filter_outliers_iqr(distances)
            kept = embeddings[mask]

            to_upsert = [
                (
                    username,
                    emb.tolist(),
                    "temporal",
                    records[i].is_correct,
                    records[i].is_fallback,
                )
                for i, emb in enumerate(kept)
            ]
            if to_upsert:
                await _run_sync(upsert_vectors, to_upsert)

            accepted = int(mask.sum())
            rejected = int((~mask).sum())
            total_accepted += accepted
            total_rejected += rejected

            drift_result = detect_drift(
                username=username,
                gallery_vectors=kept if len(kept) > 0 else embeddings,
                baseline=baseline,
                gallery_size=gallery_size + accepted,
            )
            mean_drift = drift_result.mean_drift
            max_drift = drift_result.max_drift
            is_drifting = drift_result.is_drifting

            if is_drifting:
                drift_users.append(username)

            # MLflow per-user drift
            try:
                with mlflow.start_run(
                    run_name=f"drift-{username}", nested=True
                ):
                    mlflow.log_param("username", username)
                    mlflow.log_metric("mean_drift", mean_drift)
                    mlflow.log_metric("max_drift", max_drift)
                    mlflow.log_metric("is_drifting", int(is_drifting))
                    mlflow.log_metric(
                        "gallery_size", drift_result.gallery_size
                    )
            except Exception:
                pass

    # MLflow outlier summary
    try:
        with mlflow.start_run(run_name="outlier-filter", nested=True):
            total_input = total_accepted + total_rejected
            mlflow.log_metric("total_input", total_input)
            mlflow.log_metric("accepted", total_accepted)
            mlflow.log_metric("rejected", total_rejected)
            if total_input > 0:
                mlflow.log_metric(
                    "rejection_rate", total_rejected / total_input
                )
    except Exception:
        pass

    return {
        "accepted": total_accepted,
        "rejected": total_rejected,
        "drift_users": drift_users,
    }


@router.post("/expand-gallery")
async def expand_gallery(body: ExpandGalleryRequest) -> dict:
    """Add fallback-verified vectors as secondary anchors.

    Called by the Airflow ``gallery_expansion`` task:
        POST /vectors/expand-gallery
    """
    if not body.vectors:
        return {"upserted": 0}

    to_upsert = [
        (
            v.username,
            v.embedding,
            "secondary",
            v.is_correct,
            v.is_fallback,
        )
        for v in body.vectors
    ]
    count = await _run_sync(upsert_vectors, to_upsert)

    # MLflow per-user gallery expansion
    by_user: dict[str, int] = defaultdict(int)
    for v in body.vectors:
        by_user[v.username] += 1

    for username, added in by_user.items():
        existing = await _run_sync(
            get_user_vectors, username, with_vectors=False
        )
        try:
            with mlflow.start_run(
                run_name=f"gallery-expand-{username}", nested=True
            ):
                mlflow.log_param("username", username)
                mlflow.log_metric("vectors_added", added)
                mlflow.log_metric("new_gallery_size", len(existing))
        except Exception:
            pass

    return {"upserted": count, "users": list(by_user.keys())}


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
