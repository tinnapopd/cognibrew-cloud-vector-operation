import pandas as pd
from fastapi import APIRouter
import mlflow
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity

from app.api.deps import QdrantDep
from app.core.config import settings
from app.core.qdrant import (
    get_user_baselines,
    update_vector,
    upsert_vector,
    get_vectors_by_usernames,
)
from app.models.schemas import (
    UpdateUserBaselineRequest,
    UpdateUserBaselineResponse,
    ThresholdResponse,
    DeviceVectorEntry,
    GetVectorsByDeviceIdResponse,
)

router = APIRouter(prefix="/vectors", tags=["vectors"])

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)


@router.post(
    "/update/user-baseline",
    response_model=UpdateUserBaselineResponse,
)
async def update_user_baseline(
    body: UpdateUserBaselineRequest,
    qdrant: QdrantDep,
) -> UpdateUserBaselineResponse:
    filter_vector = [v for v in body.vectors if v.is_correct]
    if not filter_vector:
        return UpdateUserBaselineResponse(
            status="skipped",
            action="skipped",
            username=body.username,
            max_similarity=0.0,
        )

    # Compute average vector from correct vectors
    avg_vector = np.mean([v.embedding for v in filter_vector], axis=0).astype(
        np.float32
    )
    # Get user's baselines - list[np.ndarray]
    baselines = get_user_baselines(qdrant, body.username)

    if not baselines:
        upsert_vector(
            qdrant,
            username=body.username,
            embedding=avg_vector.tolist(),
            is_correct=True,
        )
        return UpdateUserBaselineResponse(
            status="ok",
            action="bootstrap",
            username=body.username,
            max_similarity=0.0,
        )

    # Compute cosine similarity against every baseline
    avg_2d = avg_vector.reshape(1, -1)
    similarities = [
        float(cosine_similarity(avg_2d, b.reshape(1, -1))[0][0])
        for b in baselines
    ]
    max_similarity = max(similarities)
    best_idx = int(np.argmax(similarities))

    # Branch 1: High similarity: Update via EMA. It's the same angle/lighting,
    # just a slightly different day.
    if max_similarity > settings.UPPER_SIMILARITY_THRESHOLD:
        alpha = settings.EMA_ALPHA
        updated = (1 - alpha) * baselines[best_idx] + alpha * avg_vector
        update_vector(qdrant, body.username, best_idx, updated.tolist())
        action = "ema_update"

    # Branch 2: New look, fill empty slot. This is likely a side profile or
    # different lighting.
    elif max_similarity > settings.LOWER_SIMILARITY_THRESHOLD:
        if len(baselines) < settings.MAX_VECTORS_PER_USER:
            upsert_vector(
                qdrant,
                username=body.username,
                embedding=avg_vector.tolist(),
                is_correct=True,
            )
        action = "new_look"

    # Branch 3: Discard. Even if confirmed by the barista, the image might be
    # too blurry/dark to be a useful template.
    else:
        action = "discard"

    try:
        with mlflow.start_run(run_name=f"process-{body.username}"):
            mlflow.log_param("username", body.username)
            mlflow.log_param("device_id", body.device_id)
            mlflow.log_param("action", action)
            mlflow.log_metric("max_similarity", round(max_similarity, 4))
            mlflow.log_metric("baseline_vectors", len(baselines))
            mlflow.log_metric("input_vectors", len(filter_vector))
    except Exception:
        pass

    return UpdateUserBaselineResponse(
        status="ok",
        action=action,
        username=body.username,
        max_similarity=round(max_similarity, 4),
    )


@router.get("/threshold/{device_id}", response_model=ThresholdResponse)
async def get_device_threshold(device_id: str) -> ThresholdResponse:
    """Return the optimal cosine-similarity threshold for a given device_id.

    Queries all MLflow runs tagged with the device_id, extracts
    (max_similarity, label) pairs, and computes the threshold that maximises
    Youden's J (sensitivity + specificity - 1) on the ROC curve.

    Falls back to FALLBACK_SIMILARITY_THRESHOLD (0.5) when there are fewer
    than MIN_CALIBRATION_SAMPLES runs available for the device.
    """
    # Positive actions = person was successfully matched / enrolled.
    # "discard" = similarity too low, treated as a negative / no-match.
    POSITIVE_ACTIONS = {"ema_update", "new_look", "bootstrap"}

    experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return ThresholdResponse(
            device_id=device_id,
            optimal_threshold=settings.FALLBACK_SIMILARITY_THRESHOLD,
            sample_count=0,
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.device_id = '{device_id}'",
        output_format="pandas",
    )

    if not isinstance(runs, pd.DataFrame):
        runs = pd.DataFrame(runs)

    # Keep rows that have the data we need
    required_cols = {"params.action", "metrics.max_similarity"}
    available_cols = set(runs.columns)
    if required_cols - available_cols or runs.empty:
        return ThresholdResponse(
            device_id=device_id,
            optimal_threshold=settings.FALLBACK_SIMILARITY_THRESHOLD,
            sample_count=0,
        )

    runs = runs.dropna(subset=["params.action", "metrics.max_similarity"])
    sample_count = len(runs)

    if sample_count < settings.MIN_CALIBRATION_SAMPLES:
        return ThresholdResponse(
            device_id=device_id,
            optimal_threshold=settings.FALLBACK_SIMILARITY_THRESHOLD,
            sample_count=sample_count,
        )

    y_scores = runs["metrics.max_similarity"].tolist()
    y_true = [
        1 if action in POSITIVE_ACTIONS else 0
        for action in runs["params.action"].tolist()
    ]

    # Need both classes present for ROC analysis
    if len(set(y_true)) < 2:
        return ThresholdResponse(
            device_id=device_id,
            optimal_threshold=settings.FALLBACK_SIMILARITY_THRESHOLD,
            sample_count=sample_count,
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    optimal_threshold = round(float(thresholds[best_idx]), 4)

    return ThresholdResponse(
        device_id=device_id,
        optimal_threshold=optimal_threshold,
        sample_count=sample_count,
    )


@router.get("/{device_id}", response_model=GetVectorsByDeviceIdResponse)
async def get_vectors_by_device_id(
    device_id: str,
    qdrant: QdrantDep,
) -> GetVectorsByDeviceIdResponse:
    """Return all vector records stored in Qdrant for every user
    associated with *device_id*.

    Looks up MLflow runs to discover which usernames have been logged
    against the device, then fetches their baseline vectors from Qdrant.
    """
    experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return GetVectorsByDeviceIdResponse(
            device_id=device_id,
            users=[],
            total_users=0,
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.device_id = '{device_id}'",
        output_format="pandas",
    )

    if not isinstance(runs, pd.DataFrame):
        runs = pd.DataFrame(runs)

    if runs.empty or "params.username" not in runs.columns:
        return GetVectorsByDeviceIdResponse(
            device_id=device_id, users=[], total_users=0
        )

    usernames = runs["params.username"].dropna().unique().tolist()
    users_data = get_vectors_by_usernames(qdrant, usernames)
    entries = [
        DeviceVectorEntry(
            username=username,
            vectors=vectors,
            vector_count=len(vectors),
        )
        for username, vectors in users_data.items()
    ]

    return GetVectorsByDeviceIdResponse(
        device_id=device_id,
        users=entries,
        total_users=len(entries),
    )
