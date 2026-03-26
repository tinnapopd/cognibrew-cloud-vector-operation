from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=False,
        env_ignore_empty=True,
        case_sensitive=False,
    )

    API_PREFIX_STR: str = "/api/v1"
    PROJECT_NAME: str = "CogniBrew Vector Operation"
    ENVIRONMENT: Literal["local", "staging", "production"] = "production"

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6334
    QDRANT_COLLECTION: str = "face_embeddings"
    EMBEDDING_DIM: int = 512

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_EXPERIMENT_NAME: str = "vector-evolution"

    # Gallery Update
    MAX_VECTORS_PER_USER: int = 10

    # Industry-standard heuristics based on how modern facial recognition
    # models (like ArcFace, InsightFace, or FaceNet)
    UPPER_SIMILARITY_THRESHOLD: float = 0.92
    LOWER_SIMILARITY_THRESHOLD: float = 0.75
    # EMA smoothing factor: weight given to the incoming avg_vector
    # 0.1 = slow adaptation, 0.3 = faster adaptation
    EMA_ALPHA: float = 0.1

    # Per-device threshold calibration
    # Minimum number of MLflow runs needed before ROC calibration is applied.
    # If fewer runs exist for a device_id, the fallback threshold is returned.
    MIN_CALIBRATION_SAMPLES: int = 10
    # Returned when there is insufficient data to calibrate a device.
    FALLBACK_SIMILARITY_THRESHOLD: float = 0.5


settings = Settings()  # type: ignore
