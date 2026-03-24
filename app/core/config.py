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

    # Drift detection threshold (cosine distance)
    DRIFT_THRESHOLD: float = 0.15

    # Logging
    LOG_LEVEL: str = "INFO"


settings = Settings()  # type: ignore
