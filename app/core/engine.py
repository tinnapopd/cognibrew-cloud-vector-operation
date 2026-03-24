from dataclasses import dataclass

import numpy as np

from app.core.config import settings

_IQR_FACTOR = 1.5


def cosine_distances_to_reference(
    vectors: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Cosine distances of each row in *vectors* to *reference*."""
    ref_norm = reference / (np.linalg.norm(reference) + 1e-12)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    normed = vectors / norms
    sims = normed @ ref_norm
    return 1.0 - sims


@dataclass
class DriftResult:
    username: str
    mean_drift: float
    max_drift: float
    is_drifting: bool
    gallery_size: int


def detect_drift(
    username: str,
    gallery_vectors: np.ndarray,
    baseline: np.ndarray,
    gallery_size: int,
) -> DriftResult:
    """Measure cosine distance of gallery vectors from baseline."""
    if len(gallery_vectors) == 0:
        return DriftResult(
            username=username,
            mean_drift=0.0,
            max_drift=0.0,
            is_drifting=False,
            gallery_size=gallery_size,
        )

    distances = cosine_distances_to_reference(gallery_vectors, baseline)
    mean_d = float(np.mean(distances))
    max_d = float(np.max(distances))

    return DriftResult(
        username=username,
        mean_drift=mean_d,
        max_drift=max_d,
        is_drifting=mean_d > settings.DRIFT_THRESHOLD,
        gallery_size=gallery_size,
    )
