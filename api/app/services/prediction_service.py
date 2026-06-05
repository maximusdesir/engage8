"""Run/pass prediction service.

Reuses the trained LightGBM artifact from the sibling ``ml`` project. The
artifact is loaded once (lazily, on first request) and cached for the lifetime
of the process so we never pay the joblib load cost per request.
"""
from __future__ import annotations

import sys
from functools import lru_cache

from fastapi import HTTPException, status

from app.config import settings

# Make the engage8 ml package importable. settings.ml_root points at ../ml.
_ml_root = str(settings.ml_root)
if _ml_root not in sys.path:
    sys.path.insert(0, _ml_root)

from engage8.predict import load_model, predict_situation  # noqa: E402

# Request fields we forward to predict_situation. Anything None is dropped so
# the ml layer can fall back to its own neutral defaults.
_PASSTHROUGH = (
    "down",
    "distance",
    "yardline_100",
    "quarter",
    "game_seconds_remaining",
    "score_differential",
    "posteam_timeouts",
    "formation",
    "personnel",
    "motion_type",
    "hash",
    "offense_team",
)


@lru_cache(maxsize=1)
def _get_artifact() -> dict:
    """Load the trained artifact once.

    ``load_model`` raises ``SystemExit`` when the model file is missing; we
    translate that into a clean 503 so the API stays up even before the ml
    pipeline has been run.
    """
    try:
        return load_model()
    except SystemExit as exc:  # model file not present
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not trained yet; run the ml pipeline.",
        ) from exc


def predict(game_state: dict) -> dict:
    """Map request fields onto predict_situation kwargs and score the play.

    Returns the ml dict: pass_prob, run_prob, lean, confidence, why.
    """
    artifact = _get_artifact()

    kw = {
        key: game_state[key]
        for key in _PASSTHROUGH
        if game_state.get(key) is not None
    }

    try:
        return predict_situation(artifact, **kw)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
