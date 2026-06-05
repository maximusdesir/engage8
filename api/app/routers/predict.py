"""Prediction + defensive recommendation endpoints (public, no auth)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.deps import get_db
from app.db.models import Prediction
from app.schemas.prediction import (
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
)
from app.services import prediction_service, recommendation

router = APIRouter(tags=["prediction"])


def _persist(db: Session, game_state: dict, result: dict) -> None:
    """Best-effort logging of a prediction. A DB hiccup must never break the
    response, so everything is wrapped and rolled back on failure."""
    try:
        row = Prediction(
            user_id=None,
            game_state=json.dumps(game_state),
            result=json.dumps(result),
        )
        db.add(row)
        db.commit()
    except Exception:  # noqa: BLE001 - logging is non-critical
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    """Score a single pre-snap situation for run/pass lean."""
    game_state = req.model_dump()
    result = prediction_service.predict(game_state)
    _persist(db, game_state, result)
    return PredictResponse(**result)


@router.post("/recommend", response_model=RecommendResponse)
def recommend(
    req: RecommendRequest, db: Session = Depends(get_db)
) -> RecommendResponse:
    """Predict run/pass, then return ranked defensive calls for the situation."""
    game_state = req.model_dump()
    prediction = prediction_service.predict(game_state)
    calls = recommendation.recommend(prediction, game_state)

    payload = {"predicted": prediction, "recommendations": calls}
    _persist(db, game_state, payload)

    return RecommendResponse(
        predicted=PredictResponse(**prediction),
        recommendations=calls,
    )
