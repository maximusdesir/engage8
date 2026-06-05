"""Public tendency endpoint.

Serves the coach-facing "what do they like to do" tables, computed from the
plays stored in the database via the ml engage8 tendency engine.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.deps import get_db
from app.schemas.tendency import TendencyResponse
from app.services import tendencies as tendency_service

router = APIRouter(tags=["tendencies"])


@router.get("/tendencies", response_model=TendencyResponse)
def get_tendencies(
    team: str | None = None,
    split: str = "down_distance",
    db: Session = Depends(get_db),
) -> TendencyResponse:
    """Run/pass split + top play per bucket for the chosen dimension.

    Public endpoint (no auth). ``split`` selects the aggregation dimension;
    an unknown split returns a 400 listing the valid choices.
    """
    try:
        rows = tendency_service.get_tendencies(db, split=split, team=team)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return TendencyResponse(team=team, split=split, rows=rows)
