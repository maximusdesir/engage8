"""Play charting routes: list with simple filters, and insert a single play."""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Play, User
from app.deps import get_current_user, get_db
from app.schemas.team import PlayCreate, PlayOut

router = APIRouter(tags=["plays"])


@router.get("/plays", response_model=list[PlayOut])
def list_plays(
    offense_team: str | None = None,
    down: int | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Play]:
    stmt = select(Play)
    if offense_team is not None:
        stmt = stmt.where(Play.offense_team == offense_team)
    if down is not None:
        stmt = stmt.where(Play.down == down)
    stmt = stmt.limit(limit)
    return list(db.scalars(stmt))


@router.post("/plays", response_model=PlayOut, status_code=status.HTTP_201_CREATED)
def create_play(
    payload: PlayCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Play:
    play = Play(**payload.model_dump())
    db.add(play)
    db.commit()
    db.refresh(play)
    return play
