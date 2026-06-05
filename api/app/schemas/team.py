"""Pydantic v2 schemas for teams, opponents, and plays."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TeamCreate(BaseModel):
    name: str
    level: str = "hs"
    season: int | None = None


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    level: str
    season: int | None = None


class OpponentCreate(BaseModel):
    name: str
    oc_name: str | None = None
    notes: str | None = None


class OpponentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    team_id: int
    name: str
    oc_name: str | None = None
    notes: str | None = None


class PlayCreate(BaseModel):
    """The chartable fields for a single play."""

    # Required situational fields.
    down: int
    ydstogo: int
    yardline_100: int
    play_type: str  # run|pass

    # Optional context.
    offense_team: str | None = None
    defense_team: str | None = None
    quarter: int | None = None
    game_seconds_remaining: int | None = None
    hash: str | None = None
    score_differential: int | None = None
    personnel: str | None = None
    formation: str | None = None
    motion_type: str | None = None
    play_family: str | None = None
    pass_concept: str | None = None
    run_concept: str | None = None
    result_yards: int | None = None


class PlayOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    down: int | None = None
    ydstogo: int | None = None
    yardline_100: int | None = None
    play_type: str | None = None
    offense_team: str | None = None
    defense_team: str | None = None
    quarter: int | None = None
    game_seconds_remaining: int | None = None
    hash: str | None = None
    score_differential: int | None = None
    personnel: str | None = None
    formation: str | None = None
    motion_type: str | None = None
    play_family: str | None = None
    pass_concept: str | None = None
    run_concept: str | None = None
    result_yards: int | None = None
