"""Pydantic v2 schemas for the prediction + recommendation endpoints."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """A single pre-snap situation to score."""

    model_config = ConfigDict(extra="ignore")

    down: int = Field(..., ge=1, le=4, description="Current down (1-4).")
    distance: int = Field(..., ge=0, description="Yards to go for a first down.")
    yardline_100: int = Field(
        ..., ge=1, le=99, description="Yards from the opponent's goal line."
    )
    quarter: int = Field(2, ge=1, le=5, description="Quarter (5 = OT).")
    game_seconds_remaining: int = Field(
        1800, ge=0, le=3600, description="Seconds left in the game."
    )
    score_differential: int = Field(
        0, description="Offense score minus defense score."
    )
    formation: str | None = None
    personnel: str | None = None
    motion_type: str | None = None
    offense_team: str | None = None


class PredictResponse(BaseModel):
    """Calibrated run/pass lean for a situation."""

    pass_prob: float
    run_prob: float
    lean: str
    confidence: float
    why: str


class DefensiveCall(BaseModel):
    """A single ranked defensive recommendation."""

    front: str
    coverage: str
    pressure: str
    stunt: str
    confidence: float
    expected_epa_prevented: float
    rationale: str


class RecommendRequest(PredictRequest):
    """Recommendation requests carry the same situation fields as a predict."""


class RecommendResponse(BaseModel):
    """The prediction plus a ranked list of defensive calls."""

    predicted: PredictResponse
    recommendations: list[DefensiveCall]
