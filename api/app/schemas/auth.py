"""Pydantic v2 schemas for authentication: users and tokens."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    """Signup payload."""

    email: EmailStr
    password: str
    role: str = "coach"


class UserOut(BaseModel):
    """Public view of a user (never exposes the password hash)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    role: str


class Token(BaseModel):
    """Bearer access token returned on login."""

    access_token: str
    token_type: str = "bearer"
