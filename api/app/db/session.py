"""Database engine, session factory, and Base class."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ..config import settings

# check_same_thread is a SQLite-only flag; harmless to pass conditionally.
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create all tables. Called on app startup for the SQLite dev setup."""
    from . import models  # noqa: F401  (register models on Base.metadata)
    Base.metadata.create_all(bind=engine)
