"""Application settings.

Reads from environment variables (and a .env file if present). Defaults are
chosen so the API runs locally with zero configuration: SQLite on disk and the
trained model from the sibling ml/ project.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

API_ROOT = Path(__file__).resolve().parent.parent      # api/
PROJECT_ROOT = API_ROOT.parent                          # engage_eight/
ML_ROOT = PROJECT_ROOT / "ml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ENGAGE8_", extra="ignore")

    app_name: str = "Engage Eight API"
    debug: bool = True

    # SQLite by default; set ENGAGE8_DATABASE_URL to a Postgres URL in prod.
    database_url: str = f"sqlite:///{API_ROOT / 'engage8.db'}"

    # Auth
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # ML: where the trained run/pass artifact and the engage8 package live.
    ml_root: Path = ML_ROOT
    ml_artifact_path: Path = ML_ROOT / "artifacts" / "runpass_model.joblib"

    # CORS for the future Next.js frontend.
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


settings = Settings()
