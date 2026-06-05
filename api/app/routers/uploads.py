"""Public charting-upload endpoint.

Accepts a manually charted opponent CSV, runs it through the ml charting
loader, and persists the resulting canonical plays.
"""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.deps import get_db
from app.schemas.tendency import UploadSummary
from app.services import tendencies as tendency_service

router = APIRouter(tags=["uploads"])


@router.post("/uploads", response_model=UploadSummary)
def upload_charting(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadSummary:
    """Ingest a charting CSV into the plays table.

    Public endpoint (no auth). The uploaded file is written to a temp path,
    parsed/validated by the ml charting loader, and removed afterward. Parse
    or validation failures (including the loader's ``SystemExit``) become a
    400 with the underlying message.
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv"
        ) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        summary = tendency_service.ingest_charting_csv(db, tmp_path)
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not ingest charting file: {exc}",
        ) from exc
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)

    teams = summary["teams"]
    inserted = summary["inserted"]
    message = f"Inserted {inserted} plays for {len(teams)} team(s)."
    return UploadSummary(inserted=inserted, teams=teams, message=message)
