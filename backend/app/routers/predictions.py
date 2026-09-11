"""GET /api/predictions and /api/predictions/{id} — build spec §06."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import MAX_INPUT_CHARS
from ..db import Prediction, get_db
from ..models.schemas import PredictionDetail, PredictionPage, PredictionSummary

router = APIRouter(prefix="/api", tags=["predictions"])


def _summary(row: Prediction) -> PredictionSummary:
    return PredictionSummary(
        id=row.id,
        label=row.predicted_label,
        confidence=row.confidence,
        input_length=row.input_length,
        truncated=row.input_length > MAX_INPUT_CHARS,
        model_version=row.model_version,
        created_at=row.created_at,
    )


@router.get("/predictions", response_model=PredictionPage)
def list_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Paginated history for the results view — newest first."""
    total = db.scalar(select(func.count()).select_from(Prediction)) or 0
    rows = db.scalars(
        select(Prediction)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return PredictionPage(
        items=[_summary(r) for r in rows], page=page, page_size=page_size, total=total,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionDetail)
def get_prediction(prediction_id: UUID, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found.")
    return PredictionDetail(
        **_summary(row).model_dump(),
        input_text=row.input_text,
        probabilities=row.probabilities,
    )
