"""GET /api/health and GET /api/stats — build spec §06."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import MODEL_VERSION
from ..db import Prediction, get_db
from ..models import inference
from ..models.schemas import HealthResponse, StatsResponse

router = APIRouter(prefix="/api", tags=["health"])

LABELS = ("ai_phishing", "human_phishing", "legitimate")


@router.get("/health", response_model=HealthResponse)
def health():
    """Uptime ping. Hit ~1 min before a live demo to pre-warm Render (§09)."""
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        model_loaded=inference._predictor is not None,
    )


@router.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)):
    """Aggregate counts by class — powers the history view's breakdown."""
    rows = db.execute(
        select(Prediction.predicted_label, func.count())
        .group_by(Prediction.predicted_label)
    ).all()
    by_label = {lab: 0 for lab in LABELS}
    for label, n in rows:
        if label in by_label:
            by_label[label] = int(n)
    return StatsResponse(
        total=sum(by_label.values()), by_label=by_label, model_version=MODEL_VERSION,
    )
