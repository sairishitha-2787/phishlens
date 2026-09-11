"""
THE API CONTRACT — build spec §06.

Anyone touching the frontend's api.js reads this file instead of guessing
field names. Response shapes here match the spec's example byte-for-byte.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer

Label = Literal["ai_phishing", "human_phishing", "legitimate"]


class _Timestamped(BaseModel):
    """
    created_at is always emitted as UTC with a trailing "Z", exactly like the
    spec example. SQLite returns naive datetimes and Postgres returns "+00:00";
    without this the frontend would see three different formats.
    """
    created_at: datetime

    @field_serializer("created_at")
    def _ser_created_at(self, dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------- POST /api/predict

class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw email / message text to classify.")


class Explanation(BaseModel):
    top_features: list[str] = Field(
        ..., description='Short, human-readable reasons — shown as "why this was flagged".'
    )


class PredictResponse(_Timestamped):
    id: UUID
    label: Label
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[Label, float]
    explanation: Explanation
    model_version: str
    truncated: bool

    model_config = {"protected_namespaces": ()}  # allow the `model_version` field name


# ---------------------------------------------------- GET /api/predictions[/id]

class PredictionSummary(_Timestamped):
    """One row of the history view. No input_text — the list stays light."""
    id: UUID
    label: Label
    confidence: float
    input_length: int
    truncated: bool
    model_version: str

    model_config = {"protected_namespaces": ()}


class PredictionDetail(PredictionSummary):
    """One past result, in full — same shape as PredictResponse plus the text."""
    input_text: str
    probabilities: dict[Label, float]


class PredictionPage(BaseModel):
    items: list[PredictionSummary]
    page: int
    page_size: int
    total: int


# ------------------------------------------------------- GET /api/health, /stats

class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    model_loaded: bool

    model_config = {"protected_namespaces": ()}


class StatsResponse(BaseModel):
    total: int
    by_label: dict[Label, int]
    model_version: str

    model_config = {"protected_namespaces": ()}
