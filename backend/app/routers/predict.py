"""POST /api/predict — build spec §06, edge cases §11."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..config import MAX_INPUT_CHARS
from ..db import Prediction, get_db
from ..models.inference import Predictor, get_predictor
from ..models.schemas import Explanation, PredictRequest, PredictResponse

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(
    body: PredictRequest,
    db: Session = Depends(get_db),
    model: Predictor = Depends(get_predictor),
):
    # §11: empty / whitespace-only input → 400
    if not body.text or not body.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text is empty or whitespace-only.",
        )

    # §11: truncate over MAX_INPUT_CHARS, server-side, before inference
    original_len = len(body.text)
    truncated = original_len > MAX_INPUT_CHARS
    text = body.text[:MAX_INPUT_CHARS] if truncated else body.text

    result = model.predict(text)

    # §07: store post-truncation text only — never the un-truncated original
    row = Prediction(
        input_text=text,
        input_length=original_len,
        predicted_label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_version=model.version,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return PredictResponse(
        id=row.id,
        label=row.predicted_label,
        confidence=row.confidence,
        probabilities=row.probabilities,
        explanation=Explanation(top_features=result["top_features"]),
        model_version=row.model_version,
        truncated=truncated,
        created_at=row.created_at,
    )
