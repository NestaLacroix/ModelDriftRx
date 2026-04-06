"""
POST /predict - run inference with the current champion model.

Accepts a 2-D feature array and returns the model's predictions.
Returns 503 when no model is loaded, 422 when the input shape is invalid.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import PredictRequest, PredictResponse
from api.state import AppState, get_state

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    state: AppState = Depends(get_state),
) -> PredictResponse:
    """
    Send a batch of feature vectors to the current champion model.
    Returns one prediction per row.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="No model is currently loaded.")

    try:
        features = np.array(request.features, dtype=float)
        if features.ndim != 2:  # pragma: no cover - numpy coercion edge case
            raise ValueError("features is not 2-D")
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"features must be a valid 2-D numeric array. ({exc})",
        ) from exc

    raw = state.model.predict(features)

    # Convert numpy scalar types to plain Python floats for JSON serialisation.
    return PredictResponse(predictions=[float(v) for v in raw])

