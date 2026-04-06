"""
GET /health - system health check.

Returns whether a model and baseline are loaded, the timestamp of the last
drift check, and the number of incidents recorded so far.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import HealthResponse
from api.state import AppState, get_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Return the current operational status of the DriftRx service."""
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        baseline_loaded=state.baseline is not None,
        last_drift_check=(
            state.last_drift_check.isoformat() if state.last_drift_check else None
        ),
        incident_count=len(state.incidents),
    )

