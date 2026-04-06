"""
GET /incidents       - list all past incident reports (summary view).
GET /incidents/{id}  - retrieve a single incident in full detail.

Incidents are stored in AppState as plain dicts produced by
IncidentReport.to_dict().  The Reporter populates this store after every
healing cycle; the simulation script also writes here directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import IncidentDetail, IncidentSummary
from api.state import AppState, get_state

router = APIRouter()


@router.get("/incidents", response_model=list[IncidentSummary])
def list_incidents(state: AppState = Depends(get_state)) -> list[IncidentSummary]:
    """Return a summary of every recorded incident, newest-first."""
    result: list[IncidentSummary] = []
    for inc in reversed(state.incidents):
        result.append(
            IncidentSummary(
                id=inc["id"],
                timestamp=inc["timestamp"],
                action=inc["healing_outcome"]["action"],
                summary=inc["summary"],
            )
        )
    return result


@router.get("/incidents/{incident_id}", response_model=IncidentDetail)
def get_incident(
    incident_id: str,
    state: AppState = Depends(get_state),
) -> IncidentDetail:
    """Return full detail for one incident by its UUID."""
    for inc in state.incidents:
        if inc["id"] == incident_id:
            return IncidentDetail(
                id=inc["id"],
                timestamp=inc["timestamp"],
                summary=inc["summary"],
                charts=inc.get("charts", {}),
                healing_outcome=inc["healing_outcome"],
            )
    raise HTTPException(
        status_code=404,
        detail=f"Incident '{incident_id}' not found.",
    )

