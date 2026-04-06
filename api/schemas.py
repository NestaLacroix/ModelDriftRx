"""
DriftRx API — Pydantic request / response schemas.

All data entering or leaving the API is validated through these models.
Using Pydantic v2 syntax throughout.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description="2-D feature array: [[f1, f2, …], [f1, f2, …], …]",
        min_length=1,
    )


class PredictResponse(BaseModel):
    predictions: list[float]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str                     # "ok" | "degraded"
    model_loaded: bool
    baseline_loaded: bool
    last_drift_check: str | None    # ISO-8601 timestamp or null
    incident_count: int


# ---------------------------------------------------------------------------
# /check-drift
# ---------------------------------------------------------------------------


class DriftCheckRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description="Incoming data: [[f1, f2, …], …]",
        min_length=1,
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Column names.  Falls back to app.state.feature_names if omitted.",
    )


class FeatureDriftOut(BaseModel):
    feature_name: str
    psi_score: float
    ks_p_value: float
    severity: str           # "none" | "low" | "moderate" | "severe"
    baseline_mean: float
    current_mean: float
    shift_percentage: float


class DriftCheckResponse(BaseModel):
    timestamp: str          # ISO-8601
    overall_severity: str
    triggered_healing: bool
    feature_drifts: list[FeatureDriftOut]


# ---------------------------------------------------------------------------
# /incidents
# ---------------------------------------------------------------------------


class IncidentSummary(BaseModel):
    id: str
    timestamp: str          # ISO-8601
    action: str             # "promote" | "rollback" | "no_action"
    summary: str


class IncidentDetail(BaseModel):
    id: str
    timestamp: str
    summary: str
    charts: dict[str, str]
    healing_outcome: dict[str, Any]

