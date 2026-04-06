"""
POST /check-drift - run a drift check against the loaded baseline.

Accepts an incoming data batch, runs PSI + KS tests per feature, and
returns the full DriftReport as JSON.  Updates last_drift_check on the
app state so /health reflects when detection was last run.

Returns 503 when no baseline is loaded, 422 for shape mismatches.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import DriftCheckRequest, DriftCheckResponse, FeatureDriftOut
from api.state import AppState, get_state
from src.detector import DriftDetector

router = APIRouter()


@router.post("/check-drift", response_model=DriftCheckResponse)
def check_drift(
    request: DriftCheckRequest,
    state: AppState = Depends(get_state),
) -> DriftCheckResponse:
    """
    Compare uploaded data against the loaded baseline and return a drift report.

    Feature names are resolved in this priority order:
      1. ``request.feature_names`` (explicit, per-request)
      2. ``state.feature_names`` (pre-configured at startup)
      3. Auto-generated names: feature_0, feature_1, … (DriftDetector default)
    """
    if state.baseline is None:
        raise HTTPException(
            status_code=503,
            detail="No baseline data is loaded.  Configure app state before calling /check-drift.",
        )

    try:
        incoming = np.array(request.features, dtype=float)
        if incoming.ndim != 2:  # pragma: no cover
            raise ValueError("incoming is not 2-D")
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"features must be a valid 2-D numeric array. ({exc})",
        ) from exc

    n_expected = state.baseline.shape[1]
    if incoming.shape[1] != n_expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Feature dimension mismatch: baseline has {n_expected} features, "
                f"but incoming data has {incoming.shape[1]}."
            ),
        )

    feature_names = request.feature_names or state.feature_names
    detector = DriftDetector(state.baseline, feature_names=feature_names)
    report = detector.check(incoming)

    # Persist the check time so /health can reflect it
    state.last_drift_check = report.timestamp

    feature_drifts_out = [
        FeatureDriftOut(
            feature_name=fd.feature_name,
            psi_score=fd.psi_score,
            ks_p_value=fd.ks_p_value,
            severity=fd.severity.value,
            baseline_mean=fd.baseline_mean,
            current_mean=fd.current_mean,
            shift_percentage=fd.shift_percentage,
        )
        for fd in report.feature_drifts
    ]

    return DriftCheckResponse(
        timestamp=report.timestamp.isoformat(),
        overall_severity=report.overall_severity.value,
        triggered_healing=report.triggered_healing,
        feature_drifts=feature_drifts_out,
    )
