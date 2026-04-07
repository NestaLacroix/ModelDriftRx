"""
DriftRx Dashboard - Data layer.

Fetches data from the running FastAPI service when possible.
Falls back to deterministic synthetic data when the API is offline so the
dashboard is always runnable standalone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

_TIMEOUT = 3  # seconds per request


# ---------------------------------------------------------------------------
# Generic request helper
# ---------------------------------------------------------------------------


def _get(api_base: str, path: str) -> Any | None:
    """GET ``api_base + path``, return parsed JSON or None on any error."""
    try:
        r = requests.get(f"{api_base.rstrip('/')}{path}", timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def fetch_health(api_base: str) -> dict[str, Any]:
    data = _get(api_base, "/health")
    if data is not None:
        return data
    return {
        "status": "ok",
        "model_loaded": True,
        "baseline_loaded": True,
        "last_drift_check": (
            datetime.now(UTC) - timedelta(minutes=12)
        ).isoformat(),
        "incident_count": 5,
        "_synthetic": True,
    }


# ---------------------------------------------------------------------------
# /incidents
# ---------------------------------------------------------------------------

_SYNTH_ACTIONS = ["promote", "rollback", "no_action", "promote", "rollback"]
_SYNTH_INCIDENTS: list[dict[str, Any]] = [
    {
        "id": f"inc-{i:04d}",
        "timestamp": (
            datetime.now(UTC) - timedelta(hours=i * 7)
        ).isoformat(),
        "action": _SYNTH_ACTIONS[i],
        "summary": (
            f"Feature drift detected: transaction_amount PSI={0.32 + i * 0.04:.2f}. "
            f"account_age PSI={0.18 + i * 0.02:.2f}. "
            f"Challenger {'promoted' if _SYNTH_ACTIONS[i] == 'promote' else 'evaluated'} "
            f"after holdout comparison."
        ),
    }
    for i in range(5)
]


def fetch_incidents(api_base: str) -> tuple[list[dict[str, Any]], bool]:
    """Return ``(incidents, is_synthetic)``."""
    data = _get(api_base, "/incidents")
    if data is not None:
        return data, False
    return _SYNTH_INCIDENTS, True


def fetch_incident_detail(
    api_base: str, incident_id: str
) -> dict[str, Any] | None:
    return _get(api_base, f"/incidents/{incident_id}")


# ---------------------------------------------------------------------------
# Drift timeline (synthetic - no dedicated API endpoint yet)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "transaction_amount",
    "account_age",
    "num_transactions",
    "credit_score",
]


def build_drift_timeline() -> pd.DataFrame:
    """Synthetic PSI timeline across the last 12 checks (6-hour cadence)."""
    rng = np.random.default_rng(42)
    n = 12
    now = datetime.now(UTC)
    timestamps = [
        (now - timedelta(hours=(n - i) * 6)).strftime("%b %d %H:%M")
        for i in range(n)
    ]
    data: dict[str, Any] = {"timestamp": timestamps}
    for j, name in enumerate(FEATURE_NAMES):
        base = 0.02 + j * 0.012
        spike_at = n - 3
        values = []
        for i in range(n):
            if i >= spike_at and j < 2:
                v = base + (i - spike_at + 1) * (0.07 + j * 0.04) + rng.normal(0, 0.01)
            else:
                v = base + rng.normal(0, 0.007)
            values.append(float(max(v, 0.0)))
        data[name] = values
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Latest drift snapshot (synthetic)
# ---------------------------------------------------------------------------


def build_synthetic_drift_check() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "overall_severity": "severe",
        "triggered_healing": True,
        "feature_drifts": [
            {
                "feature_name": "transaction_amount",
                "psi_score": 0.38,
                "ks_p_value": 0.001,
                "severity": "severe",
                "baseline_mean": 250.0,
                "current_mean": 410.0,
                "shift_percentage": 64.0,
            },
            {
                "feature_name": "account_age",
                "psi_score": 0.18,
                "ks_p_value": 0.031,
                "severity": "moderate",
                "baseline_mean": 3.2,
                "current_mean": 2.5,
                "shift_percentage": -21.9,
            },
            {
                "feature_name": "num_transactions",
                "psi_score": 0.07,
                "ks_p_value": 0.21,
                "severity": "low",
                "baseline_mean": 12.1,
                "current_mean": 13.4,
                "shift_percentage": 10.7,
            },
            {
                "feature_name": "credit_score",
                "psi_score": 0.02,
                "ks_p_value": 0.89,
                "severity": "none",
                "baseline_mean": 682.0,
                "current_mean": 685.0,
                "shift_percentage": 0.4,
            },
        ],
        "_synthetic": True,
    }


# ---------------------------------------------------------------------------
# Champion vs challenger (synthetic)
# ---------------------------------------------------------------------------


def build_synthetic_champ_vs_chall() -> dict[str, Any]:
    return {
        "metric_names": ["accuracy", "precision", "recall", "f1_score"],
        "champion": [0.823, 0.811, 0.796, 0.803],
        "challenger": [0.851, 0.839, 0.844, 0.841],
        "action": "promote",
        "reason": (
            "Challenger exceeded champion by 2.8% accuracy on the holdout set, "
            "surpassing the minimum improvement threshold of 2.0%."
        ),
        "_synthetic": True,
    }
