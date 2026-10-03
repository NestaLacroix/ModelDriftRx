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
# Drift history
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "transaction_amount",
    "account_age",
    "num_transactions",
    "credit_score",
]


def _synthetic_drift_timeline() -> pd.DataFrame:
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


def fetch_drift_history(api_base: str) -> tuple[list[dict[str, Any]], bool]:
    """Return recorded checks and whether synthetic fallback data is used."""
    data = _get(api_base, "/drift-history")
    if data is not None:
        return data, False
    return [], True


def build_drift_timeline(api_base: str | None = None) -> pd.DataFrame:
    """Build PSI history from the API, falling back only when it is offline."""
    if api_base is None:
        return _synthetic_drift_timeline()

    history, is_synthetic = fetch_drift_history(api_base)
    if is_synthetic:
        return _synthetic_drift_timeline()
    if not history:
        return pd.DataFrame(columns=["timestamp"])

    timestamps: list[str] = []
    feature_names: list[str] = []
    for entry in history:
        timestamps.append(entry["timestamp"])
        for feature in entry["feature_drifts"]:
            if feature["feature_name"] not in feature_names:
                feature_names.append(feature["feature_name"])

    rows: list[dict[str, Any]] = []
    for entry, timestamp in zip(history, timestamps):
        row: dict[str, Any] = {"timestamp": timestamp}
        row.update({name: np.nan for name in feature_names})
        row.update(
            {
                feature["feature_name"]: feature["psi_score"]
                for feature in entry["feature_drifts"]
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Latest drift snapshot
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


def fetch_latest_drift_check(api_base: str) -> dict[str, Any] | None:
    """Return the latest live drift report, or synthetic data if the API is offline."""
    history, is_synthetic = fetch_drift_history(api_base)
    if is_synthetic:
        return build_synthetic_drift_check()
    return history[0] if history else None


# ---------------------------------------------------------------------------
# Champion vs challenger
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


def fetch_champ_vs_chall(api_base: str) -> dict[str, Any] | None:
    """Return the newest real model comparison, or synthetic data when offline."""
    incidents, is_synthetic = fetch_incidents(api_base)
    if is_synthetic:
        return build_synthetic_champ_vs_chall()
    if not incidents:
        return None

    detail = fetch_incident_detail(api_base, incidents[0]["id"])
    if detail is None:
        return None
    outcome = detail.get("healing_outcome", {})
    champion = outcome.get("champion_metrics", {})
    challenger = outcome.get("challenger_metrics", {})
    names = ["accuracy", "precision", "recall", "f1"]
    return {
        "metric_names": names,
        "champion": [float(champion.get(name, 0.0)) for name in names],
        "challenger": [float(challenger.get(name, 0.0)) for name in names],
        "action": outcome.get("action", incidents[0].get("action", "no_action")),
        "reason": outcome.get("reason", "No decision reason was recorded."),
        "_synthetic": False,
    }
