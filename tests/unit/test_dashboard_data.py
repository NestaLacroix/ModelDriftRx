from __future__ import annotations

import pandas as pd

from dashboard import data


def test_drift_timeline_uses_api_history(monkeypatch):
    history = [
        {
            "timestamp": "2026-09-22T12:00:00+00:00",
            "overall_severity": "severe",
            "triggered_healing": True,
            "healing_started": False,
            "healing_status": "labels_required",
            "feature_drifts": [
                {"feature_name": "amount", "psi_score": 0.8},
                {"feature_name": "age", "psi_score": 0.1},
            ],
        }
    ]
    monkeypatch.setattr(data, "_get", lambda api_base, path: history)

    timeline = data.build_drift_timeline("http://api")

    assert isinstance(timeline, pd.DataFrame)
    assert timeline.loc[0, "amount"] == 0.8
    assert timeline.loc[0, "age"] == 0.1


def test_drift_timeline_is_empty_for_online_api_without_history(monkeypatch):
    monkeypatch.setattr(data, "_get", lambda api_base, path: [])

    timeline = data.build_drift_timeline("http://api")

    assert timeline.empty
    assert list(timeline.columns) == ["timestamp"]


def test_latest_drift_returns_none_for_online_api_without_checks(monkeypatch):
    monkeypatch.setattr(data, "_get", lambda api_base, path: [])

    assert data.fetch_latest_drift_check("http://api") is None


def test_champion_comparison_uses_latest_incident(monkeypatch):
    incident = {"id": "incident-1", "action": "promote"}
    detail = {
        "healing_outcome": {
            "action": "promote",
            "reason": "challenger improved",
            "champion_metrics": {"accuracy": 0.7, "precision": 0.6, "recall": 0.5, "f1": 0.55},
            "challenger_metrics": {"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65},
        }
    }
    monkeypatch.setattr(data, "_get", lambda api_base, path: [incident] if path == "/incidents" else detail)

    result = data.fetch_champ_vs_chall("http://api")

    assert result is not None
    assert result["_synthetic"] is False
    assert result["champion"] == [0.7, 0.6, 0.5, 0.55]
    assert result["challenger"] == [0.8, 0.7, 0.6, 0.65]
