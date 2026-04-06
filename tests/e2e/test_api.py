"""
Phase 6 - End-to-end tests for the FastAPI service.

Every test uses FastAPI's TestClient (synchronous) with the `get_state`
dependency overridden to inject a fresh, isolated AppState.  No real ML
framework is required - FakeModel from tests/mocks.py is used throughout.

Test scope:
  GET  /health            - no model, with model, baseline flag
  POST /predict           - happy path, no model (503), bad shape (422)
  POST /check-drift       - no baseline (503), clean data, drifted data,
                            wrong feature count (422), custom feature names
  GET  /incidents         - empty store, with incidents (ordered newest-first)
  GET  /incidents/{id}    - found, not found (404)
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.mocks import FakeModel

from api.main import app
from api.state import AppState, get_state

# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def state() -> AppState:
    """Blank AppState for a single test."""
    return AppState()


@pytest.fixture()
def client(state: AppState) -> TestClient:
    """TestClient wired to a blank state (no model, no baseline)."""
    app.dependency_overrides[get_state] = lambda: state
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def loaded_state() -> AppState:
    """AppState pre-loaded with a FakeModel and baseline data."""
    s = AppState()
    s.model = FakeModel(accuracy=0.90)
    s.baseline = RNG.normal(50, 10, (200, 4)).astype(float)
    s.feature_names = ["f0", "f1", "f2", "f3"]
    return s


@pytest.fixture()
def loaded_client(loaded_state: AppState) -> TestClient:
    """TestClient with a FakeModel and baseline pre-loaded."""
    app.dependency_overrides[get_state] = lambda: loaded_state
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_incident_dict(
    incident_id: str | None = None,
    action: str = "promote",
    timestamp: str = "2026-02-11T14:31:00",
    summary: str = "Drift detected. Model promoted.",
) -> dict[str, Any]:
    """Build a minimal incident dict matching IncidentReport.to_dict() format."""
    return {
        "id": incident_id or str(uuid.uuid4()),
        "timestamp": timestamp,
        "summary": summary,
        "charts": {},
        "healing_outcome": {
            "diagnosis": {
                "drift_report": {
                    "timestamp": "2026-02-11T14:30:00",
                    "overall_severity": "severe",
                    "feature_drifts": [],
                    "triggered_healing": True,
                    "num_drifted_features": 1,
                },
                "top_contributors": [],
                "estimated_accuracy_drop": 0.12,
                "root_cause_summary": "f0 shifted +20%",
            },
            "champion_metrics": {
                "accuracy": 0.78, "f1": 0.72, "precision": 0.70,
                "recall": 0.74, "loss": 0.45,
            },
            "challenger_metrics": {
                "accuracy": 0.92, "f1": 0.89, "precision": 0.87,
                "recall": 0.91, "loss": 0.12,
            },
            "action": action,
            "reason": "challenger better",
            "improvement": 0.14,
        },
    }


# ===========================================================================
# GET /health
# ===========================================================================


class TestHealth:
    def test_health_ok_when_empty(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert body["baseline_loaded"] is False
        assert body["last_drift_check"] is None
        assert body["incident_count"] == 0

    def test_health_reflects_loaded_model(self, loaded_client: TestClient) -> None:
        resp = loaded_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["baseline_loaded"] is True

    def test_health_incident_count(self, state: AppState, client: TestClient) -> None:
        state.incidents.append(_make_incident_dict())
        state.incidents.append(_make_incident_dict())
        resp = client.get("/health")
        assert resp.json()["incident_count"] == 2

    def test_health_last_drift_check_after_check_drift(
        self, loaded_client: TestClient, loaded_state: AppState
    ) -> None:
        # Trigger a drift check
        payload = {"features": loaded_state.baseline[:5].tolist()}
        loaded_client.post("/check-drift", json=payload)

        resp = loaded_client.get("/health")
        assert resp.json()["last_drift_check"] is not None


# ===========================================================================
# POST /predict
# ===========================================================================


class TestPredict:
    def test_predict_returns_predictions(self, loaded_client: TestClient) -> None:
        payload = {"features": [[50.0, 50.0, 50.0, 50.0]] * 5}
        resp = loaded_client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 5

    def test_predict_single_row(self, loaded_client: TestClient) -> None:
        payload = {"features": [[1.0, 2.0, 3.0, 4.0]]}
        resp = loaded_client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 1

    def test_predict_503_when_no_model(self, client: TestClient) -> None:
        payload = {"features": [[1.0, 2.0, 3.0, 4.0]]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 503
        assert "No model" in resp.json()["detail"]

    def test_predict_422_on_empty_features(self, loaded_client: TestClient) -> None:
        # Pydantic min_length=1 validation catches this
        payload = {"features": []}
        resp = loaded_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_422_on_missing_body(self, loaded_client: TestClient) -> None:
        resp = loaded_client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_all_values_are_floats(self, loaded_client: TestClient) -> None:
        payload = {"features": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]}
        body = loaded_client.post("/predict", json=payload).json()
        for v in body["predictions"]:
            assert isinstance(v, (int, float))


# ===========================================================================
# POST /check-drift
# ===========================================================================


class TestCheckDrift:
    def test_drift_503_when_no_baseline(self, client: TestClient) -> None:
        payload = {"features": [[50.0, 50.0, 50.0, 50.0]]}
        resp = client.post("/check-drift", json=payload)
        assert resp.status_code == 503
        assert "baseline" in resp.json()["detail"].lower()

    def test_drift_returns_report_structure(self, loaded_client: TestClient, loaded_state: AppState) -> None:
        payload = {"features": loaded_state.baseline[:10].tolist()}
        resp = loaded_client.post("/check-drift", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "overall_severity" in body
        assert "triggered_healing" in body
        assert "feature_drifts" in body
        assert "timestamp" in body

    def test_drift_feature_count_matches(self, loaded_client: TestClient, loaded_state: AppState) -> None:
        payload = {"features": loaded_state.baseline[:10].tolist()}
        body = loaded_client.post("/check-drift", json=payload).json()
        assert len(body["feature_drifts"]) == 4  # 4 features in loaded_state

    def test_drift_feature_drift_fields(self, loaded_client: TestClient, loaded_state: AppState) -> None:
        payload = {"features": loaded_state.baseline[:20].tolist()}
        body = loaded_client.post("/check-drift", json=payload).json()
        first = body["feature_drifts"][0]
        assert "feature_name" in first
        assert "psi_score" in first
        assert "ks_p_value" in first
        assert "severity" in first
        assert "baseline_mean" in first
        assert "current_mean" in first
        assert "shift_percentage" in first

    def test_drift_detects_severe_shift(
        self, loaded_client: TestClient, loaded_state: AppState
    ) -> None:
        # Shift all features by a large amount - should cause high PSI
        drifted = loaded_state.baseline.copy()
        drifted[:, 0] += 100.0   # massive shift in feature f0
        payload = {"features": drifted.tolist()}
        body = loaded_client.post("/check-drift", json=payload).json()

        # At least one feature should be moderate or severe
        severities = [fd["severity"] for fd in body["feature_drifts"]]
        assert any(s in ("moderate", "severe") for s in severities)

    def test_drift_no_change_reports_no_drift(
        self, loaded_client: TestClient, loaded_state: AppState
    ) -> None:
        # Send baseline itself - should produce very low / no drift
        payload = {"features": loaded_state.baseline.tolist()}
        body = loaded_client.post("/check-drift", json=payload).json()
        assert body["overall_severity"] in ("none", "low")

    def test_drift_updates_last_drift_check(
        self, loaded_client: TestClient, loaded_state: AppState
    ) -> None:
        assert loaded_state.last_drift_check is None
        payload = {"features": loaded_state.baseline[:5].tolist()}
        loaded_client.post("/check-drift", json=payload)
        assert loaded_state.last_drift_check is not None

    def test_drift_422_on_wrong_feature_count(
        self, loaded_client: TestClient
    ) -> None:
        # 4 features expected; send 3
        payload = {"features": [[1.0, 2.0, 3.0]] * 5}
        resp = loaded_client.post("/check-drift", json=payload)
        assert resp.status_code == 422
        assert "mismatch" in resp.json()["detail"].lower()

    def test_drift_uses_custom_feature_names(
        self, loaded_client: TestClient, loaded_state: AppState
    ) -> None:
        custom_names = ["alpha", "beta", "gamma", "delta"]
        payload = {
            "features": loaded_state.baseline[:10].tolist(),
            "feature_names": custom_names,
        }
        body = loaded_client.post("/check-drift", json=payload).json()
        returned_names = [fd["feature_name"] for fd in body["feature_drifts"]]
        assert returned_names == custom_names

    def test_drift_falls_back_to_state_feature_names(
        self, loaded_client: TestClient, loaded_state: AppState
    ) -> None:
        # Do not pass feature_names in the request - should use state.feature_names
        payload = {"features": loaded_state.baseline[:10].tolist()}
        body = loaded_client.post("/check-drift", json=payload).json()
        returned_names = [fd["feature_name"] for fd in body["feature_drifts"]]
        assert returned_names == loaded_state.feature_names

    def test_drift_422_on_empty_features(self, loaded_client: TestClient) -> None:
        resp = loaded_client.post("/check-drift", json={"features": []})
        assert resp.status_code == 422


# ===========================================================================
# GET /incidents  &  GET /incidents/{id}
# ===========================================================================


class TestIncidents:
    def test_incidents_empty_returns_empty_list(self, client: TestClient) -> None:
        resp = client.get("/incidents")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_incidents_returns_all_entries(self, state: AppState, client: TestClient) -> None:
        state.incidents.append(_make_incident_dict(action="promote"))
        state.incidents.append(_make_incident_dict(action="rollback"))
        resp = client.get("/incidents")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_incidents_newest_first(self, state: AppState, client: TestClient) -> None:
        old = _make_incident_dict(timestamp="2026-01-01T00:00:00", summary="old")
        new = _make_incident_dict(timestamp="2026-03-01T00:00:00", summary="new")
        state.incidents.append(old)
        state.incidents.append(new)

        body = client.get("/incidents").json()
        # newest-first means the last-added item appears at index 0
        assert body[0]["summary"] == "new"
        assert body[1]["summary"] == "old"

    def test_incidents_summary_fields(self, state: AppState, client: TestClient) -> None:
        state.incidents.append(_make_incident_dict(action="promote", summary="test summary"))
        body = client.get("/incidents").json()
        inc = body[0]
        assert "id" in inc
        assert "timestamp" in inc
        assert "action" in inc
        assert "summary" in inc
        assert inc["action"] == "promote"
        assert inc["summary"] == "test summary"

    def test_get_incident_by_id(self, state: AppState, client: TestClient) -> None:
        known_id = "abc-123"
        state.incidents.append(_make_incident_dict(incident_id=known_id))
        resp = client.get(f"/incidents/{known_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == known_id

    def test_get_incident_detail_fields(self, state: AppState, client: TestClient) -> None:
        inc = _make_incident_dict(incident_id="xyz-789")
        state.incidents.append(inc)
        body = client.get("/incidents/xyz-789").json()
        assert "id" in body
        assert "timestamp" in body
        assert "summary" in body
        assert "charts" in body
        assert "healing_outcome" in body

    def test_get_incident_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.get("/incidents/does-not-exist")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_get_incident_healing_outcome_has_action(
        self, state: AppState, client: TestClient
    ) -> None:
        inc = _make_incident_dict(incident_id="ho-test", action="rollback")
        state.incidents.append(inc)
        body = client.get("/incidents/ho-test").json()
        assert body["healing_outcome"]["action"] == "rollback"

    def test_multiple_incidents_correct_id_returned(
        self, state: AppState, client: TestClient
    ) -> None:
        id_a, id_b = "aaa", "bbb"
        state.incidents.append(_make_incident_dict(incident_id=id_a, summary="A"))
        state.incidents.append(_make_incident_dict(incident_id=id_b, summary="B"))

        assert client.get(f"/incidents/{id_a}").json()["summary"] == "A"
        assert client.get(f"/incidents/{id_b}").json()["summary"] == "B"

