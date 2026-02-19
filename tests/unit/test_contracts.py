"""
DriftRx - Unit Tests for Core Data Contracts

These verify that every dataclass, enum, and the model interface
behave correctly.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from tests.mocks import FakeModel

from src.contracts import (
    DiagnosisResult,
    DriftReport,
    DriftSeverity,
    FeatureDrift,
    HealAction,
    HealingOutcome,
    IncidentReport,
    ModelMetrics,
)
from src.protocols import MonitorableModel

# Enums

class TestDriftSeverity:
    """Verify severity enum values and ordering."""

    def test_values(self) -> None:
        assert DriftSeverity.NONE.value == "none"
        assert DriftSeverity.LOW.value == "low"
        assert DriftSeverity.MODERATE.value == "moderate"
        assert DriftSeverity.SEVERE.value == "severe"

    def test_ordering(self) -> None:
        assert DriftSeverity.SEVERE > DriftSeverity.MODERATE
        assert DriftSeverity.MODERATE > DriftSeverity.LOW
        assert DriftSeverity.LOW > DriftSeverity.NONE
        assert DriftSeverity.NONE < DriftSeverity.SEVERE

    def test_ge_le(self) -> None:
        assert DriftSeverity.SEVERE >= DriftSeverity.SEVERE
        assert DriftSeverity.NONE <= DriftSeverity.NONE
        assert DriftSeverity.MODERATE >= DriftSeverity.LOW
        assert DriftSeverity.LOW <= DriftSeverity.MODERATE


class TestHealAction:
    """Verify heal action enum values."""

    def test_values(self) -> None:
        assert HealAction.PROMOTE.value == "promote"
        assert HealAction.ROLLBACK.value == "rollback"
        assert HealAction.NO_ACTION.value == "no_action"


# Model Metrics

class TestModelMetrics:
    """Verify metrics storage and serialization."""

    def test_creation(self, sample_metrics: ModelMetrics) -> None:
        assert sample_metrics.accuracy == 0.95
        assert sample_metrics.f1 == 0.90
        assert sample_metrics.precision == 0.88
        assert sample_metrics.recall == 0.92
        assert sample_metrics.loss == 0.1

    def test_extra_defaults_empty(self) -> None:
        m = ModelMetrics(accuracy=0.9, f1=0.8, precision=0.7, recall=0.6, loss=0.2)
        assert m.extra == {}

    def test_extra_field(self) -> None:
        m = ModelMetrics(
            accuracy=0.9, f1=0.8, precision=0.7,
            recall=0.6, loss=0.2, extra={"auc": 0.95},
        )
        assert m.extra["auc"] == 0.95

    def test_to_dict(self, sample_metrics: ModelMetrics) -> None:
        d = sample_metrics.to_dict()
        assert d["accuracy"] == 0.95
        assert d["f1"] == 0.90
        assert d["loss"] == 0.1
        assert "extra" not in d

    def test_to_dict_merges_extra(self) -> None:
        m = ModelMetrics(
            accuracy=0.9, f1=0.8, precision=0.7,
            recall=0.6, loss=0.2, extra={"auc": 0.95, "mcc": 0.88},
        )
        d = m.to_dict()
        assert d["auc"] == 0.95
        assert d["mcc"] == 0.88
        assert d["accuracy"] == 0.9

    def test_edge_values(self) -> None:
        m = ModelMetrics(accuracy=0.0, f1=0.0, precision=0.0, recall=0.0, loss=0.0)
        assert m.accuracy == 0.0
        m2 = ModelMetrics(accuracy=1.0, f1=1.0, precision=1.0, recall=1.0, loss=0.0)
        assert m2.accuracy == 1.0


# Feature Drift

class TestFeatureDrift:
    """Verify per-feature drift data and shift calculation."""

    def test_creation(self, sample_feature_drift_severe: FeatureDrift) -> None:
        assert sample_feature_drift_severe.feature_name == "feature_0"
        assert sample_feature_drift_severe.psi_score == 0.42
        assert sample_feature_drift_severe.severity == DriftSeverity.SEVERE

    def test_shift_percentage_positive(self) -> None:
        fd = FeatureDrift(
            feature_name="feature_0", psi_score=0.4, ks_p_value=0.001,
            severity=DriftSeverity.SEVERE, baseline_mean=50.0, current_mean=67.0,
        )
        assert fd.shift_percentage == 34.0

    def test_shift_percentage_negative(self) -> None:
        fd = FeatureDrift(
            feature_name="feature_1", psi_score=0.15, ks_p_value=0.02,
            severity=DriftSeverity.LOW, baseline_mean=100.0, current_mean=80.0,
        )
        assert fd.shift_percentage == -20.0

    def test_shift_percentage_zero_baseline(self) -> None:
        fd = FeatureDrift(
            feature_name="feature_2", psi_score=0.1, ks_p_value=0.05,
            severity=DriftSeverity.LOW, baseline_mean=0.0, current_mean=10.0,
        )
        assert fd.shift_percentage == 0.0

    def test_to_dict(self, sample_feature_drift_severe: FeatureDrift) -> None:
        d = sample_feature_drift_severe.to_dict()
        assert d["feature_name"] == "feature_0"
        assert d["severity"] == "severe"
        assert d["shift_percentage"] == 40.0
        assert d["psi_score"] == 0.42


# Drift Report

class TestDriftReport:
    """Verify report aggregation and filtering."""

    def test_overall_severity(self, sample_drift_report: DriftReport) -> None:
        assert sample_drift_report.overall_severity == DriftSeverity.SEVERE

    def test_drifted_features(self, sample_drift_report: DriftReport) -> None:
        drifted = sample_drift_report.drifted_features
        assert len(drifted) == 2
        names = [fd.feature_name for fd in drifted]
        assert "feature_0" in names
        assert "feature_1" in names
        assert "feature_3" not in names

    def test_top_drifted_features(self, sample_drift_report: DriftReport) -> None:
        top = sample_drift_report.top_drifted_features(1)
        assert len(top) == 1
        assert top[0].feature_name == "feature_0"

    def test_triggered_healing_true(self, sample_drift_report: DriftReport) -> None:
        assert sample_drift_report.triggered_healing is True

    def test_triggered_healing_false(
        self, sample_feature_drift_none: FeatureDrift,
    ) -> None:
        report = DriftReport(
            timestamp=datetime(2026, 2, 11, 14, 30, 0),
            overall_severity=DriftSeverity.NONE,
            feature_drifts=[sample_feature_drift_none],
            triggered_healing=False,
        )
        assert report.triggered_healing is False

    def test_empty_features(self) -> None:
        report = DriftReport(
            timestamp=datetime(2026, 2, 11, 14, 30, 0),
            overall_severity=DriftSeverity.NONE,
            feature_drifts=[],
            triggered_healing=False,
        )
        assert report.drifted_features == []
        assert report.top_drifted_features(5) == []

    def test_to_dict(self, sample_drift_report: DriftReport) -> None:
        d = sample_drift_report.to_dict()
        assert d["overall_severity"] == "severe"
        assert d["triggered_healing"] is True
        assert d["num_drifted_features"] == 2
        assert len(d["feature_drifts"]) == 3
        assert d["feature_drifts"][0]["feature_name"] == "feature_3"


# Diagnosis Result

class TestDiagnosisResult:
    """Verify diagnosis wraps drift report with root cause info."""

    def test_creation(self, sample_diagnosis: DiagnosisResult) -> None:
        assert sample_diagnosis.estimated_accuracy_drop == 0.15
        assert len(sample_diagnosis.top_contributors) == 1
        assert sample_diagnosis.top_contributors[0].feature_name == "feature_0"

    def test_root_cause_summary(self, sample_diagnosis: DiagnosisResult) -> None:
        assert "feature_0" in sample_diagnosis.root_cause_summary

    def test_to_dict(self, sample_diagnosis: DiagnosisResult) -> None:
        d = sample_diagnosis.to_dict()
        assert "drift_report" in d
        assert "top_contributors" in d
        assert d["estimated_accuracy_drop"] == 0.15
        assert d["root_cause_summary"] == sample_diagnosis.root_cause_summary


# Healing Outcome

class TestHealingOutcome:
    """Verify healing decisions and improvement calculation."""

    def test_promote(self, sample_healing_promote: HealingOutcome) -> None:
        assert sample_healing_promote.action == HealAction.PROMOTE
        assert sample_healing_promote.improvement > 0

    def test_rollback(self, sample_healing_rollback: HealingOutcome) -> None:
        assert sample_healing_rollback.action == HealAction.ROLLBACK
        assert sample_healing_rollback.improvement < 0

    def test_improvement_value(self, sample_healing_promote: HealingOutcome) -> None:
        expected = 0.92 - 0.78
        assert abs(sample_healing_promote.improvement - expected) < 1e-9

    def test_to_dict(self, sample_healing_promote: HealingOutcome) -> None:
        d = sample_healing_promote.to_dict()
        assert d["action"] == "promote"
        assert "improvement" in d
        assert "champion_metrics" in d
        assert "challenger_metrics" in d
        assert d["champion_metrics"]["accuracy"] == 0.78


# Incident Report

class TestIncidentReport:
    """Verify incident records and unique IDs."""

    def test_has_id(self, sample_incident: IncidentReport) -> None:
        assert sample_incident.id is not None
        assert len(sample_incident.id) > 0

    def test_unique_ids(self, sample_healing_promote: HealingOutcome) -> None:
        incident1 = IncidentReport(
            healing_outcome=sample_healing_promote,
            timestamp=datetime(2026, 2, 11, 14, 31, 0),
            summary="first",
        )
        incident2 = IncidentReport(
            healing_outcome=sample_healing_promote,
            timestamp=datetime(2026, 2, 11, 14, 32, 0),
            summary="second",
        )
        assert incident1.id != incident2.id

    def test_charts_default_empty(self, sample_incident: IncidentReport) -> None:
        assert sample_incident.charts == {}

    def test_to_dict(self, sample_incident: IncidentReport) -> None:
        d = sample_incident.to_dict()
        assert "id" in d
        assert "healing_outcome" in d
        assert "summary" in d
        assert "charts" in d
        assert d["summary"].startswith("Drift detected")


# Protocol

class TestMonitorableModel:
    """Verify the Protocol works with FakeModel."""

    def test_fake_model_satisfies_protocol(self) -> None:
        model = FakeModel()
        assert isinstance(model, MonitorableModel)

    def test_predict_returns_array(self, fake_model: FakeModel) -> None:
        x = np.ones((10, 5))
        predictions = fake_model.predict(x)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10

    def test_evaluate_returns_metrics(
        self, fake_model: FakeModel,
    ) -> None:
        x = np.ones((10, 5))
        y = np.ones(10)
        metrics = fake_model.evaluate(x, y)
        assert isinstance(metrics, ModelMetrics)
        assert metrics.accuracy == 0.95

    def test_retrain_returns_new_model(
        self, fake_model: FakeModel,
    ) -> None:
        x = np.ones((10, 5))
        y = np.ones(10)
        new_model = fake_model.retrain(x, y)
        assert new_model is not fake_model
        assert isinstance(new_model, MonitorableModel)
        assert new_model.accuracy == 1.0

    def test_retrain_does_not_mutate_original(
        self, fake_model: FakeModel,
    ) -> None:
        x = np.ones((10, 5))
        y = np.ones(10)
        fake_model.retrain(x, y)
        assert fake_model.accuracy == 0.95
