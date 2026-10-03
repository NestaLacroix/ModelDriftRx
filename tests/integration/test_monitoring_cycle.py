from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from src.contracts import DriftReport, DriftSeverity, FeatureDrift, HealAction
from src.pipeline import run_monitoring_cycle
from src.reporter import Reporter
from tests.mocks import FakeModel


class NoOpTracker:
    def log_healing(self, outcome):
        return None

    def log_incident(self, report):
        return None


def test_monitoring_cycle_runs_diagnosis_healing_and_reporting(tmp_path):
    rng = np.random.default_rng(4)
    baseline = rng.normal(50, 10, size=(80, 2))
    incoming = baseline + np.array([25.0, 2.0])
    labels = np.tile(np.array([0, 1]), 40)
    report = DriftReport(
        timestamp=datetime.now(UTC),
        overall_severity=DriftSeverity.SEVERE,
        feature_drifts=[
            FeatureDrift(
                feature_name="amount",
                psi_score=0.8,
                ks_p_value=0.001,
                severity=DriftSeverity.SEVERE,
                baseline_mean=float(baseline[:, 0].mean()),
                current_mean=float(incoming[:, 0].mean()),
            ),
            FeatureDrift(
                feature_name="age",
                psi_score=0.02,
                ks_p_value=0.9,
                severity=DriftSeverity.NONE,
                baseline_mean=float(baseline[:, 1].mean()),
                current_mean=float(incoming[:, 1].mean()),
            ),
        ],
        triggered_healing=True,
    )

    result = run_monitoring_cycle(
        drift_report=report,
        champion_model=FakeModel(accuracy=0.80, retrain_accuracy_boost=0.05),
        baseline=baseline,
        incoming=incoming,
        labels=labels,
        feature_names=["amount", "age"],
        reporter=Reporter(reports_dir=str(tmp_path)),
        tracker=NoOpTracker(),
    )

    assert result.diagnosis.root_cause_summary
    assert result.healing_outcome.action == HealAction.PROMOTE
    assert result.incident.healing_outcome.action == HealAction.PROMOTE
    assert result.incident.mlflow_run_id is None
    assert result.incident.charts
    assert result.champion_model is not None
