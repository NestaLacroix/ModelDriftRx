"""
DriftRx - Shared Test Config
This module defines reusable pytest fixtures for testing the DriftRx system. It includes:
- Sample models (healthy and degraded)
- Sample data (baseline and drifted)
- Sample labels
- Sample metrics
- Sample feature drift reports
- Sample drift reports
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
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

# Fixed seed so every test run produces identical data
RNG = np.random.default_rng(42)

# Model agnostic names for testing
FEATURE_NAMES = [
    "feature_0",
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
    "feature_8",
    "feature_9",
]


# Models

@pytest.fixture()
def fake_model() -> FakeModel:
    """Healthy model with good metrics."""
    return FakeModel(accuracy=0.95)

@pytest.fixture()
def degraded_model() -> FakeModel:
    """Model that has lost performance."""
    return FakeModel(accuracy=0.78, f1=0.72, precision=0.70, recall=0.74, loss=0.45)


# Data

@pytest.fixture()
def sample_feature_names() -> list[str]:
    """10 generic feature names matching the sample data columns."""
    return FEATURE_NAMES.copy()

@pytest.fixture()
def sample_baseline_data() -> np.ndarray:
    """1000 rows x 10 features, normal distribution, mean=50, std=10."""
    return RNG.normal(loc=50, scale=10, size=(1000, 10))

@pytest.fixture()
def sample_drifted_data() -> np.ndarray:
    """
    Same shape as baseline but with known drift:
        feature_0: mean shifted 50 -> 70 (severe drift)
        feature_1: mean shifted 50 -> 55 (mild drift)
        features 2-9: unchanged
    """
    data = RNG.normal(loc=50, scale=10, size=(1000, 10))
    data[:, 0] = RNG.normal(loc=70, scale=10, size=1000)
    data[:, 1] = RNG.normal(loc=55, scale=10, size=1000)
    return data

@pytest.fixture()
def sample_labels() -> np.ndarray:
    """1000 binary labels (0 or 1)."""
    return RNG.integers(0, 2, size=1000)


# Contracts

@pytest.fixture()
def sample_metrics() -> ModelMetrics:
    """Typical healthy model metrics."""
    return ModelMetrics(
        accuracy=0.95,
        f1=0.90,
        precision=0.88,
        recall=0.92,
        loss=0.1,
    )

@pytest.fixture()
def sample_feature_drift_none() -> FeatureDrift:
    """A feature with no drift."""
    return FeatureDrift(
        feature_name="feature_3",
        psi_score=0.03,
        ks_p_value=0.45,
        severity=DriftSeverity.NONE,
        baseline_mean=50.0,
        current_mean=51.0,
    )

@pytest.fixture()
def sample_feature_drift_low() -> FeatureDrift:
    """A feature with low drift."""
    return FeatureDrift(
        feature_name="feature_1",
        psi_score=0.15,
        ks_p_value=0.03,
        severity=DriftSeverity.LOW,
        baseline_mean=50.0,
        current_mean=55.0,
    )

@pytest.fixture()
def sample_feature_drift_severe() -> FeatureDrift:
    """A feature with severe drift."""
    return FeatureDrift(
        feature_name="feature_0",
        psi_score=0.42,
        ks_p_value=0.0001,
        severity=DriftSeverity.SEVERE,
        baseline_mean=50.0,
        current_mean=70.0,
    )

@pytest.fixture()
def sample_drift_report(
    sample_feature_drift_none: FeatureDrift,
    sample_feature_drift_low: FeatureDrift,
    sample_feature_drift_severe: FeatureDrift,
) -> DriftReport:
    """Report with 3 features: one NONE, one LOW, one SEVERE."""
    return DriftReport(
        timestamp=datetime(2026, 2, 11, 14, 30, 0),
        overall_severity=DriftSeverity.SEVERE,
        feature_drifts=[
            sample_feature_drift_none,
            sample_feature_drift_low,
            sample_feature_drift_severe,
        ],
        triggered_healing=True,
    )

@pytest.fixture()
def sample_diagnosis(
    sample_drift_report: DriftReport,
    sample_feature_drift_severe: FeatureDrift,
) -> DiagnosisResult:
    """Diagnosis pointing to feature_0 as the root cause."""
    return DiagnosisResult(
        drift_report=sample_drift_report,
        top_contributors=[sample_feature_drift_severe],
        estimated_accuracy_drop=0.15,
        root_cause_summary="feature_0 shifted +40%",
    )

@pytest.fixture()
def sample_healing_promote(sample_diagnosis: DiagnosisResult) -> HealingOutcome:
    """Healing where challenger wins and gets promoted."""
    return HealingOutcome(
        diagnosis=sample_diagnosis,
        champion_metrics=ModelMetrics(
            accuracy=0.78, f1=0.72, precision=0.70, recall=0.74, loss=0.45,
        ),
        challenger_metrics=ModelMetrics(
            accuracy=0.92, f1=0.89, precision=0.87, recall=0.91, loss=0.12,
        ),
        action=HealAction.PROMOTE,
        reason="challenger accuracy 0.92 > champion 0.78 + 0.02 threshold",
    )

@pytest.fixture()
def sample_healing_rollback(sample_diagnosis: DiagnosisResult) -> HealingOutcome:
    """Healing where challenger loses and champion is kept."""
    return HealingOutcome(
        diagnosis=sample_diagnosis,
        champion_metrics=ModelMetrics(
            accuracy=0.90, f1=0.87, precision=0.85, recall=0.89, loss=0.15,
        ),
        challenger_metrics=ModelMetrics(
            accuracy=0.85, f1=0.82, precision=0.80, recall=0.84, loss=0.22,
        ),
        action=HealAction.ROLLBACK,
        reason="challenger accuracy 0.85 < champion 0.90",
    )

@pytest.fixture()
def sample_incident(sample_healing_promote: HealingOutcome) -> IncidentReport:
    """Complete incident record."""
    return IncidentReport(
        healing_outcome=sample_healing_promote,
        timestamp=datetime(2026, 2, 11, 14, 31, 0),
        summary=(
            "Drift detected on feature 0 (shifted +40%)."
            "Accuracy degraded from 95% to 78%."
            "Challenger retrained, new accuracy 92%."
            "Model promoted."
        ),
    )
