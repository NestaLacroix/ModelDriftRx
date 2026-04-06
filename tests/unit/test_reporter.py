"""
Phase 5 - Unit tests for Reporter.

Verifies:
  - Summary text content (mentions right features, right numbers, right decision)
  - Chart files are created and their paths are recorded in charts dict
  - No-drift case produces a clean summary
  - Rollback decision is reflected in text
  - Distribution-comparison chart is only generated when arrays are supplied
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.contracts import (
    DiagnosisResult,
    DriftReport,
    DriftSeverity,
    FeatureDrift,
    HealAction,
    HealingOutcome,
    ModelMetrics,
)
from src.reporter import Reporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outcome(
    action: HealAction = HealAction.PROMOTE,
    champion_acc: float = 0.78,
    challenger_acc: float = 0.92,
    drifted: bool = True,
) -> HealingOutcome:
    """Build a minimal but realistic HealingOutcome for testing."""
    from datetime import datetime

    if drifted:
        feature_drifts = [
            FeatureDrift(
                feature_name="feature_0",
                psi_score=0.42,
                ks_p_value=0.0001,
                severity=DriftSeverity.SEVERE,
                baseline_mean=50.0,
                current_mean=70.0,
            ),
            FeatureDrift(
                feature_name="feature_1",
                psi_score=0.15,
                ks_p_value=0.03,
                severity=DriftSeverity.LOW,
                baseline_mean=50.0,
                current_mean=55.0,
            ),
        ]
        overall = DriftSeverity.SEVERE
        top_contributors = [feature_drifts[0]]
        root_cause = "feature_0 shifted +40%"
    else:
        feature_drifts = [
            FeatureDrift(
                feature_name="feature_0",
                psi_score=0.02,
                ks_p_value=0.80,
                severity=DriftSeverity.NONE,
                baseline_mean=50.0,
                current_mean=50.5,
            ),
        ]
        overall = DriftSeverity.NONE
        top_contributors = []
        root_cause = ""

    drift_report = DriftReport(
        timestamp=datetime(2026, 2, 11, 14, 30, 0),
        overall_severity=overall,
        feature_drifts=feature_drifts,
        triggered_healing=drifted,
    )
    diagnosis = DiagnosisResult(
        drift_report=drift_report,
        top_contributors=top_contributors,
        estimated_accuracy_drop=0.15 if drifted else 0.0,
        root_cause_summary=root_cause,
    )
    reason = (
        f"Challenger accuracy {challenger_acc:.4f} >= champion {champion_acc:.4f} + 0.0200; promote."
        if action == HealAction.PROMOTE
        else f"Challenger accuracy {challenger_acc:.4f} is worse than champion {champion_acc:.4f}; rollback."
    )
    return HealingOutcome(
        diagnosis=diagnosis,
        champion_metrics=ModelMetrics(
            accuracy=champion_acc, f1=0.72, precision=0.70, recall=0.74, loss=0.45
        ),
        challenger_metrics=ModelMetrics(
            accuracy=challenger_acc, f1=0.89, precision=0.87, recall=0.91, loss=0.12
        ),
        action=action,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Summary text tests
# ---------------------------------------------------------------------------

def test_summary_mentions_drifted_feature(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    report = reporter.generate(outcome)

    assert "feature_0" in report.summary


def test_summary_mentions_psi(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    report = reporter.generate(outcome)

    # PSI value 0.42 should appear in the summary
    assert "0.4200" in report.summary or "0.42" in report.summary


def test_summary_mentions_promote_action(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome(action=HealAction.PROMOTE)
    report = reporter.generate(outcome)

    assert "PROMOTE" in report.summary.upper()


def test_summary_mentions_rollback_action(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome(
        action=HealAction.ROLLBACK,
        champion_acc=0.90,
        challenger_acc=0.85,
    )
    report = reporter.generate(outcome)

    assert "ROLLBACK" in report.summary.upper()


def test_summary_mentions_champion_accuracy(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome(champion_acc=0.78, challenger_acc=0.92)
    report = reporter.generate(outcome)

    assert "0.7800" in report.summary


def test_summary_mentions_challenger_accuracy(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome(champion_acc=0.78, challenger_acc=0.92)
    report = reporter.generate(outcome)

    assert "0.9200" in report.summary


def test_summary_no_drift_is_clean(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome(drifted=False)
    report = reporter.generate(outcome)

    lower = report.summary.lower()
    assert "no meaningful drift" in lower or "stable" in lower


def test_summary_mentions_root_cause(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    report = reporter.generate(outcome)

    assert "feature_0 shifted" in report.summary


def test_summary_mentions_accuracy_drop(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    report = reporter.generate(outcome)

    # estimated_accuracy_drop is 0.15 → 15.0%
    assert "15.0%" in report.summary


# ---------------------------------------------------------------------------
# Chart generation tests
# ---------------------------------------------------------------------------

def test_drift_bar_chart_created(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    report = reporter.generate(outcome)

    assert "drift_bar" in report.charts
    assert Path(report.charts["drift_bar"]).exists()


def test_champ_vs_chall_chart_created(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    report = reporter.generate(outcome)

    assert "champion_vs_challenger" in report.charts
    assert Path(report.charts["champion_vs_challenger"]).exists()


def test_dist_compare_chart_not_created_without_arrays(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    # No baseline / current supplied
    report = reporter.generate(outcome)

    assert "dist_compare" not in report.charts


def test_dist_compare_chart_created_with_arrays(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()

    rng = np.random.default_rng(0)
    baseline = rng.normal(50, 10, (200, 2))
    current = rng.normal(70, 10, (200, 2))
    feature_names = ["feature_0", "feature_1"]

    report = reporter.generate(outcome, baseline=baseline, current=current,
                                feature_names=feature_names)

    assert "dist_compare" in report.charts
    assert Path(report.charts["dist_compare"]).exists()


def test_no_dist_compare_when_top_contributors_empty(tmp_path):
    """If top_contributors is empty the dist_compare chart should be skipped."""
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome(drifted=False)

    rng = np.random.default_rng(1)
    baseline = rng.normal(50, 10, (100, 1))
    current = rng.normal(50, 10, (100, 1))

    report = reporter.generate(outcome, baseline=baseline, current=current,
                                feature_names=["feature_0"])

    # no top contributors → no dist_compare
    assert "dist_compare" not in report.charts


# ---------------------------------------------------------------------------
# IncidentReport structure
# ---------------------------------------------------------------------------

def test_incident_report_has_id(tmp_path):
    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    report = reporter.generate(outcome)

    assert report.id is not None
    assert len(report.id) > 0


def test_incident_report_timestamp_is_recent(tmp_path):
    from datetime import UTC, datetime

    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    before = datetime.now(tz=UTC)
    report = reporter.generate(outcome)
    after = datetime.now(tz=UTC)

    assert before <= report.timestamp <= after


def test_incident_report_to_dict_is_serializable(tmp_path):
    """to_dict() should return a plain dict with no non-serializable values."""
    import json

    reporter = Reporter(reports_dir=str(tmp_path))
    outcome = _make_outcome()
    report = reporter.generate(outcome)
    d = report.to_dict()

    # Should serialize without error
    json.dumps(d)
    assert isinstance(d["id"], str)
    assert isinstance(d["summary"], str)
    assert isinstance(d["charts"], dict)


# ---------------------------------------------------------------------------
# Conftest-fixture based smoke test
# ---------------------------------------------------------------------------

def test_generate_with_conftest_fixtures(tmp_path, sample_healing_promote):
    """Smoke test using the shared conftest fixtures."""
    reporter = Reporter(reports_dir=str(tmp_path))
    report = reporter.generate(sample_healing_promote)

    assert report.summary
    assert "drift_bar" in report.charts
    assert "champion_vs_challenger" in report.charts
